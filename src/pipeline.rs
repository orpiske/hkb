use std::{
    fs::{self, File},
    io::BufWriter,
    path::{Path, PathBuf},
    process::Command,
};

use jiff::Timestamp;
use thiserror::Error;

use crate::{
    cache::{CacheError, GenerationBatch, GenerationCache},
    chunker::{ChunkError, chunk_markdown},
    discover::{DiscoveryError, discover_markdown},
    export::{ExportError, write_alpaca_jsonl, write_manifest},
    identity::sha256_hex,
    llm::{GeneratedQa, LlmClient, LlmError},
    progress::{NoopProgress, ProgressEvent, ProgressReporter},
    prompt::{PROMPT_TEMPLATE, PROMPT_VERSION, build_qa_prompt},
    qa::{normalize_question, validate_and_deduplicate},
    types::{
        BuildConfig, BuildManifest, BuildStats, Chunk, InputScope, PromptMetadata, QaItem,
        RepositoryMetadata, SourceRef,
    },
};

#[derive(Debug)]
pub struct BuildOutcome {
    pub dataset_path: PathBuf,
    pub manifest_path: PathBuf,
    pub manifest: BuildManifest,
}

#[derive(Debug, Error)]
pub enum BuildError {
    #[error("input scope {0:?} is not supported by the docs-only MVP")]
    UnsupportedInputScope(InputScope),
    #[error("invalid build configuration: {0}")]
    InvalidConfig(&'static str),
    #[error("no Markdown documents were discovered")]
    NoDocuments,
    #[error("the discovered Markdown documents contained no chunkable text")]
    NoChunks,
    #[error(transparent)]
    Discovery(#[from] DiscoveryError),
    #[error(transparent)]
    Chunk(#[from] ChunkError),
    #[error("LLM generation failed for {path}:{start_line}-{end_line}: {source}")]
    Llm {
        path: String,
        start_line: usize,
        end_line: usize,
        #[source]
        source: LlmError,
    },
    #[error(transparent)]
    Cache(#[from] CacheError),
    #[error(transparent)]
    Export(#[from] ExportError),
    #[error("output I/O failed for {path}")]
    OutputIo {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

pub async fn build_dataset(
    client: &dyn LlmClient,
    config: &BuildConfig,
) -> Result<BuildOutcome, BuildError> {
    build_dataset_with_progress(client, config, &NoopProgress).await
}

pub async fn build_dataset_with_progress(
    client: &dyn LlmClient,
    config: &BuildConfig,
    progress: &dyn ProgressReporter,
) -> Result<BuildOutcome, BuildError> {
    if config.include != InputScope::Docs {
        return Err(BuildError::UnsupportedInputScope(config.include));
    }
    if config.chunking.max_characters == 0 {
        return Err(BuildError::InvalidConfig(
            "max_characters must be greater than zero",
        ));
    }
    if config.generation.questions_per_chunk == 0 {
        return Err(BuildError::InvalidConfig(
            "questions_per_chunk must be greater than zero",
        ));
    }
    if config.generation.model.trim().is_empty() {
        return Err(BuildError::InvalidConfig("model must not be empty"));
    }
    if !config.generation.temperature.is_finite() {
        return Err(BuildError::InvalidConfig("temperature must be finite"));
    }

    progress.report(ProgressEvent::DiscoveryStarted {
        repository: config.repository.clone(),
    });
    let discovery = discover_markdown(&config.repository, &config.discovery)?;
    if discovery.documents.is_empty() {
        return Err(BuildError::NoDocuments);
    }
    progress.report(ProgressEvent::DiscoveryFinished {
        discovered_files: discovery.discovered_files,
        selected_files: discovery.documents.len(),
        skipped_files: discovery.skipped.len(),
    });

    let mut chunks = Vec::new();
    for document in &discovery.documents {
        let document_chunks = chunk_markdown(&document.path, &document.text, &config.chunking)?;
        progress.report(ProgressEvent::FileChunked {
            path: document.path.clone(),
            chunks: document_chunks.len(),
        });
        chunks.extend(document_chunks);
    }
    if chunks.is_empty() {
        return Err(BuildError::NoChunks);
    }

    progress.report(ProgressEvent::GenerationStarted {
        total_chunks: chunks.len(),
    });
    let cache = GenerationCache::new(&config.repository, &config.cache);
    let mut generated_items = Vec::new();
    for (chunk_index, chunk) in chunks.iter().enumerate() {
        progress.report(ProgressEvent::ChunkStarted {
            index: chunk_index + 1,
            total: chunks.len(),
            path: chunk.path.clone(),
            start_line: chunk.start_line,
            end_line: chunk.end_line,
        });
        let batch = if config.cache.enabled {
            match cache.load(chunk, &config.generation)? {
                Some(batch) => {
                    progress.report(ProgressEvent::CacheHit);
                    batch
                }
                None => {
                    progress.report(ProgressEvent::LlmRequestStarted);
                    let batch = generate_batch(client, chunk, config)
                        .await
                        .map_err(|source| llm_build_error(chunk, source))?;
                    cache.store(chunk, &config.generation, &batch)?;
                    batch
                }
            }
        } else {
            progress.report(ProgressEvent::LlmRequestStarted);
            generate_batch(client, chunk, config)
                .await
                .map_err(|source| llm_build_error(chunk, source))?
        };

        progress.report(ProgressEvent::ChunkFinished {
            generated_items: batch.items.len(),
        });
        generated_items.extend(to_qa_items(
            chunk,
            batch.items,
            &config.generation.model,
            &batch.generated_at,
        ));
    }

    let validation = validate_and_deduplicate(generated_items);
    progress.report(ProgressEvent::ValidationFinished {
        accepted_items: validation.accepted.len(),
        rejected_items: validation.rejected.len(),
        duplicate_items: validation.duplicate_items,
    });
    let stats = BuildStats {
        discovered_files: discovery.discovered_files,
        processed_files: discovery.documents.len(),
        skipped_files: discovery.skipped.len(),
        chunks: chunks.len(),
        generated_items: validation.accepted.len(),
        rejected_items: validation.rejected.len(),
        duplicate_items: validation.duplicate_items,
    };
    let generated_at = Timestamp::now().to_string();
    let manifest = BuildManifest {
        schema_version: "1".to_owned(),
        generated_at,
        repository: RepositoryMetadata {
            root: config.repository.clone(),
            commit: git_commit(&config.repository),
        },
        config: config.clone(),
        prompt: PromptMetadata {
            version: PROMPT_VERSION.to_owned(),
            template: PROMPT_TEMPLATE.to_owned(),
        },
        stats,
    };

    let dataset_path = config.export.output.clone();
    let manifest_path = dataset_path
        .parent()
        .unwrap_or_else(|| Path::new(""))
        .join("manifest.json");
    progress.report(ProgressEvent::WritingOutput {
        dataset_path: dataset_path.clone(),
        manifest_path: manifest_path.clone(),
    });
    create_parent_directory(&dataset_path)?;
    let dataset_file = File::create(&dataset_path).map_err(|source| BuildError::OutputIo {
        path: dataset_path.clone(),
        source,
    })?;
    write_alpaca_jsonl(BufWriter::new(dataset_file), &validation.accepted)?;

    let manifest_file = File::create(&manifest_path).map_err(|source| BuildError::OutputIo {
        path: manifest_path.clone(),
        source,
    })?;
    write_manifest(BufWriter::new(manifest_file), &manifest)?;
    progress.report(ProgressEvent::Finished);

    Ok(BuildOutcome {
        dataset_path,
        manifest_path,
        manifest,
    })
}

async fn generate_batch(
    client: &dyn LlmClient,
    chunk: &Chunk,
    config: &BuildConfig,
) -> Result<GenerationBatch, LlmError> {
    let prompt = build_qa_prompt(chunk, config.generation.questions_per_chunk);
    let items = client
        .generate_questions(&prompt, &config.generation)
        .await?;

    Ok(GenerationBatch {
        generated_at: Timestamp::now().to_string(),
        items,
    })
}

fn to_qa_items(
    chunk: &Chunk,
    items: Vec<GeneratedQa>,
    model: &str,
    generated_at: &str,
) -> Vec<QaItem> {
    items
        .into_iter()
        .map(|item| {
            let question = item.question.trim().to_owned();
            let answer = item.answer.trim().to_owned();
            let identity = format!("{}\0{}", chunk.chunk_id, normalize_question(&question));

            QaItem {
                id: sha256_hex(identity.as_bytes()),
                question,
                answer,
                source: SourceRef {
                    path: chunk.path.clone(),
                    start_line: chunk.start_line,
                    end_line: chunk.end_line,
                },
                chunk_id: chunk.chunk_id.clone(),
                model: model.to_owned(),
                generated_at: generated_at.to_owned(),
                tags: item
                    .tags
                    .into_iter()
                    .map(|tag| tag.trim().to_owned())
                    .filter(|tag| !tag.is_empty())
                    .collect(),
                confidence: item.confidence,
            }
        })
        .collect()
}

fn llm_build_error(chunk: &Chunk, source: LlmError) -> BuildError {
    BuildError::Llm {
        path: chunk.path.clone(),
        start_line: chunk.start_line,
        end_line: chunk.end_line,
        source,
    }
}

fn create_parent_directory(path: &Path) -> Result<(), BuildError> {
    let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    else {
        return Ok(());
    };

    fs::create_dir_all(parent).map_err(|source| BuildError::OutputIo {
        path: parent.to_path_buf(),
        source,
    })
}

fn git_commit(repository: &Path) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repository)
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;

    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        sync::{
            Mutex,
            atomic::{AtomicUsize, Ordering},
        },
    };

    use async_trait::async_trait;

    use super::{build_dataset, build_dataset_with_progress};
    use crate::{
        llm::{GeneratedQa, LlmClient, LlmError},
        progress::{ProgressEvent, ProgressReporter},
        types::{BuildConfig, ExportConfig, GenerationConfig},
    };

    #[derive(Debug, Default)]
    struct FakeClient {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl LlmClient for FakeClient {
        async fn generate_questions(
            &self,
            _prompt: &str,
            _config: &GenerationConfig,
        ) -> Result<Vec<GeneratedQa>, LlmError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(vec![GeneratedQa {
                question: "What does HKB build?".to_owned(),
                answer: "It builds knowledge datasets.".to_owned(),
                tags: vec!["overview".to_owned()],
                confidence: Some(0.9),
            }])
        }
    }

    #[derive(Debug, Default)]
    struct RecordingProgress {
        events: Mutex<Vec<ProgressEvent>>,
    }

    impl RecordingProgress {
        fn events(&self) -> Vec<ProgressEvent> {
            match self.events.lock() {
                Ok(events) => events.clone(),
                Err(poisoned) => poisoned.into_inner().clone(),
            }
        }
    }

    impl ProgressReporter for RecordingProgress {
        fn report(&self, event: ProgressEvent) {
            match self.events.lock() {
                Ok(mut events) => events.push(event),
                Err(poisoned) => poisoned.into_inner().push(event),
            }
        }
    }

    #[tokio::test]
    async fn builds_dataset_and_reuses_cached_generation() -> Result<(), Box<dyn std::error::Error>>
    {
        let repository = tempfile::tempdir()?;
        fs::write(
            repository.path().join("README.md"),
            "# HKB\nHKB builds knowledge datasets.",
        )?;
        let output = repository.path().join("out/dataset.jsonl");
        let config = BuildConfig {
            repository: repository.path().to_path_buf(),
            export: ExportConfig {
                output: output.clone(),
                ..ExportConfig::default()
            },
            ..BuildConfig::default()
        };
        let client = FakeClient::default();
        let first_progress = RecordingProgress::default();

        let first = build_dataset_with_progress(&client, &config, &first_progress).await?;
        let first_dataset = fs::read_to_string(&first.dataset_path)?;
        let second_progress = RecordingProgress::default();
        let second = build_dataset_with_progress(&client, &config, &second_progress).await?;
        let second_dataset = fs::read_to_string(&second.dataset_path)?;

        assert_eq!(client.calls.load(Ordering::SeqCst), 1);
        assert!(
            first_progress
                .events()
                .contains(&ProgressEvent::LlmRequestStarted)
        );
        assert!(second_progress.events().contains(&ProgressEvent::CacheHit));
        assert_eq!(first_dataset, second_dataset);
        assert!(first_dataset.contains("\"instruction\":\"What does HKB build?\""));
        assert_eq!(second.manifest.stats.generated_items, 1);
        assert!(second.manifest_path.is_file());
        Ok(())
    }

    #[tokio::test]
    async fn rejects_invalid_configuration_before_calling_llm() {
        let config = BuildConfig {
            generation: GenerationConfig {
                questions_per_chunk: 0,
                ..GenerationConfig::default()
            },
            ..BuildConfig::default()
        };
        let client = FakeClient::default();

        let result = build_dataset(&client, &config).await;

        assert!(matches!(result, Err(super::BuildError::InvalidConfig(_))));
        assert_eq!(client.calls.load(Ordering::SeqCst), 0);
    }
}
