use std::{
    fs::{self, File},
    io::BufWriter,
    path::{Path, PathBuf},
    process::Command,
    time::Duration,
};

use futures_util::{StreamExt, TryStreamExt, stream};
use jiff::Timestamp;
use thiserror::Error;
use tokio::time::sleep;

use crate::{
    cache::{CacheError, GenerationBatch, GenerationCache},
    chunker::{ChunkError, chunk_markdown},
    discover::{DiscoveryError, discover_markdown_excluding},
    export::{ExportError, write_alpaca_jsonl, write_manifest},
    identity::sha256_hex,
    llm::{GeneratedQa, LlmClient, LlmError, generate_questions},
    progress::{GenerationSource, NoopProgress, ProgressEvent, ProgressReporter},
    prompt::{PromptError, PromptTemplate, build_qa_prompt, load_qa_prompt},
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
    #[error(transparent)]
    Prompt(#[from] PromptError),
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
    #[error("output I/O failed for {path}: {source}")]
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
    if config.generation.concurrency == 0 {
        return Err(BuildError::InvalidConfig(
            "concurrency must be greater than zero",
        ));
    }
    if config.generation.model.trim().is_empty() {
        return Err(BuildError::InvalidConfig("model must not be empty"));
    }
    if !config.generation.temperature.is_finite() {
        return Err(BuildError::InvalidConfig("temperature must be finite"));
    }
    let prompt_template = load_qa_prompt(&config.repository, config.prompt_file.as_deref())?;
    let (dataset_path, manifest_path) = resolve_output_paths(&config.export.output);
    create_parent_directory(&dataset_path)?;

    progress.report(ProgressEvent::DiscoveryStarted {
        repository: config.repository.clone(),
    });
    let prompt_files = prompt_template
        .source_path
        .iter()
        .cloned()
        .collect::<Vec<_>>();
    let discovery =
        discover_markdown_excluding(&config.repository, &config.discovery, &prompt_files)?;
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
    let generation_context = ChunkGenerationContext {
        total: chunks.len(),
        client,
        config,
        prompt_template: &prompt_template,
        cache: &cache,
        progress,
    };
    let mut batches = stream::iter(chunks.iter().enumerate())
        .map(|(index, chunk)| process_chunk(index, chunk, &generation_context))
        .buffer_unordered(config.generation.concurrency)
        .try_collect::<Vec<_>>()
        .await?;
    batches.sort_by_key(|(index, _)| *index);

    let mut generated_items = Vec::new();
    for (chunk_index, batch) in batches {
        let chunk = &chunks[chunk_index];
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
            version: prompt_template.version,
            template: prompt_template.template,
        },
        stats,
    };

    progress.report(ProgressEvent::WritingOutput {
        dataset_path: dataset_path.clone(),
        manifest_path: manifest_path.clone(),
    });
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

fn resolve_output_paths(output: &Path) -> (PathBuf, PathBuf) {
    let dataset_path = if output.is_dir() {
        output.join("dataset.jsonl")
    } else {
        output.to_path_buf()
    };
    let manifest_path = dataset_path
        .parent()
        .unwrap_or_else(|| Path::new(""))
        .join("manifest.json");
    (dataset_path, manifest_path)
}

struct ChunkGenerationContext<'a> {
    total: usize,
    client: &'a dyn LlmClient,
    config: &'a BuildConfig,
    prompt_template: &'a PromptTemplate,
    cache: &'a GenerationCache,
    progress: &'a dyn ProgressReporter,
}

async fn process_chunk(
    index: usize,
    chunk: &Chunk,
    context: &ChunkGenerationContext<'_>,
) -> Result<(usize, GenerationBatch), BuildError> {
    if context.config.cache.enabled
        && let Some(batch) = context.cache.load(
            chunk,
            &context.config.generation,
            &context.prompt_template.template,
        )?
    {
        report_chunk_started(
            index,
            context.total,
            chunk,
            GenerationSource::Cache,
            context.progress,
        );
        report_chunk_finished(
            index,
            chunk,
            GenerationSource::Cache,
            &batch,
            context.progress,
        );
        return Ok((index, batch));
    }

    report_chunk_started(
        index,
        context.total,
        chunk,
        GenerationSource::Llm,
        context.progress,
    );
    let batch = generate_batch_with_retry(
        index,
        chunk,
        context.client,
        context.config,
        &context.prompt_template.template,
        context.progress,
    )
    .await
    .map_err(|source| llm_build_error(chunk, source))?;

    // Store inside the chunk future so completed work survives a later failure or interruption.
    if context.config.cache.enabled {
        context.cache.store(
            chunk,
            &context.config.generation,
            &context.prompt_template.template,
            &batch,
        )?;
    }
    report_chunk_finished(
        index,
        chunk,
        GenerationSource::Llm,
        &batch,
        context.progress,
    );

    Ok((index, batch))
}

fn report_chunk_started(
    index: usize,
    total: usize,
    chunk: &Chunk,
    source: GenerationSource,
    progress: &dyn ProgressReporter,
) {
    progress.report(ProgressEvent::ChunkStarted {
        index: index + 1,
        total,
        path: chunk.path.clone(),
        start_line: chunk.start_line,
        end_line: chunk.end_line,
        source,
    });
}

fn report_chunk_finished(
    index: usize,
    chunk: &Chunk,
    source: GenerationSource,
    batch: &GenerationBatch,
    progress: &dyn ProgressReporter,
) {
    progress.report(ProgressEvent::ChunkFinished {
        index: index + 1,
        path: chunk.path.clone(),
        source,
        generated_items: batch.items.len(),
    });
}

async fn generate_batch_with_retry(
    index: usize,
    chunk: &Chunk,
    client: &dyn LlmClient,
    config: &BuildConfig,
    prompt_template: &str,
    progress: &dyn ProgressReporter,
) -> Result<GenerationBatch, LlmError> {
    let mut retries = 0;
    loop {
        match generate_batch(client, chunk, config, prompt_template).await {
            Ok(batch) => return Ok(batch),
            Err(error) if error.is_retryable() && retries < config.generation.max_retries => {
                retries += 1;
                let delay = retry_delay(retries);
                progress.report(ProgressEvent::RetryScheduled {
                    index: index + 1,
                    path: chunk.path.clone(),
                    attempt: retries,
                    max_retries: config.generation.max_retries,
                    delay_ms: delay.as_millis() as u64,
                });
                sleep(delay).await;
            }
            Err(error) => return Err(error),
        }
    }
}

fn retry_delay(attempt: usize) -> Duration {
    let exponent = attempt.saturating_sub(1).min(6) as u32;
    Duration::from_millis(500 * 2_u64.pow(exponent))
}

async fn generate_batch(
    client: &dyn LlmClient,
    chunk: &Chunk,
    config: &BuildConfig,
    prompt_template: &str,
) -> Result<GenerationBatch, LlmError> {
    let prompt = build_qa_prompt(
        prompt_template,
        chunk,
        config.generation.questions_per_chunk,
    );
    let items = generate_questions(client, &prompt, &config.generation).await?;

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
        time::Duration,
    };

    use async_trait::async_trait;
    use reqwest::StatusCode;
    use tokio::time::sleep;

    use super::{build_dataset, build_dataset_with_progress};
    use crate::{
        export::AlpacaRecord,
        llm::{LlmClient, LlmError, LlmRequestConfig},
        progress::{GenerationSource, ProgressEvent, ProgressReporter},
        types::{BuildConfig, CacheConfig, ExportConfig, GenerationConfig},
    };

    #[derive(Debug, Default)]
    struct FakeClient {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl LlmClient for FakeClient {
        async fn complete_structured(
            &self,
            _prompt: &str,
            _config: &LlmRequestConfig<'_>,
            _response_schema: &serde_json::Value,
        ) -> Result<String, LlmError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(serde_json::json!({
                "items": [{
                    "question": "What does HKB build?",
                    "answer": "It builds knowledge datasets.",
                    "tags": ["overview"],
                    "confidence": 0.9
                }]
            })
            .to_string())
        }
    }

    #[derive(Debug, Default)]
    struct PromptCapturingClient {
        prompts: Mutex<Vec<String>>,
    }

    #[async_trait]
    impl LlmClient for PromptCapturingClient {
        async fn complete_structured(
            &self,
            prompt: &str,
            _config: &LlmRequestConfig<'_>,
            _response_schema: &serde_json::Value,
        ) -> Result<String, LlmError> {
            match self.prompts.lock() {
                Ok(mut prompts) => prompts.push(prompt.to_owned()),
                Err(poisoned) => poisoned.into_inner().push(prompt.to_owned()),
            }
            Ok(serde_json::json!({
                "items": [{"question": "What is documented?", "answer": "Project knowledge."}]
            })
            .to_string())
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
        assert!(first_progress.events().iter().any(|event| matches!(
            event,
            ProgressEvent::ChunkStarted {
                source: GenerationSource::Llm,
                ..
            }
        )));
        assert!(second_progress.events().iter().any(|event| matches!(
            event,
            ProgressEvent::ChunkStarted {
                source: GenerationSource::Cache,
                ..
            }
        )));
        assert_eq!(first_dataset, second_dataset);
        assert!(first_dataset.contains("\"instruction\":\"What does HKB build?\""));
        assert_eq!(second.manifest.stats.generated_items, 1);
        assert!(second.manifest_path.is_file());
        Ok(())
    }

    #[tokio::test]
    async fn writes_default_filenames_when_output_is_a_directory()
    -> Result<(), Box<dyn std::error::Error>> {
        let repository = tempfile::tempdir()?;
        fs::write(
            repository.path().join("README.md"),
            "# HKB\nBuild datasets.",
        )?;
        let output_directory = repository.path().join("generated");
        fs::create_dir(&output_directory)?;
        let config = BuildConfig {
            repository: repository.path().to_path_buf(),
            export: ExportConfig {
                output: output_directory.clone(),
                ..ExportConfig::default()
            },
            ..BuildConfig::default()
        };

        let outcome = build_dataset(&FakeClient::default(), &config).await?;

        assert_eq!(outcome.dataset_path, output_directory.join("dataset.jsonl"));
        assert_eq!(
            outcome.manifest_path,
            output_directory.join("manifest.json")
        );
        assert!(outcome.dataset_path.is_file());
        assert!(outcome.manifest_path.is_file());
        Ok(())
    }

    #[tokio::test]
    async fn uses_and_records_a_project_specific_prompt() -> Result<(), Box<dyn std::error::Error>>
    {
        let repository = tempfile::tempdir()?;
        fs::write(
            repository.path().join("README.md"),
            "# HKB\nProject knowledge.",
        )?;
        let template = "Create one project question from {{path}}:\n{{chunk_text}}";
        fs::write(repository.path().join("hkb-prompt.md"), template)?;
        let config = BuildConfig {
            repository: repository.path().to_path_buf(),
            prompt_file: Some("hkb-prompt.md".into()),
            export: ExportConfig {
                output: repository.path().join("dataset.jsonl"),
                ..ExportConfig::default()
            },
            cache: CacheConfig {
                enabled: false,
                ..CacheConfig::default()
            },
            ..BuildConfig::default()
        };
        let client = PromptCapturingClient::default();

        let outcome = build_dataset(&client, &config).await?;

        let prompts = match client.prompts.lock() {
            Ok(prompts) => prompts,
            Err(poisoned) => poisoned.into_inner(),
        };
        assert_eq!(
            prompts.as_slice(),
            ["Create one project question from README.md:\n# HKB\nProject knowledge."]
        );
        assert_eq!(outcome.manifest.prompt.version, "custom");
        assert_eq!(outcome.manifest.prompt.template, template);
        assert_eq!(outcome.manifest.stats.processed_files, 1);
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

    #[derive(Debug, Default)]
    struct ParallelClient {
        active: AtomicUsize,
        peak_active: AtomicUsize,
    }

    #[async_trait]
    impl LlmClient for ParallelClient {
        async fn complete_structured(
            &self,
            prompt: &str,
            _config: &LlmRequestConfig<'_>,
            _response_schema: &serde_json::Value,
        ) -> Result<String, LlmError> {
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak_active.fetch_max(active, Ordering::SeqCst);

            let (section, delay_ms) = if prompt.contains("# First") {
                ("First", 80)
            } else if prompt.contains("# Second") {
                ("Second", 10)
            } else {
                ("Third", 20)
            };
            sleep(Duration::from_millis(delay_ms)).await;
            self.active.fetch_sub(1, Ordering::SeqCst);

            Ok(serde_json::json!({
                "items": [{
                    "question": format!("What is in the {section} section?"),
                    "answer": format!("The {section} section content.")
                }]
            })
            .to_string())
        }
    }

    #[tokio::test]
    async fn bounds_parallel_requests_and_preserves_source_order()
    -> Result<(), Box<dyn std::error::Error>> {
        let repository = tempfile::tempdir()?;
        fs::write(
            repository.path().join("README.md"),
            "# First\nFirst content.\n\n# Second\nSecond content.\n\n# Third\nThird content.",
        )?;
        let output = repository.path().join("dataset.jsonl");
        let config = BuildConfig {
            repository: repository.path().to_path_buf(),
            generation: GenerationConfig {
                concurrency: 2,
                ..GenerationConfig::default()
            },
            export: ExportConfig {
                output: output.clone(),
                ..ExportConfig::default()
            },
            cache: CacheConfig {
                enabled: false,
                ..CacheConfig::default()
            },
            ..BuildConfig::default()
        };
        let client = ParallelClient::default();

        build_dataset(&client, &config).await?;

        let records = fs::read_to_string(output)?
            .lines()
            .map(serde_json::from_str::<AlpacaRecord>)
            .collect::<Result<Vec<_>, _>>()?;
        let questions = records
            .iter()
            .map(|record| record.instruction.as_str())
            .collect::<Vec<_>>();
        assert_eq!(client.peak_active.load(Ordering::SeqCst), 2);
        assert_eq!(
            questions,
            [
                "What is in the First section?",
                "What is in the Second section?",
                "What is in the Third section?"
            ]
        );
        Ok(())
    }

    #[derive(Debug, Default)]
    struct FlakyClient {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl LlmClient for FlakyClient {
        async fn complete_structured(
            &self,
            _prompt: &str,
            _config: &LlmRequestConfig<'_>,
            _response_schema: &serde_json::Value,
        ) -> Result<String, LlmError> {
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                return Err(LlmError::UnexpectedSchema {
                    reason: "temporary malformed response".to_owned(),
                    excerpt: "not JSON".to_owned(),
                });
            }

            Ok(serde_json::json!({
                "items": [{"question": "What was retried?", "answer": "The LLM request."}]
            })
            .to_string())
        }
    }

    #[tokio::test]
    async fn retries_retryable_llm_errors() -> Result<(), Box<dyn std::error::Error>> {
        let repository = tempfile::tempdir()?;
        fs::write(repository.path().join("README.md"), "# Retry\nContent.")?;
        let config = BuildConfig {
            repository: repository.path().to_path_buf(),
            generation: GenerationConfig {
                max_retries: 1,
                ..GenerationConfig::default()
            },
            export: ExportConfig {
                output: repository.path().join("dataset.jsonl"),
                ..ExportConfig::default()
            },
            cache: CacheConfig {
                enabled: false,
                ..CacheConfig::default()
            },
            ..BuildConfig::default()
        };
        let client = FlakyClient::default();
        let progress = RecordingProgress::default();

        build_dataset_with_progress(&client, &config, &progress).await?;

        assert_eq!(client.calls.load(Ordering::SeqCst), 2);
        assert!(
            progress
                .events()
                .iter()
                .any(|event| matches!(event, ProgressEvent::RetryScheduled { attempt: 1, .. }))
        );
        Ok(())
    }

    #[derive(Debug, Default)]
    struct PartiallyFailingClient;

    #[async_trait]
    impl LlmClient for PartiallyFailingClient {
        async fn complete_structured(
            &self,
            prompt: &str,
            _config: &LlmRequestConfig<'_>,
            _response_schema: &serde_json::Value,
        ) -> Result<String, LlmError> {
            if prompt.contains("# Successful") {
                sleep(Duration::from_millis(10)).await;
                return Ok(serde_json::json!({
                    "items": [{"question": "Which chunk succeeded?", "answer": "The first chunk."}]
                })
                .to_string());
            }

            sleep(Duration::from_millis(50)).await;
            Err(LlmError::Api {
                status: StatusCode::UNAUTHORIZED,
                body: "test failure".to_owned(),
            })
        }
    }

    #[tokio::test]
    async fn keeps_completed_cache_entries_when_another_chunk_fails()
    -> Result<(), Box<dyn std::error::Error>> {
        let repository = tempfile::tempdir()?;
        fs::write(
            repository.path().join("README.md"),
            "# Successful\nThis chunk succeeds.\n\n# Failing\nThis chunk fails.",
        )?;
        let config = BuildConfig {
            repository: repository.path().to_path_buf(),
            generation: GenerationConfig {
                concurrency: 2,
                ..GenerationConfig::default()
            },
            export: ExportConfig {
                output: repository.path().join("dataset.jsonl"),
                ..ExportConfig::default()
            },
            ..BuildConfig::default()
        };

        let result = build_dataset(&PartiallyFailingClient, &config).await;

        assert!(matches!(result, Err(super::BuildError::Llm { .. })));
        let cache_entries =
            fs::read_dir(repository.path().join(".hkb"))?.collect::<Result<Vec<_>, _>>()?;
        assert_eq!(cache_entries.len(), 1);
        assert_eq!(
            cache_entries[0]
                .path()
                .extension()
                .and_then(|value| value.to_str()),
            Some("json")
        );
        Ok(())
    }
}
