use std::{
    collections::{HashMap, HashSet},
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    time::Duration,
};

use futures_util::{StreamExt, TryStreamExt, stream};
use jiff::Timestamp;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::time::sleep;

use crate::{
    chunker::{ChunkError, chunk_markdown},
    export::AlpacaRecord,
    identity::sha256_hex,
    llm::{LlmClient, LlmError, LlmRequestConfig, parse_json_completion},
    prompt::{PromptError, PromptTemplate, build_verification_prompt, load_verification_prompt},
    types::{BuildManifest, CacheConfig, Chunk, LlmProvider, PromptMetadata, QaItem, SourceRef},
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationCheck {
    pub passed: bool,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationChecks {
    pub grounded: VerificationCheck,
    pub self_contained: VerificationCheck,
    pub answer_relevant: VerificationCheck,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum VerificationResult {
    Accepted {
        checks: VerificationChecks,
        #[serde(skip_serializing_if = "Option::is_none")]
        evidence: Option<String>,
    },
    Rejected {
        checks: VerificationChecks,
        #[serde(skip_serializing_if = "Option::is_none")]
        evidence: Option<String>,
    },
    Unverifiable {
        reason: String,
    },
}

impl VerificationResult {
    pub fn verdict(&self) -> VerificationVerdict {
        match self {
            Self::Accepted { .. } => VerificationVerdict::Accepted,
            Self::Rejected { .. } => VerificationVerdict::Rejected,
            Self::Unverifiable { .. } => VerificationVerdict::Unverifiable,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationVerdict {
    Accepted,
    Rejected,
    Unverifiable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifierMetadata {
    pub provider: LlmProvider,
    pub model: String,
    pub verified_at: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationRecord {
    pub qa_id: String,
    pub question: String,
    pub answer: String,
    pub source: SourceRef,
    #[serde(flatten)]
    pub result: VerificationResult,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verifier: Option<VerifierMetadata>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationStats {
    pub total_items: usize,
    pub accepted_items: usize,
    pub rejected_items: usize,
    pub unverifiable_items: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationConfig {
    pub dataset: PathBuf,
    pub build_manifest: PathBuf,
    pub repository: PathBuf,
    pub output: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_file: Option<PathBuf>,
    pub verifier: VerifierConfig,
    pub cache: CacheConfig,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerifierConfig {
    pub provider: LlmProvider,
    pub model: String,
    pub endpoint: Option<String>,
    pub temperature: f32,
    pub concurrency: usize,
    pub max_retries: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationManifest {
    pub schema_version: String,
    pub verified_at: String,
    pub config: VerificationConfig,
    pub prompt: PromptMetadata,
    pub source_build: BuildManifest,
    pub stats: VerificationStats,
}

#[derive(Debug)]
pub struct VerificationOutcome {
    pub report_path: PathBuf,
    pub manifest_path: PathBuf,
    pub manifest: VerificationManifest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationSource {
    Cache,
    Llm,
    SourceUnavailable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerificationProgressEvent {
    Started {
        total_items: usize,
    },
    ItemStarted {
        index: usize,
        total: usize,
        qa_id: String,
        source: VerificationSource,
    },
    RetryScheduled {
        index: usize,
        qa_id: String,
        attempt: usize,
        max_retries: usize,
        delay_ms: u64,
    },
    ItemFinished {
        index: usize,
        qa_id: String,
        source: VerificationSource,
        verdict: VerificationVerdict,
    },
    WritingOutput {
        report_path: PathBuf,
        manifest_path: PathBuf,
    },
    Finished,
}

pub trait VerificationProgressReporter: Send + Sync {
    fn report(&self, event: VerificationProgressEvent);
}

#[derive(Debug, Default)]
pub struct NoopVerificationProgress;

impl VerificationProgressReporter for NoopVerificationProgress {
    fn report(&self, _event: VerificationProgressEvent) {}
}

#[derive(Debug, Error)]
pub enum VerificationError {
    #[error("failed to read {kind} file {path}")]
    ReadFile {
        kind: &'static str,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid build manifest JSON in {path}")]
    ManifestJson {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("invalid dataset JSON on line {line} of {path}")]
    DatasetJson {
        path: PathBuf,
        line: usize,
        #[source]
        source: serde_json::Error,
    },
    #[error("dataset contains an empty record on line {line} of {path}")]
    EmptyDatasetLine { path: PathBuf, line: usize },
    #[error("dataset contains no records: {0}")]
    EmptyDataset(PathBuf),
    #[error("invalid verification configuration: {0}")]
    InvalidConfig(&'static str),
    #[error(transparent)]
    Prompt(#[from] PromptError),
    #[error(transparent)]
    Chunk(#[from] ChunkError),
    #[error("verification LLM failed for Q&A {qa_id}: {source}")]
    Llm {
        qa_id: String,
        #[source]
        source: LlmError,
    },
    #[error(transparent)]
    Cache(#[from] VerificationCacheError),
    #[error("failed to create output directory {path}")]
    CreateOutputDirectory {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to create output file {path}")]
    CreateOutputFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to write verification output")]
    WriteOutput(#[from] std::io::Error),
    #[error("failed to serialize verification output")]
    OutputJson(#[from] serde_json::Error),
}

pub async fn verify_dataset(
    client: &dyn LlmClient,
    config: &VerificationConfig,
) -> Result<VerificationOutcome, VerificationError> {
    verify_dataset_with_progress(client, config, &NoopVerificationProgress).await
}

pub async fn verify_dataset_with_progress(
    client: &dyn LlmClient,
    config: &VerificationConfig,
    progress: &dyn VerificationProgressReporter,
) -> Result<VerificationOutcome, VerificationError> {
    validate_config(config)?;
    let source_build = read_build_manifest(&config.build_manifest)?;
    let items = read_dataset(&config.dataset)?;
    let prompt = load_verification_prompt(&config.repository, config.prompt_file.as_deref())?;
    let chunks = reconstruct_chunks(&config.repository, &source_build, &items)?;
    let cache = VerificationCache::new(&config.repository, &config.cache);

    progress.report(VerificationProgressEvent::Started {
        total_items: items.len(),
    });
    let context = VerificationContext {
        total: items.len(),
        client,
        config,
        prompt: &prompt,
        chunks: &chunks,
        cache: &cache,
        progress,
    };
    let mut records = stream::iter(items.iter().enumerate())
        .map(|(index, item)| verify_item(index, item, &context))
        .buffer_unordered(config.verifier.concurrency)
        .try_collect::<Vec<_>>()
        .await?;
    records.sort_by_key(|(index, _)| *index);
    let records = records
        .into_iter()
        .map(|(_, record)| record)
        .collect::<Vec<_>>();

    let stats = verification_stats(&records);
    let verified_at = Timestamp::now().to_string();
    let manifest = VerificationManifest {
        schema_version: "1".to_owned(),
        verified_at,
        config: config.clone(),
        prompt: PromptMetadata {
            version: prompt.version,
            template: prompt.template,
        },
        source_build,
        stats,
    };
    let report_path = config.output.clone();
    let manifest_path = report_path
        .parent()
        .unwrap_or_else(|| Path::new(""))
        .join("verification-manifest.json");
    progress.report(VerificationProgressEvent::WritingOutput {
        report_path: report_path.clone(),
        manifest_path: manifest_path.clone(),
    });
    create_parent_directory(&report_path)?;
    write_report(&report_path, &records)?;
    write_verification_manifest(&manifest_path, &manifest)?;
    progress.report(VerificationProgressEvent::Finished);

    Ok(VerificationOutcome {
        report_path,
        manifest_path,
        manifest,
    })
}

fn validate_config(config: &VerificationConfig) -> Result<(), VerificationError> {
    if config.verifier.concurrency == 0 {
        return Err(VerificationError::InvalidConfig(
            "concurrency must be greater than zero",
        ));
    }
    if config.verifier.model.trim().is_empty() {
        return Err(VerificationError::InvalidConfig("model must not be empty"));
    }
    if !config.verifier.temperature.is_finite() {
        return Err(VerificationError::InvalidConfig(
            "temperature must be finite",
        ));
    }
    Ok(())
}

fn read_build_manifest(path: &Path) -> Result<BuildManifest, VerificationError> {
    let bytes = fs::read(path).map_err(|source| VerificationError::ReadFile {
        kind: "build manifest",
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_slice(&bytes).map_err(|source| VerificationError::ManifestJson {
        path: path.to_path_buf(),
        source,
    })
}

fn read_dataset(path: &Path) -> Result<Vec<QaItem>, VerificationError> {
    let file = File::open(path).map_err(|source| VerificationError::ReadFile {
        kind: "dataset",
        path: path.to_path_buf(),
        source,
    })?;
    let mut items = Vec::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line_number = index + 1;
        let line = line.map_err(|source| VerificationError::ReadFile {
            kind: "dataset",
            path: path.to_path_buf(),
            source,
        })?;
        if line.trim().is_empty() {
            return Err(VerificationError::EmptyDatasetLine {
                path: path.to_path_buf(),
                line: line_number,
            });
        }
        let record: AlpacaRecord =
            serde_json::from_str(&line).map_err(|source| VerificationError::DatasetJson {
                path: path.to_path_buf(),
                line: line_number,
                source,
            })?;
        items.push(record.into());
    }
    if items.is_empty() {
        return Err(VerificationError::EmptyDataset(path.to_path_buf()));
    }
    Ok(items)
}

fn reconstruct_chunks(
    repository: &Path,
    source_build: &BuildManifest,
    items: &[QaItem],
) -> Result<HashMap<String, Chunk>, VerificationError> {
    let repository = match fs::canonicalize(repository) {
        Ok(repository) => repository,
        Err(_) => return Ok(HashMap::new()),
    };
    let source_paths = items
        .iter()
        .map(|item| item.source.path.as_str())
        .collect::<HashSet<_>>();
    let mut chunks = HashMap::new();
    for source_path in source_paths {
        let Some(path) = repository_source_path(&repository, source_path) else {
            continue;
        };
        let Ok(metadata) = fs::metadata(&path) else {
            continue;
        };
        if metadata.len() > source_build.config.discovery.max_bytes_per_file {
            continue;
        }
        let Ok(bytes) = fs::read(path) else {
            continue;
        };
        let Ok(text) = String::from_utf8(bytes) else {
            continue;
        };
        let text = text.replace("\r\n", "\n").replace('\r', "\n");
        for chunk in chunk_markdown(Path::new(source_path), &text, &source_build.config.chunking)? {
            chunks.insert(chunk.chunk_id.clone(), chunk);
        }
    }
    Ok(chunks)
}

fn repository_source_path(repository: &Path, source_path: &str) -> Option<PathBuf> {
    let source_path = Path::new(source_path);
    if source_path.is_absolute() {
        return None;
    }
    fs::canonicalize(repository.join(source_path))
        .ok()
        .filter(|path| path.starts_with(repository) && path.is_file())
}

struct VerificationContext<'a> {
    total: usize,
    client: &'a dyn LlmClient,
    config: &'a VerificationConfig,
    prompt: &'a PromptTemplate,
    chunks: &'a HashMap<String, Chunk>,
    cache: &'a VerificationCache,
    progress: &'a dyn VerificationProgressReporter,
}

async fn verify_item(
    index: usize,
    item: &QaItem,
    context: &VerificationContext<'_>,
) -> Result<(usize, VerificationRecord), VerificationError> {
    let Some(chunk) = context.chunks.get(&item.chunk_id) else {
        let result = VerificationResult::Unverifiable {
            reason: "the original source chunk could not be reconstructed".to_owned(),
        };
        report_started(index, item, VerificationSource::SourceUnavailable, context);
        report_finished(
            index,
            item,
            VerificationSource::SourceUnavailable,
            result.verdict(),
            context,
        );
        return Ok((index, verification_record(item, result, None)));
    };
    if item.source.path != chunk.path
        || item.source.start_line != chunk.start_line
        || item.source.end_line != chunk.end_line
    {
        let result = VerificationResult::Unverifiable {
            reason: "dataset provenance does not match the reconstructed source chunk".to_owned(),
        };
        report_started(index, item, VerificationSource::SourceUnavailable, context);
        report_finished(
            index,
            item,
            VerificationSource::SourceUnavailable,
            result.verdict(),
            context,
        );
        return Ok((index, verification_record(item, result, None)));
    }

    if context.config.cache.enabled
        && let Some(cached) = context.cache.load(
            item,
            chunk,
            &context.config.verifier,
            &context.prompt.template,
        )?
    {
        let result = cached.result();
        let verifier = cached.verifier(&context.config.verifier);
        report_started(index, item, VerificationSource::Cache, context);
        report_finished(
            index,
            item,
            VerificationSource::Cache,
            result.verdict(),
            context,
        );
        return Ok((index, verification_record(item, result, Some(verifier))));
    }

    report_started(index, item, VerificationSource::Llm, context);
    let generated = verify_with_retry(index, item, chunk, context).await?;
    let cached = CachedVerification {
        checks: generated.checks,
        evidence: generated.evidence,
        verified_at: Timestamp::now().to_string(),
    };
    if context.config.cache.enabled {
        context.cache.store(
            item,
            chunk,
            &context.config.verifier,
            &context.prompt.template,
            &cached,
        )?;
    }
    let result = cached.result();
    let verifier = cached.verifier(&context.config.verifier);
    report_finished(
        index,
        item,
        VerificationSource::Llm,
        result.verdict(),
        context,
    );
    Ok((index, verification_record(item, result, Some(verifier))))
}

fn report_started(
    index: usize,
    item: &QaItem,
    source: VerificationSource,
    context: &VerificationContext<'_>,
) {
    context
        .progress
        .report(VerificationProgressEvent::ItemStarted {
            index: index + 1,
            total: context.total,
            qa_id: item.id.clone(),
            source,
        });
}

fn report_finished(
    index: usize,
    item: &QaItem,
    source: VerificationSource,
    verdict: VerificationVerdict,
    context: &VerificationContext<'_>,
) {
    context
        .progress
        .report(VerificationProgressEvent::ItemFinished {
            index: index + 1,
            qa_id: item.id.clone(),
            source,
            verdict,
        });
}

async fn verify_with_retry(
    index: usize,
    item: &QaItem,
    chunk: &Chunk,
    context: &VerificationContext<'_>,
) -> Result<GeneratedVerification, VerificationError> {
    let mut retries = 0;
    loop {
        match verify_once(context.client, item, chunk, context.config, context.prompt).await {
            Ok(generated) => return Ok(generated),
            Err(error) if error.is_retryable() && retries < context.config.verifier.max_retries => {
                retries += 1;
                let delay = retry_delay(retries);
                context
                    .progress
                    .report(VerificationProgressEvent::RetryScheduled {
                        index: index + 1,
                        qa_id: item.id.clone(),
                        attempt: retries,
                        max_retries: context.config.verifier.max_retries,
                        delay_ms: delay.as_millis() as u64,
                    });
                sleep(delay).await;
            }
            Err(source) => {
                return Err(VerificationError::Llm {
                    qa_id: item.id.clone(),
                    source,
                });
            }
        }
    }
}

async fn verify_once(
    client: &dyn LlmClient,
    item: &QaItem,
    chunk: &Chunk,
    config: &VerificationConfig,
    prompt: &PromptTemplate,
) -> Result<GeneratedVerification, LlmError> {
    let rendered = build_verification_prompt(&prompt.template, &item.question, &item.answer, chunk);
    let completion = client
        .complete_structured(
            &rendered,
            &LlmRequestConfig {
                model: &config.verifier.model,
                temperature: config.verifier.temperature,
            },
            &verification_schema(),
        )
        .await?;
    let generated: GeneratedVerification =
        serde_json::from_value(parse_json_completion(&completion)?).map_err(|source| {
            LlmError::UnexpectedSchema {
                reason: source.to_string(),
                excerpt: completion.chars().take(2_000).collect(),
            }
        })?;
    generated.validate()?;
    Ok(generated)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct GeneratedVerification {
    #[serde(flatten)]
    checks: VerificationChecks,
    #[serde(default)]
    evidence: Option<String>,
}

impl GeneratedVerification {
    fn validate(&self) -> Result<(), LlmError> {
        for (name, check) in [
            ("grounded", &self.checks.grounded),
            ("self_contained", &self.checks.self_contained),
            ("answer_relevant", &self.checks.answer_relevant),
        ] {
            if check.reason.trim().is_empty() {
                return Err(LlmError::UnexpectedSchema {
                    reason: format!("{name}.reason must not be empty"),
                    excerpt: String::new(),
                });
            }
        }
        if self
            .evidence
            .as_deref()
            .is_some_and(|evidence| evidence.trim().is_empty())
        {
            return Err(LlmError::UnexpectedSchema {
                reason: "evidence must be non-empty when present".to_owned(),
                excerpt: String::new(),
            });
        }
        Ok(())
    }
}

fn verification_schema() -> serde_json::Value {
    let check = serde_json::json!({
        "type": "object",
        "properties": {
            "passed": { "type": "boolean" },
            "reason": { "type": "string", "minLength": 1 }
        },
        "required": ["passed", "reason"],
        "additionalProperties": false
    });
    serde_json::json!({
        "type": "object",
        "properties": {
            "grounded": check,
            "self_contained": check,
            "answer_relevant": check,
            "evidence": {
                "anyOf": [
                    { "type": "string", "minLength": 1 },
                    { "type": "null" }
                ]
            }
        },
        "required": ["grounded", "self_contained", "answer_relevant", "evidence"],
        "additionalProperties": false
    })
}

fn verification_record(
    item: &QaItem,
    result: VerificationResult,
    verifier: Option<VerifierMetadata>,
) -> VerificationRecord {
    VerificationRecord {
        qa_id: item.id.clone(),
        question: item.question.clone(),
        answer: item.answer.clone(),
        source: item.source.clone(),
        result,
        verifier,
    }
}

fn verification_stats(records: &[VerificationRecord]) -> VerificationStats {
    let mut stats = VerificationStats {
        total_items: records.len(),
        ..VerificationStats::default()
    };
    for record in records {
        match record.result.verdict() {
            VerificationVerdict::Accepted => stats.accepted_items += 1,
            VerificationVerdict::Rejected => stats.rejected_items += 1,
            VerificationVerdict::Unverifiable => stats.unverifiable_items += 1,
        }
    }
    stats
}

fn write_report(path: &Path, records: &[VerificationRecord]) -> Result<(), VerificationError> {
    let file = File::create(path).map_err(|source| VerificationError::CreateOutputFile {
        path: path.to_path_buf(),
        source,
    })?;
    let mut writer = BufWriter::new(file);
    for record in records {
        serde_json::to_writer(&mut writer, record)?;
        writer.write_all(b"\n")?;
    }
    Ok(())
}

fn write_verification_manifest(
    path: &Path,
    manifest: &VerificationManifest,
) -> Result<(), VerificationError> {
    let file = File::create(path).map_err(|source| VerificationError::CreateOutputFile {
        path: path.to_path_buf(),
        source,
    })?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, manifest)?;
    writer.write_all(b"\n")?;
    Ok(())
}

fn create_parent_directory(path: &Path) -> Result<(), VerificationError> {
    let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    else {
        return Ok(());
    };
    fs::create_dir_all(parent).map_err(|source| VerificationError::CreateOutputDirectory {
        path: parent.to_path_buf(),
        source,
    })
}

fn retry_delay(attempt: usize) -> Duration {
    let exponent = attempt.saturating_sub(1).min(6) as u32;
    Duration::from_millis(500 * 2_u64.pow(exponent))
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct CachedVerification {
    checks: VerificationChecks,
    evidence: Option<String>,
    verified_at: String,
}

impl CachedVerification {
    fn result(&self) -> VerificationResult {
        let checks = self.checks.clone();
        let evidence = self.evidence.clone();
        if checks.grounded.passed && checks.self_contained.passed && checks.answer_relevant.passed {
            VerificationResult::Accepted { checks, evidence }
        } else {
            VerificationResult::Rejected { checks, evidence }
        }
    }

    fn verifier(&self, config: &VerifierConfig) -> VerifierMetadata {
        VerifierMetadata {
            provider: config.provider,
            model: config.model.clone(),
            verified_at: self.verified_at.clone(),
        }
    }
}

#[derive(Debug, Clone)]
struct VerificationCache {
    directory: PathBuf,
}

impl VerificationCache {
    fn new(repository: &Path, config: &CacheConfig) -> Self {
        let base = if config.directory.is_absolute() {
            config.directory.clone()
        } else {
            repository.join(&config.directory)
        };
        Self {
            directory: base.join("verify"),
        }
    }

    fn load(
        &self,
        item: &QaItem,
        chunk: &Chunk,
        config: &VerifierConfig,
        prompt: &str,
    ) -> Result<Option<CachedVerification>, VerificationCacheError> {
        let path = self.entry_path(item, chunk, config, prompt)?;
        match fs::read(&path) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map(Some)
                .map_err(|source| VerificationCacheError::Json { path, source }),
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(source) => Err(VerificationCacheError::Io { path, source }),
        }
    }

    fn store(
        &self,
        item: &QaItem,
        chunk: &Chunk,
        config: &VerifierConfig,
        prompt: &str,
        verification: &CachedVerification,
    ) -> Result<(), VerificationCacheError> {
        fs::create_dir_all(&self.directory).map_err(|source| VerificationCacheError::Io {
            path: self.directory.clone(),
            source,
        })?;
        let path = self.entry_path(item, chunk, config, prompt)?;
        let temporary_path = path.with_extension("json.tmp");
        let bytes = serde_json::to_vec_pretty(verification).map_err(|source| {
            VerificationCacheError::Json {
                path: path.clone(),
                source,
            }
        })?;
        fs::write(&temporary_path, bytes).map_err(|source| VerificationCacheError::Io {
            path: temporary_path.clone(),
            source,
        })?;
        fs::rename(&temporary_path, &path)
            .map_err(|source| VerificationCacheError::Io { path, source })?;
        Ok(())
    }

    fn entry_path(
        &self,
        item: &QaItem,
        chunk: &Chunk,
        config: &VerifierConfig,
        prompt: &str,
    ) -> Result<PathBuf, VerificationCacheError> {
        let identity = VerificationCacheIdentity {
            qa_id: &item.id,
            question: &item.question,
            answer: &item.answer,
            chunk_id: &chunk.chunk_id,
            prompt_hash: sha256_hex(prompt.as_bytes()),
            provider: config.provider,
            model: &config.model,
            endpoint: config.endpoint.as_deref(),
            temperature: config.temperature,
        };
        let bytes = serde_json::to_vec(&identity).map_err(VerificationCacheError::Identity)?;
        Ok(self.directory.join(format!("{}.json", sha256_hex(&bytes))))
    }
}

#[derive(Debug, Serialize)]
struct VerificationCacheIdentity<'a> {
    qa_id: &'a str,
    question: &'a str,
    answer: &'a str,
    chunk_id: &'a str,
    prompt_hash: String,
    provider: LlmProvider,
    model: &'a str,
    endpoint: Option<&'a str>,
    temperature: f32,
}

#[derive(Debug, Error)]
pub enum VerificationCacheError {
    #[error("failed to serialize verification cache identity")]
    Identity(#[source] serde_json::Error),
    #[error("verification cache I/O failed for {path}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("verification cache JSON is invalid at {path}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
}

#[cfg(test)]
mod tests {
    use std::{
        fs::{self, File},
        io::BufWriter,
        sync::atomic::{AtomicUsize, Ordering},
        time::Duration,
    };

    use async_trait::async_trait;
    use tokio::time::sleep;

    use super::{
        VerificationConfig, VerificationRecord, VerificationResult, VerifierConfig, verify_dataset,
    };
    use crate::{
        chunker::chunk_markdown,
        export::{write_alpaca_jsonl, write_manifest},
        llm::{LlmClient, LlmError, LlmRequestConfig},
        prompt::{DEFAULT_PROMPT_TEMPLATE, PROMPT_VERSION},
        types::{
            BuildConfig, BuildManifest, BuildStats, CacheConfig, ExportConfig, LlmProvider,
            PromptMetadata, QaItem, RepositoryMetadata, SourceRef,
        },
    };

    #[derive(Debug, Default)]
    struct FakeVerifier {
        calls: AtomicUsize,
        active: AtomicUsize,
        peak_active: AtomicUsize,
    }

    #[async_trait]
    impl LlmClient for FakeVerifier {
        async fn complete_structured(
            &self,
            prompt: &str,
            _config: &LlmRequestConfig<'_>,
            _response_schema: &serde_json::Value,
        ) -> Result<String, LlmError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak_active.fetch_max(active, Ordering::SeqCst);
            let self_contained = !prompt.contains("Question: What does this do?");
            sleep(Duration::from_millis(if self_contained { 20 } else { 5 })).await;
            self.active.fetch_sub(1, Ordering::SeqCst);
            Ok(serde_json::json!({
                "grounded": {
                    "passed": true,
                    "reason": "The source supports the answer."
                },
                "self_contained": {
                    "passed": self_contained,
                    "reason": if self_contained {
                        "The subject is named."
                    } else {
                        "The word this has no referent outside the source."
                    }
                },
                "answer_relevant": {
                    "passed": true,
                    "reason": "The answer addresses the question."
                },
                "evidence": "The documentation states that HKB creates datasets."
            })
            .to_string())
        }
    }

    #[tokio::test]
    async fn verifies_records_reuses_cache_and_detects_source_drift()
    -> Result<(), Box<dyn std::error::Error>> {
        let repository = tempfile::tempdir()?;
        let dataset_path = repository.path().join("dataset.jsonl");
        let build_manifest_path = repository.path().join("build-manifest.json");
        let report_path = repository.path().join("verification.jsonl");
        let readme = "# HKB\nHKB builds datasets.";
        fs::write(repository.path().join("README.md"), readme)?;
        let chunk = chunk_markdown(
            std::path::Path::new("README.md"),
            readme,
            &crate::types::ChunkConfig::default(),
        )?
        .remove(0);
        let items = [
            qa_item(
                "qa-accepted",
                "What does HKB build?",
                "HKB builds datasets.",
                &chunk,
            ),
            qa_item(
                "qa-rejected",
                "What does this do?",
                "HKB builds datasets.",
                &chunk,
            ),
        ];
        write_alpaca_jsonl(BufWriter::new(File::create(&dataset_path)?), &items)?;
        let build_config = BuildConfig {
            repository: repository.path().to_path_buf(),
            export: ExportConfig {
                output: dataset_path.clone(),
                ..ExportConfig::default()
            },
            ..BuildConfig::default()
        };
        write_manifest(
            BufWriter::new(File::create(&build_manifest_path)?),
            &BuildManifest {
                schema_version: "1".to_owned(),
                generated_at: "2026-08-01T10:00:00Z".to_owned(),
                repository: RepositoryMetadata {
                    root: repository.path().to_path_buf(),
                    commit: None,
                },
                config: build_config,
                prompt: PromptMetadata {
                    version: PROMPT_VERSION.to_owned(),
                    template: DEFAULT_PROMPT_TEMPLATE.to_owned(),
                },
                stats: BuildStats {
                    generated_items: items.len(),
                    ..BuildStats::default()
                },
            },
        )?;
        let config = VerificationConfig {
            dataset: dataset_path,
            build_manifest: build_manifest_path,
            repository: repository.path().to_path_buf(),
            output: report_path.clone(),
            prompt_file: None,
            verifier: VerifierConfig {
                provider: LlmProvider::Ollama,
                model: "test-verifier".to_owned(),
                endpoint: None,
                temperature: 0.0,
                concurrency: 2,
                max_retries: 2,
            },
            cache: CacheConfig::default(),
        };
        let client = FakeVerifier::default();

        let first = verify_dataset(&client, &config).await?;
        let second = verify_dataset(&client, &config).await?;

        assert_eq!(client.calls.load(Ordering::SeqCst), 2);
        assert_eq!(client.peak_active.load(Ordering::SeqCst), 2);
        assert_eq!(first.manifest.stats.accepted_items, 1);
        assert_eq!(first.manifest.stats.rejected_items, 1);
        assert_eq!(second.manifest.stats, first.manifest.stats);
        let records = read_records(&report_path)?;
        assert!(matches!(
            records[0].result,
            VerificationResult::Accepted { .. }
        ));
        assert!(matches!(
            records[1].result,
            VerificationResult::Rejected { .. }
        ));

        fs::write(
            repository.path().join("README.md"),
            "# HKB\nThe source changed.",
        )?;
        let drift = verify_dataset(&client, &config).await?;

        assert_eq!(client.calls.load(Ordering::SeqCst), 2);
        assert_eq!(drift.manifest.stats.unverifiable_items, 2);
        Ok(())
    }

    #[test]
    fn rejects_dataset_source_paths_outside_the_repository()
    -> Result<(), Box<dyn std::error::Error>> {
        let parent = tempfile::tempdir()?;
        let repository = parent.path().join("repository");
        fs::create_dir(&repository)?;
        fs::write(parent.path().join("outside.md"), "# Outside")?;

        let resolved = super::repository_source_path(&repository, "../outside.md");

        assert!(resolved.is_none());
        Ok(())
    }

    fn qa_item(id: &str, question: &str, answer: &str, chunk: &crate::types::Chunk) -> QaItem {
        QaItem {
            id: id.to_owned(),
            question: question.to_owned(),
            answer: answer.to_owned(),
            source: SourceRef {
                path: chunk.path.clone(),
                start_line: chunk.start_line,
                end_line: chunk.end_line,
            },
            chunk_id: chunk.chunk_id.clone(),
            model: "generator".to_owned(),
            generated_at: "2026-08-01T10:00:00Z".to_owned(),
            tags: Vec::new(),
            confidence: None,
        }
    }

    fn read_records(path: &std::path::Path) -> Result<Vec<VerificationRecord>, std::io::Error> {
        fs::read_to_string(path).and_then(|contents| {
            contents
                .lines()
                .map(|line| {
                    serde_json::from_str(line).map_err(|error| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData, error)
                    })
                })
                .collect()
        })
    }
}
