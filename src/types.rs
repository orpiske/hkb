use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Chunk {
    pub chunk_id: String,
    pub path: String,
    pub language: String,
    pub start_line: usize,
    pub end_line: usize,
    pub content_hash: String,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceRef {
    pub path: String,
    pub start_line: usize,
    pub end_line: usize,
}

/// A provider-independent question and answer generated from one source chunk.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QaItem {
    pub id: String,
    pub question: String,
    pub answer: String,
    pub source: SourceRef,
    pub chunk_id: String,
    pub model: String,
    pub generated_at: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputScope {
    #[default]
    Docs,
    Code,
    All,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LlmProvider {
    #[default]
    Ollama,
    OpenAiCompatible,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExportFormat {
    #[default]
    Alpaca,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscoveryConfig {
    pub respect_gitignore: bool,
    pub ignore_files: Vec<PathBuf>,
    pub max_files: Option<usize>,
    pub max_bytes_per_file: u64,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            respect_gitignore: true,
            ignore_files: Vec::new(),
            max_files: None,
            max_bytes_per_file: 1_000_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkConfig {
    pub max_characters: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            max_characters: 4_000,
        }
    }
}

/// Public LLM settings that are safe to persist in a build manifest.
///
/// Authentication credentials intentionally do not belong in this type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub provider: LlmProvider,
    pub model: String,
    pub endpoint: Option<String>,
    pub questions_per_chunk: usize,
    pub temperature: f32,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            provider: LlmProvider::Ollama,
            model: "llama3.2".to_owned(),
            endpoint: None,
            questions_per_chunk: 3,
            temperature: 0.2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportConfig {
    pub format: ExportFormat,
    pub output: PathBuf,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            format: ExportFormat::Alpaca,
            output: PathBuf::from("dataset.jsonl"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheConfig {
    pub enabled: bool,
    pub directory: PathBuf,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            directory: PathBuf::from(".hkb"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BuildConfig {
    pub repository: PathBuf,
    pub include: InputScope,
    pub discovery: DiscoveryConfig,
    pub chunking: ChunkConfig,
    pub generation: GenerationConfig,
    pub export: ExportConfig,
    pub cache: CacheConfig,
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self {
            repository: PathBuf::from("."),
            include: InputScope::default(),
            discovery: DiscoveryConfig::default(),
            chunking: ChunkConfig::default(),
            generation: GenerationConfig::default(),
            export: ExportConfig::default(),
            cache: CacheConfig::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepositoryMetadata {
    pub root: PathBuf,
    pub commit: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptMetadata {
    pub version: String,
    pub template: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuildStats {
    pub discovered_files: usize,
    pub processed_files: usize,
    pub skipped_files: usize,
    pub chunks: usize,
    pub generated_items: usize,
    pub rejected_items: usize,
    pub duplicate_items: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BuildManifest {
    pub schema_version: String,
    pub generated_at: String,
    pub repository: RepositoryMetadata,
    pub config: BuildConfig,
    pub prompt: PromptMetadata,
    pub stats: BuildStats,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{
        BuildConfig, BuildManifest, BuildStats, InputScope, PromptMetadata, QaItem,
        RepositoryMetadata, SourceRef,
    };

    #[test]
    fn default_build_config_matches_the_mvp_scope() {
        let config = BuildConfig::default();

        assert_eq!(config.repository, std::path::PathBuf::from("."));
        assert_eq!(config.include, InputScope::Docs);
        assert!(config.discovery.respect_gitignore);
        assert!(config.discovery.ignore_files.is_empty());
        assert_eq!(config.chunking.max_characters, 4_000);
        assert_eq!(config.generation.questions_per_chunk, 3);
        assert_eq!(
            config.export.output,
            std::path::PathBuf::from("dataset.jsonl")
        );
        assert!(config.cache.enabled);
    }

    #[test]
    fn qa_item_omits_empty_optional_metadata() -> Result<(), serde_json::Error> {
        let item = QaItem {
            id: "qa-1".to_owned(),
            question: "What does HKB build?".to_owned(),
            answer: "A dataset derived from a repository.".to_owned(),
            source: SourceRef {
                path: "README.md".to_owned(),
                start_line: 1,
                end_line: 5,
            },
            chunk_id: "chunk-1".to_owned(),
            model: "llama3.2".to_owned(),
            generated_at: "2026-07-26T12:00:00Z".to_owned(),
            tags: Vec::new(),
            confidence: None,
        };

        let value = serde_json::to_value(item)?;

        assert_eq!(value.get("tags"), None);
        assert_eq!(value.get("confidence"), None);
        Ok(())
    }

    #[test]
    fn manifest_round_trips_through_json() -> Result<(), serde_json::Error> {
        let manifest = BuildManifest {
            schema_version: "1".to_owned(),
            generated_at: "2026-07-26T12:00:00Z".to_owned(),
            repository: RepositoryMetadata {
                root: std::path::PathBuf::from("."),
                commit: Some("abc123".to_owned()),
            },
            config: BuildConfig::default(),
            prompt: PromptMetadata {
                version: "1".to_owned(),
                template: "Generate questions from this chunk.".to_owned(),
            },
            stats: BuildStats {
                discovered_files: 2,
                processed_files: 2,
                chunks: 4,
                generated_items: 12,
                ..BuildStats::default()
            },
        };

        let value = serde_json::to_value(&manifest)?;
        assert_eq!(value["config"]["include"], json!("docs"));
        assert_eq!(value["config"]["generation"]["provider"], json!("ollama"));
        assert_eq!(value["config"]["export"]["format"], json!("alpaca"));

        let decoded = serde_json::from_value(value)?;
        assert_eq!(manifest, decoded);
        Ok(())
    }
}
