use std::{
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    identity::sha256_hex,
    llm::GeneratedQa,
    prompt::PROMPT_VERSION,
    types::{CacheConfig, Chunk, GenerationConfig},
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationBatch {
    pub generated_at: String,
    pub items: Vec<GeneratedQa>,
}

#[derive(Debug, Clone)]
pub struct GenerationCache {
    directory: PathBuf,
}

impl GenerationCache {
    pub fn new(repository: &Path, config: &CacheConfig) -> Self {
        let directory = if config.directory.is_absolute() {
            config.directory.clone()
        } else {
            repository.join(&config.directory)
        };

        Self { directory }
    }

    pub fn load(
        &self,
        chunk: &Chunk,
        config: &GenerationConfig,
    ) -> Result<Option<GenerationBatch>, CacheError> {
        let path = self.entry_path(chunk, config)?;
        match fs::read(&path) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map(Some)
                .map_err(|source| CacheError::Json { path, source }),
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(source) => Err(CacheError::Io { path, source }),
        }
    }

    pub fn store(
        &self,
        chunk: &Chunk,
        config: &GenerationConfig,
        batch: &GenerationBatch,
    ) -> Result<(), CacheError> {
        fs::create_dir_all(&self.directory).map_err(|source| CacheError::Io {
            path: self.directory.clone(),
            source,
        })?;

        let path = self.entry_path(chunk, config)?;
        let temporary_path = path.with_extension("json.tmp");
        let bytes = serde_json::to_vec_pretty(batch).map_err(|source| CacheError::Json {
            path: path.clone(),
            source,
        })?;
        fs::write(&temporary_path, bytes).map_err(|source| CacheError::Io {
            path: temporary_path.clone(),
            source,
        })?;
        fs::rename(&temporary_path, &path).map_err(|source| CacheError::Io { path, source })?;
        Ok(())
    }

    fn entry_path(&self, chunk: &Chunk, config: &GenerationConfig) -> Result<PathBuf, CacheError> {
        let identity = CacheIdentity {
            chunk_id: &chunk.chunk_id,
            prompt_version: PROMPT_VERSION,
            provider: config.provider,
            model: &config.model,
            endpoint: config.endpoint.as_deref(),
            questions_per_chunk: config.questions_per_chunk,
            temperature: config.temperature,
        };
        let bytes = serde_json::to_vec(&identity).map_err(CacheError::Identity)?;
        Ok(self.directory.join(format!("{}.json", sha256_hex(&bytes))))
    }
}

#[derive(Debug, Serialize)]
struct CacheIdentity<'a> {
    chunk_id: &'a str,
    prompt_version: &'a str,
    provider: crate::types::LlmProvider,
    model: &'a str,
    endpoint: Option<&'a str>,
    questions_per_chunk: usize,
    temperature: f32,
}

#[derive(Debug, Error)]
pub enum CacheError {
    #[error("failed to serialize cache identity")]
    Identity(#[source] serde_json::Error),
    #[error("cache I/O failed for {path}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("cache JSON is invalid at {path}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
}

#[cfg(test)]
mod tests {
    use super::{GenerationBatch, GenerationCache};
    use crate::{
        llm::GeneratedQa,
        types::{CacheConfig, Chunk, GenerationConfig},
    };

    #[test]
    fn round_trips_a_generation_batch() -> Result<(), Box<dyn std::error::Error>> {
        let repository = tempfile::tempdir()?;
        let cache = GenerationCache::new(repository.path(), &CacheConfig::default());
        let chunk = Chunk {
            chunk_id: "chunk-1".to_owned(),
            path: "README.md".to_owned(),
            language: "markdown".to_owned(),
            start_line: 1,
            end_line: 2,
            content_hash: "content-hash".to_owned(),
            text: "# HKB".to_owned(),
        };
        let batch = GenerationBatch {
            generated_at: "2026-07-30T10:00:00Z".to_owned(),
            items: vec![GeneratedQa {
                question: "What is HKB?".to_owned(),
                answer: "A dataset builder.".to_owned(),
                tags: Vec::new(),
                confidence: None,
            }],
        };

        assert_eq!(cache.load(&chunk, &GenerationConfig::default())?, None);
        cache.store(&chunk, &GenerationConfig::default(), &batch)?;
        assert_eq!(
            cache.load(&chunk, &GenerationConfig::default())?,
            Some(batch)
        );
        Ok(())
    }
}
