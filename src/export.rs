use std::io::Write;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::types::{BuildManifest, QaItem, SourceRef};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlpacaRecord {
    pub instruction: String,
    pub input: String,
    pub output: String,
    pub metadata: AlpacaMetadata,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlpacaMetadata {
    pub id: String,
    pub source: SourceRef,
    pub chunk_id: String,
    pub model: String,
    pub generated_at: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

impl From<&QaItem> for AlpacaRecord {
    fn from(item: &QaItem) -> Self {
        Self {
            instruction: item.question.clone(),
            input: String::new(),
            output: item.answer.clone(),
            metadata: AlpacaMetadata {
                id: item.id.clone(),
                source: item.source.clone(),
                chunk_id: item.chunk_id.clone(),
                model: item.model.clone(),
                generated_at: item.generated_at.clone(),
                tags: item.tags.clone(),
                confidence: item.confidence,
            },
        }
    }
}

#[derive(Debug, Error)]
pub enum ExportError {
    #[error("failed to write export data")]
    Io(#[from] std::io::Error),
    #[error("failed to serialize export data")]
    Json(#[from] serde_json::Error),
}

pub fn write_alpaca_jsonl(mut writer: impl Write, items: &[QaItem]) -> Result<(), ExportError> {
    for item in items {
        serde_json::to_writer(&mut writer, &AlpacaRecord::from(item))?;
        writer.write_all(b"\n")?;
    }

    Ok(())
}

pub fn write_manifest(mut writer: impl Write, manifest: &BuildManifest) -> Result<(), ExportError> {
    serde_json::to_writer_pretty(&mut writer, manifest)?;
    writer.write_all(b"\n")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{AlpacaRecord, write_alpaca_jsonl};
    use crate::types::{QaItem, SourceRef};

    #[test]
    fn writes_one_alpaca_record_per_line_with_provenance() -> Result<(), Box<dyn std::error::Error>>
    {
        let item = QaItem {
            id: "qa-1".to_owned(),
            question: "What is HKB?".to_owned(),
            answer: "A knowledge dataset builder.".to_owned(),
            source: SourceRef {
                path: "README.md".to_owned(),
                start_line: 1,
                end_line: 2,
            },
            chunk_id: "chunk-1".to_owned(),
            model: "test-model".to_owned(),
            generated_at: "2026-07-30T10:00:00Z".to_owned(),
            tags: vec!["overview".to_owned()],
            confidence: Some(0.9),
        };
        let mut output = Vec::new();

        write_alpaca_jsonl(&mut output, &[item])?;

        let line = std::str::from_utf8(&output)?.trim_end();
        let record: AlpacaRecord = serde_json::from_str(line)?;
        assert_eq!(record.instruction, "What is HKB?");
        assert_eq!(record.output, "A knowledge dataset builder.");
        assert_eq!(record.metadata.source.path, "README.md");
        Ok(())
    }
}
