use std::collections::HashSet;

use thiserror::Error;

use crate::types::QaItem;

#[derive(Debug, Clone, PartialEq)]
pub struct RejectedQa {
    pub item: QaItem,
    pub reason: QaValidationError,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ValidationReport {
    pub accepted: Vec<QaItem>,
    pub rejected: Vec<RejectedQa>,
    pub duplicate_items: usize,
}

#[derive(Debug, Clone, Error, PartialEq)]
pub enum QaValidationError {
    #[error("question is empty")]
    EmptyQuestion,
    #[error("answer is empty")]
    EmptyAnswer,
    #[error("source path is empty")]
    EmptySourcePath,
    #[error("source line range is invalid: {start_line}..={end_line}")]
    InvalidSourceRange { start_line: usize, end_line: usize },
    #[error("chunk ID is empty")]
    EmptyChunkId,
    #[error("confidence must be finite and between 0 and 1")]
    InvalidConfidence,
}

pub fn validate_qa(item: &QaItem) -> Result<(), QaValidationError> {
    if item.question.trim().is_empty() {
        return Err(QaValidationError::EmptyQuestion);
    }
    if item.answer.trim().is_empty() {
        return Err(QaValidationError::EmptyAnswer);
    }
    if item.source.path.trim().is_empty() {
        return Err(QaValidationError::EmptySourcePath);
    }
    if item.source.start_line == 0 || item.source.end_line < item.source.start_line {
        return Err(QaValidationError::InvalidSourceRange {
            start_line: item.source.start_line,
            end_line: item.source.end_line,
        });
    }
    if item.chunk_id.trim().is_empty() {
        return Err(QaValidationError::EmptyChunkId);
    }
    if item
        .confidence
        .is_some_and(|confidence| !confidence.is_finite() || !(0.0..=1.0).contains(&confidence))
    {
        return Err(QaValidationError::InvalidConfidence);
    }

    Ok(())
}

pub fn normalize_question(question: &str) -> String {
    question
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim_end_matches(['?', '.', '!', ':', ';'])
        .to_lowercase()
}

pub fn validate_and_deduplicate(items: impl IntoIterator<Item = QaItem>) -> ValidationReport {
    let mut report = ValidationReport::default();
    let mut seen_questions = HashSet::new();

    for item in items {
        if let Err(reason) = validate_qa(&item) {
            report.rejected.push(RejectedQa { item, reason });
            continue;
        }

        if !seen_questions.insert(normalize_question(&item.question)) {
            report.duplicate_items += 1;
            continue;
        }

        report.accepted.push(item);
    }

    report
}

#[cfg(test)]
mod tests {
    use super::{QaValidationError, normalize_question, validate_and_deduplicate, validate_qa};
    use crate::types::{QaItem, SourceRef};

    fn item(id: &str, question: &str, answer: &str) -> QaItem {
        QaItem {
            id: id.to_owned(),
            question: question.to_owned(),
            answer: answer.to_owned(),
            source: SourceRef {
                path: "README.md".to_owned(),
                start_line: 1,
                end_line: 3,
            },
            chunk_id: "chunk-1".to_owned(),
            model: "test-model".to_owned(),
            generated_at: "2026-07-30T10:00:00Z".to_owned(),
            tags: Vec::new(),
            confidence: None,
        }
    }

    #[test]
    fn normalizes_case_whitespace_and_trailing_punctuation() {
        assert_eq!(
            normalize_question("  What   IS HKB?  "),
            normalize_question("what is hkb")
        );
    }

    #[test]
    fn rejects_empty_answers() {
        let result = validate_qa(&item("qa-1", "What is HKB?", " \n "));

        assert_eq!(result, Err(QaValidationError::EmptyAnswer));
    }

    #[test]
    fn rejects_invalid_confidence() {
        let mut qa = item("qa-1", "What is HKB?", "A dataset builder.");
        qa.confidence = Some(f32::NAN);

        assert_eq!(validate_qa(&qa), Err(QaValidationError::InvalidConfidence));
    }

    #[test]
    fn preserves_first_question_and_counts_duplicates() {
        let report = validate_and_deduplicate([
            item("qa-1", "What is HKB?", "A dataset builder."),
            item("qa-2", " what  is hkb ", "A Rust program."),
            item("qa-3", "How is it built?", "With Rust."),
        ]);

        assert_eq!(report.accepted.len(), 2);
        assert_eq!(report.accepted[0].id, "qa-1");
        assert_eq!(report.duplicate_items, 1);
        assert!(report.rejected.is_empty());
    }
}
