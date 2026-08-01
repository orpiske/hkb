use std::{
    fs,
    path::{Path, PathBuf},
};

use thiserror::Error;

use crate::types::Chunk;

pub const PROMPT_VERSION: &str = "qa-v1";
pub const CUSTOM_PROMPT_VERSION: &str = "custom";
pub const DEFAULT_PROMPT_TEMPLATE: &str = include_str!("../prompts/qa-v1.md");

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QaPromptTemplate {
    pub version: String,
    pub template: String,
    pub source_path: Option<PathBuf>,
}

#[derive(Debug, Error)]
pub enum PromptError {
    #[error("failed to read prompt file {path}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("prompt template must not be empty")]
    Empty,
    #[error("prompt template must contain the {{chunk_text}} placeholder")]
    MissingChunkText,
}

pub fn load_qa_prompt(
    repository: &Path,
    prompt_file: Option<&Path>,
) -> Result<QaPromptTemplate, PromptError> {
    let (version, template, source_path) = match prompt_file {
        Some(path) => {
            let path = if path.is_absolute() {
                path.to_path_buf()
            } else {
                repository.join(path)
            };
            let template = fs::read_to_string(&path).map_err(|source| PromptError::Io {
                path: path.clone(),
                source,
            })?;
            (CUSTOM_PROMPT_VERSION, template, Some(path))
        }
        None => (PROMPT_VERSION, DEFAULT_PROMPT_TEMPLATE.to_owned(), None),
    };

    validate_template(&template)?;
    Ok(QaPromptTemplate {
        version: version.to_owned(),
        template,
        source_path,
    })
}

pub fn build_qa_prompt(template: &str, chunk: &Chunk, questions_per_chunk: usize) -> String {
    template
        .replace("{{questions_per_chunk}}", &questions_per_chunk.to_string())
        .replace("{{path}}", &chunk.path)
        .replace("{{start_line}}", &chunk.start_line.to_string())
        .replace("{{end_line}}", &chunk.end_line.to_string())
        .replace("{{chunk_text}}", &chunk.text)
}

fn validate_template(template: &str) -> Result<(), PromptError> {
    if template.trim().is_empty() {
        return Err(PromptError::Empty);
    }
    if !template.contains("{{chunk_text}}") {
        return Err(PromptError::MissingChunkText);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::{DEFAULT_PROMPT_TEMPLATE, PromptError, build_qa_prompt, load_qa_prompt};
    use crate::types::Chunk;

    #[test]
    fn prompt_contains_generation_rules_and_source_provenance() {
        let chunk = Chunk {
            chunk_id: "chunk-1".to_owned(),
            path: "README.md".to_owned(),
            language: "markdown".to_owned(),
            start_line: 4,
            end_line: 8,
            content_hash: "hash".to_owned(),
            text: "HKB builds datasets.".to_owned(),
        };

        let prompt = build_qa_prompt(DEFAULT_PROMPT_TEMPLATE, &chunk, 3);

        assert!(prompt.contains("exactly 3 items"));
        assert!(prompt.contains("README.md:4-8"));
        assert!(prompt.contains("<source>\nHKB builds datasets.\n</source>"));
    }

    #[test]
    fn loads_a_repository_relative_custom_prompt() -> Result<(), Box<dyn std::error::Error>> {
        let repository = tempfile::tempdir()?;
        fs::write(
            repository.path().join("hkb-prompt.md"),
            "Project-specific instructions\n{{chunk_text}}",
        )?;

        let prompt = load_qa_prompt(
            repository.path(),
            Some(std::path::Path::new("hkb-prompt.md")),
        )?;

        assert_eq!(prompt.version, "custom");
        assert!(prompt.template.starts_with("Project-specific"));
        Ok(())
    }

    #[test]
    fn rejects_a_custom_prompt_without_source_content() -> Result<(), Box<dyn std::error::Error>> {
        let repository = tempfile::tempdir()?;
        fs::write(
            repository.path().join("hkb-prompt.md"),
            "Generate questions",
        )?;

        let result = load_qa_prompt(
            repository.path(),
            Some(std::path::Path::new("hkb-prompt.md")),
        );

        assert!(matches!(result, Err(PromptError::MissingChunkText)));
        Ok(())
    }
}
