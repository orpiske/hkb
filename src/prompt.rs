use std::{
    fs,
    path::{Path, PathBuf},
};

use thiserror::Error;

use crate::types::Chunk;

pub const PROMPT_VERSION: &str = "qa-v1";
pub const VERIFICATION_PROMPT_VERSION: &str = "verify-v1";
pub const CUSTOM_PROMPT_VERSION: &str = "custom";
pub const DEFAULT_PROMPT_TEMPLATE: &str = include_str!("../prompts/qa-v1.md");
pub const DEFAULT_VERIFICATION_PROMPT_TEMPLATE: &str = include_str!("../prompts/verify-v1.md");

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptTemplate {
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
    #[error("prompt template must contain the {0} placeholder")]
    MissingPlaceholder(&'static str),
}

pub fn load_qa_prompt(
    repository: &Path,
    prompt_file: Option<&Path>,
) -> Result<PromptTemplate, PromptError> {
    load_prompt(
        repository,
        prompt_file,
        PROMPT_VERSION,
        DEFAULT_PROMPT_TEMPLATE,
        &["{{chunk_text}}"],
    )
}

pub fn load_verification_prompt(
    repository: &Path,
    prompt_file: Option<&Path>,
) -> Result<PromptTemplate, PromptError> {
    load_prompt(
        repository,
        prompt_file,
        VERIFICATION_PROMPT_VERSION,
        DEFAULT_VERIFICATION_PROMPT_TEMPLATE,
        &["{{question}}", "{{answer}}", "{{chunk_text}}"],
    )
}

fn load_prompt(
    repository: &Path,
    prompt_file: Option<&Path>,
    default_version: &'static str,
    default_template: &'static str,
    required_placeholders: &[&'static str],
) -> Result<PromptTemplate, PromptError> {
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
        None => (default_version, default_template.to_owned(), None),
    };

    validate_template(&template, required_placeholders)?;
    Ok(PromptTemplate {
        version: version.to_owned(),
        template,
        source_path,
    })
}

pub fn build_verification_prompt(
    template: &str,
    question: &str,
    answer: &str,
    chunk: &Chunk,
) -> String {
    let start_line = chunk.start_line.to_string();
    let end_line = chunk.end_line.to_string();
    render_template(
        template,
        &[
            ("{{question}}", question),
            ("{{answer}}", answer),
            ("{{path}}", &chunk.path),
            ("{{start_line}}", &start_line),
            ("{{end_line}}", &end_line),
            ("{{chunk_text}}", &chunk.text),
        ],
    )
}

pub fn build_qa_prompt(template: &str, chunk: &Chunk, questions_per_chunk: usize) -> String {
    let questions_per_chunk = questions_per_chunk.to_string();
    let start_line = chunk.start_line.to_string();
    let end_line = chunk.end_line.to_string();
    render_template(
        template,
        &[
            ("{{questions_per_chunk}}", &questions_per_chunk),
            ("{{path}}", &chunk.path),
            ("{{start_line}}", &start_line),
            ("{{end_line}}", &end_line),
            ("{{chunk_text}}", &chunk.text),
        ],
    )
}

fn render_template(template: &str, replacements: &[(&str, &str)]) -> String {
    let mut rendered = String::with_capacity(template.len());
    let mut remaining = template;
    while let Some((offset, placeholder, value)) = replacements
        .iter()
        .filter_map(|(placeholder, value)| {
            remaining
                .find(placeholder)
                .map(|offset| (offset, *placeholder, *value))
        })
        .min_by_key(|(offset, _, _)| *offset)
    {
        rendered.push_str(&remaining[..offset]);
        rendered.push_str(value);
        remaining = &remaining[offset + placeholder.len()..];
    }
    rendered.push_str(remaining);
    rendered
}

fn validate_template(
    template: &str,
    required_placeholders: &[&'static str],
) -> Result<(), PromptError> {
    if template.trim().is_empty() {
        return Err(PromptError::Empty);
    }
    if let Some(placeholder) = required_placeholders
        .iter()
        .find(|placeholder| !template.contains(**placeholder))
    {
        return Err(PromptError::MissingPlaceholder(placeholder));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::{
        DEFAULT_PROMPT_TEMPLATE, PromptError, build_qa_prompt, build_verification_prompt,
        load_qa_prompt,
    };
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

        assert!(matches!(
            result,
            Err(PromptError::MissingPlaceholder("{{chunk_text}}"))
        ));
        Ok(())
    }

    #[test]
    fn does_not_expand_placeholder_text_inside_inserted_values() {
        let chunk = Chunk {
            chunk_id: "chunk-1".to_owned(),
            path: "README.md".to_owned(),
            language: "markdown".to_owned(),
            start_line: 1,
            end_line: 1,
            content_hash: "hash".to_owned(),
            text: "The literal {{answer}} is documented.".to_owned(),
        };

        let prompt = build_verification_prompt(
            "Question={{question}}\nAnswer={{answer}}\nSource={{chunk_text}}",
            "What does {{answer}} mean?",
            "It is a placeholder.",
            &chunk,
        );

        assert!(prompt.contains("Question=What does {{answer}} mean?"));
        assert!(prompt.contains("Source=The literal {{answer}} is documented."));
    }
}
