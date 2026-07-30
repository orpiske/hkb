use std::{
    fs,
    path::{Path, PathBuf},
};

use ignore::WalkBuilder;
use thiserror::Error;

use crate::types::DiscoveryConfig;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Document {
    pub path: PathBuf,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkippedFile {
    pub path: PathBuf,
    pub reason: SkipReason,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkipReason {
    FileLimit,
    TooLarge { bytes: u64, max_bytes: u64 },
    InvalidUtf8,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DiscoveryReport {
    pub discovered_files: usize,
    pub documents: Vec<Document>,
    pub skipped: Vec<SkippedFile>,
}

#[derive(Debug, Error)]
pub enum DiscoveryError {
    #[error("repository path is not a directory: {0}")]
    NotDirectory(PathBuf),
    #[error("failed while walking the repository")]
    Walk(#[from] ignore::Error),
    #[error("failed to read metadata for {path}")]
    Metadata {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to read {path}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

pub fn discover_markdown(
    repository: &Path,
    config: &DiscoveryConfig,
) -> Result<DiscoveryReport, DiscoveryError> {
    if !repository.is_dir() {
        return Err(DiscoveryError::NotDirectory(repository.to_path_buf()));
    }

    let mut builder = WalkBuilder::new(repository);
    builder
        .hidden(false)
        .parents(false)
        .ignore(false)
        .require_git(false)
        .git_global(false)
        .git_ignore(config.respect_gitignore)
        .git_exclude(config.respect_gitignore)
        .filter_entry(|entry| !matches!(entry.file_name().to_str(), Some(".git" | ".hkb")));

    let mut candidates = Vec::new();
    for entry in builder.build() {
        let entry = entry?;
        if entry
            .file_type()
            .is_some_and(|file_type| file_type.is_file())
            && is_markdown(entry.path())
        {
            candidates.push(entry.into_path());
        }
    }
    candidates.sort();

    let mut report = DiscoveryReport {
        discovered_files: candidates.len(),
        ..DiscoveryReport::default()
    };

    for (index, absolute_path) in candidates.into_iter().enumerate() {
        let relative_path = absolute_path
            .strip_prefix(repository)
            .unwrap_or(&absolute_path)
            .to_path_buf();

        if config.max_files.is_some_and(|max_files| index >= max_files) {
            report.skipped.push(SkippedFile {
                path: relative_path,
                reason: SkipReason::FileLimit,
            });
            continue;
        }

        let metadata = fs::metadata(&absolute_path).map_err(|source| DiscoveryError::Metadata {
            path: relative_path.clone(),
            source,
        })?;
        if metadata.len() > config.max_bytes_per_file {
            report.skipped.push(SkippedFile {
                path: relative_path,
                reason: SkipReason::TooLarge {
                    bytes: metadata.len(),
                    max_bytes: config.max_bytes_per_file,
                },
            });
            continue;
        }

        let bytes = fs::read(&absolute_path).map_err(|source| DiscoveryError::Read {
            path: relative_path.clone(),
            source,
        })?;
        let Ok(text) = String::from_utf8(bytes) else {
            report.skipped.push(SkippedFile {
                path: relative_path,
                reason: SkipReason::InvalidUtf8,
            });
            continue;
        };

        report.documents.push(Document {
            path: relative_path,
            text: normalize_line_endings(&text),
        });
    }

    Ok(report)
}

fn is_markdown(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("md"))
}

fn normalize_line_endings(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::{SkipReason, discover_markdown};
    use crate::types::DiscoveryConfig;

    #[test]
    fn discovers_sorted_markdown_and_respects_gitignore() -> Result<(), Box<dyn std::error::Error>>
    {
        let repository = tempfile::tempdir()?;
        fs::create_dir(repository.path().join("docs"))?;
        fs::write(repository.path().join(".gitignore"), "ignored.md\n")?;
        fs::write(repository.path().join("README.md"), "# HKB\r\nBuilder\r\n")?;
        fs::write(repository.path().join("docs/guide.md"), "# Guide")?;
        fs::write(repository.path().join("ignored.md"), "# Ignore me")?;
        fs::write(repository.path().join("src.rs"), "fn main() {}")?;

        let report = discover_markdown(repository.path(), &DiscoveryConfig::default())?;

        assert_eq!(report.discovered_files, 2);
        assert_eq!(report.documents.len(), 2);
        assert_eq!(
            report.documents[0].path,
            std::path::PathBuf::from("README.md")
        );
        assert_eq!(report.documents[0].text, "# HKB\nBuilder\n");
        assert_eq!(
            report.documents[1].path,
            std::path::PathBuf::from("docs/guide.md")
        );
        Ok(())
    }

    #[test]
    fn can_disable_gitignore_filtering() -> Result<(), Box<dyn std::error::Error>> {
        let repository = tempfile::tempdir()?;
        fs::write(repository.path().join(".gitignore"), "ignored.md\n")?;
        fs::write(repository.path().join("ignored.md"), "# Included")?;
        let config = DiscoveryConfig {
            respect_gitignore: false,
            ..DiscoveryConfig::default()
        };

        let report = discover_markdown(repository.path(), &config)?;

        assert_eq!(report.documents.len(), 1);
        assert_eq!(
            report.documents[0].path,
            std::path::PathBuf::from("ignored.md")
        );
        Ok(())
    }

    #[test]
    fn reports_size_and_file_limit_skips() -> Result<(), Box<dyn std::error::Error>> {
        let repository = tempfile::tempdir()?;
        fs::write(repository.path().join("a.md"), "too large")?;
        fs::write(repository.path().join("b.md"), "small")?;
        let config = DiscoveryConfig {
            max_files: Some(1),
            max_bytes_per_file: 3,
            ..DiscoveryConfig::default()
        };

        let report = discover_markdown(repository.path(), &config)?;

        assert!(report.documents.is_empty());
        assert!(matches!(
            report.skipped[0].reason,
            SkipReason::TooLarge { .. }
        ));
        assert_eq!(report.skipped[1].reason, SkipReason::FileLimit);
        Ok(())
    }
}
