use std::{
    ops::Range,
    path::{Path, PathBuf},
};

use pulldown_cmark::{Event, Parser, Tag};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::types::{Chunk, ChunkConfig};

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ChunkError {
    #[error("max_characters must be greater than zero")]
    InvalidMaxCharacters,
    #[error("path is not valid UTF-8: {0:?}")]
    NonUtf8Path(PathBuf),
}

/// Splits normalized Markdown into heading-aware chunks.
///
/// The inputs are borrowed, while every returned chunk owns its text and metadata.
pub fn chunk_markdown(
    path: &Path,
    text: &str,
    config: &ChunkConfig,
) -> Result<Vec<Chunk>, ChunkError> {
    if config.max_characters == 0 {
        return Err(ChunkError::InvalidMaxCharacters);
    }

    let path = path
        .to_str()
        .ok_or_else(|| ChunkError::NonUtf8Path(path.to_path_buf()))?;

    let chunks = section_ranges(text)
        .into_iter()
        .flat_map(|range| split_oversized_range(text, range, config.max_characters))
        .filter_map(|range| trim_range(text, range))
        .map(|range| build_chunk(path, text, range))
        .collect();

    Ok(chunks)
}

fn section_ranges(text: &str) -> Vec<Range<usize>> {
    let mut starts = vec![0];

    starts.extend(
        Parser::new(text)
            .into_offset_iter()
            .filter_map(|(event, range)| match event {
                Event::Start(Tag::Heading { .. }) if range.start > 0 => Some(range.start),
                _ => None,
            }),
    );

    starts.sort_unstable();
    starts.dedup();

    starts
        .iter()
        .copied()
        .zip(
            starts
                .iter()
                .copied()
                .skip(1)
                .chain(std::iter::once(text.len())),
        )
        .map(|(start, end)| start..end)
        .collect()
}

fn split_oversized_range(
    text: &str,
    range: Range<usize>,
    max_characters: usize,
) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = range.start;

    while text[start..range.end].chars().count() > max_characters {
        let candidate = byte_offset_after_characters(&text[start..range.end], max_characters);
        let candidate_end = start + candidate;
        let end = text[start..candidate_end]
            .rfind('\n')
            .map_or(candidate_end, |newline| start + newline + 1);

        ranges.push(start..end);
        start = end;
    }

    ranges.push(start..range.end);
    ranges
}

fn byte_offset_after_characters(text: &str, character_count: usize) -> usize {
    text.char_indices()
        .nth(character_count)
        .map_or(text.len(), |(offset, _)| offset)
}

fn trim_range(text: &str, range: Range<usize>) -> Option<Range<usize>> {
    let slice = &text[range.clone()];
    let trimmed_start = slice.trim_start();
    let start = range.start + slice.len() - trimmed_start.len();
    let trimmed = trimmed_start.trim_end();

    (!trimmed.is_empty()).then_some(start..start + trimmed.len())
}

fn build_chunk(path: &str, text: &str, range: Range<usize>) -> Chunk {
    let chunk_text = &text[range.clone()];
    let start_line = line_number_at(text, range.start);
    let end_line = start_line + chunk_text.bytes().filter(|byte| *byte == b'\n').count();
    let content_hash = sha256(chunk_text.as_bytes());
    let identity = format!("{path}\0{start_line}\0{end_line}\0{content_hash}");

    Chunk {
        chunk_id: sha256(identity.as_bytes()),
        path: path.to_owned(),
        language: "markdown".to_owned(),
        start_line,
        end_line,
        content_hash,
        text: chunk_text.to_owned(),
    }
}

fn line_number_at(text: &str, byte_offset: usize) -> usize {
    text[..byte_offset]
        .bytes()
        .filter(|byte| *byte == b'\n')
        .count()
        + 1
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{ChunkError, chunk_markdown};
    use crate::types::ChunkConfig;

    fn config(max_characters: usize) -> ChunkConfig {
        ChunkConfig { max_characters }
    }

    #[test]
    fn splits_markdown_at_heading_boundaries() -> Result<(), ChunkError> {
        let text = "# Introduction\nHKB builds datasets.\n\n## Usage\nRun HKB.";

        let chunks = chunk_markdown(Path::new("README.md"), text, &config(1_000))?;

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].text, "# Introduction\nHKB builds datasets.");
        assert_eq!((chunks[0].start_line, chunks[0].end_line), (1, 2));
        assert_eq!(chunks[1].text, "## Usage\nRun HKB.");
        assert_eq!((chunks[1].start_line, chunks[1].end_line), (4, 5));
        Ok(())
    }

    #[test]
    fn markdown_parser_ignores_heading_syntax_in_fenced_code() -> Result<(), ChunkError> {
        let text = "# Example\n\n```markdown\n# Not a section\n```";

        let chunks = chunk_markdown(Path::new("README.md"), text, &config(1_000))?;

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, text);
        Ok(())
    }

    #[test]
    fn splits_oversized_lines_at_unicode_character_boundaries() -> Result<(), ChunkError> {
        let chunks = chunk_markdown(Path::new("README.md"), "ééééé", &config(3))?;

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].text, "ééé");
        assert_eq!(chunks[1].text, "éé");
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[1].start_line, 1);
        Ok(())
    }

    #[test]
    fn prefers_newline_boundaries_and_preserves_line_numbers() -> Result<(), ChunkError> {
        let chunks = chunk_markdown(Path::new("README.md"), "1234\n5678", &config(5))?;

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].text, "1234");
        assert_eq!((chunks[0].start_line, chunks[0].end_line), (1, 1));
        assert_eq!(chunks[1].text, "5678");
        assert_eq!((chunks[1].start_line, chunks[1].end_line), (2, 2));
        Ok(())
    }

    #[test]
    fn stable_id_changes_when_content_changes() -> Result<(), ChunkError> {
        let original = chunk_markdown(Path::new("README.md"), "# HKB", &config(1_000))?;
        let repeated = chunk_markdown(Path::new("README.md"), "# HKB", &config(1_000))?;
        let changed = chunk_markdown(Path::new("README.md"), "# HKB!", &config(1_000))?;

        assert_eq!(original[0].chunk_id, repeated[0].chunk_id);
        assert_ne!(original[0].chunk_id, changed[0].chunk_id);
        assert_ne!(original[0].content_hash, changed[0].content_hash);
        Ok(())
    }

    #[test]
    fn whitespace_only_input_produces_no_chunks() -> Result<(), ChunkError> {
        let chunks = chunk_markdown(Path::new("README.md"), " \n\n ", &config(1_000))?;

        assert!(chunks.is_empty());
        Ok(())
    }

    #[test]
    fn rejects_a_zero_character_limit() {
        let result = chunk_markdown(Path::new("README.md"), "# HKB", &config(0));

        assert_eq!(result, Err(ChunkError::InvalidMaxCharacters));
    }
}
