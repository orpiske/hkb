use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub chunk_id: String,
    pub path: String,
    pub language: String,
    pub start_line: usize,
    pub end_line: usize,
    pub content_hash: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceRef {
    pub path: String,
    pub start_line: usize,
    pub end_line: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QaItem {
    pub id: String,
    pub question: String,
    pub answer: String,
    pub source: SourceRef,
    pub chunk_id: String,
}
