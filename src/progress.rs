use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationSource {
    Cache,
    Llm,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProgressEvent {
    DiscoveryStarted {
        repository: PathBuf,
    },
    DiscoveryFinished {
        discovered_files: usize,
        selected_files: usize,
        skipped_files: usize,
    },
    FileChunked {
        path: PathBuf,
        chunks: usize,
    },
    GenerationStarted {
        total_chunks: usize,
    },
    ChunkStarted {
        index: usize,
        total: usize,
        path: String,
        start_line: usize,
        end_line: usize,
        source: GenerationSource,
    },
    RetryScheduled {
        index: usize,
        path: String,
        attempt: usize,
        max_retries: usize,
        delay_ms: u64,
    },
    ChunkFinished {
        index: usize,
        path: String,
        source: GenerationSource,
        generated_items: usize,
    },
    ValidationFinished {
        accepted_items: usize,
        rejected_items: usize,
        duplicate_items: usize,
    },
    WritingOutput {
        dataset_path: PathBuf,
        manifest_path: PathBuf,
    },
    Finished,
}

pub trait ProgressReporter: Send + Sync {
    fn report(&self, event: ProgressEvent);
}

#[derive(Debug, Default)]
pub struct NoopProgress;

impl ProgressReporter for NoopProgress {
    fn report(&self, _event: ProgressEvent) {}
}
