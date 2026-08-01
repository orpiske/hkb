use std::{
    env, fs,
    path::{Path, PathBuf},
    process::ExitCode,
    sync::atomic::{AtomicUsize, Ordering},
    time::Duration,
};

use clap::{Args, Parser, Subcommand, ValueEnum};
use hkb::{
    llm::{
        DEFAULT_OLLAMA_ENDPOINT, DEFAULT_OPENAI_ENDPOINT, LlmClient, OllamaClient,
        OpenAiCompatibleClient,
    },
    pipeline::{BuildError, BuildOutcome, build_dataset_with_progress},
    progress::{GenerationSource, ProgressEvent, ProgressReporter},
    types::{
        BuildConfig, CacheConfig, ChunkConfig, DiscoveryConfig, ExportConfig, GenerationConfig,
        InputScope, LlmProvider,
    },
};
use indicatif::{ProgressBar, ProgressStyle};
use thiserror::Error;

#[derive(Debug, Parser)]
#[command(name = "hkb", version, about = "Humble Knowledge Builder")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Build a Q&A dataset from repository documentation.
    Build(BuildArgs),
}

#[derive(Debug, Args)]
struct BuildArgs {
    /// Repository to process.
    #[arg(long, default_value = ".")]
    repo: PathBuf,

    /// Alpaca JSONL output path.
    #[arg(long, default_value = "dataset.jsonl")]
    out: PathBuf,

    /// LLM API family.
    #[arg(long, value_enum, default_value_t)]
    provider: ProviderArg,

    /// Model name. Defaults to llama3.2 for Ollama; required otherwise.
    #[arg(long)]
    model: Option<String>,

    /// Override the provider API endpoint.
    #[arg(long)]
    endpoint: Option<String>,

    /// Environment variable containing the API key.
    #[arg(long, default_value = "OPENAI_API_KEY")]
    api_key_env: String,

    /// Read the API key from a file instead of the environment.
    #[arg(long)]
    api_key_file: Option<PathBuf>,

    /// Apply additional gitignore-style rules from a file. May be repeated.
    #[arg(long = "ignore-file", value_name = "PATH")]
    ignore_files: Vec<PathBuf>,

    /// Use a custom prompt template, relative to the repository unless absolute.
    #[arg(long, value_name = "PATH")]
    prompt_file: Option<PathBuf>,

    /// Maximum number of Markdown files to process.
    #[arg(long)]
    max_files: Option<usize>,

    /// Maximum size of one input file.
    #[arg(long, default_value_t = 1_000_000)]
    max_bytes_per_file: u64,

    /// Maximum number of Unicode characters in one chunk.
    #[arg(long, default_value_t = 4_000)]
    max_characters: usize,

    /// Number of questions requested for each chunk.
    #[arg(long, default_value_t = 3)]
    questions_per_chunk: usize,

    /// LLM sampling temperature.
    #[arg(long, default_value_t = 0.2)]
    temperature: f32,

    /// Maximum number of simultaneous LLM requests.
    #[arg(long, default_value_t = 1)]
    concurrency: usize,

    /// Number of retries for transient or malformed LLM responses.
    #[arg(long, default_value_t = 2)]
    max_retries: usize,

    /// Cache directory, relative to the repository unless absolute.
    #[arg(long, default_value = ".hkb")]
    cache_dir: PathBuf,

    /// Disable cached LLM generations.
    #[arg(long)]
    no_cache: bool,

    /// Include files excluded by .gitignore.
    #[arg(long)]
    no_gitignore: bool,
}

#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum ProviderArg {
    #[default]
    Ollama,
    OpenaiCompatible,
}

impl From<ProviderArg> for LlmProvider {
    fn from(provider: ProviderArg) -> Self {
        match provider {
            ProviderArg::Ollama => Self::Ollama,
            ProviderArg::OpenaiCompatible => Self::OpenAiCompatible,
        }
    }
}

#[derive(Debug, Error)]
enum CliError {
    #[error("--model is required for the OpenAI-compatible provider")]
    ModelRequired,
    #[error("failed to read API key file {path}")]
    ApiKeyFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("API key file is empty: {0}")]
    EmptyApiKeyFile(PathBuf),
    #[error("API key environment variable {name} is not valid Unicode")]
    InvalidApiKeyEnvironment {
        name: String,
        #[source]
        source: env::VarError,
    },
    #[error(transparent)]
    Build(#[from] BuildError),
}

#[tokio::main]
async fn main() -> ExitCode {
    match run(Cli::parse()).await {
        Ok(outcome) => {
            print_summary(&outcome);
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

async fn run(cli: Cli) -> Result<BuildOutcome, CliError> {
    match cli.command {
        Command::Build(args) => {
            let provider = LlmProvider::from(args.provider);
            let (model, endpoint, client): (String, String, Box<dyn LlmClient>) = match provider {
                LlmProvider::Ollama => {
                    let model = args.model.unwrap_or_else(|| "llama3.2".to_owned());
                    let endpoint = args
                        .endpoint
                        .unwrap_or_else(|| DEFAULT_OLLAMA_ENDPOINT.to_owned());
                    let client = Box::new(OllamaClient::new(&endpoint));
                    (model, endpoint, client)
                }
                LlmProvider::OpenAiCompatible => {
                    let api_key = resolve_api_key(args.api_key_file.as_deref(), &args.api_key_env)?;
                    let model = args.model.ok_or(CliError::ModelRequired)?;
                    let endpoint = args
                        .endpoint
                        .unwrap_or_else(|| DEFAULT_OPENAI_ENDPOINT.to_owned());
                    let client = Box::new(OpenAiCompatibleClient::new(&endpoint, api_key));
                    (model, endpoint, client)
                }
            };
            let config = BuildConfig {
                repository: args.repo,
                include: InputScope::Docs,
                prompt_file: args.prompt_file,
                discovery: DiscoveryConfig {
                    respect_gitignore: !args.no_gitignore,
                    ignore_files: args.ignore_files,
                    max_files: args.max_files,
                    max_bytes_per_file: args.max_bytes_per_file,
                },
                chunking: ChunkConfig {
                    max_characters: args.max_characters,
                },
                generation: GenerationConfig {
                    provider,
                    model,
                    endpoint: Some(endpoint),
                    questions_per_chunk: args.questions_per_chunk,
                    temperature: args.temperature,
                    concurrency: args.concurrency,
                    max_retries: args.max_retries,
                },
                export: ExportConfig {
                    output: args.out,
                    ..ExportConfig::default()
                },
                cache: CacheConfig {
                    enabled: !args.no_cache,
                    directory: args.cache_dir,
                },
            };
            let progress = ConsoleProgress::new();

            Ok(build_dataset_with_progress(client.as_ref(), &config, &progress).await?)
        }
    }
}

fn resolve_api_key(
    api_key_file: Option<&Path>,
    api_key_environment: &str,
) -> Result<Option<String>, CliError> {
    if let Some(path) = api_key_file {
        let value = fs::read_to_string(path).map_err(|source| CliError::ApiKeyFile {
            path: path.to_path_buf(),
            source,
        })?;
        let key = value.trim();
        if key.is_empty() {
            return Err(CliError::EmptyApiKeyFile(path.to_path_buf()));
        }
        return Ok(Some(key.to_owned()));
    }

    match env::var(api_key_environment) {
        Ok(value) if !value.trim().is_empty() => Ok(Some(value.trim().to_owned())),
        Ok(_) | Err(env::VarError::NotPresent) => Ok(None),
        Err(source) => Err(CliError::InvalidApiKeyEnvironment {
            name: api_key_environment.to_owned(),
            source,
        }),
    }
}

fn print_summary(outcome: &BuildOutcome) {
    let stats = &outcome.manifest.stats;
    println!("Dataset: {}", outcome.dataset_path.display());
    println!("Manifest: {}", outcome.manifest_path.display());
    println!(
        "Processed {} files into {} chunks and exported {} items",
        stats.processed_files, stats.chunks, stats.generated_items
    );
    if stats.skipped_files + stats.rejected_items + stats.duplicate_items > 0 {
        println!(
            "Skipped: {} files, {} rejected items, {} duplicates",
            stats.skipped_files, stats.rejected_items, stats.duplicate_items
        );
    }
}

#[derive(Debug)]
struct ConsoleProgress {
    bar: ProgressBar,
    active_llm: AtomicUsize,
}

impl ConsoleProgress {
    fn new() -> Self {
        let bar = ProgressBar::new_spinner();
        bar.set_style(spinner_style());
        bar.enable_steady_tick(Duration::from_millis(100));
        Self {
            bar,
            active_llm: AtomicUsize::new(0),
        }
    }

    fn log(&self, message: impl AsRef<str>) {
        self.bar.suspend(|| eprintln!("{}", message.as_ref()));
    }
}

impl ProgressReporter for ConsoleProgress {
    fn report(&self, event: ProgressEvent) {
        match event {
            ProgressEvent::DiscoveryStarted { repository } => {
                self.bar.set_style(spinner_style());
                self.bar
                    .set_message(format!("Discovering Markdown in {}", repository.display()));
            }
            ProgressEvent::DiscoveryFinished {
                discovered_files,
                selected_files,
                skipped_files,
            } => {
                self.log(format!(
                    "Discovered {discovered_files} Markdown files: {selected_files} selected, \
                     {skipped_files} skipped"
                ));
            }
            ProgressEvent::FileChunked { path, chunks } => {
                self.log(format!("Chunked {} into {chunks} chunk(s)", path.display()));
            }
            ProgressEvent::GenerationStarted { total_chunks } => {
                self.active_llm.store(0, Ordering::SeqCst);
                self.bar.disable_steady_tick();
                self.bar.set_length(total_chunks as u64);
                self.bar.set_position(0);
                self.bar.set_style(generation_style());
                self.bar.set_prefix("ready");
                self.bar.enable_steady_tick(Duration::from_millis(100));
                self.bar
                    .set_message(format!("{total_chunks} chunks queued"));
            }
            ProgressEvent::ChunkStarted {
                index,
                total,
                path,
                start_line,
                end_line,
                source,
            } => {
                self.bar.set_length(total as u64);
                match source {
                    GenerationSource::Cache => self.bar.set_prefix("cache"),
                    GenerationSource::Llm => {
                        let active = self.active_llm.fetch_add(1, Ordering::SeqCst) + 1;
                        self.bar.set_prefix(format!("LLM {active} active"));
                    }
                }
                self.bar
                    .set_message(format!("{path}:{start_line}-{end_line} ({index}/{total})"));
            }
            ProgressEvent::RetryScheduled {
                index,
                path,
                attempt,
                max_retries,
                delay_ms,
            } => {
                self.log(format!(
                    "Retry {attempt}/{max_retries} for {path} (chunk {index}) in {delay_ms} ms"
                ));
            }
            ProgressEvent::ChunkFinished {
                index,
                path,
                source,
                generated_items,
            } => {
                if source == GenerationSource::Llm {
                    self.active_llm.fetch_sub(1, Ordering::SeqCst);
                }
                self.bar.inc(1);
                let active = self.active_llm.load(Ordering::SeqCst);
                self.bar.set_prefix(format!("LLM {active} active"));
                self.bar.set_message(format!(
                    "{path} (chunk {index}) produced {generated_items} item(s)"
                ));
            }
            ProgressEvent::ValidationFinished {
                accepted_items,
                rejected_items,
                duplicate_items,
            } => {
                self.log(format!(
                    "Validated {accepted_items} items: {rejected_items} rejected, \
                     {duplicate_items} duplicates"
                ));
            }
            ProgressEvent::WritingOutput {
                dataset_path,
                manifest_path,
            } => {
                self.bar.set_prefix("write");
                self.bar.set_message(format!(
                    "{} and {}",
                    dataset_path.display(),
                    manifest_path.display()
                ));
            }
            ProgressEvent::Finished => {
                self.bar.finish_and_clear();
            }
        }
    }
}

fn spinner_style() -> ProgressStyle {
    ProgressStyle::with_template("{spinner:.cyan} [{elapsed_precise}] {wide_msg}")
        .unwrap_or_else(|_| ProgressStyle::default_spinner())
        .tick_strings(&["-", "\\", "|", "/"])
}

fn generation_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{spinner:.cyan} [{elapsed_precise}] [{bar:30.cyan/blue}] {pos}/{len} {prefix:.bold} \
         {wide_msg}",
    )
    .unwrap_or_else(|_| ProgressStyle::default_bar())
    .progress_chars("=>-")
    .tick_strings(&["-", "\\", "|", "/"])
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::{Cli, Command, ProviderArg, resolve_api_key};

    #[test]
    fn parses_build_defaults() -> Result<(), clap::Error> {
        let cli = Cli::try_parse_from(["hkb", "build"])?;
        let Command::Build(args) = cli.command;

        assert!(matches!(args.provider, ProviderArg::Ollama));
        assert_eq!(args.repo, std::path::PathBuf::from("."));
        assert_eq!(args.max_characters, 4_000);
        assert_eq!(args.api_key_env, "OPENAI_API_KEY");
        assert!(args.api_key_file.is_none());
        assert!(args.ignore_files.is_empty());
        assert!(args.prompt_file.is_none());
        assert_eq!(args.concurrency, 1);
        assert_eq!(args.max_retries, 2);
        assert!(!args.no_cache);
        Ok(())
    }

    #[test]
    fn parses_openai_compatible_provider() -> Result<(), clap::Error> {
        let cli = Cli::try_parse_from([
            "hkb",
            "build",
            "--provider",
            "openai-compatible",
            "--model",
            "local-model",
        ])?;
        let Command::Build(args) = cli.command;

        assert!(matches!(args.provider, ProviderArg::OpenaiCompatible));
        assert_eq!(args.model.as_deref(), Some("local-model"));
        Ok(())
    }

    #[test]
    fn parses_multiple_custom_ignore_files() -> Result<(), clap::Error> {
        let cli = Cli::try_parse_from([
            "hkb",
            "build",
            "--ignore-file",
            "first.ignore",
            "--ignore-file",
            "second.ignore",
        ])?;
        let Command::Build(args) = cli.command;

        assert_eq!(
            args.ignore_files,
            [
                std::path::PathBuf::from("first.ignore"),
                std::path::PathBuf::from("second.ignore")
            ]
        );
        Ok(())
    }

    #[test]
    fn parses_a_project_prompt_file() -> Result<(), clap::Error> {
        let cli = Cli::try_parse_from(["hkb", "build", "--prompt-file", "hkb-prompt.md"])?;
        let Command::Build(args) = cli.command;

        assert_eq!(
            args.prompt_file,
            Some(std::path::PathBuf::from("hkb-prompt.md"))
        );
        Ok(())
    }

    #[test]
    fn reads_and_trims_api_key_file() -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("api-key");
        std::fs::write(&path, " secret-value \n")?;

        let key = resolve_api_key(Some(&path), "UNUSED_ENVIRONMENT")?;

        assert_eq!(key.as_deref(), Some("secret-value"));
        Ok(())
    }

    #[test]
    fn rejects_empty_api_key_file() -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("api-key");
        std::fs::write(&path, " \n")?;

        let result = resolve_api_key(Some(&path), "UNUSED_ENVIRONMENT");

        assert!(matches!(result, Err(super::CliError::EmptyApiKeyFile(_))));
        Ok(())
    }
}
