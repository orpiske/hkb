# Humble Knowledge Builder

HKB builds traceable question-and-answer datasets from repository documentation. The current MVP
discovers Markdown files, splits them into stable chunks, generates grounded questions and answers
with an LLM, validates and deduplicates the results, and exports extended Alpaca JSONL plus a build
manifest.

## Build

```bash
cargo build
```

## Ollama

Start Ollama with the model you want to use, then run:

```bash
cargo run -- build \
  --repo /path/to/repository \
  --out dataset.jsonl \
  --provider ollama \
  --model llama3.2
```

The default Ollama endpoint is `http://localhost:11434`.

During a build, HKB reports every Markdown file it chunks and shows a live per-chunk progress bar.
The active status distinguishes LLM requests from cache hits and includes the source file and line
range currently being processed.

## OpenAI-Compatible APIs

Set an API key when the server requires one:

```bash
export OPENAI_API_KEY="..."
```

Then provide the model and, for non-default servers, the endpoint:

```bash
cargo run -- build \
  --repo /path/to/repository \
  --out dataset.jsonl \
  --provider openai-compatible \
  --model your-model \
  --endpoint https://example.com/v1
```

The key is read from the environment and is never stored in the manifest or cache.

## Output

`dataset.jsonl` contains one extended Alpaca record per line:

```json
{
  "instruction": "What does HKB build?",
  "input": "",
  "output": "HKB builds question-and-answer datasets.",
  "metadata": {
    "id": "...",
    "source": {
      "path": "README.md",
      "start_line": 1,
      "end_line": 4
    },
    "chunk_id": "...",
    "model": "llama3.2",
    "generated_at": "2026-07-30T12:00:00Z"
  }
}
```

`manifest.json` records the repository, public build configuration, prompt version and template,
Git commit when available, and processing statistics.

Cached generations are stored under `.hkb` by default. A cache entry is invalidated when the chunk,
provider, endpoint, model, prompt version, temperature, or requested question count changes.

## Quality Checks

```bash
cargo fmt --check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
```
