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

### Project-Specific Prompts

The default Q&A prompt lives in `prompts/qa-v1.md`. To customize generation for a project, create a
prompt file in that repository and pass its repository-relative path:

```bash
cargo run -- build \
  --repo /path/to/repository \
  --prompt-file hkb-prompt.md
```

Custom templates must contain `{{chunk_text}}` so generated answers remain grounded in the source.
They may also use these placeholders:

- `{{questions_per_chunk}}`
- `{{path}}`
- `{{start_line}}`
- `{{end_line}}`

Absolute prompt paths are also accepted. The resolved prompt text is recorded in `manifest.json`
and included in the cache identity, so changing the prompt invalidates only the affected cached
generations.

### Parallel Generation

Use `--concurrency` to allow multiple chunks to be sent to the LLM at the same time:

```bash
cargo run -- build \
  --repo /path/to/repository \
  --model llama3.2 \
  --concurrency 2
```

The default is `1`, preserving sequential behavior. Completed chunks are cached immediately, even
while other requests are still running. HKB restores the original chunk order before validation
and export, so changing concurrency does not reorder the dataset. Retryable failures use exponential
backoff; configure the retry count with `--max-retries` (default: `2`).

Ollama also limits parallel work on the server. To serve two requests concurrently, start it with a
matching limit:

```bash
OLLAMA_NUM_PARALLEL=2 ollama serve
```

Higher concurrency can increase throughput, but Ollama's memory use grows with the number of
parallel requests and their context lengths. Increase this setting gradually for the available
hardware.

### Additional Ignore Rules

Use `--ignore-file` to apply repository-relative gitignore-style exclusions without modifying the
target repository:

```text
path/to/module/
path/to/another/**test.md
```

```bash
cargo run -- build \
  --repo /path/to/repository \
  --ignore-file ~/.config/hkb/java.ignore
```

The option may be repeated. Custom rules apply even with `--no-gitignore` and take final exclusion
precedence over repository rules. Relative ignore-file paths are resolved from the directory where
HKB is launched; patterns inside each file are rooted at `--repo`.

## OpenAI-Compatible APIs

Set an API key when the server requires one:

```bash
export OPENAI_API_KEY="..."
```

To use a different environment variable:

```bash
cargo run -- build \
  --provider openai-compatible \
  --model your-model \
  --api-key-env HKB_API_KEY
```

Alternatively, read the key from a file:

```bash
cargo run -- build \
  --provider openai-compatible \
  --model your-model \
  --api-key-file ~/.config/hkb/api-key
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

`--api-key-file` takes precedence over `--api-key-env`. The key is never stored in the manifest,
cache, or progress output. HKB deliberately does not accept a literal `--api-key` value because
command-line arguments can be exposed through shell history and process inspection.

## Output

`--out` accepts either a JSONL file path or an existing directory. For a directory, HKB writes
`dataset.jsonl` and `manifest.json` inside it:

```bash
cargo run -- build --repo /path/to/repository --out /path/to/output-directory
```

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
provider, endpoint, model, prompt contents, temperature, or requested question count changes.

## Verify A Dataset

Use `verify` to evaluate every generated Q&A pair against its original source chunk:

```bash
cargo run -- verify \
  --dataset /path/to/dataset.jsonl \
  --manifest /path/to/manifest.json \
  --repo /path/to/repository \
  --model llama3.2 \
  --out verification.jsonl \
  --concurrency 2
```

The verifier independently checks that the answer is grounded in the source, the question is
self-contained, and the answer directly addresses the question. Each report record has one of
three verdicts:

- `accepted`: all checks passed.
- `rejected`: at least one quality check failed.
- `unverifiable`: the original chunk could not be reconstructed, usually because the repository
  changed after the dataset was built.

Verification never modifies the input dataset. It writes `verification.jsonl` and a neighboring
`verification-manifest.json`. Decisions are cached under `.hkb/verify` and are invalidated when the
question, answer, source chunk, verifier model, or verification prompt changes.

The default verification prompt lives in `prompts/verify-v1.md`. A project-specific prompt can be
selected with `--prompt-file`; it must contain `{{question}}`, `{{answer}}`, and `{{chunk_text}}`.

By default, quality findings do not make the command fail. For CI, use `--fail-on rejected` to fail
when a pair is rejected, or `--fail-on any` to fail for either rejected or unverifiable records. The
reports are written before the nonzero exit code is returned.

## Quality Checks

```bash
cargo fmt --check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
```
