# Ethi Model Runner (`ethirun`)

Run and verify Hugging Face ASR models at a pinned commit inside disposable,
isolated worker environments — without installing model runtimes into the host
project.

`ethirun` is a small CLI and Python toolkit that takes an ASR repo from
“interesting model on Hugging Face” to a reproducible environment and a
machine-readable verification report.

> **This repository ships the tool only.** Model weights, generated recipes,
> lockfiles, smoke audio, and worker virtualenvs stay outside git. They are
> generated locally under `recipes/` and `~/.ethirun/`.

---

## The problem

A Hugging Face model repo is not a normal package. It usually contains:

- configuration and tokenizer files;
- large PyTorch/safetensors weights;
- arbitrary Python that may be auto-loaded as remote code;
- dependency constraints written for the author's machine.

Installing a model's runtime into the host app creates a familiar failure: one
repo pins `transformers<5`, another wants a newer torch, and an old repo wants a
`huggingface_hub` API that no longer exists. A single “successful” install can
break the host application and its test suite.

There is a second problem: “it installed” is not the same as “it imported”,
“it built”, “it loaded real weights”, or “it produced text”. These are different
claims and need different evidence.

`ethirun` solves both:

1. **Isolation** — model libraries live only in per-recipe worker venvs.
2. **Evidence** — every status rung is backed by a check that actually ran.

## What we set out to do

- Pin remote repos to full commit SHAs.
- Read repo code statically with `ast`, never execute it by default.
- Resolve dependencies with date bounds and fallback candidates.
- Build reproducible environments from lockfiles.
- Probe imports, architecture resolution, meta-device construction, processor
  loading, and safetensors headers.
- Load real weights and run a smoke transcription.
- Record every result as machine-readable evidence.
- Prove the host environment was never modified.

## What we achieved

### A working end-to-end CLI

```text
ethirun inspect  stages 0–1  pinned SHA + allowlisted metadata + AST evidence
ethirun plan     stages 0–2  recipe + candidate lockfiles
ethirun build    stages 3–4  isolated env + P1–P5 probes
ethirun verify   stages 5–6  weights + load + smoke transcription
```

All four commands work against remote repos, with `--local-dir` available for
inspection of fixtures or offline snapshots.

### Host/worker isolation

The host process never imports torch or transformers. Worker dependencies are
installed only into:

```text
~/.ethirun/envs/<env_hash>/
```

The hash covers lockfile bytes, Python version, and worker version. A valid env
is reused without invoking uv again; partial or corrupt envs are rebuilt from
the lockfile, never patched.

The isolation test snapshots every installed distribution before and after a
real env build and requires the two lists to be identical. The host dependency
tree contains no torch, transformers, tokenizers, soundfile, or datasets.

### Enforced evidence ladder

The status ladder is a first-class model, not a display label:

| Rung | Proven by |
| --- | --- |
| `inspected` | Stage 0 inventory + Stage 1 AST scan |
| `resolved` | `uv pip compile` candidate |
| `installed` | `uv venv` + `uv pip sync` + `uv pip check` |
| `imports` | P1 imports + P2 architecture class |
| `constructs` | P3 meta-device build + P4 processor/tokenizer |
| `weights_compatible` | P5 safetensors name/shape comparison |
| `loads` | Real weight load |
| `transcribes` | Non-empty smoke transcription |
| `accuracy_checked` | Labeled accuracy slice — **not implemented** |

A rung other than `untested` requires a matching check record. The recipe and
report ladders are validated together, and hand-editing a TOML status without
its proof is rejected.

### Static-first inspection

Stage 1 parses every downloaded `.py` with `ast`. It never imports or executes
repo code. It classifies standard-library, local, and third-party imports, maps
modules to PyPI names, compares them with `requirements.txt`, flags `auto_map`
as remote code, and selects an adapter family with a confidence score.

### Candidate dependency resolution

The resolver compiles up to three candidates and keeps every outcome:

| Candidate | Python | Package world |
| --- | --- | --- |
| A | 3.11 | commit date + 120 days |
| B | 3.12 by default | same date bound |
| C | 3.11 | unbounded |

Seeds come from adapter hints plus the model config's `transformers_version` as
a hint, not a hard pin. Failed candidates retain uv's conflict message. If an
environment or probe fails during `build`, `ethirun` can fall back to the next
resolved candidate and update the recipe selection.

### Worker protocol and probes

The worker is a separate process speaking one JSON object per line. It reserves
a private stdout descriptor for protocol frames and redirects `sys.stdout` and
fd 1 to stderr, so library prints cannot corrupt the protocol. The host drains
stderr continuously, enforces deadlines by killing the process group, and
supports graceful SIGTERM plus hard SIGKILL.

P1–P5 run inside the worker env:

1. import torch + transformers;
2. resolve the architecture class named in `config.json`;
3. build the model on `torch.device("meta")`;
4. load processor and tokenizer from local files;
5. fetch safetensors headers over HTTP and compare tensor names/shapes.

P5 understands tied weights, known weight-norm renames, sharded safetensors, and
not-applicable cases for local models or repos without safetensors.

### Two adapters, real pipeline behavior

- `transformers_whisper`
- `transformers_ctc`

Both use the transformers ASR pipeline on a numpy array prepared in the worker.
Audio is loaded with soundfile, downmixed, and linearly resampled to 16 kHz;
ffmpeg is not required. The worker source imports model libraries lazily, so the
source itself stays importable on `PYTHONPATH` without installing torch.

### Tested, including real inference

```text
170 host tests pass
2 tiny-model inference tests pass in a separate CI job
ruff, ruff format, mypy --strict: clean
```

The tiny-model CI job creates a disposable worker venv, installs CPU
torch/transformers/soundfile only there, builds random-weight Whisper and CTC
models inside that venv, and requires both to reach `transcribes` with non-empty
text. The model artifacts are never committed.

### Real-world issues resolved while building

| Problem | Fix |
| --- | --- |
| Tied output weights omitted from safetensors | P5 consults `_tied_weights_keys` (Transformers 5 dict and 4.x list forms) |
| Newer torch renames weight-norm tensors | Normalize `parametrizations.weight.original0/1` → `weight_g/weight_v` |
| Old huggingface-hub has no `huggingface_hub.errors` | Worker falls back to `huggingface_hub.utils` |
| Old torch wheel fails `uv pip check` on the current platform | `build` records the attempt and falls back to the next resolved candidate |
| Nested `None` cannot be written to TOML | Drop nulls recursively in TOML while keeping JSON nulls |
| Repos publish both safetensors and `.bin` | Prefer safetensors during weight staging, avoiding duplicate downloads |
| Library output corrupts worker protocol | Private stdout fd; all other writes go to stderr |

## Quick start

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked
uv run ethirun --help
```

Inspect a repo without downloading weights:

```bash
uv run ethirun inspect openai/whisper-tiny
```

Build a recipe and isolated env, then verify with a smoke clip:

```bash
uv run ethirun plan openai/whisper-tiny
uv run ethirun build openai/whisper-tiny
uv run ethirun verify openai/whisper-tiny --audio clip.wav --language en
```

Generated recipes and reports land under `recipes/`; weights and envs land under
`~/.ethirun/`. Both are local artifacts and are not committed here.

For an offline fixture:

```bash
uv run ethirun inspect local/whisper \
  --local-dir tests/fixtures/repos/whisper_local --json
```

## How it works

```text
                     host process (no model runtimes)
  ┌─────────────────────────────────────────────────────────────┐
  │ Inspector → Resolver → Env Manager → Verifier → recipes/    │
  │   stages 0–1     2          3           4–6                 │
  └───────────────┬─────────────────────────────────────────────┘
                  │ JSON lines over stdin/stdout; audio as paths
  ┌───────────────▼─────────────────────────────────────────────┐
  │ worker venv: torch / transformers / adapters                │
  └─────────────────────────────────────────────────────────────┘
```

| Stage | Behavior |
| --- | --- |
| 0 Inspect | Resolve SHA; fetch allowlisted metadata only; list weights without downloading |
| 1 Static scan | AST-only import scan; declared/undeclared packages; remote-code flag; adapter scoring |
| 2 Resolve | A/B/C candidate matrix; lockfiles and conflict messages |
| 3 Env | Content-addressed venv from `uv venv` + `uv pip sync` + `uv pip check` |
| 4 Probes | P1–P5 inside the worker, no weights except P5 headers |
| 5 Load | Download weights at the pinned SHA and load in the worker |
| 6 Smoke | Normalize audio and transcribe; require non-empty text |
| 7 Accuracy | Not implemented |

### Adapter support

| Family | Status |
| --- | --- |
| `transformers_whisper` | Supported and tested |
| `transformers_ctc` | Supported and tested |
| `fake` | Test adapter used by protocol/isolation tests |
| `nemo`, `onnx`, `custom_script` | Detected and reported as unsupported; never executed |

## CLI reference

| Command | Description |
| --- | --- |
| `ethirun inspect <repo>[@rev]` | Stages 0–1 evidence report |
| `ethirun plan <repo>` | Stages 0–2 recipe + candidate locks |
| `ethirun build <repo>` | Stages 3–4 env + P1–P5 |
| `ethirun verify <repo> --audio clip.wav` | Stages 5–6 weights + smoke transcription |

Useful flags: `--local-dir`, `--revision`, `--recipes-dir`, `--cache-dir`,
`--envs-dir`, `--models-dir`, `--language`, `--device`, `--json`, `--output`.

## Development

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

Tiny-model inference tests need a disposable worker venv:

```bash
uv venv "$TMPDIR/ethirun-tiny-worker" --python 3.11
uv pip install --python "$TMPDIR/ethirun-tiny-worker/bin/python" \
  torch transformers soundfile
ETHIRUN_RUN_TINY_MODEL_TESTS=1 \
ETHIRUN_TINY_WORKER_PYTHON="$TMPDIR/ethirun-tiny-worker/bin/python" \
uv run pytest -m tiny_models
```

CI runs the same path in a separate job with `HF_HUB_OFFLINE=1`, so the main
test job never gains model runtimes.

## What is not claimed

- **Accuracy.** CER/WER on a labeled slice is not implemented. A non-empty
  transcript is not an accuracy claim.
- **GPU.** Verified runs were CPU-only. The worker accepts a device option, but
  CUDA was not verified here.
- **Arbitrary models.** NeMo, ONNX, and custom `auto_map` repos are reported as
  unsupported. Remote code is never executed.
- **Packaged worker for `uv tool install`.** The worker source is resolved from
  the checkout or `ETHIRUN_WORKER_DIR`; wheel packaging is future work.
- **`ethirun run` / `Transcriber` API.** Not implemented yet.

## Repository layout

```text
src/ethirun/         CLI, inspector, resolver, env manager, verifier, store
worker/              stdlib-only protocol + lazy transformer adapters
tests/               unit/integration tests, fixtures, tiny-model paths
.github/workflows/   checks, host isolation, tiny-model inference
recipes/             generated locally; git-ignored in this tool repo
```

## License

MIT. See [LICENSE](LICENSE).
