# Ethio Runner (`ethirun`)

**Find out whether a speech model actually works — before you download it, install it, or ship it.**

[![CI](https://github.com/Henok-Tekeba/ethirun/actions/workflows/ci.yml/badge.svg)](https://github.com/Henok-Tekeba/ethirun/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

`ethirun` takes a Hugging Face ASR repo, pins it to a commit, inspects it without
running its code, builds its runtime in a disposable environment, and records
exactly how far it got — from “the files look plausible” to “it produced a
transcript.”

It was built for the messy side of speech AI: Amharic and other
Ethiopian-language checkpoints, old model repos, and machines that should never
have `torch` installed globally.

## The problem

A Hugging Face model repo is not a package. It can be missing its tokenizer, pin
a `torch` that fights your app, ship Python that auto-loads as remote code, or
load successfully and return an empty string.

“Installed” is also not “works”:

- installed ≠ imports
- imports ≠ the model constructs
- constructs ≠ real weights load
- loads ≠ produces text
- produces text ≠ accurate

Normally you discover this after downloading gigabytes, waiting, and breaking
your environment. For low-resource languages, often nobody has checked the model
at all.

## What it does

Four commands, each one adding evidence:

```bash
ethirun inspect  openai/whisper-tiny   # pin SHA + static scan. No weights.
ethirun plan     openai/whisper-tiny   # resolve deps, write recipe + lockfiles.
ethirun build    openai/whisper-tiny   # isolated env + probes P1–P5.
ethirun verify   openai/whisper-tiny --audio clip.wav --language en
```

```text
Inspected openai/whisper-tiny @ 169d4a4…
Python files: 0 scanned; remote code: no
Adapter: transformers_whisper confidence 0.95 (supported)
Status: inspected = passed
```

`plan`, `build`, and `verify` write `recipe.toml`, `report.json`, and dependency
lockfiles under:

```text
recipes/<org>__<model>/<pinned-sha>/
```

`inspect` prints its report and can save one with `--output`. Re-running uses the
same SHA and the same content-addressed environment. If a recipe says
`transcribes`, there is a check record showing the transcription that proved it.

## The evidence ladder

| Rung | Proven by |
| --- | --- |
| `inspected` | repo inventory + AST-only Python scan |
| `resolved` | dependency candidates compiled by `uv` |
| `installed` | isolated venv synced from the lockfile + `uv pip check` |
| `imports` | torch/transformers import + architecture class resolution |
| `constructs` | meta-device model build + processor/tokenizer load |
| `weights_compatible` | safetensors names/shapes compared with the model |
| `loads` | real weights downloaded and loaded |
| `transcribes` | non-empty smoke transcription |
| `accuracy_checked` | labeled CER/WER slice — **not implemented** |

A rung other than `untested` must carry a matching check record. Hand-edit a
TOML status without its proof, and loading the recipe fails.

## Quick start

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Henok-Tekeba/ethirun.git
cd ethirun
uv sync --locked

uv run ethirun inspect openai/whisper-tiny
uv run ethirun plan    openai/whisper-tiny
uv run ethirun build   openai/whisper-tiny
uv run ethirun verify  openai/whisper-tiny --audio clip.wav --language en
```

Model runtimes go only into `~/.ethirun/envs/<hash>/`; weights go into
`~/.ethirun/models/`. Your host Python never gets `torch`, `transformers`, or
`soundfile`.

Runs are CPU-verified. `--device` is available, but GPU runs are not yet
validated.

## Why not just `pip install transformers`?

Sometimes that is the right answer. Use the `transformers` pipeline directly if
you have one known-good model and a disposable environment.

Use `ethirun` when you need to:

- choose between models for a language and want evidence, not vibes;
- find out a repo is broken before spending bandwidth on weights;
- reproduce a run from a pinned commit and lockfile;
- keep model dependencies out of your app, CI, and laptop;
- keep a failure record with a clear reason instead of a stack trace.

It is not a faster installer. It does more work up front so you can fail early
and leave proof.

## Under the hood

- **Static first.** Repo Python is parsed with `ast`, never executed. `auto_map`
  is flagged as remote code, not trusted.
- **Isolated.** The host process never imports model libraries. Worker envs are
  content-addressed and rebuilt from lockfiles, never patched.
- **Evidence first.** Status is a validated model, not a display label. Stages
  that did not run cannot claim a rung.

```text
host:    inspect → resolve → build env → probes → weights → transcribe
                         │ JSON lines over stdin/stdout
worker:                  torch / transformers / ASR adapters
```

## Current limits

- **No accuracy claim.** `transcribes` means non-empty text, not correct text.
  CER/WER on a labeled slice is the next milestone.
- **No `ethirun run`.** The Python `Transcriber` API and a serving endpoint are
  not implemented yet.
- **Two adapters.** Transformers Whisper and Transformers CTC (Wav2Vec2,
  HuBERT, Wav2Vec2-BERT). NeMo, ONNX, and custom `auto_map` repos are detected
  and reported as unsupported; their code is never executed.
- **CPU only so far.** No verified CUDA/MPS results.

## Development

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

The tiny-model inference tests need a disposable worker venv with
`torch`, `transformers`, and `soundfile`; see
[tests/test_tiny_models.py](tests/test_tiny_models.py) and the `tiny-models` CI
job.

## Roadmap

- CER/WER accuracy rung on a versioned Amharic evaluation set
- `ethirun run` and an OpenAI-compatible local transcription endpoint
- More adapters behind the same evidence ladder

## License

MIT. See [LICENSE](LICENSE).
