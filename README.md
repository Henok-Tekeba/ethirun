# Ethi Model Runner (`ethirun`)

Run Hugging Face ASR models at a pinned commit inside isolated, reproducible
worker environments — without ever changing your own project's dependencies.

> Status: **Milestone 4 — resolver and env manager**. Inspected models can be
> compiled into bounded candidate lockfiles and built into cached, isolated
> worker environments without changing the host environment.

## Why

Model repositories ship conflicting dependency constraints. Installing their
runtime (`torch`, `transformers`, ...) into the host project is how one model's
pin breaks an entire test suite. `ethirun` keeps the host small and pushes every
model runtime into its own virtualenv, rebuilt from a lockfile on demand.

## Install (development)

```bash
uv sync --locked
uv run ethirun --help
```

## Quality gates

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

All four run in CI (`.github/workflows/ci.yml`) on Python 3.11.

## Architecture (short)

| Layer | Contents | Isolation |
| --- | --- | --- |
| Host CLI | `typer`, `pydantic`, `rich`, `huggingface_hub` | installed via `uv tool`, never shares a site-packages with your app |
| Worker | adapter code on `PYTHONPATH` | one venv per recipe at `~/.ethirun/envs/<env_hash>/` |
| Contract | JSON lines over stdin/stdout, audio as paths | host-enforced timeouts, kill on overrun |

Evidence is tracked as a status ladder (`inspected` → `resolved` → ... →
`accuracy_checked`); a rung is only ever set by code that ran the matching
check.

## Schemas and status proof

`ethirun.schemas` defines frozen Pydantic models for `Recipe`, `Report`,
`Evidence`, `CheckRecord` and `StatusLadder`. Every rung from `inspected` to
`accuracy_checked` is present. The only state allowed without proof is
`untested`:

```python
from ethirun.schemas import RungResult, RungState

RungResult(state=RungState.PASSED)  # raises: requires a check record
```

Because `recipe.toml` keeps the documented `[status]` strings, a verified rung
also carries its proof in an additive `[status_checks.<rung>]` table. A
hand-edited recipe cannot promote a rung without that proof. Reports nest the
same checks under `status.<rung>.check` in JSON.

`ethirun.store` provides TOML/JSON round trips plus `RecipeStore`, which writes
`recipes/<org>__<model>/<sha>/{recipe.toml,report.json}` and refuses to save a
recipe and report that disagree.

## Worker protocol

The worker is a separate process (one venv per recipe later) speaking one JSON
object per line:

```text
<- {"id":0,"ok":true,"result":{"protocol":1,"worker_version":"0.1.0",...}}
-> {"id":1,"op":"probe","name":"meta_build","args":{"model_dir":"..."}}
<- {"id":1,"ok":true,"result":{"verdict":"pass",...}}
-> {"id":2,"op":"shutdown"}
```

Before any adapter is imported, the worker duplicates fd 1 for protocol frames
and points fd 1 plus `sys.stdout` at stderr. Library `print` calls and raw
`os.write(1, ...)` therefore become logs instead of corrupting the stream.
`ethirun.client.WorkerClient` drains stderr continuously, enforces per-request
deadlines by killing the process group, and exposes `terminate()`/`kill()` for
hard shutdown. Errors travel as `{type, message, traceback}` and become
`WorkerOperationError` on the host.

Tests launch the real worker with the deterministic fake adapter and cover
success, structured errors, timeouts, crashes, protocol violations, noisy
stdout, startup timeout, and hard kills.

## Inspection

```bash
ethirun inspect openai/whisper-tiny --json
ethirun inspect org/model@<sha> --output report.json
ethirun inspect local/whisper --local-dir tests/fixtures/repos/whisper_local
```

Stage 0 resolves a remote revision to a full commit SHA through `HfApi` and
downloads only an allowlist: config/generation/preprocessor files, tokenizer and
vocab files, README, requirements, safetensors index, and `.py` files under a
256 KiB cap. Weight files are listed in the inventory but never downloaded. A
local directory is pinned by a deterministic 40-hex content digest, or by an
explicit `--revision`.

Stage 1 parses every downloaded `.py` with `ast`; nothing is imported or
executed. It classifies stdlib/local/third-party imports, maps modules to PyPI
names, compares third-party packages against `requirements.txt`, flags
`auto_map` as remote code, and selects `transformers_whisper`,
`transformers_ctc`, or a named unsupported family with a confidence score.
Results are typed `Report` JSON including `inventory` and `scan` sections, with
the `inspected` rung backed by a `stage0.inspect` check record.

## Resolution and isolated environments

`ethirun.resolver.Resolver` compiles the three-candidate matrix with
`uv pip compile` from adapter hints plus the `transformers_version` hint:

| Candidate | Python | Date bound |
| --- | --- | --- |
| A | 3.11 | reference commit date + 120 days |
| B | 3.12 (configurable) | same date bound |
| C | 3.11 | unbounded |

Every attempt is kept, including uv's conflict message. Successful candidates
write `requirements-A/B/C.lock`; the first success is also copied to
`requirements.lock`. Failures do not stop later candidates.

`ethirun.envs.EnvManager` then runs:

```text
uv venv ~/.ethirun/envs/<env_hash> --python 3.11
uv pip sync --python <env>/bin/python requirements.lock
uv pip check --python <env>/bin/python
```

`env_hash` is SHA-256 over lockfile bytes, Python version, and worker version.
A completion marker makes a matching environment a no-op on the next build: uv
is not invoked and the environment is never patched in place. Corrupt or partial
environments are deleted and rebuilt. The manager passes `--python` explicitly
for every sync/check so the host `.venv` can never be the install target.

Tests use locally built wheels with `--no-index --find-links`, so resolution and
env construction are exercised for real without network access. A dedicated
test snapshots the host distribution list before and after a build and requires
it to be identical.

## Roadmap

- [x] **M0** Scaffold: uv, ruff, mypy strict, pytest, CI
- [x] **M1** Schemas and store: recipe/report/evidence, enforced ladder, TOML/JSON
- [x] **M2** Worker and protocol: stdlib-only runtime, host client, fake adapter, timeouts
- [x] **M3** Inspector: pinned SHA, allowlisted files, AST-only scan, adapter evidence
- [x] **M4** Resolver and env manager: candidate locks, cached isolated venvs, uv check
- [ ] M5 Probes and adapters
- [ ] M6 Real models

## Host dependency policy

`tests/test_dependency_boundary.py` fails if `torch`, `transformers`, or any
other model runtime is added to the host project metadata. That failure is the
point: those packages belong in worker environments only.
