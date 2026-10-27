# Ethi Model Runner (`ethirun`)

Run Hugging Face ASR models at a pinned commit inside isolated, reproducible
worker environments — without ever changing your own project's dependencies.

> Status: **Milestone 2 — worker and protocol**. The CLI shell, schemas/store,
> and isolated worker process with host client are in place; model commands land
> milestone by milestone.

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

## Roadmap

- [x] **M0** Scaffold: uv, ruff, mypy strict, pytest, CI
- [x] **M1** Schemas and store: recipe/report/evidence, enforced ladder, TOML/JSON
- [x] **M2** Worker and protocol: stdlib-only runtime, host client, fake adapter, timeouts
- [ ] M3 Inspector
- [ ] M3 Inspector
- [ ] M4 Resolver and env manager
- [ ] M5 Probes and adapters
- [ ] M6 Real models

## Host dependency policy

`tests/test_dependency_boundary.py` fails if `torch`, `transformers`, or any
other model runtime is added to the host project metadata. That failure is the
point: those packages belong in worker environments only.
