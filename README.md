# Ethi Model Runner (`ethirun`)

Run Hugging Face ASR models at a pinned commit inside isolated, reproducible
worker environments — without ever changing your own project's dependencies.

> Status: **Milestone 0 — scaffold**. The CLI shell exists; model commands land
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

## Roadmap

- [x] **M0** Scaffold: uv, ruff, mypy strict, pytest, CI
- [ ] M1 Schemas and store
- [ ] M2 Worker and protocol
- [ ] M3 Inspector
- [ ] M4 Resolver and env manager
- [ ] M5 Probes and adapters
- [ ] M6 Real models

## Host dependency policy

`tests/test_dependency_boundary.py` fails if `torch`, `transformers`, or any
other model runtime is added to the host project metadata. That failure is the
point: those packages belong in worker environments only.
