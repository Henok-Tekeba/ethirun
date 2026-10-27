# Worker (stdlib-only)

The isolated runtime for Milestone 2 onward. It is placed on the worker venv's
`PYTHONPATH` and launched as `python -m ethirun_worker`. It imports only the
standard library and its own package, guarded by
`tests/test_worker_isolation.py`.

- `protocol.py` — JSON-lines framing and structured errors
- `streams.py` — private stdout descriptor; `sys.stdout` and fd 1 point at stderr
- `server.py` — request dispatch over stdin, one JSON object per line
- `adapters/` — `WorkerAdapter` contract and the deterministic fake adapter

The worker is never imported by the host client; the contract is JSON lines over
stdin/stdout. Future transformer adapters arrive in Milestone 5.
