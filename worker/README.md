# Worker (stdlib-only at import time)

The isolated runtime for Milestone 2 onward. It is placed on the worker venv's
`PYTHONPATH` and launched as `python -m ethirun_worker`. Runtime libraries
(torch, transformers, numpy, soundfile) are imported lazily inside probes and
adapter methods, so the source itself needs no installation and top-level
imports stay standard-library only. `tests/test_worker_isolation.py` enforces
that top-level boundary with AST.

- `protocol.py` — JSON-lines framing and structured errors
- `streams.py` — private stdout descriptor; `sys.stdout` and fd 1 point at stderr
- `server.py` — request dispatch over stdin, one JSON object per line
- `probes.py` — pure P5 tensor-name/shape comparison
- `audio.py` — soundfile/numpy 16 kHz mono normalization, no ffmpeg
- `adapters/fake.py` — deterministic protocol adapter
- `adapters/transformers_asr.py` — shared P1–P5 and pipeline adapter for:
  - `transformers_whisper`
  - `transformers_ctc`

The worker is never imported by the host client; the contract is JSON lines over
stdin/stdout. `ethirun.verifier` runs P1–P5, load and smoke transcription on the
host, setting evidence-rung checks only after each op succeeds.
