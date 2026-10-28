"""Ethi Model Runner: run Hugging Face ASR models in isolated workers."""

from __future__ import annotations

__all__ = ["WORKER_VERSION", "__version__"]

__version__ = "0.1.0"

# Keep in sync with worker/ethirun_worker; the env hash includes this value.
WORKER_VERSION = "0.2.0"
