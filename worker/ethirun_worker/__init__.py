"""Stdlib-only worker runtime for ethirun.

This package is placed on the worker environment's ``PYTHONPATH`` and run with
``python -m ethirun_worker``. It must never import the host package or any
third-party library from the host environment.
"""

from __future__ import annotations

from ethirun_worker.protocol import WORKER_VERSION

__all__ = ["WORKER_VERSION"]
