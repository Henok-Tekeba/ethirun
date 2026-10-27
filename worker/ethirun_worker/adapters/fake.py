"""Deterministic fake adapter for protocol and client tests.

It deliberately prints to both Python-level stdout and file descriptor 1, so
tests can prove those writes are diverted to stderr and never corrupt protocol
frames. Test control knobs are passed through request args.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Mapping
from typing import Any


class FakeAdapter:
    """Adapter with no model dependencies and controllable failures."""

    name = "fake"

    def probe(self, name: str, args: Mapping[str, Any]) -> dict[str, Any]:
        self._test_controls(args)

        if name == "explode":
            raise RuntimeError("intentional probe failure")
        if name == "crash":
            os._exit(7)
        return {
            "verdict": "pass",
            "name": name,
            "args": dict(args),
        }

    def load(self, model_dir: str, device: str) -> dict[str, Any]:
        self._test_controls({})

        if model_dir == "explode":
            raise RuntimeError("intentional load failure")
        return {"loaded": True, "model_dir": model_dir, "device": device}

    def transcribe(self, audio_path: str, language: str | None) -> str:
        self._test_controls({})

        if audio_path.endswith("explode.wav"):
            raise ValueError("intentional transcription failure")
        text = os.environ.get("ETHIRUN_FAKE_TEXT", "ሰላም ዓለም")
        if language is None:
            return text
        return f"{text} [{language}]"

    @staticmethod
    def _test_controls(args: Mapping[str, Any]) -> None:
        delay = args.get("delay_seconds", 0)
        if delay:
            if not isinstance(delay, int | float) or delay < 0:
                raise ValueError("delay_seconds must be a non-negative number")
            time.sleep(float(delay))

        if args.get("noise"):
            print("fake library noise on stdout", flush=True)
            os.write(1, b"fake raw fd noise\n")
            print("fake library noise on stderr", file=sys.stderr, flush=True)
