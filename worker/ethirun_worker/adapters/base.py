"""Worker adapter contract shared by all worker-side adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol


class WorkerAdapter(Protocol):
    """How one model family loads and transcribes inside a worker venv."""

    name: str

    def probe(self, name: str, args: Mapping[str, Any]) -> Any:
        """Run one named probe without downloading weights."""
        ...

    def load(self, model_dir: str, device: str) -> Any:
        """Load a model from a local directory."""
        ...

    def transcribe(self, audio_path: str, language: str | None) -> str:
        """Transcribe one audio file and return text."""
        ...
