"""Audio loading for worker smoke tests.

soundfile and numpy are imported lazily so the worker source remains importable
without model-runtime dependencies. Stereo is downmixed and audio is linearly
resampled to 16 kHz; ffmpeg is not required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

TARGET_SAMPLE_RATE = 16_000


class AudioError(RuntimeError):
    """The audio file could not be read or normalized."""


def load_audio_16k_mono(path: str) -> Any:
    """Return float32 mono audio at 16 kHz."""
    import numpy as np
    import soundfile as sf

    try:
        samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    except Exception as exc:
        raise AudioError(f"could not read {Path(path).name}: {exc}") from exc

    if samples.size == 0:
        raise AudioError("audio file is empty")

    mono = samples.mean(axis=1, dtype=np.float32)
    if sample_rate != TARGET_SAMPLE_RATE:
        duration = mono.shape[0] / float(sample_rate)
        target_length = max(1, round(duration * TARGET_SAMPLE_RATE))
        source_positions = np.linspace(0.0, duration, num=mono.shape[0], endpoint=False)
        target_positions = np.linspace(0.0, duration, num=target_length, endpoint=False)
        mono = np.interp(target_positions, source_positions, mono).astype(np.float32)

    return mono
