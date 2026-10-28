"""End-to-end tiny-model tests: probes, load, and real transcription.

These tests are skipped in the normal host suite because torch/transformers are
not host dependencies. CI runs them in a separate job with:

    ETHIRUN_RUN_TINY_MODEL_TESTS=1
    ETHIRUN_TINY_WORKER_PYTHON=<worker-venv>/bin/python

The worker venv is disposable; the host never imports torch or transformers.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest

from ethirun.client import WorkerClient, WorkerConfig
from ethirun.schemas import Rung, RungState
from ethirun.verifier import verify_worker
from tests.worker_harness import WORKER_DIR

RUN_TINY = os.environ.get("ETHIRUN_RUN_TINY_MODEL_TESTS") == "1"
TINY_PYTHON = os.environ.get("ETHIRUN_TINY_WORKER_PYTHON")

pytestmark = [
    pytest.mark.tiny_models,
    pytest.mark.skipif(
        not RUN_TINY or not TINY_PYTHON,
        reason="set ETHIRUN_RUN_TINY_MODEL_TESTS=1 and ETHIRUN_TINY_WORKER_PYTHON",
    ),
]


@dataclass(frozen=True)
class TinyModels:
    whisper: Path
    ctc: Path
    clip: Path


@pytest.fixture(scope="session")
def tiny_models(tmp_path_factory: pytest.TempPathFactory) -> TinyModels:
    assert TINY_PYTHON is not None
    python = Path(TINY_PYTHON)
    if not python.is_file():
        pytest.fail(f"tiny worker python not found: {python}")

    root = tmp_path_factory.mktemp("ethirun-tiny-models")
    script = Path(__file__).resolve().parent / "scripts" / "make_tiny_models.py"
    completed = subprocess.run(
        [str(python), str(script), str(root)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
    )
    if completed.returncode != 0:
        pytest.fail(f"tiny model generation failed:\n{completed.stdout}\n{completed.stderr}")

    return TinyModels(
        whisper=root / "whisper",
        ctc=root / "ctc",
        clip=root / "clip.wav",
    )


@contextmanager
def _worker(adapter: str) -> Iterator[WorkerClient]:
    assert TINY_PYTHON is not None
    client = WorkerClient(
        WorkerConfig(
            python=TINY_PYTHON,
            worker_dir=WORKER_DIR,
            adapter=adapter,
            startup_timeout=60.0,
            default_timeout=300.0,
        )
    )
    client.start()
    try:
        yield client
    finally:
        client.close()


def test_whisper_tiny_reaches_transcribes(tiny_models: TinyModels) -> None:
    with _worker("transformers_whisper") as client:
        result = verify_worker(
            client,
            model_dir=str(tiny_models.whisper),
            audio_path=str(tiny_models.clip),
        )

        assert result.error is None, result.error
        assert result.status.state_of(Rung.IMPORTS) is RungState.PASSED
        assert result.status.state_of(Rung.CONSTRUCTS) is RungState.PASSED
        assert result.status.state_of(Rung.WEIGHTS_COMPATIBLE) is RungState.NOT_APPLICABLE
        assert result.status.state_of(Rung.LOADS) is RungState.PASSED
        assert result.status.state_of(Rung.TRANSCRIBES) is RungState.PASSED
        assert result.text is not None
        assert result.text.strip()

        architecture = result.probes["architecture"]
        assert architecture["detail"]["class"] == "WhisperForConditionalGeneration"
        meta = result.probes["meta_build"]
        assert meta["detail"]["meta_device"] is True
        assert meta["detail"]["parameters"] > 0
        assert result.probes["safetensors"]["verdict"] == "not_applicable"
        assert result.load["loaded"] is True


def test_ctc_tiny_reaches_transcribes(tiny_models: TinyModels) -> None:
    with _worker("transformers_ctc") as client:
        result = verify_worker(
            client,
            model_dir=str(tiny_models.ctc),
            audio_path=str(tiny_models.clip),
        )

        assert result.error is None, result.error
        assert result.status.state_of(Rung.IMPORTS) is RungState.PASSED
        assert result.status.state_of(Rung.CONSTRUCTS) is RungState.PASSED
        assert result.status.state_of(Rung.WEIGHTS_COMPATIBLE) is RungState.NOT_APPLICABLE
        assert result.status.state_of(Rung.LOADS) is RungState.PASSED
        assert result.status.state_of(Rung.TRANSCRIBES) is RungState.PASSED
        assert result.text is not None
        assert result.text.strip()

        architecture = result.probes["architecture"]
        assert architecture["detail"]["class"] == "Wav2Vec2ForCTC"
        meta = result.probes["meta_build"]
        assert meta["detail"]["meta_device"] is True
        assert meta["detail"]["parameters"] > 0
        assert result.probes["safetensors"]["verdict"] == "not_applicable"
        assert result.load["loaded"] is True
