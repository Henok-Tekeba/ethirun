"""Verifier tests with a fake worker; no torch or subprocess required."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from ethirun.schemas import Rung, RungState
from ethirun.verifier import verify_worker

NOW = datetime(2026, 10, 28, 9, 0, tzinfo=UTC)


class FakeClient:
    def __init__(
        self,
        *,
        fail_probe: str | None = None,
        p5_verdict: str = "not_applicable",
        text: str = "ሰላም",
        fail_load: bool = False,
    ) -> None:
        self.fail_probe = fail_probe
        self.p5_verdict = p5_verdict
        self.text = text
        self.fail_load = fail_load
        self.calls: list[tuple[str, str | None, dict[str, Any]]] = []

    def probe(
        self,
        name: str,
        *,
        timeout: float | None = None,
        **args: Any,
    ) -> Any:
        self.calls.append(("probe", name, dict(args)))
        if name == self.fail_probe:
            raise RuntimeError(f"{name} failed on purpose")
        if name == "imports":
            return {"verdict": "pass", "detail": {"torch": "2.6.0"}}
        if name == "architecture":
            return {"verdict": "pass", "detail": {"class": "WhisperForConditionalGeneration"}}
        if name == "meta_build":
            return {"verdict": "pass", "detail": {"parameters": 12}}
        if name == "processor":
            return {"verdict": "pass", "detail": {"processor": "WhisperProcessor"}}
        if name == "safetensors":
            return {"verdict": self.p5_verdict, "detail": {"reason": "test"}}
        raise AssertionError(f"unexpected probe {name}")

    def load(
        self,
        model_dir: str,
        *,
        device: str = "cpu",
        timeout: float | None = None,
    ) -> Any:
        self.calls.append(("load", model_dir, {"device": device}))
        if self.fail_load:
            raise RuntimeError("load failed on purpose")
        return {"loaded": True, "device": device}

    def transcribe(
        self,
        audio_path: str,
        *,
        language: str | None = None,
        timeout: float | None = None,
    ) -> str:
        self.calls.append(("transcribe", audio_path, {"language": language}))
        return self.text


class FailedProbeClient(FakeClient):
    def probe(
        self,
        name: str,
        *,
        timeout: float | None = None,
        **args: Any,
    ) -> Any:
        if name == "processor":
            self.calls.append(("probe", name, dict(args)))
            return {"verdict": "failed", "detail": {}}
        return super().probe(name, timeout=timeout, **args)


class FalseLoadClient(FakeClient):
    def load(
        self,
        model_dir: str,
        *,
        device: str = "cpu",
        timeout: float | None = None,
    ) -> Any:
        return {"loaded": False}


def test_successful_verification_sets_every_rung() -> None:
    client = FakeClient()

    result = verify_worker(
        client,
        model_dir="/models/tiny",
        audio_path="clip.wav",
        repo_id=None,
        now=NOW,
    )

    assert result.error is None
    assert result.text == "ሰላም"
    assert result.status.state_of(Rung.IMPORTS) is RungState.PASSED
    assert result.status.state_of(Rung.CONSTRUCTS) is RungState.PASSED
    assert result.status.state_of(Rung.WEIGHTS_COMPATIBLE) is RungState.NOT_APPLICABLE
    assert result.status.state_of(Rung.LOADS) is RungState.PASSED
    assert result.status.state_of(Rung.TRANSCRIBES) is RungState.PASSED
    assert result.status.state_of(Rung.ACCURACY_CHECKED) is RungState.UNTESTED
    transcribe_check = result.status.root[Rung.TRANSCRIBES].check
    imports_check = result.status.root[Rung.IMPORTS].check
    assert transcribe_check is not None
    assert imports_check is not None
    assert transcribe_check.check == "stage6.smoke"
    assert imports_check.ran_at == NOW


def test_import_failure_stops_the_ladder() -> None:
    result = verify_worker(
        FakeClient(fail_probe="architecture"),
        model_dir="/models/tiny",
        audio_path="clip.wav",
        now=NOW,
    )

    assert result.error is not None
    assert "architecture failed" in result.error
    assert result.status.state_of(Rung.IMPORTS) is RungState.FAILED
    assert result.status.state_of(Rung.CONSTRUCTS) is RungState.UNTESTED
    assert result.status.state_of(Rung.TRANSCRIBES) is RungState.UNTESTED


def test_weights_mismatch_is_investigate_but_load_and_smoke_still_run() -> None:
    client = FakeClient(p5_verdict="mismatch")

    result = verify_worker(
        client,
        model_dir="/models/tiny",
        audio_path="clip.wav",
        now=NOW,
    )

    assert result.status.state_of(Rung.WEIGHTS_COMPATIBLE) is RungState.FAILED
    assert result.status.state_of(Rung.LOADS) is RungState.PASSED
    assert result.status.state_of(Rung.TRANSCRIBES) is RungState.PASSED
    assert result.error is None


def test_p5_receives_repo_and_revision() -> None:
    client = FakeClient()

    verify_worker(
        client,
        model_dir="/models/tiny",
        audio_path="clip.wav",
        repo_id="org/model",
        revision="a" * 40,
        now=NOW,
    )

    p5_calls = [call for call in client.calls if call[0] == "probe" and call[1] == "safetensors"]
    assert p5_calls
    assert p5_calls[0][2] == {
        "model_dir": "/models/tiny",
        "repo_id": "org/model",
        "revision": "a" * 40,
    }


def test_load_failure_leaves_transcribes_untested() -> None:
    result = verify_worker(
        FakeClient(fail_load=True),
        model_dir="/models/tiny",
        audio_path="clip.wav",
        now=NOW,
    )

    assert result.status.state_of(Rung.LOADS) is RungState.FAILED
    assert result.status.state_of(Rung.TRANSCRIBES) is RungState.UNTESTED
    assert "load failed" in (result.error or "")


def test_non_pass_probe_verdict_fails_constructs() -> None:
    result = verify_worker(
        FailedProbeClient(),
        model_dir="/models/tiny",
        audio_path="clip.wav",
        now=NOW,
    )

    assert result.status.state_of(Rung.CONSTRUCTS) is RungState.FAILED
    assert result.status.state_of(Rung.LOADS) is RungState.UNTESTED
    assert "verdict" in (result.error or "")


def test_load_must_report_loaded_true() -> None:
    result = verify_worker(
        FalseLoadClient(),
        model_dir="/models/tiny",
        audio_path="clip.wav",
        now=NOW,
    )

    assert result.status.state_of(Rung.LOADS) is RungState.FAILED
    assert "loaded=true" in (result.error or "")


def test_empty_transcription_fails_the_smoke_rung() -> None:
    result = verify_worker(
        FakeClient(text="   "),
        model_dir="/models/tiny",
        audio_path="clip.wav",
        now=NOW,
    )

    assert result.status.state_of(Rung.TRANSCRIBES) is RungState.FAILED
    assert result.error == "transcription was empty"


@pytest.mark.parametrize("verdict", ["match", "match_with_prefix"])
def test_matching_safetensors_passes_weights_rung(verdict: str) -> None:
    result = verify_worker(
        FakeClient(p5_verdict=verdict),
        model_dir="/models/tiny",
        audio_path="clip.wav",
        now=NOW,
    )

    assert result.status.state_of(Rung.WEIGHTS_COMPATIBLE) is RungState.PASSED
