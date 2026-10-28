"""Host-side verification runner for probes, load and smoke transcription.

Each rung is set only by the worker operation that actually ran: P1/P2 for
imports, P3/P4 for constructs, P5 for weights_compatible, load for loads, and a
non-empty transcription for transcribes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from ethirun.schemas import CheckRecord, Rung, RungState, StatusLadder


class VerificationClient(Protocol):
    """Minimal worker surface needed by the verifier."""

    def probe(self, name: str, *, timeout: float | None = None, **args: Any) -> Any: ...

    def load(
        self,
        model_dir: str,
        *,
        device: str = "cpu",
        timeout: float | None = None,
    ) -> Any: ...

    def transcribe(
        self,
        audio_path: str,
        *,
        language: str | None = None,
        timeout: float | None = None,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """P1-P5 outcome, without loading weights or transcribing."""

    status: StatusLadder
    probes: dict[str, Any]
    error: str | None


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Evidence ladder plus the artifacts observed during verification."""

    status: StatusLadder
    probes: dict[str, Any]
    load: Any
    text: str | None
    error: str | None


def _detail(probe: Any) -> dict[str, Any]:
    if isinstance(probe, dict):
        detail = probe.get("detail")
        if isinstance(detail, dict):
            return {str(key): value for key, value in detail.items()}
    return {}


def _require_pass(probe: Any, name: str) -> None:
    if not isinstance(probe, dict) or probe.get("verdict") != "pass":
        raise ValueError(f"{name} probe did not report verdict 'pass'")


def _check(
    rung: Rung,
    check: str,
    stage: str,
    outcome: RungState,
    now: datetime,
    detail: dict[str, Any],
) -> CheckRecord:
    return CheckRecord(
        rung=rung,
        check=check,
        stage=stage,
        outcome=outcome,
        ran_at=now,
        detail=detail,
    )


def run_probes(
    client: VerificationClient,
    *,
    model_dir: str,
    repo_id: str | None = None,
    revision: str | None = None,
    now: datetime | None = None,
    probe_timeout: float | None = None,
    initial_status: StatusLadder | None = None,
) -> ProbeResult:
    """Run P1-P5 and record imports/constructs/weights rungs."""
    ran_at = now or datetime.now(UTC)
    status = initial_status if initial_status is not None else StatusLadder.initial()
    probes: dict[str, Any] = {}

    try:
        probes["imports"] = client.probe("imports", timeout=probe_timeout)
        probes["architecture"] = client.probe(
            "architecture",
            timeout=probe_timeout,
            model_dir=model_dir,
        )
        _require_pass(probes["imports"], "imports")
        _require_pass(probes["architecture"], "architecture")
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        status = status.with_check(
            _check(
                Rung.IMPORTS,
                "stage4.imports",
                "stage4",
                RungState.FAILED,
                ran_at,
                {"probes": probes, "error": message},
            )
        )
        return ProbeResult(status, probes, message)

    status = status.with_check(
        _check(
            Rung.IMPORTS,
            "stage4.imports",
            "stage4",
            RungState.PASSED,
            ran_at,
            {
                "imports": _detail(probes["imports"]),
                "architecture": _detail(probes["architecture"]),
            },
        )
    )

    try:
        probes["meta_build"] = client.probe(
            "meta_build",
            timeout=probe_timeout,
            model_dir=model_dir,
        )
        probes["processor"] = client.probe(
            "processor",
            timeout=probe_timeout,
            model_dir=model_dir,
        )
        _require_pass(probes["meta_build"], "meta_build")
        _require_pass(probes["processor"], "processor")
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        status = status.with_check(
            _check(
                Rung.CONSTRUCTS,
                "stage4.constructs",
                "stage4",
                RungState.FAILED,
                ran_at,
                {"probes": probes, "error": message},
            )
        )
        return ProbeResult(status, probes, message)

    status = status.with_check(
        _check(
            Rung.CONSTRUCTS,
            "stage4.constructs",
            "stage4",
            RungState.PASSED,
            ran_at,
            {
                "meta_build": _detail(probes["meta_build"]),
                "processor": _detail(probes["processor"]),
            },
        )
    )

    try:
        p5_args: dict[str, Any] = {"model_dir": model_dir}
        if repo_id is not None:
            p5_args["repo_id"] = repo_id
        if revision is not None:
            p5_args["revision"] = revision
        probes["safetensors"] = client.probe(
            "safetensors",
            timeout=probe_timeout,
            **p5_args,
        )
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        status = status.with_check(
            _check(
                Rung.WEIGHTS_COMPATIBLE,
                "stage4.weights",
                "stage4",
                RungState.FAILED,
                ran_at,
                {"error": message},
            )
        )
        return ProbeResult(status, probes, message)

    p5 = probes["safetensors"]
    verdict = p5.get("verdict") if isinstance(p5, dict) else None
    if verdict in {"match", "match_with_prefix"}:
        weights_state = RungState.PASSED
    elif verdict == "not_applicable":
        weights_state = RungState.NOT_APPLICABLE
    else:
        weights_state = RungState.FAILED
    status = status.with_check(
        _check(
            Rung.WEIGHTS_COMPATIBLE,
            "stage4.weights",
            "stage4",
            weights_state,
            ran_at,
            {"probe": p5},
        )
    )

    return ProbeResult(status, probes, None)


def verify_worker(
    client: VerificationClient,
    *,
    model_dir: str,
    audio_path: str,
    repo_id: str | None = None,
    revision: str | None = None,
    language: str | None = None,
    device: str = "cpu",
    now: datetime | None = None,
    probe_timeout: float | None = None,
    transcribe_timeout: float | None = None,
    initial_status: StatusLadder | None = None,
) -> VerificationResult:
    """Run P1-P5, load and smoke transcription, recording every rung."""
    ran_at = now or datetime.now(UTC)
    probe_result = run_probes(
        client,
        model_dir=model_dir,
        repo_id=repo_id,
        revision=revision,
        now=ran_at,
        probe_timeout=probe_timeout,
        initial_status=initial_status,
    )
    status = probe_result.status
    probes = probe_result.probes
    if probe_result.error is not None:
        return VerificationResult(status, probes, None, None, probe_result.error)

    load_result: Any = None
    text: str | None = None

    try:
        load_result = client.load(model_dir, device=device, timeout=probe_timeout)
        if not isinstance(load_result, dict) or load_result.get("loaded") is not True:
            raise ValueError("load did not report loaded=true")
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        status = status.with_check(
            _check(
                Rung.LOADS,
                "stage5.load",
                "stage5",
                RungState.FAILED,
                ran_at,
                {"error": message},
            )
        )
        return VerificationResult(status, probes, load_result, text, message)

    status = status.with_check(
        _check(
            Rung.LOADS,
            "stage5.load",
            "stage5",
            RungState.PASSED,
            ran_at,
            {"load": load_result},
        )
    )

    try:
        text = client.transcribe(
            audio_path,
            language=language,
            timeout=transcribe_timeout,
        )
        if not isinstance(text, str):
            raise ValueError("transcribe returned a non-string result")
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        status = status.with_check(
            _check(
                Rung.TRANSCRIBES,
                "stage6.smoke",
                "stage6",
                RungState.FAILED,
                ran_at,
                {"error": message},
            )
        )
        return VerificationResult(status, probes, load_result, text, message)

    transcribe_state = RungState.PASSED if text.strip() else RungState.FAILED
    error = None if transcribe_state is RungState.PASSED else "transcription was empty"
    detail: dict[str, Any] = {"text_length": len(text), "text": text}
    if language is not None:
        detail["language"] = language
    status = status.with_check(
        _check(
            Rung.TRANSCRIBES,
            "stage6.smoke",
            "stage6",
            transcribe_state,
            ran_at,
            detail,
        )
    )
    return VerificationResult(status, probes, load_result, text, error)
