"""Host client tests: handshakes, errors, timeouts, crashes, hard kills."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from ethirun.client import (
    PROTOCOL_VERSION,
    WorkerClient,
    WorkerClosedError,
    WorkerConfig,
    WorkerCrashedError,
    WorkerOperationError,
    WorkerProtocolError,
    WorkerStartupError,
    WorkerTimeoutError,
)
from ethirun_worker.protocol import PROTOCOL_VERSION as WORKER_PROTOCOL_VERSION
from tests.worker_harness import WORKER_DIR


def _config(**overrides: object) -> WorkerConfig:
    values: dict[str, object] = {
        "python": sys.executable,
        "worker_dir": WORKER_DIR,
        "startup_timeout": 5.0,
        "default_timeout": 5.0,
        "env": {"ETHIRUN_FAKE_TEXT": "ሰላም ዓለም"},
    }
    values.update(overrides)
    return WorkerConfig(**values)  # type: ignore[arg-type]


@pytest.fixture
def client() -> Iterator[WorkerClient]:
    worker = WorkerClient(_config())
    worker.start()
    try:
        yield worker
    finally:
        worker.close()


def _wait_until(predicate: Callable[[], bool], timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _write_worker(tmp_path: Path, name: str, source: str) -> Path:
    root = tmp_path / "workers"
    package = root / name
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "__main__.py").write_text(source, encoding="utf-8")
    return root


def test_protocol_versions_agree() -> None:
    assert PROTOCOL_VERSION == WORKER_PROTOCOL_VERSION == 1


def test_handshake(client: WorkerClient) -> None:
    assert client.protocol_version == PROTOCOL_VERSION
    assert client.worker_version is not None
    assert client.adapter_name == "fake"
    assert client.is_running
    assert client.pid is not None


def test_probe_success(client: WorkerClient) -> None:
    result = client.probe("meta_build", model_dir="/models/tiny")

    assert result == {
        "verdict": "pass",
        "name": "meta_build",
        "args": {"model_dir": "/models/tiny"},
    }


def test_load_and_transcribe(client: WorkerClient) -> None:
    loaded = client.load("/models/tiny", device="cpu")
    text = client.transcribe("clip.wav", language="am")

    assert loaded == {
        "loaded": True,
        "model_dir": "/models/tiny",
        "device": "cpu",
    }
    assert text == "ሰላም ዓለም [am]"


def test_operation_error_is_classified_and_worker_survives(
    client: WorkerClient,
) -> None:
    with pytest.raises(WorkerOperationError) as info:
        client.probe("explode")

    exc = info.value
    assert exc.op == "probe"
    assert exc.error_type == "RuntimeError"
    assert exc.error_message == "intentional probe failure"
    assert "Traceback" in exc.traceback_text
    assert "fake.py" in exc.traceback_text
    assert "intentional probe failure" in str(exc)

    assert client.is_running
    assert client.probe("after_error")["verdict"] == "pass"


def test_library_noise_goes_to_stderr_and_protocol_stays_clean(
    client: WorkerClient,
) -> None:
    result = client.probe("echo", noise=True)

    assert result["verdict"] == "pass"
    assert _wait_until(lambda: "fake library noise on stdout" in client.stderr_text())
    assert "fake raw fd noise" in client.stderr_text()
    assert client.probe("after_noise")["name"] == "after_noise"


def test_timeout_kills_the_worker_process() -> None:
    worker = WorkerClient(_config())
    worker.start()
    pid = worker.pid
    assert pid is not None

    started = time.monotonic()
    with pytest.raises(WorkerTimeoutError) as info:
        worker.probe("slow", timeout=0.2, delay_seconds=5)

    exc = info.value
    assert exc.op == "probe"
    assert exc.timeout == pytest.approx(0.2)
    assert "timed out" in str(exc)

    elapsed = time.monotonic() - started
    assert elapsed < 3.0
    assert not worker.is_running
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)

    worker.close()


def test_hard_kill_is_immediate() -> None:
    worker = WorkerClient(_config())
    worker.start()
    pid = worker.pid
    assert pid is not None

    worker.kill()

    assert not worker.is_running
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    worker.close()


def test_terminate_escalates_cleanly() -> None:
    worker = WorkerClient(_config())
    worker.start()
    pid = worker.pid
    assert pid is not None

    worker.terminate(grace=1.0)

    assert not worker.is_running
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    worker.close()


def test_shutdown_reaps_the_worker(client: WorkerClient) -> None:
    client.shutdown()

    assert not client.is_running


def test_closed_client_rejects_requests() -> None:
    worker = WorkerClient(_config())
    worker.close()

    with pytest.raises(WorkerClosedError):
        worker.request("probe", name="echo")


def test_context_manager_closes_worker() -> None:
    with WorkerClient(_config()) as worker:
        assert worker.is_running
    assert not worker.is_running


def test_worker_crash_is_reported_with_return_code() -> None:
    worker = WorkerClient(_config())
    worker.start()

    with pytest.raises(WorkerCrashedError) as info:
        worker.probe("crash")

    assert info.value.returncode == 7
    assert not worker.is_running
    worker.close()


def test_unknown_adapter_fails_startup_cleanly() -> None:
    worker = WorkerClient(_config(adapter="does_not_exist"))

    with pytest.raises(WorkerStartupError, match="unknown worker adapter"):
        worker.start()

    assert not worker.is_running
    worker.close()


def test_startup_timeout_kills_a_silent_worker(tmp_path: Path) -> None:
    root = _write_worker(
        tmp_path,
        "silent_worker",
        "import time\n\ntime.sleep(30)\n",
    )
    worker = WorkerClient(
        WorkerConfig(
            python=sys.executable,
            worker_dir=root,
            module="silent_worker",
            startup_timeout=0.3,
        )
    )

    started = time.monotonic()
    with pytest.raises(WorkerTimeoutError):
        worker.start()

    assert time.monotonic() - started < 3.0
    assert not worker.is_running
    worker.close()


def test_protocol_violation_is_rejected(tmp_path: Path) -> None:
    root = _write_worker(
        tmp_path,
        "bad_worker",
        "import sys\n\nsys.stdout.write('this is not a protocol frame\\n')\nsys.stdout.flush()\n",
    )
    worker = WorkerClient(
        WorkerConfig(
            python=sys.executable,
            worker_dir=root,
            module="bad_worker",
            startup_timeout=3.0,
        )
    )

    with pytest.raises(WorkerProtocolError, match="non-protocol line"):
        worker.start()

    assert not worker.is_running
    worker.close()
