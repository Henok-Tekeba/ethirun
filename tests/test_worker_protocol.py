"""Protocol-level tests against a real worker process."""

from __future__ import annotations

from typing import Any

import pytest

from ethirun_worker.protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    decode,
    encode,
    error_response,
    success_response,
)
from tests.worker_harness import protocol_messages, run_worker


def test_encode_is_one_line_and_preserves_unicode() -> None:
    line = encode({"id": 1, "text": "ሰላም"})

    assert line.endswith("\n")
    assert line.count("\n") == 1
    assert "ሰላም" in line


def test_decode_ignores_blank_and_rejects_bad_lines() -> None:
    assert decode("   \n") is None

    with pytest.raises(ProtocolError, match="invalid JSON"):
        decode("{not json")

    with pytest.raises(ProtocolError, match="JSON object"):
        decode("[1, 2, 3]")


def test_response_builders_include_structured_errors() -> None:
    success = success_response(3, {"text": "ok"})
    assert success == {"id": 3, "ok": True, "result": {"text": "ok"}}

    try:
        raise ValueError("boom")
    except ValueError as exc:
        error = error_response(4, exc)

    assert error["id"] == 4
    assert error["ok"] is False
    assert error["error"]["type"] == "ValueError"
    assert error["error"]["message"] == "boom"
    assert "Traceback" in error["error"]["traceback"]


def test_worker_handshake_and_successful_requests() -> None:
    result = run_worker(
        [
            {"id": 1, "op": "probe", "name": "echo", "args": {"value": 1}},
            {
                "id": 2,
                "op": "load",
                "args": {"model_dir": "/models/tiny", "device": "cpu"},
            },
            {
                "id": 3,
                "op": "transcribe",
                "args": {"audio_path": "clip.wav", "language": "am"},
            },
            {"id": 4, "op": "shutdown"},
        ]
    )

    assert result.returncode == 0, result.stderr
    messages = protocol_messages(result.stdout)
    assert len(messages) == 5

    hello = messages[0]
    assert hello["id"] == 0
    assert hello["ok"] is True
    assert hello["result"]["protocol"] == PROTOCOL_VERSION
    assert hello["result"]["adapter"] == "fake"
    assert isinstance(hello["result"]["worker_version"], str)
    assert hello["result"]["pid"] > 0

    assert messages[1]["result"] == {"verdict": "pass", "name": "echo", "args": {"value": 1}}
    assert messages[2]["result"] == {
        "loaded": True,
        "model_dir": "/models/tiny",
        "device": "cpu",
    }
    assert messages[3]["result"]["text"] == "ሰላም ዓለም [am]"
    assert messages[4]["result"] == {"shutdown": True}


def test_library_stdout_noise_does_not_corrupt_protocol() -> None:
    result = run_worker(
        [
            {"id": 1, "op": "probe", "name": "echo", "args": {"noise": True}},
            {"id": 2, "op": "shutdown"},
        ]
    )

    assert result.returncode == 0, result.stderr
    messages = protocol_messages(result.stdout)
    assert [message["id"] for message in messages] == [0, 1, 2]
    assert all(message["ok"] is True for message in messages)

    assert "fake library noise on stdout" in result.stderr
    assert "fake raw fd noise" in result.stderr
    assert "fake library noise on stderr" in result.stderr


def test_worker_errors_are_returned_as_data_and_do_not_kill_the_server() -> None:
    result = run_worker(
        [
            {"id": 1, "op": "probe", "name": "explode"},
            {"id": 2, "op": "probe", "name": "after_error"},
            {"id": 3, "op": "shutdown"},
        ]
    )

    assert result.returncode == 0, result.stderr
    messages = protocol_messages(result.stdout)

    error = messages[1]
    assert error["id"] == 1
    assert error["ok"] is False
    assert error["error"]["type"] == "RuntimeError"
    assert error["error"]["message"] == "intentional probe failure"
    assert "Traceback" in error["error"]["traceback"]
    assert "fake.py" in error["error"]["traceback"]

    assert messages[2]["ok"] is True
    assert messages[2]["result"]["name"] == "after_error"
    assert messages[3]["result"] == {"shutdown": True}


def test_malformed_requests_return_errors_and_server_continues() -> None:
    result = run_worker(
        [
            "this is not json",
            {"id": 2, "op": "unknown_op"},
            {"id": 3, "op": "load", "args": {}},
            {"id": 4, "op": "probe"},
            {"id": 5, "op": "shutdown"},
        ]
    )

    assert result.returncode == 0, result.stderr
    messages = protocol_messages(result.stdout)

    assert messages[1]["id"] is None
    assert messages[1]["error"]["type"] == "ProtocolError"
    assert "invalid JSON" in messages[1]["error"]["message"]

    assert messages[2]["error"]["type"] == "ProtocolError"
    assert "unknown op" in messages[2]["error"]["message"]

    assert messages[3]["error"]["type"] == "ProtocolError"
    assert "model_dir" in messages[3]["error"]["message"]

    assert messages[4]["error"]["type"] == "ProtocolError"
    assert "name" in messages[4]["error"]["message"]

    assert messages[5]["id"] == 5
    assert messages[5]["ok"] is True


def test_fake_adapter_failures_are_typed() -> None:
    result = run_worker(
        [
            {
                "id": 1,
                "op": "transcribe",
                "args": {"audio_path": "explode.wav", "language": None},
            },
            {"id": 2, "op": "load", "args": {"model_dir": "explode"}},
            {"id": 3, "op": "shutdown"},
        ]
    )

    assert result.returncode == 0, result.stderr
    messages = protocol_messages(result.stdout)

    assert messages[1]["error"]["type"] == "ValueError"
    assert "intentional transcription failure" in messages[1]["error"]["message"]
    assert messages[2]["error"]["type"] == "RuntimeError"
    assert "intentional load failure" in messages[2]["error"]["message"]


def test_every_stdout_line_is_a_json_object_and_stderr_holds_logs() -> None:
    result = run_worker(
        [
            {"id": 1, "op": "probe", "name": "echo", "args": {"noise": True}},
            {"id": 2, "op": "shutdown"},
        ]
    )

    stdout_lines = [line for line in result.stdout.splitlines() if line.strip()]
    parsed: list[dict[str, Any]] = protocol_messages(result.stdout)

    assert len(stdout_lines) == len(parsed)
    assert result.stderr
