"""Worker entry point: claim stdout for protocol, then serve JSON lines."""

from __future__ import annotations

import os
import sys
from typing import Any, TextIO

from ethirun_worker import WORKER_VERSION
from ethirun_worker.adapters.base import WorkerAdapter
from ethirun_worker.protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    decode,
    error_response,
    success_response,
)
from ethirun_worker.streams import ProtocolWriter


class ShutdownRequestedError(Exception):
    """Internal control flow: reply, then leave the serve loop."""


def _request_id(message: dict[str, Any]) -> int | None:
    value = message.get("id")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _request_args(message: dict[str, Any]) -> dict[str, Any]:
    args = message.get("args", {})
    if args is None:
        return {}
    if not isinstance(args, dict):
        raise ProtocolError("'args' must be a JSON object")
    return args


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"'{field}' must be a non-empty string")
    return value


def dispatch(adapter: WorkerAdapter, message: dict[str, Any]) -> Any:
    """Execute one request against the active adapter."""
    op = message.get("op")
    if not isinstance(op, str) or not op:
        raise ProtocolError("request is missing a string 'op'")

    args = _request_args(message)

    if op == "shutdown":
        raise ShutdownRequestedError

    if op == "probe":
        name = _require_text(message.get("name", args.get("name")), "name")
        return adapter.probe(name, args)

    if op == "load":
        model_dir = _require_text(args.get("model_dir"), "model_dir")
        device = args.get("device", "cpu")
        if not isinstance(device, str) or not device:
            raise ProtocolError("'device' must be a non-empty string")
        return adapter.load(model_dir, device)

    if op == "transcribe":
        audio_path = _require_text(args.get("audio_path"), "audio_path")
        language = args.get("language")
        if language is not None and not isinstance(language, str):
            raise ProtocolError("'language' must be a string or null")
        text = adapter.transcribe(audio_path, language)
        return {"text": text}

    raise ProtocolError(f"unknown op: {op!r}")


def serve(stream: TextIO, writer: ProtocolWriter, adapter: WorkerAdapter) -> int:
    """Read requests until EOF or shutdown, replying to every request."""
    for raw_line in stream:
        try:
            message = decode(raw_line)
        except ProtocolError as exc:
            writer.send(error_response(None, exc))
            continue

        if message is None:
            continue

        request_id = _request_id(message)
        try:
            result = dispatch(adapter, message)
        except ShutdownRequestedError:
            writer.send(success_response(request_id, {"shutdown": True}))
            return 0
        except Exception as exc:  # deliberate boundary: errors travel as data
            writer.send(error_response(request_id, exc))
        else:
            writer.send(success_response(request_id, result))

    return 0


def main(writer: ProtocolWriter) -> int:
    """Start the worker protocol loop."""
    adapter_name = os.environ.get("ETHIRUN_WORKER_ADAPTER", "fake")

    try:
        from ethirun_worker.adapters import create_adapter

        adapter = create_adapter(adapter_name)
    except Exception as exc:  # adapter startup failure must be reportable
        writer.send(error_response(None, exc))
        return 2

    writer.send(
        success_response(
            0,
            {
                "protocol": PROTOCOL_VERSION,
                "worker_version": WORKER_VERSION,
                "adapter": adapter.name,
                "pid": os.getpid(),
            },
        )
    )

    return serve(sys.stdin, writer, adapter)
