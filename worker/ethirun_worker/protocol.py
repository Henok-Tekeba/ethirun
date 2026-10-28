"""Protocol primitives: one JSON object per line, errors as data."""

from __future__ import annotations

import json
import traceback
from typing import Any

PROTOCOL_VERSION = 1
WORKER_VERSION = "0.2.0"


class ProtocolError(Exception):
    """A line on the protocol stream was not a valid request object."""


def encode(message: dict[str, Any]) -> str:
    """Encode one protocol message as a single line, preserving Unicode."""
    return json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n"


def decode(line: str) -> dict[str, Any] | None:
    """Decode one protocol line.

    Blank lines are ignored. Anything that is not a JSON object raises
    :class:`ProtocolError`, which the server converts into an error response.
    """
    stripped = line.strip()
    if not stripped:
        return None

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid JSON: {exc.msg} at column {exc.colno}") from exc

    if not isinstance(payload, dict):
        raise ProtocolError("protocol line must be a JSON object")
    return payload


def success_response(request_id: int | None, result: Any) -> dict[str, Any]:
    """Build a successful response."""
    return {"id": request_id, "ok": True, "result": result}


def error_response(request_id: int | None, error: BaseException) -> dict[str, Any]:
    """Build an error response with type, message and traceback."""
    return {
        "id": request_id,
        "ok": False,
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
        },
    }
