"""Helpers for launching the real stdlib-only worker in tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKER_DIR = PROJECT_ROOT / "worker"


def worker_env(overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORKER_DIR)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["ETHIRUN_WORKER_ADAPTER"] = "fake"
    if overrides:
        env.update(overrides)
    return env


def run_worker(
    lines: Sequence[object],
    *,
    env_overrides: Mapping[str, str] | None = None,
    timeout: float = 10.0,
) -> subprocess.CompletedProcess[str]:
    payload = "".join(
        (line if isinstance(line, str) else json.dumps(line, ensure_ascii=False)) + "\n"
        for line in lines
    )
    return subprocess.run(
        [sys.executable, "-u", "-m", "ethirun_worker"],
        input=payload,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=worker_env(env_overrides),
        timeout=timeout,
        check=False,
    )


def protocol_messages(stdout: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        assert isinstance(payload, dict), f"protocol line is not an object: {line!r}"
        messages.append(payload)
    return messages
