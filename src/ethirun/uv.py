"""Small subprocess boundary around the ``uv`` executable.

All model-toolchain commands go through here so tests can inject a fake and so
failures retain the exact command and stderr instead of a bare traceback.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class UvError(RuntimeError):
    """Base class for uv invocation failures."""


class UvNotFoundError(UvError):
    """No uv executable was found."""


class UvTimeoutError(UvError):
    """uv did not finish before the deadline."""

    def __init__(self, command: tuple[str, ...], timeout: float) -> None:
        self.command = command
        self.timeout = timeout
        super().__init__(f"uv command timed out after {timeout:g}s: {' '.join(command)}")


@dataclass(frozen=True, slots=True)
class UvResult:
    """Captured result of one uv invocation."""

    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def failure_message(self, *, max_chars: int = 4000) -> str:
        detail = self.stderr.strip() or self.stdout.strip() or "uv returned no output"
        if len(detail) > max_chars:
            detail = "..." + detail[-max_chars:]
        return detail


class UvRunner(Protocol):
    """Injectable command runner used by the resolver and env manager."""

    def run(
        self,
        args: Sequence[str],
        *,
        stdin: str | None = None,
        cwd: Path | None = None,
    ) -> UvResult:
        """Run uv with arguments and capture text output."""
        ...


@dataclass(frozen=True, slots=True)
class UvIndexOptions:
    """Optional package-index controls shared by compile and sync."""

    index_url: str | None = None
    find_links: Path | None = None
    no_index: bool = False
    offline: bool = False

    def args(self) -> tuple[str, ...]:
        options: list[str] = []
        if self.no_index:
            options.append("--no-index")
        if self.index_url is not None:
            options.extend(("--index-url", self.index_url))
        if self.find_links is not None:
            options.extend(("--find-links", str(self.find_links)))
        if self.offline:
            options.append("--offline")
        return tuple(options)


class Uv:
    """Real uv runner."""

    def __init__(
        self,
        binary: str | Path | None = None,
        *,
        timeout: float = 900.0,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if binary is None:
            configured = os.environ.get("ETHIRUN_UV")
            if configured:
                binary = configured
            else:
                discovered = shutil.which("uv")
                if discovered is None:
                    raise UvNotFoundError("uv was not found on PATH; install uv or set ETHIRUN_UV")
                binary = discovered

        self._binary = str(binary)
        self._timeout = timeout
        self._env = dict(env or {})

    @property
    def binary(self) -> str:
        return self._binary

    def run(
        self,
        args: Sequence[str],
        *,
        stdin: str | None = None,
        cwd: Path | None = None,
    ) -> UvResult:
        command = (self._binary, *args)
        env = os.environ.copy()
        env.update(self._env)

        try:
            completed = subprocess.run(
                command,
                input=stdin,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self._timeout,
                cwd=cwd,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise UvTimeoutError(command, self._timeout) from exc
        except OSError as exc:
            raise UvError(f"could not run uv: {exc}") from exc

        return UvResult(
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
