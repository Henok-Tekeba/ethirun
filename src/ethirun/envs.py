"""Stage 3: build one disposable virtualenv per lockfile, and cache it.

The environment hash covers the lockfile bytes, Python version and worker
version. An env with a matching completion marker is reused without invoking
uv at all. A partial or mismatched environment is removed, never patched.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ethirun import WORKER_VERSION
from ethirun.uv import Uv, UvIndexOptions, UvRunner

DEFAULT_ENVS_DIR = Path.home() / ".ethirun" / "envs"
ENV_MARKER = ".ethirun-env.json"
MARKER_SCHEMA = 1


class EnvManagerError(RuntimeError):
    """Base class for environment management failures."""


class EnvBuildError(EnvManagerError):
    """One stage of building a worker env failed."""

    def __init__(
        self,
        stage: str,
        command: tuple[str, ...],
        message: str,
    ) -> None:
        self.stage = stage
        self.command = command
        self.message = message
        command_text = " ".join(command) if command else "(none)"
        super().__init__(f"{stage} failed: {message}\ncommand: {command_text}")


@dataclass(frozen=True, slots=True)
class EnvResult:
    """A valid worker environment, freshly built or reused."""

    env_hash: str
    env_dir: Path
    python: Path
    lockfile_sha256: str
    cached: bool


class EnvManager:
    """Build and cache isolated worker virtualenvs from lockfiles."""

    def __init__(
        self,
        *,
        root: Path | None = None,
        uv: UvRunner | None = None,
        index: UvIndexOptions | None = None,
        worker_version: str = WORKER_VERSION,
    ) -> None:
        self.root = (root or DEFAULT_ENVS_DIR).expanduser()
        self._uv = uv or Uv()
        self._index = index or UvIndexOptions()
        self.worker_version = worker_version

    @staticmethod
    def compute_hash(
        lockfile_bytes: bytes,
        python_version: str,
        worker_version: str,
    ) -> str:
        """Hash exactly what determines an env's contents."""
        digest = hashlib.sha256()
        digest.update(lockfile_bytes)
        digest.update(b"\0")
        digest.update(python_version.encode("utf-8"))
        digest.update(b"\0")
        digest.update(worker_version.encode("utf-8"))
        return digest.hexdigest()

    def env_dir(
        self,
        lockfile: Path,
        python_version: str,
        *,
        worker_version: str | None = None,
    ) -> Path:
        """Return the cache path for a lockfile without building it."""
        data = lockfile.read_bytes()
        worker = self.worker_version if worker_version is None else worker_version
        env_hash = self.compute_hash(data, python_version, worker)
        return self.root / env_hash

    @staticmethod
    def python_path(env_dir: Path) -> Path:
        if os.name == "nt":
            return env_dir / "Scripts" / "python.exe"
        return env_dir / "bin" / "python"

    def build(
        self,
        lockfile: Path,
        python_version: str,
        *,
        worker_version: str | None = None,
    ) -> EnvResult:
        """Build or reuse the env for this exact lockfile and Python."""
        lockfile = lockfile.expanduser().resolve()
        try:
            lock_bytes = lockfile.read_bytes()
        except OSError as exc:
            raise EnvBuildError(
                "lockfile",
                (),
                f"could not read {lockfile}: {exc}",
            ) from exc

        if not lock_bytes.strip():
            raise EnvBuildError("lockfile", (), f"{lockfile} is empty")

        worker = self.worker_version if worker_version is None else worker_version
        env_hash = self.compute_hash(lock_bytes, python_version, worker)
        lockfile_sha256 = hashlib.sha256(lock_bytes).hexdigest()
        env_dir = self.root / env_hash

        if self._is_cached(env_dir, env_hash, lockfile_sha256, python_version, worker):
            return EnvResult(
                env_hash=env_hash,
                env_dir=env_dir,
                python=self.python_path(env_dir),
                lockfile_sha256=lockfile_sha256,
                cached=True,
            )

        self.root.mkdir(parents=True, exist_ok=True)
        if env_dir.exists():
            shutil.rmtree(env_dir, ignore_errors=True)

        venv_result = self._uv.run(
            ("venv", str(env_dir), "--python", python_version),
            cwd=self.root,
        )
        if not venv_result.ok:
            self._cleanup(env_dir)
            raise EnvBuildError(
                "uv venv",
                venv_result.command,
                venv_result.failure_message(),
            )

        python = self.python_path(env_dir)
        if not python.is_file():
            self._cleanup(env_dir)
            raise EnvBuildError(
                "uv venv",
                venv_result.command,
                f"venv was created without a Python executable at {python}",
            )

        sync_command = (
            "pip",
            "sync",
            "--python",
            str(python),
            *self._index.args(),
            str(lockfile),
        )
        sync_result = self._uv.run(sync_command, cwd=self.root)
        if not sync_result.ok:
            self._cleanup(env_dir)
            raise EnvBuildError(
                "uv pip sync",
                sync_result.command,
                sync_result.failure_message(),
            )

        check_command = ("pip", "check", "--python", str(python))
        check_result = self._uv.run(check_command, cwd=self.root)
        if not check_result.ok:
            self._cleanup(env_dir)
            raise EnvBuildError(
                "uv pip check",
                check_result.command,
                check_result.failure_message(),
            )

        marker = {
            "schema": MARKER_SCHEMA,
            "env_hash": env_hash,
            "lockfile_sha256": lockfile_sha256,
            "python_version": python_version,
            "worker_version": worker,
            "created_at": datetime.now(UTC).isoformat(),
        }
        (env_dir / ENV_MARKER).write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        return EnvResult(
            env_hash=env_hash,
            env_dir=env_dir,
            python=python,
            lockfile_sha256=lockfile_sha256,
            cached=False,
        )

    def _is_cached(
        self,
        env_dir: Path,
        env_hash: str,
        lockfile_sha256: str,
        python_version: str,
        worker_version: str,
    ) -> bool:
        marker_path = env_dir / ENV_MARKER
        if not marker_path.is_file() or not self.python_path(env_dir).is_file():
            return False

        try:
            marker: Any = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(marker, dict):
            return False

        return (
            marker.get("schema") == MARKER_SCHEMA
            and marker.get("env_hash") == env_hash
            and marker.get("lockfile_sha256") == lockfile_sha256
            and marker.get("python_version") == python_version
            and marker.get("worker_version") == worker_version
        )

    @staticmethod
    def _cleanup(env_dir: Path) -> None:
        shutil.rmtree(env_dir, ignore_errors=True)
