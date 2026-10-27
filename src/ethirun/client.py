"""Host-side worker client: spawn, request, timeout, and hard termination.

The worker is a separate process with its own virtualenv. The only host
knowledge of it is the JSON-lines protocol on stdin/stdout; stderr is drained
continuously so noisy libraries cannot fill a pipe and deadlock the worker.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import signal
import subprocess
import sys
import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, TypeAlias

PROTOCOL_VERSION = 1
DEFAULT_WORKER_MODULE = "ethirun_worker"
_STDERR_LINES = 500
_WAIT_FOR_EXIT_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    """Everything needed to launch one worker process."""

    python: str | Path = field(default_factory=lambda: sys.executable)
    worker_dir: Path | None = None
    adapter: str = "fake"
    module: str = DEFAULT_WORKER_MODULE
    startup_timeout: float = 10.0
    default_timeout: float = 60.0
    cwd: Path | None = None
    env: Mapping[str, str] = field(default_factory=dict)


class WorkerError(RuntimeError):
    """Base class for all worker process failures."""


class WorkerClosedError(WorkerError):
    """The client is closed or the worker is no longer running."""


class WorkerStartupError(WorkerError):
    """The worker failed its handshake or could not be launched."""


class WorkerProtocolError(WorkerError):
    """The worker violated the JSON-lines contract."""


class WorkerOperationError(WorkerError):
    """The worker returned a structured error for one request."""

    def __init__(self, op: str, error: Mapping[str, Any], stderr: str) -> None:
        raw_type = error.get("type")
        raw_message = error.get("message")
        raw_traceback = error.get("traceback")

        self.op = op
        self.error_type = raw_type if isinstance(raw_type, str) else "UnknownError"
        self.error_message = raw_message if isinstance(raw_message, str) else ""
        self.traceback_text = raw_traceback if isinstance(raw_traceback, str) else ""
        self.stderr = stderr
        super().__init__(f"{op} failed: {self.error_type}: {self.error_message}{_tail(stderr)}")


class WorkerTimeoutError(WorkerError):
    """The request exceeded its deadline and the worker was killed."""

    def __init__(self, op: str, timeout: float, stderr: str) -> None:
        self.op = op
        self.timeout = timeout
        self.stderr = stderr
        super().__init__(f"{op} timed out after {timeout:g}s; worker was killed{_tail(stderr)}")


class WorkerCrashedError(WorkerError):
    """The worker exited without producing a response."""

    def __init__(self, op: str, returncode: int | None, stderr: str) -> None:
        self.op = op
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(f"{op} failed: worker exited with code {returncode!r}{_tail(stderr)}")


@dataclass(frozen=True, slots=True)
class _InvalidLine:
    line: str


class _EndOfStream:
    __slots__ = ()


_EOF = _EndOfStream()
_Incoming: TypeAlias = dict[str, Any] | _InvalidLine | _EndOfStream


def _tail(text: str, *, max_lines: int = 20, max_chars: int = 4000) -> str:
    lines = text.strip().splitlines()
    if not lines:
        return ""
    tail = "\n".join(lines[-max_lines:])
    if len(tail) > max_chars:
        tail = "..." + tail[-max_chars:]
    return f"\n--- worker stderr (tail) ---\n{tail}"


def default_worker_dir() -> Path:
    """Return the source-checkout worker directory."""
    return Path(__file__).resolve().parents[2] / "worker"


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            process.kill()
    else:
        process.kill()

    try:
        process.wait(timeout=_WAIT_FOR_EXIT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=_WAIT_FOR_EXIT_SECONDS)


def _terminate_process_group(process: subprocess.Popen[str], grace: float) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except PermissionError:
            process.terminate()
    else:
        process.terminate()

    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        _kill_process_group(process)


class WorkerClient:
    """Synchronous, single-flight client for one worker process."""

    def __init__(self, config: WorkerConfig | None = None) -> None:
        self._config = config or WorkerConfig()
        self._process: subprocess.Popen[str] | None = None
        self._incoming: queue.Queue[_Incoming] = queue.Queue()
        self._stderr_lines: deque[str] = deque(maxlen=_STDERR_LINES)
        self._stderr_lock = threading.Lock()
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._request_lock = threading.Lock()
        self._next_id = 0
        self._closed = True
        self._protocol_version: int | None = None
        self._worker_version: str | None = None
        self._adapter_name: str | None = None

    # ---------------------------- lifecycle ----------------------------

    def start(self) -> None:
        """Launch the worker and complete the version handshake."""
        if self._process is not None:
            if self.is_running:
                raise WorkerError("worker is already running")
            self._finalize_streams()

        worker_dir = self._resolve_worker_dir()
        command = [str(self._config.python), "-u", "-m", self._config.module]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(worker_dir)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["ETHIRUN_WORKER_ADAPTER"] = self._config.adapter
        env.update(self._config.env)

        self._incoming = queue.Queue()
        with self._stderr_lock:
            self._stderr_lines.clear()
        self._next_id = 0
        self._protocol_version = None
        self._worker_version = None
        self._adapter_name = None

        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                cwd=self._config.cwd,
                start_new_session=os.name == "posix",
            )
        except OSError as exc:
            raise WorkerStartupError(f"could not start worker: {exc}") from exc

        self._closed = False
        self._stdout_thread = threading.Thread(
            target=self._pump_stdout,
            name="ethirun-worker-stdout",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._pump_stderr,
            name="ethirun-worker-stderr",
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

        try:
            self._handshake()
        except WorkerError:
            self.kill()
            raise

    def close(self) -> None:
        """Shut down cleanly, escalating to termination if needed."""
        if not self.is_running:
            self._closed = True
            self._finalize_streams()
            return

        try:
            self.shutdown()
        except WorkerError:
            self.kill()

    def terminate(self, grace: float = 2.0) -> None:
        """Send SIGTERM, then SIGKILL if the process does not exit."""
        process = self._process
        self._closed = True
        if process is not None and process.poll() is None:
            _terminate_process_group(process, grace)
        self._finalize_streams()

    def kill(self) -> None:
        """Hard-kill the worker process group immediately."""
        process = self._process
        self._closed = True
        if process is not None and process.poll() is None:
            _kill_process_group(process)
        self._finalize_streams()

    def __enter__(self) -> WorkerClient:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    # ---------------------------- requests -----------------------------

    def request(
        self,
        op: str,
        *,
        name: str | None = None,
        args: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Send one request and wait for its matching response."""
        if not self.is_running:
            raise WorkerClosedError("worker is not running")

        effective_timeout = self._config.default_timeout if timeout is None else timeout
        if effective_timeout <= 0:
            raise ValueError("timeout must be positive")

        with self._request_lock:
            if not self.is_running:
                raise WorkerClosedError("worker is not running")

            self._next_id += 1
            request_id = self._next_id
            payload: dict[str, Any] = {
                "id": request_id,
                "op": op,
                "args": dict(args or {}),
            }
            if name is not None:
                payload["name"] = name

            try:
                self._send(payload)
            except WorkerClosedError as exc:
                raise self._crashed_error(op) from exc

            message = self._get_incoming(effective_timeout, op)

            if message.get("id") != request_id:
                raise self._protocol_failure(
                    f"{op}: expected response id {request_id}, got {message.get('id')!r}"
                )

            ok = message.get("ok")
            if ok is True:
                return message.get("result")
            if ok is False:
                raw_error = message.get("error")
                error: Mapping[str, Any] = raw_error if isinstance(raw_error, Mapping) else {}
                raise WorkerOperationError(op, error, self.stderr_text())

            raise self._protocol_failure(f"{op}: response is missing a boolean 'ok' field")

    def probe(self, name: str, *, timeout: float | None = None, **args: Any) -> Any:
        """Run a named probe in the worker."""
        return self.request("probe", name=name, args=args, timeout=timeout)

    def load(
        self,
        model_dir: str,
        *,
        device: str = "cpu",
        timeout: float | None = None,
    ) -> Any:
        """Load a model inside the worker."""
        return self.request(
            "load",
            args={"model_dir": model_dir, "device": device},
            timeout=timeout,
        )

    def transcribe(
        self,
        audio_path: str,
        *,
        language: str | None = None,
        timeout: float | None = None,
    ) -> str:
        """Transcribe an audio file path and return its text."""
        result = self.request(
            "transcribe",
            args={"audio_path": audio_path, "language": language},
            timeout=timeout,
        )
        if isinstance(result, Mapping):
            text = result.get("text")
            if isinstance(text, str):
                return text
        raise self._protocol_failure("transcribe response is missing string 'text'")

    def shutdown(self, *, timeout: float = 5.0) -> None:
        """Ask the worker to exit, then reap it."""
        if not self.is_running:
            self._closed = True
            self._finalize_streams()
            return

        try:
            self.request("shutdown", timeout=timeout)
        finally:
            process = self._process
            if process is not None:
                try:
                    process.wait(timeout=min(timeout, 2.0))
                except subprocess.TimeoutExpired:
                    _terminate_process_group(process, grace=1.0)
            self._closed = True
            self._finalize_streams()

    # ---------------------------- state --------------------------------

    @property
    def is_running(self) -> bool:
        process = self._process
        return not self._closed and process is not None and process.poll() is None

    @property
    def pid(self) -> int | None:
        return None if self._process is None else self._process.pid

    @property
    def protocol_version(self) -> int | None:
        return self._protocol_version

    @property
    def worker_version(self) -> str | None:
        return self._worker_version

    @property
    def adapter_name(self) -> str | None:
        return self._adapter_name

    def stderr_text(self) -> str:
        """Return the drained worker stderr tail."""
        with self._stderr_lock:
            return "\n".join(self._stderr_lines)

    # ---------------------------- internals ----------------------------

    def _resolve_worker_dir(self) -> Path:
        if self._config.worker_dir is not None:
            candidate = self._config.worker_dir
        else:
            from_env = os.environ.get("ETHIRUN_WORKER_DIR")
            candidate = Path(from_env) if from_env else default_worker_dir()

        path = candidate.expanduser().resolve()
        if not path.is_dir():
            raise WorkerStartupError(f"worker source directory not found: {path}; set worker_dir")
        return path

    def _pump_stdout(self) -> None:
        process = self._process
        stream = None if process is None else process.stdout
        if stream is None:
            self._incoming.put(_EOF)
            return

        try:
            for line in stream:
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    self._incoming.put(_InvalidLine(text[:500]))
                    return
                if not isinstance(payload, dict):
                    self._incoming.put(_InvalidLine(text[:500]))
                    return
                self._incoming.put(payload)
        except (OSError, ValueError):
            pass
        finally:
            self._incoming.put(_EOF)

    def _pump_stderr(self) -> None:
        process = self._process
        stream = None if process is None else process.stderr
        if stream is None:
            return

        try:
            for line in stream:
                with self._stderr_lock:
                    self._stderr_lines.append(line.rstrip("\n"))
        except (OSError, ValueError):
            pass

    def _handshake(self) -> None:
        message = self._get_incoming(self._config.startup_timeout, "handshake")

        if message.get("ok") is not True:
            raw_error = message.get("error")
            error: Mapping[str, Any] = raw_error if isinstance(raw_error, Mapping) else {}
            error_type = error.get("type", "UnknownError")
            error_message = error.get("message", "")
            raise WorkerStartupError(
                f"worker rejected startup: {error_type}: {error_message}{_tail(self.stderr_text())}"
            )

        if message.get("id") != 0:
            raise self._protocol_failure(f"handshake: expected id 0, got {message.get('id')!r}")

        result = message.get("result")
        if not isinstance(result, Mapping):
            raise self._protocol_failure("handshake: result must be an object")

        protocol = result.get("protocol")
        worker_version = result.get("worker_version")
        adapter = result.get("adapter")
        if (
            not isinstance(protocol, int)
            or isinstance(protocol, bool)
            or not isinstance(worker_version, str)
            or not isinstance(adapter, str)
        ):
            raise self._protocol_failure("handshake: malformed result fields")

        if protocol != PROTOCOL_VERSION:
            raise self._protocol_failure(
                f"protocol mismatch: host speaks {PROTOCOL_VERSION}, worker speaks {protocol}"
            )

        self._protocol_version = protocol
        self._worker_version = worker_version
        self._adapter_name = adapter

    def _send(self, payload: dict[str, Any]) -> None:
        process = self._process
        stdin = None if process is None else process.stdin
        if stdin is None:
            raise WorkerClosedError("worker stdin is unavailable")

        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        try:
            stdin.write(line)
            stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise WorkerClosedError("worker stdin is closed") from exc

    def _get_incoming(self, timeout: float, op: str) -> dict[str, Any]:
        try:
            item = self._incoming.get(timeout=timeout)
        except queue.Empty as exc:
            self.kill()
            raise WorkerTimeoutError(op, timeout, self.stderr_text()) from exc

        if isinstance(item, _EndOfStream):
            error = self._crashed_error(op)
            self._finalize_streams()
            raise error

        if isinstance(item, _InvalidLine):
            raise self._protocol_failure(f"{op}: worker wrote a non-protocol line: {item.line!r}")

        return item

    def _crashed_error(self, op: str) -> WorkerCrashedError:
        process = self._process
        returncode: int | None = None
        if process is not None:
            try:
                returncode = process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                returncode = process.poll()
                if returncode is None:
                    _kill_process_group(process)
                    returncode = process.poll()

        stderr_thread = self._stderr_thread
        if stderr_thread is not None and stderr_thread.is_alive():
            stderr_thread.join(timeout=0.5)

        return WorkerCrashedError(op, returncode, self.stderr_text())

    def _protocol_failure(self, message: str) -> WorkerProtocolError:
        self.kill()
        return WorkerProtocolError(message + _tail(self.stderr_text()))

    def _finalize_streams(self) -> None:
        process = self._process
        if process is not None:
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is None:
                    continue
                with contextlib.suppress(OSError, ValueError):
                    stream.close()

        for thread in (self._stdout_thread, self._stderr_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=1.0)

        self._closed = True
