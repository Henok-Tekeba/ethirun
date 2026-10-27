"""Test-only local wheel builder and uv instrumentation."""

from __future__ import annotations

import base64
import hashlib
import re
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ethirun.uv import UvResult, UvRunner


def _digest(data: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")


def build_wheel(
    directory: Path,
    name: str,
    version: str,
    *,
    requires: Sequence[str] = (),
    module: str | None = None,
) -> Path:
    """Write a minimal valid pure-Python wheel and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    dist = re.sub(r"[-_.]+", "_", name)
    module_name = module or dist
    dist_info = f"{dist}-{version}.dist-info"

    files: dict[str, bytes] = {
        f"{module_name}/__init__.py": (
            f'"""Test wheel {name} {version}."""\n__version__ = "{version}"\n'
        ).encode(),
        f"{dist_info}/METADATA": (
            "Metadata-Version: 2.1\n"
            f"Name: {name}\n"
            f"Version: {version}\n"
            "Summary: ethirun test wheel\n"
            "Requires-Python: >=3.9\n"
            + "".join(f"Requires-Dist: {requirement}\n" for requirement in requires)
        ).encode(),
        f"{dist_info}/WHEEL": (
            b"Wheel-Version: 1.0\n"
            b"Generator: ethirun-tests\n"
            b"Root-Is-Purelib: true\n"
            b"Tag: py3-none-any\n"
        ),
    }

    record_path = f"{dist_info}/RECORD"
    record_lines = [
        f"{path},sha256={_digest(data)},{len(data)}" for path, data in sorted(files.items())
    ]
    record_lines.append(f"{record_path},,")
    files[record_path] = ("\n".join(record_lines) + "\n").encode()

    wheel_path = directory / f"{dist}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, data in sorted(files.items()):
            archive.writestr(path, data)
    return wheel_path


@dataclass
class FakeUv:
    """Resolver fake that records commands and writes synthetic lockfiles."""

    fail_date_bound: bool = False
    calls: list[tuple[str, ...]] = field(default_factory=list)

    def run(
        self,
        args: Sequence[str],
        *,
        stdin: str | None = None,
        cwd: Path | None = None,
    ) -> UvResult:
        command = ("uv", *args)
        self.calls.append(command)

        if "compile" in args and self.fail_date_bound and "--exclude-newer" in args:
            return UvResult(
                command=command,
                returncode=1,
                stdout="",
                stderr="No solution found: date bound too tight",
            )

        if "compile" in args:
            output_index = args.index("--output-file") + 1
            Path(args[output_index]).write_text(
                "demo-package==1.0.0\n",
                encoding="utf-8",
            )

        return UvResult(command=command, returncode=0, stdout="", stderr="")


@dataclass
class SpyUv:
    """Wrap a real uv runner and record every invocation."""

    inner: UvRunner
    calls: list[tuple[str, ...]] = field(default_factory=list)

    def run(
        self,
        args: Sequence[str],
        *,
        stdin: str | None = None,
        cwd: Path | None = None,
    ) -> UvResult:
        self.calls.append(tuple(args))
        return self.inner.run(args, stdin=stdin, cwd=cwd)
