"""Stage 2: resolve adapter hints through a bounded candidate matrix.

Candidates are tried in order:

A. Python 3.11, packages as of commit date + 120 days
B. an alternative Python, same date bound
C. Python 3.11, no date bound

Every attempt is recorded. The first successful lockfile becomes the recipe's
``requirements.lock``; failed attempts keep uv's conflict message.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

from ethirun.static_scan import CTC_FAMILY, WHISPER_FAMILY
from ethirun.uv import Uv, UvError, UvIndexOptions, UvRunner, UvTimeoutError

CANDIDATE_DATE_BOUND_DAYS = 120
DEFAULT_PYTHON = "3.11"
DEFAULT_ALTERNATE_PYTHON = "3.12"

_TRANSFORMERS_VERSION = re.compile(r"^(\d+\.\d+(?:\.\d+)?)")


class ResolverError(RuntimeError):
    """Resolution could not start, or uv itself failed unexpectedly."""


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    """One row of the resolver matrix."""

    key: Literal["A", "B", "C"]
    python: str
    exclude_newer: date | None


@dataclass(frozen=True, slots=True)
class CandidateOutcome:
    """Result of compiling one candidate."""

    candidate: CandidateSpec
    status: Literal["resolved", "failed"]
    lockfile: Path | None
    stdout: str
    stderr: str
    message: str

    @property
    def resolved(self) -> bool:
        return self.status == "resolved"


@dataclass(frozen=True, slots=True)
class ResolutionResult:
    """All candidate attempts plus the selected recipe lockfile."""

    attempts: tuple[CandidateOutcome, ...]
    selected: CandidateOutcome | None
    recipe_lockfile: Path | None
    requirements_in: Path

    @property
    def resolved(self) -> bool:
        return self.selected is not None


def default_candidates(
    reference_date: date,
    *,
    python_b: str = DEFAULT_ALTERNATE_PYTHON,
) -> tuple[CandidateSpec, ...]:
    """Build the A/B/C matrix relative to a commit date."""
    excluded = reference_date + timedelta(days=CANDIDATE_DATE_BOUND_DAYS)
    return (
        CandidateSpec(key="A", python=DEFAULT_PYTHON, exclude_newer=excluded),
        CandidateSpec(key="B", python=python_b, exclude_newer=excluded),
        CandidateSpec(key="C", python=DEFAULT_PYTHON, exclude_newer=None),
    )


def _transformers_hint(config: Mapping[str, Any]) -> str:
    raw = config.get("transformers_version")
    if isinstance(raw, str):
        match = _TRANSFORMERS_VERSION.match(raw)
        if match:
            return f"transformers>={match.group(1)}"
    return "transformers"


def seeds_for_family(
    family: str | None,
    config: Mapping[str, Any],
) -> tuple[str, ...]:
    """Adapter package hints, with the saved transformers version as a hint."""
    if family not in {WHISPER_FAMILY, CTC_FAMILY}:
        raise ResolverError(f"adapter {family!r} is not supported for resolution yet")

    return (
        _transformers_hint(config),
        "torch",
        "numpy",
        "soundfile",
    )


class Resolver:
    """Compile candidate lockfiles without installing anything."""

    def __init__(
        self,
        uv: UvRunner | None = None,
        *,
        index: UvIndexOptions | None = None,
        generate_hashes: bool = True,
    ) -> None:
        self._uv = uv or Uv()
        self._index = index or UvIndexOptions()
        self._generate_hashes = generate_hashes

    def resolve(
        self,
        seeds: Sequence[str],
        *,
        output_dir: Path,
        reference_date: date,
        python_b: str = DEFAULT_ALTERNATE_PYTHON,
        generate_hashes: bool | None = None,
        candidates: Sequence[CandidateSpec] | None = None,
    ) -> ResolutionResult:
        """Run every candidate and write lockfiles plus ``requirements.lock``."""
        cleaned = [seed.strip() for seed in seeds if seed.strip()]
        if not cleaned:
            raise ResolverError("resolver needs at least one seed requirement")
        if any("\n" in seed for seed in cleaned):
            raise ResolverError("seed requirements must be single lines")

        matrix = (
            tuple(candidates)
            if candidates is not None
            else default_candidates(reference_date, python_b=python_b)
        )
        if not matrix:
            raise ResolverError("candidate matrix is empty")

        output_dir.mkdir(parents=True, exist_ok=True)
        requirements_in = output_dir / "requirements.in"
        input_text = "\n".join(cleaned) + "\n"
        requirements_in.write_text(input_text, encoding="utf-8")

        recipe_lockfile = output_dir / "requirements.lock"
        recipe_lockfile.unlink(missing_ok=True)

        use_hashes = self._generate_hashes if generate_hashes is None else generate_hashes
        attempts: list[CandidateOutcome] = []

        for candidate in matrix:
            lockfile = output_dir / f"requirements-{candidate.key}.lock"
            lockfile.unlink(missing_ok=True)
            command = self._compile_command(
                candidate,
                lockfile=lockfile,
                generate_hashes=use_hashes,
            )

            try:
                result = self._uv.run(command, stdin=input_text)
            except UvTimeoutError as exc:
                attempts.append(
                    CandidateOutcome(
                        candidate=candidate,
                        status="failed",
                        lockfile=None,
                        stdout="",
                        stderr=str(exc),
                        message=str(exc),
                    )
                )
                continue
            except UvError as exc:
                raise ResolverError(str(exc)) from exc

            if result.ok and lockfile.is_file():
                attempts.append(
                    CandidateOutcome(
                        candidate=candidate,
                        status="resolved",
                        lockfile=lockfile,
                        stdout=result.stdout,
                        stderr=result.stderr,
                        message="resolved",
                    )
                )
            else:
                lockfile.unlink(missing_ok=True)
                message = result.failure_message()
                attempts.append(
                    CandidateOutcome(
                        candidate=candidate,
                        status="failed",
                        lockfile=None,
                        stdout=result.stdout,
                        stderr=result.stderr,
                        message=message,
                    )
                )

        selected = next((attempt for attempt in attempts if attempt.resolved), None)
        if selected is not None and selected.lockfile is not None:
            recipe_lockfile.write_bytes(selected.lockfile.read_bytes())

        return ResolutionResult(
            attempts=tuple(attempts),
            selected=selected,
            recipe_lockfile=recipe_lockfile if selected is not None else None,
            requirements_in=requirements_in,
        )

    def _compile_command(
        self,
        candidate: CandidateSpec,
        *,
        lockfile: Path,
        generate_hashes: bool,
    ) -> tuple[str, ...]:
        command = [
            "pip",
            "compile",
            "--python-version",
            candidate.python,
            "--output-file",
            str(lockfile),
            "--no-header",
            *self._index.args(),
        ]
        if candidate.exclude_newer is not None:
            command.extend(("--exclude-newer", candidate.exclude_newer.isoformat()))
        if generate_hashes:
            command.append("--generate-hashes")
        command.append("-")
        return tuple(command)
