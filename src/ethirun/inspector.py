"""Stage 0 inspection plus Stage 1 static scan orchestration.

No model weights are downloaded and no repository code is executed. Remote runs
fetch only an allowlisted set of small metadata files at a pinned commit; local
runs read a directory directly.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import JsonValue, ValidationError

from ethirun.hub import HubClient, HubError, HubSnapshot, HuggingFaceHub
from ethirun.schemas import (
    AdapterChoice,
    CheckRecord,
    Evidence,
    EvidenceKind,
    FileKind,
    FileRecord,
    ModelRef,
    Report,
    Rung,
    RungState,
    StaticScanResult,
    StatusLadder,
)
from ethirun.static_scan import MAX_PYTHON_FILE_BYTES, scan_python

DEFAULT_CACHE_DIR = Path.home() / ".ethirun" / "inspected"

_READABLE_KINDS = frozenset(
    {
        FileKind.CONFIG,
        FileKind.GENERATION_CONFIG,
        FileKind.PREPROCESSOR_CONFIG,
        FileKind.TOKENIZER,
        FileKind.README,
        FileKind.REQUIREMENTS,
        FileKind.SAFETENSORS_INDEX,
    }
)

_TOKENIZER_NAMES = frozenset(
    {
        "added_tokens.json",
        "chat_template.jinja",
        "chat_template.json",
        "feature_extractor_config.json",
        "merges.txt",
        "normalizer.json",
        "processor_config.json",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
        "vocab.txt",
    }
)

_WEIGHT_SUFFIXES = frozenset({".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".h5", ".msgpack"})


class InspectionError(RuntimeError):
    """Inspection could not produce a trustworthy report."""


def classify_path(path: str) -> FileKind:
    """Classify a repository-relative path by inspection relevance."""
    name = PurePosixPath(path).name
    lower = name.lower()

    if name == "config.json":
        return FileKind.CONFIG
    if name == "generation_config.json":
        return FileKind.GENERATION_CONFIG
    if name == "preprocessor_config.json":
        return FileKind.PREPROCESSOR_CONFIG
    if lower in _TOKENIZER_NAMES:
        return FileKind.TOKENIZER
    if lower.startswith("readme"):
        return FileKind.README
    if lower == "requirements.txt":
        return FileKind.REQUIREMENTS
    if lower.endswith(".safetensors.index.json"):
        return FileKind.SAFETENSORS_INDEX
    if lower.endswith(".nemo"):
        return FileKind.NEMO
    if lower.endswith(".onnx"):
        return FileKind.ONNX
    if lower.endswith(".py"):
        return FileKind.PYTHON
    if PurePosixPath(lower).suffix in _WEIGHT_SUFFIXES:
        return FileKind.WEIGHTS
    return FileKind.OTHER


def inspection_target(
    path: str,
    size: int | None,
    *,
    max_python_bytes: int = MAX_PYTHON_FILE_BYTES,
) -> tuple[bool, FileKind, str | None]:
    """Return whether a file is read, its kind, and why it was skipped."""
    kind = classify_path(path)

    if kind in _READABLE_KINDS:
        return True, kind, None

    if kind is FileKind.PYTHON:
        if size is not None and size > max_python_bytes:
            return False, kind, f"Python file exceeds the {max_python_bytes}-byte cap"
        return True, kind, None

    if kind in {FileKind.WEIGHTS, FileKind.NEMO, FileKind.ONNX}:
        return False, kind, "weights/archives are never downloaded during inspection"

    return False, kind, "not on the inspection allowlist"


def _validate_relative(path: str) -> PurePosixPath:
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise InspectionError(f"unsafe repository path: {path!r}")
    return pure


def _read_text(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _local_digest(records: Sequence[FileRecord]) -> str:
    """Deterministic 40-hex identity for a local snapshot.

    Selected file contents are included; weights contribute path and size only,
    so pointing ``--local-dir`` at a large model does not hash gigabytes. This
    is an identity digest, not a security boundary.
    """
    hasher = hashlib.sha1()  # content identity, not authenticity
    for record in sorted(records, key=lambda item: item.path):
        hasher.update(record.path.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(str(record.size).encode("ascii"))
        hasher.update(b"\0")
        if record.sha256 is not None:
            hasher.update(record.sha256.encode("ascii"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _iter_local_files(root: Path) -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []
    for current, directories, filenames in os.walk(root, followlinks=False):
        directories[:] = sorted(
            directory
            for directory in directories
            if directory != ".git" and not (Path(current) / directory).is_symlink()
        )
        for filename in sorted(filenames):
            path = Path(current) / filename
            if path.is_symlink() or not path.is_file():
                continue
            found.append((path.relative_to(root).as_posix(), path.stat().st_size))
    return found


def _preferred_path(inventory: Sequence[FileRecord], kind: FileKind) -> str | None:
    candidates = [record.path for record in inventory if record.kind is kind and record.read]
    if not candidates:
        return None
    return min(candidates, key=lambda path: (len(PurePosixPath(path).parts), path))


def _config_from(
    contents: dict[str, str], inventory: Sequence[FileRecord]
) -> tuple[dict[str, Any], str | None]:
    config_path = _preferred_path(inventory, FileKind.CONFIG)
    if config_path is None:
        return {}, None

    raw = contents.get(config_path, "")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InspectionError(f"{config_path} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise InspectionError(f"{config_path} must contain a JSON object")
    return value, config_path


def _requirements_from(
    contents: dict[str, str], inventory: Sequence[FileRecord]
) -> tuple[str | None, str | None]:
    path = _preferred_path(inventory, FileKind.REQUIREMENTS)
    return (contents.get(path), path) if path is not None else (None, None)


def _make_evidence(
    *,
    repo: str,
    revision: str,
    local: bool,
    inventory: Sequence[FileRecord],
    config: dict[str, Any],
    config_path: str | None,
    scan: StaticScanResult,
    pipeline_tag: str | None,
    library_name: str | None,
) -> tuple[Evidence, ...]:
    items: list[Evidence] = []

    if local:
        items.append(
            Evidence(
                claim=f"Local snapshot hashed to revision {revision}",
                source_file="local_dir",
                kind=EvidenceKind.OBSERVED,
                confidence=1.0,
            )
        )
    else:
        items.append(
            Evidence(
                claim=f"Resolved {repo} to commit {revision}",
                source_file="hf_api",
                kind=EvidenceKind.OBSERVED,
                confidence=1.0,
            )
        )

    source = config_path or "config.json"
    architectures = config.get("architectures")
    if architectures:
        items.append(
            Evidence(
                claim=f"config declares architectures: {architectures}",
                source_file=source,
                kind=EvidenceKind.REPO,
                confidence=0.99,
            )
        )

    model_type = config.get("model_type")
    if model_type:
        items.append(
            Evidence(
                claim=f"config model_type is {model_type}",
                source_file=source,
                kind=EvidenceKind.REPO,
                confidence=0.99,
            )
        )

    transformers_version = config.get("transformers_version")
    if transformers_version:
        items.append(
            Evidence(
                claim=f"author saved with transformers {transformers_version}",
                source_file=source,
                kind=EvidenceKind.REPO,
                confidence=0.95,
            )
        )

    auto_map = config.get("auto_map")
    if auto_map:
        items.append(
            Evidence(
                claim=f"config auto_map requires trust-remote-code: {auto_map}",
                source_file=source,
                kind=EvidenceKind.REPO,
                confidence=0.99,
            )
        )

    if pipeline_tag is not None:
        items.append(
            Evidence(
                claim=f"hub pipeline tag is {pipeline_tag}",
                source_file="hf_api",
                kind=EvidenceKind.EXPLICIT,
                confidence=0.9,
            )
        )
    if library_name is not None:
        items.append(
            Evidence(
                claim=f"hub library_name is {library_name}",
                source_file="hf_api",
                kind=EvidenceKind.EXPLICIT,
                confidence=0.9,
            )
        )

    read_count = sum(1 for record in inventory if record.read)
    items.append(
        Evidence(
            claim=(
                f"file inventory has {len(inventory)} files; "
                f"{read_count} small files were read, weights were skipped"
            ),
            source_file="file_inventory",
            kind=EvidenceKind.OBSERVED,
            confidence=1.0,
        )
    )

    weight_paths = [record.path for record in inventory if record.kind is FileKind.WEIGHTS]
    if weight_paths:
        items.append(
            Evidence(
                claim=f"repository contains {len(weight_paths)} weight files; none downloaded",
                source_file="file_inventory",
                kind=EvidenceKind.REPO,
                confidence=1.0,
            )
        )

    for special in (FileKind.NEMO, FileKind.ONNX):
        paths = [record.path for record in inventory if record.kind is special]
        if paths:
            items.append(
                Evidence(
                    claim=f"repository contains {special.value} artifacts: {paths}",
                    source_file="file_inventory",
                    kind=EvidenceKind.REPO,
                    confidence=1.0,
                )
            )

    items.append(
        Evidence(
            claim=(
                f"{len(scan.python_files)} Python files parsed with AST; none imported or executed"
            ),
            source_file="static_scan",
            kind=EvidenceKind.OBSERVED,
            confidence=1.0,
        )
    )

    if scan.remote_code:
        items.append(
            Evidence(
                claim="remote code is required before this repo can be loaded",
                source_file="config.json",
                kind=EvidenceKind.REPO,
                confidence=0.99,
            )
        )

    if scan.undeclared_imports:
        items.append(
            Evidence(
                claim=f"undeclared imports: {list(scan.undeclared_imports)}",
                source_file="requirements.txt",
                kind=EvidenceKind.OBSERVED,
                confidence=0.95,
            )
        )

    if scan.scan_errors:
        items.append(
            Evidence(
                claim=f"AST scan errors: {list(scan.scan_errors)}",
                source_file="static_scan",
                kind=EvidenceKind.OBSERVED,
                confidence=1.0,
            )
        )

    adapter = scan.adapter
    if adapter.family is None:
        claim = "no supported adapter family was identified"
    else:
        support = "supported" if adapter.supported else "not implemented"
        claim = f"adapter family {adapter.family} selected ({support})"
    items.append(
        Evidence(
            claim=claim,
            source_file="static_scan",
            kind=EvidenceKind.INFERRED,
            confidence=adapter.confidence,
        )
    )

    return tuple(items)


def _adapter_detail(adapter: AdapterChoice) -> dict[str, JsonValue]:
    return {
        "family": adapter.family,
        "confidence": adapter.confidence,
        "supported": adapter.supported,
        "requires_trust_remote_code": adapter.requires_trust_remote_code,
        "reasons": list(adapter.reasons),
    }


class Inspector:
    """Run Stage 0 and Stage 1 for one repository snapshot."""

    def __init__(
        self,
        *,
        hub: HubClient | None = None,
        cache_dir: Path | None = None,
        max_python_bytes: int = MAX_PYTHON_FILE_BYTES,
    ) -> None:
        self._hub = hub or HuggingFaceHub()
        self._cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self._max_python_bytes = max_python_bytes

    def inspect(
        self,
        repo: str,
        *,
        revision: str | None = None,
        local_dir: Path | None = None,
        now: datetime | None = None,
    ) -> Report:
        """Inspect a remote repo or a local directory and return a report."""
        generated_at = now or datetime.now(UTC)
        snapshot: HubSnapshot | None = None
        if local_dir is not None:
            inventory, contents, resolved_revision = self._inspect_local(
                local_dir,
                revision=revision,
            )
        else:
            snapshot = self._snapshot(repo, revision or "main")
            inventory, contents = self._inspect_remote(repo, snapshot)
            resolved_revision = snapshot.revision

        config, config_path = _config_from(contents, inventory)
        requirements, _ = _requirements_from(contents, inventory)
        python_sources = {
            record.path: contents[record.path]
            for record in inventory
            if record.kind is FileKind.PYTHON and record.read and record.path in contents
        }
        scan = scan_python(
            python_sources,
            file_paths=[record.path for record in inventory],
            requirements=requirements,
            config=config,
            pipeline_tag=None if snapshot is None else snapshot.pipeline_tag,
            library_name=None if snapshot is None else snapshot.library_name,
        )

        evidence = _make_evidence(
            repo=repo,
            revision=resolved_revision,
            local=local_dir is not None,
            inventory=inventory,
            config=config,
            config_path=config_path,
            scan=scan,
            pipeline_tag=None if snapshot is None else snapshot.pipeline_tag,
            library_name=None if snapshot is None else snapshot.library_name,
        )

        detail: dict[str, JsonValue] = {
            "repo": repo,
            "revision": resolved_revision,
            "files_listed": len(inventory),
            "files_read": sum(1 for record in inventory if record.read),
            "python_files": len(scan.python_files),
            "imports": len(scan.imports),
            "undeclared_imports": list(scan.undeclared_imports),
            "remote_code": scan.remote_code,
            "adapter": _adapter_detail(scan.adapter),
        }
        check = CheckRecord(
            rung=Rung.INSPECTED,
            check="stage0.inspect",
            stage="stage0",
            outcome=RungState.PASSED,
            ran_at=generated_at,
            detail=detail,
        )
        status = StatusLadder.initial().with_check(check)

        try:
            return Report.create(
                model=ModelRef(
                    repo=repo,
                    revision=resolved_revision,
                    family=scan.adapter.family,
                ),
                generated_at=generated_at,
                evidence=evidence,
                inventory=inventory,
                scan=scan,
                status=status,
            )
        except ValidationError as exc:
            raise InspectionError(f"inspection produced an invalid report: {exc}") from exc

    def _snapshot(self, repo: str, revision: str) -> HubSnapshot:
        try:
            return self._hub.snapshot(repo, revision)
        except HubError as exc:
            raise InspectionError(str(exc)) from exc

    def _inspect_remote(
        self,
        repo: str,
        snapshot: HubSnapshot,
    ) -> tuple[tuple[FileRecord, ...], dict[str, str]]:
        destination = self._cache_dir / repo.replace("/", "__") / snapshot.revision
        records: list[FileRecord] = []
        contents: dict[str, str] = {}

        for remote in sorted(snapshot.files, key=lambda item: item.path):
            _validate_relative(remote.path)
            selected, kind, skip_reason = inspection_target(
                remote.path,
                remote.size,
                max_python_bytes=self._max_python_bytes,
            )
            if not selected:
                records.append(
                    FileRecord(
                        path=remote.path,
                        size=remote.size,
                        kind=kind,
                        read=False,
                        skip_reason=skip_reason,
                    )
                )
                continue

            try:
                downloaded = self._hub.download(
                    repo,
                    snapshot.revision,
                    remote.path,
                    destination,
                )
                data = downloaded.read_bytes()
            except (HubError, OSError) as exc:
                raise InspectionError(f"could not read {remote.path}: {exc}") from exc

            if kind is FileKind.PYTHON and len(data) > self._max_python_bytes:
                records.append(
                    FileRecord(
                        path=remote.path,
                        size=len(data),
                        kind=kind,
                        read=False,
                        skip_reason=(f"Python file exceeds the {self._max_python_bytes}-byte cap"),
                    )
                )
                continue

            contents[remote.path] = _read_text(data)
            records.append(
                FileRecord(
                    path=remote.path,
                    size=len(data),
                    kind=kind,
                    read=True,
                    sha256=_sha256(data),
                )
            )

        return tuple(records), contents

    def _inspect_local(
        self,
        local_dir: Path,
        *,
        revision: str | None,
    ) -> tuple[tuple[FileRecord, ...], dict[str, str], str]:
        root = local_dir.expanduser().resolve()
        if not root.is_dir():
            raise InspectionError(f"local directory not found: {root}")

        records: list[FileRecord] = []
        contents: dict[str, str] = {}

        for path, size in _iter_local_files(root):
            _validate_relative(path)
            selected, kind, skip_reason = inspection_target(
                path,
                size,
                max_python_bytes=self._max_python_bytes,
            )
            if not selected:
                records.append(
                    FileRecord(
                        path=path,
                        size=size,
                        kind=kind,
                        read=False,
                        skip_reason=skip_reason,
                    )
                )
                continue

            try:
                data = (root / path).read_bytes()
            except OSError as exc:
                raise InspectionError(f"could not read {path}: {exc}") from exc

            contents[path] = _read_text(data)
            records.append(
                FileRecord(
                    path=path,
                    size=len(data),
                    kind=kind,
                    read=True,
                    sha256=_sha256(data),
                )
            )

        inventory = tuple(records)
        resolved_revision = revision or _local_digest(inventory)
        return inventory, contents, resolved_revision
