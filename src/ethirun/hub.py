"""Thin, injectable boundary around the Hugging Face Hub.

Stage 0 only needs two things: resolve/list a model at a revision, and fetch a
small allowlisted file. Keeping this behind a protocol lets the unit tests run
with a fake hub and no network.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from huggingface_hub import HfApi, hf_hub_download


class HubError(RuntimeError):
    """The hub could not resolve or download an inspected file."""


@dataclass(frozen=True, slots=True)
class RemoteFile:
    """A file advertised by the hub, with its size when known."""

    path: str
    size: int | None = None


@dataclass(frozen=True, slots=True)
class HubSnapshot:
    """A repository resolved to one immutable commit."""

    revision: str
    files: tuple[RemoteFile, ...]
    pipeline_tag: str | None = None
    library_name: str | None = None
    tags: tuple[str, ...] = ()
    last_modified: datetime | None = None


class HubClient(Protocol):
    """The small hub surface the inspector depends on."""

    def snapshot(self, repo: str, revision: str) -> HubSnapshot:
        """Resolve a repo/revision to a full commit SHA and file inventory."""
        ...

    def download(
        self,
        repo: str,
        revision: str,
        filename: str,
        destination: Path,
    ) -> Path:
        """Download one file at a pinned revision into ``destination``."""
        ...


class HuggingFaceHub:
    """Production hub client backed by ``huggingface_hub``."""

    def __init__(self, api: HfApi | None = None) -> None:
        self._api = api or HfApi()

    def snapshot(self, repo: str, revision: str) -> HubSnapshot:
        try:
            info = self._api.model_info(
                repo_id=repo,
                revision=revision,
                files_metadata=True,
            )
        except Exception as exc:
            raise HubError(f"could not resolve {repo}@{revision}: {exc}") from exc

        if not isinstance(info.sha, str) or not info.sha:
            raise HubError(f"hub did not return a commit SHA for {repo}@{revision}")

        files = tuple(
            RemoteFile(path=sibling.rfilename, size=sibling.size)
            for sibling in (info.siblings or [])
        )
        return HubSnapshot(
            revision=info.sha,
            files=files,
            pipeline_tag=info.pipeline_tag,
            library_name=info.library_name,
            tags=tuple(info.tags or ()),
            last_modified=info.last_modified,
        )

    def download(
        self,
        repo: str,
        revision: str,
        filename: str,
        destination: Path,
    ) -> Path:
        destination.mkdir(parents=True, exist_ok=True)
        try:
            downloaded = hf_hub_download(
                repo_id=repo,
                filename=filename,
                revision=revision,
                local_dir=str(destination),
            )
        except Exception as exc:
            raise HubError(f"could not download {repo}@{revision}/{filename}: {exc}") from exc

        if not isinstance(downloaded, str):
            raise HubError(f"hub returned no local path for {filename}")
        return Path(downloaded)
