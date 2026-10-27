"""Network-free fake hub for inspector tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from ethirun.hub import HubError, HubSnapshot, RemoteFile


@dataclass
class FakeHub:
    snapshot_result: HubSnapshot
    contents: Mapping[str, bytes]
    snapshot_error: HubError | None = None
    requests: list[tuple[str, str]] = field(default_factory=list)
    downloads: list[str] = field(default_factory=list)

    def snapshot(self, repo: str, revision: str) -> HubSnapshot:
        self.requests.append((repo, revision))
        if self.snapshot_error is not None:
            raise self.snapshot_error
        return self.snapshot_result

    def download(
        self,
        repo: str,
        revision: str,
        filename: str,
        destination: Path,
    ) -> Path:
        self.downloads.append(filename)
        try:
            data = self.contents[filename]
        except KeyError as exc:
            raise HubError(f"fake hub has no content for {filename}") from exc

        target = destination / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return target


def make_snapshot(
    contents: Mapping[str, bytes],
    *,
    revision: str = "a" * 40,
    extra_files: tuple[RemoteFile, ...] = (),
    pipeline_tag: str | None = "automatic-speech-recognition",
    library_name: str | None = "transformers",
) -> HubSnapshot:
    files = (
        tuple(RemoteFile(path=path, size=len(data)) for path, data in contents.items())
        + extra_files
    )
    return HubSnapshot(
        revision=revision,
        files=files,
        pipeline_tag=pipeline_tag,
        library_name=library_name,
    )
