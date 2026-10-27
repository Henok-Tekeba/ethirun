"""Stage 0 tests: local and network-free remote inspection."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ethirun.hub import HubError, RemoteFile
from ethirun.inspector import (
    InspectionError,
    Inspector,
    classify_path,
    inspection_target,
)
from ethirun.schemas import FileKind, Rung, RungState
from ethirun.static_scan import CTC_FAMILY, CUSTOM_SCRIPT_FAMILY, NEMO_FAMILY, WHISPER_FAMILY
from ethirun.store import dumps_report_json, loads_report_json
from tests.fake_hub import FakeHub, make_snapshot

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "repos"
WHISPER = FIXTURES / "whisper_local"
CTC = FIXTURES / "ctc_local"
CUSTOM = FIXTURES / "custom_whisper"
NOW = datetime(2026, 10, 27, 12, 30, tzinfo=UTC)
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def test_local_whisper_inspection_produces_a_full_report(tmp_path: Path) -> None:
    report = Inspector(cache_dir=tmp_path).inspect(
        "local/whisper",
        local_dir=WHISPER,
        now=NOW,
    )

    assert report.model.repo == "local/whisper"
    assert SHA_PATTERN.match(report.model.revision)
    assert report.model.family == WHISPER_FAMILY
    assert report.generated_at == NOW
    assert report.status.state_of(Rung.INSPECTED) is RungState.PASSED
    assert report.status.state_of(Rung.TRANSCRIBES) is RungState.UNTESTED

    inspected = report.status.root[Rung.INSPECTED]
    assert inspected.check is not None
    assert inspected.check.check == "stage0.inspect"
    assert inspected.check.stage == "stage0"
    assert inspected.check.ran_at == NOW
    assert inspected.check.detail["remote_code"] is False
    assert inspected.check.detail["python_files"] == 0

    scan = report.scan
    assert scan is not None
    assert scan.adapter.family == WHISPER_FAMILY
    assert scan.adapter.supported is True
    assert scan.remote_code is False
    assert scan.python_files == ()
    assert scan.declared_packages == ("numpy", "torch", "transformers")

    inventory = {record.path: record for record in report.inventory}
    assert inventory["config.json"].read is True
    assert inventory["config.json"].sha256 is not None
    assert inventory["tokenizer.json"].kind is FileKind.TOKENIZER
    assert inventory["tokenizer.json"].read is True
    assert inventory["model.safetensors.index.json"].kind is FileKind.SAFETENSORS_INDEX
    assert inventory["model.safetensors.index.json"].read is True
    assert inventory["pytorch_model.bin"].kind is FileKind.WEIGHTS
    assert inventory["pytorch_model.bin"].read is False
    assert "never downloaded" in (inventory["pytorch_model.bin"].skip_reason or "")

    claims = [item.claim for item in report.evidence]
    assert any("Local snapshot hashed" in claim for claim in claims)
    assert any("architectures" in claim for claim in claims)
    assert any("transformers_whisper" in claim for claim in claims)


def test_report_with_inventory_and_scan_round_trips_through_json() -> None:
    report = Inspector().inspect("local/whisper", local_dir=WHISPER, now=NOW)

    loaded = loads_report_json(dumps_report_json(report))

    assert loaded == report
    assert loaded.inventory
    assert loaded.scan is not None


def test_local_revision_is_deterministic_and_can_be_overridden() -> None:
    inspector = Inspector()
    first = inspector.inspect("local/whisper", local_dir=WHISPER, now=NOW)
    second = inspector.inspect("local/whisper", local_dir=WHISPER, now=NOW)
    overridden = inspector.inspect(
        "local/whisper",
        local_dir=WHISPER,
        revision="b" * 40,
        now=NOW,
    )

    assert first.model.revision == second.model.revision
    assert overridden.model.revision == "b" * 40

    with pytest.raises(InspectionError, match="invalid report"):
        inspector.inspect("local/whisper", local_dir=WHISPER, revision="main")


def test_local_ctc_fixture_selects_ctc_adapter() -> None:
    report = Inspector().inspect("local/ctc", local_dir=CTC, now=NOW)

    assert report.scan is not None
    assert report.scan.adapter.family == CTC_FAMILY
    assert report.scan.adapter.supported is True


def test_custom_code_repo_is_flagged_and_never_executed() -> None:
    report = Inspector().inspect("local/custom", local_dir=CUSTOM, now=NOW)

    assert report.scan is not None
    scan = report.scan
    assert scan.remote_code is True
    assert scan.adapter.family == CUSTOM_SCRIPT_FAMILY
    assert scan.adapter.supported is False
    assert scan.adapter.requires_trust_remote_code is True
    assert scan.python_files == ("configuration_custom.py", "modeling_custom.py")
    assert scan.undeclared_imports == ("numpy", "pillow")

    relative = [record for record in scan.imports if record.module.startswith(".")]
    assert relative
    assert all(record.kind.value == "local" for record in relative)


def test_python_size_cap_skips_sources_but_preserves_the_adapter(tmp_path: Path) -> None:
    report = Inspector(max_python_bytes=10).inspect(
        "local/custom",
        local_dir=CUSTOM,
        now=NOW,
    )

    assert report.scan is not None
    assert report.scan.python_files == ()
    assert report.scan.adapter.family == CUSTOM_SCRIPT_FAMILY
    skipped = [record for record in report.inventory if record.kind is FileKind.PYTHON]
    assert skipped
    assert all(record.read is False for record in skipped)
    assert all("cap" in (record.skip_reason or "") for record in skipped)


def test_nemo_artifact_is_a_valid_unsupported_result(tmp_path: Path) -> None:
    repo = tmp_path / "nemo_repo"
    repo.mkdir()
    (repo / "config.json").write_text(
        '{"architectures": ["WhisperForConditionalGeneration"], "model_type": "whisper"}',
        encoding="utf-8",
    )
    (repo / "voice.nemo").write_bytes(b"not a real nemo archive")

    report = Inspector().inspect("local/nemo", local_dir=repo, now=NOW)

    assert report.scan is not None
    assert report.scan.adapter.family == NEMO_FAMILY
    assert report.scan.adapter.supported is False
    assert report.status.state_of(Rung.INSPECTED) is RungState.PASSED
    nemo = next(record for record in report.inventory if record.kind is FileKind.NEMO)
    assert nemo.read is False


def test_unknown_family_still_produces_a_passed_inspection(tmp_path: Path) -> None:
    repo = tmp_path / "unknown_repo"
    repo.mkdir()
    (repo / "config.json").write_text(
        '{"architectures": ["MysterySpeechModel"], "model_type": "mystery"}',
        encoding="utf-8",
    )

    report = Inspector().inspect("local/unknown", local_dir=repo, now=NOW)

    assert report.scan is not None
    assert report.scan.adapter.family is None
    assert report.scan.adapter.supported is False
    assert report.status.state_of(Rung.INSPECTED) is RungState.PASSED

    loaded = loads_report_json(dumps_report_json(report))
    assert loaded == report


def test_remote_inspection_downloads_only_allowlisted_files(tmp_path: Path) -> None:
    sha = "c" * 40
    contents = {
        "config.json": b'{"architectures": ["WhisperForConditionalGeneration"], '
        b'"model_type": "whisper"}',
        "requirements.txt": b"transformers\ntorch\n",
        "README.md": b"# remote fixture\n",
        "tokenizer.json": b"{}",
        "model.safetensors.index.json": b'{"weight_map": {}}',
        "script.py": b"import numpy\nimport torch\n",
    }
    snapshot = make_snapshot(
        contents,
        revision=sha,
        extra_files=(
            RemoteFile("model.safetensors", 10_000),
            RemoteFile("huge.py", 500_000),
            RemoteFile("notes.txt", 20),
        ),
    )
    hub = FakeHub(snapshot, contents)
    inspector = Inspector(hub=hub, cache_dir=tmp_path / "cache", max_python_bytes=1000)

    report = inspector.inspect("org/remote", now=NOW)

    assert report.model.revision == sha
    assert hub.requests == [("org/remote", "main")]
    assert hub.downloads == sorted(
        [
            "README.md",
            "config.json",
            "model.safetensors.index.json",
            "requirements.txt",
            "script.py",
            "tokenizer.json",
        ]
    )
    assert "model.safetensors" not in hub.downloads
    assert "huge.py" not in hub.downloads
    assert "notes.txt" not in hub.downloads

    scan = report.scan
    assert scan is not None
    assert scan.adapter.family == WHISPER_FAMILY
    assert scan.undeclared_imports == ("numpy",)
    assert scan.python_files == ("script.py",)

    inventory = {record.path: record for record in report.inventory}
    assert inventory["model.safetensors"].read is False
    assert inventory["huge.py"].read is False
    assert "cap" in (inventory["huge.py"].skip_reason or "")
    assert inventory["notes.txt"].read is False

    cache_file = tmp_path / "cache" / "org__remote" / sha / "config.json"
    assert cache_file.is_file()
    assert any("automatic-speech-recognition" in item.claim for item in report.evidence)


def test_remote_hub_errors_are_wrapped(tmp_path: Path) -> None:
    hub = FakeHub(
        make_snapshot({}),
        {},
        snapshot_error=HubError("network unavailable"),
    )

    with pytest.raises(InspectionError, match="network unavailable"):
        Inspector(hub=hub, cache_dir=tmp_path).inspect("org/model")


def test_unsafe_remote_paths_are_rejected(tmp_path: Path) -> None:
    contents = {"config.json": b"{}"}
    snapshot = make_snapshot(
        contents,
        extra_files=(RemoteFile("../escape.py", 10),),
    )
    hub = FakeHub(snapshot, contents)

    with pytest.raises(InspectionError, match="unsafe repository path"):
        Inspector(hub=hub, cache_dir=tmp_path).inspect("org/model")

    assert "../escape.py" not in hub.downloads


@pytest.mark.parametrize(
    ("path", "size", "expected_kind", "expected_read"),
    [
        ("config.json", 10, FileKind.CONFIG, True),
        ("README.md", 10, FileKind.README, True),
        ("sub/vocab.json", 10, FileKind.TOKENIZER, True),
        ("model.safetensors.index.json", 10, FileKind.SAFETENSORS_INDEX, True),
        ("script.py", 10, FileKind.PYTHON, True),
        ("script.py", 1000, FileKind.PYTHON, False),
        ("pytorch_model.bin", 10, FileKind.WEIGHTS, False),
        ("model.onnx", 10, FileKind.ONNX, False),
        ("voice.nemo", 10, FileKind.NEMO, False),
        ("notes.txt", 10, FileKind.OTHER, False),
    ],
)
def test_inspection_target_allowlist(
    path: str,
    size: int,
    expected_kind: FileKind,
    expected_read: bool,
) -> None:
    selected, kind, _ = inspection_target(path, size, max_python_bytes=100)

    assert selected is expected_read
    assert kind is expected_kind
    assert classify_path(path) is expected_kind
