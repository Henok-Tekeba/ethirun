"""CLI coverage for `ethirun inspect` using local fixtures only."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from ethirun.cli import app
from ethirun.schemas import Report, Rung, RungState
from tests.test_inspector import WHISPER

runner = CliRunner()


def test_inspect_json_output_is_a_report() -> None:
    result = runner.invoke(
        app,
        [
            "inspect",
            "local/whisper",
            "--local-dir",
            str(WHISPER),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    report = Report.model_validate_json(result.output)
    assert report.model.repo == "local/whisper"
    assert report.scan is not None
    assert report.scan.adapter.family == "transformers_whisper"
    assert report.status.state_of(Rung.INSPECTED) is RungState.PASSED


def test_inspect_prints_a_human_summary() -> None:
    result = runner.invoke(
        app,
        ["inspect", "local/whisper", "--local-dir", str(WHISPER)],
    )

    assert result.exit_code == 0, result.output
    assert "transformers_whisper" in result.output
    assert "Python files: 0 scanned" in result.output
    assert "Status: inspected = passed" in result.output


def test_inspect_writes_the_report_file(tmp_path: Path) -> None:
    target = tmp_path / "report.json"

    result = runner.invoke(
        app,
        [
            "inspect",
            "local/whisper",
            "--local-dir",
            str(WHISPER),
            "--output",
            str(target),
        ],
    )

    assert result.exit_code == 0, result.output
    assert target.is_file()
    report = Report.model_validate_json(target.read_text(encoding="utf-8"))
    assert report.inventory
    assert "Report:" in result.output


def test_inspect_accepts_repo_at_revision() -> None:
    revision = "d" * 40

    result = runner.invoke(
        app,
        [
            "inspect",
            f"local/whisper@{revision}",
            "--local-dir",
            str(WHISPER),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    report = Report.model_validate_json(result.output)
    assert report.model.revision == revision


def test_inspect_fails_cleanly_for_missing_local_dir(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["inspect", "local/missing", "--local-dir", str(tmp_path / "nope")],
    )

    assert result.exit_code == 1
    assert "inspection failed" in result.output


def test_inspect_reports_unsupported_family_as_a_valid_result(tmp_path: Path) -> None:
    repo = tmp_path / "unknown"
    repo.mkdir()
    (repo / "config.json").write_text(
        '{"architectures": ["MysterySpeechModel"], "model_type": "mystery"}',
        encoding="utf-8",
    )

    result = runner.invoke(app, ["inspect", "local/unknown", "--local-dir", str(repo)])

    assert result.exit_code == 0, result.output
    assert "unsupported family" in result.output
