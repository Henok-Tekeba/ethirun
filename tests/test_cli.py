"""CLI smoke tests for the Milestone 0 application shell."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from ethirun import __version__
from ethirun.cli import app

runner = CliRunner()


def test_help_exits_successfully() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Usage: ethirun [OPTIONS] COMMAND [ARGS]..." in result.stdout
    assert "isolated, reproducible worker environments" in result.stdout


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])

    assert result.exit_code == 2
    assert "Usage: ethirun" in result.stdout


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == f"ethirun {__version__}"


@pytest.mark.parametrize("module", ["ethirun", "ethirun.cli"])
def test_package_is_importable(module: str) -> None:
    __import__(module)
