"""Command-line entry point for ethirun.

Milestone 0 exposes only the application shell. Model commands are added in
later milestones, one verified stage at a time.
"""

from __future__ import annotations

import typer
from rich.console import Console

from ethirun import __version__

app = typer.Typer(
    name="ethirun",
    help="Run Hugging Face ASR models in isolated, reproducible worker environments.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"ethirun {__version__}")
        raise typer.Exit


@app.callback()
def root(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show the ethirun version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Run and verify Hugging Face ASR models without polluting the host project."""


def main() -> None:
    """Console-script entry point."""
    app()
