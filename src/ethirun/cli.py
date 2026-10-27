"""Command-line entry point for ethirun."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from ethirun import __version__
from ethirun.inspector import DEFAULT_CACHE_DIR, InspectionError, Inspector
from ethirun.schemas import Report, Rung
from ethirun.store import dumps_report_json

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


def _split_revision(value: str, revision: str | None) -> tuple[str, str | None]:
    """Support both ``repo@revision`` and ``--revision``."""
    repo, separator, embedded = value.rpartition("@")
    if not separator:
        return value, revision
    if not repo or not embedded:
        raise typer.BadParameter(f"invalid repo@revision: {value!r}")
    if revision is not None and revision != embedded:
        raise typer.BadParameter("revision given twice with different values")
    return repo, embedded


def _print_summary(report: Report, output: Path | None) -> None:
    console.print(
        f"[bold]Inspected[/bold] {report.model.repo} @ [cyan]{report.model.revision}[/cyan]"
    )
    console.print(
        f"Files: {len(report.inventory)} listed, "
        f"{sum(1 for record in report.inventory if record.read)} read"
    )

    scan = report.scan
    if scan is not None:
        adapter = scan.adapter
        if adapter.family is None:
            console.print("[yellow]Adapter: unsupported family (no confident match)[/yellow]")
        else:
            support = "supported" if adapter.supported else "not implemented"
            colour = "green" if adapter.supported else "yellow"
            console.print(
                f"Adapter: [{colour}]{adapter.family}[/{colour}] "
                f"confidence {adapter.confidence:.2f} ({support})"
            )
        console.print(
            f"Python files: {len(scan.python_files)} scanned; "
            f"remote code: {'yes' if scan.remote_code else 'no'}"
        )
        if scan.undeclared_imports:
            console.print(
                "[yellow]Undeclared imports:[/yellow] " + ", ".join(scan.undeclared_imports)
            )
        if scan.scan_errors:
            console.print(f"[yellow]Scan errors:[/yellow] {len(scan.scan_errors)}")

        for reason in adapter.reasons:
            console.print(f"  [dim]- {reason}[/dim]")

    console.print(f"Status: inspected = {report.status.state_of(Rung.INSPECTED).value}")
    if output is not None:
        console.print(f"Report: [cyan]{output}[/cyan]")


@app.command()
def inspect(
    repo: Annotated[
        str,
        typer.Argument(
            help="Hugging Face repo, or org/model@revision.",
            metavar="REPO",
        ),
    ],
    revision: Annotated[
        str | None,
        typer.Option(
            "--revision",
            "-r",
            help="Revision to resolve. Defaults to main for remote repos.",
        ),
    ] = None,
    local_dir: Annotated[
        Path | None,
        typer.Option(
            "--local-dir",
            help="Inspect a local repository directory instead of the hub.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Write the full report JSON to this path.",
        ),
    ] = None,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Print the full report JSON instead of a summary."),
    ] = False,
    cache_dir: Annotated[
        Path | None,
        typer.Option(
            "--cache-dir",
            help=f"Download cache for inspected files (default: {DEFAULT_CACHE_DIR}).",
        ),
    ] = None,
    max_python_bytes: Annotated[
        int,
        typer.Option(
            "--max-python-bytes",
            min=1,
            help="Maximum size of a .py file to scan.",
        ),
    ] = 256 * 1024,
) -> None:
    """Run stages 0-1 and emit an evidence report. No weights are downloaded."""
    repo_name, resolved_revision = _split_revision(repo, revision)
    inspector = Inspector(
        cache_dir=cache_dir,
        max_python_bytes=max_python_bytes,
    )

    try:
        report = inspector.inspect(
            repo_name,
            revision=resolved_revision,
            local_dir=local_dir,
        )
    except (InspectionError, OSError) as exc:
        console.print(f"[red]inspection failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(dumps_report_json(report), encoding="utf-8")

    if as_json:
        typer.echo(dumps_report_json(report), nl=False)
        return

    _print_summary(report, output)


def main() -> None:
    """Console-script entry point."""
    app()
