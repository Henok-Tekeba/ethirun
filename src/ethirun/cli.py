"""Command-line entry point for ethirun."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from ethirun import __version__
from ethirun.envs import DEFAULT_ENVS_DIR
from ethirun.inspector import DEFAULT_CACHE_DIR, InspectionError, Inspector
from ethirun.models import DEFAULT_MODELS_DIR
from ethirun.schemas import Recipe, Report, Rung, StatusLadder
from ethirun.store import dumps_report_json
from ethirun.workflow import (
    DEFAULT_RECIPES_DIR,
    build_model,
    plan_model,
    verify_model,
)

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


def _print_ladder(status: StatusLadder) -> None:
    for rung in (
        Rung.INSPECTED,
        Rung.RESOLVED,
        Rung.INSTALLED,
        Rung.IMPORTS,
        Rung.CONSTRUCTS,
        Rung.WEIGHTS_COMPATIBLE,
        Rung.LOADS,
        Rung.TRANSCRIBES,
        Rung.ACCURACY_CHECKED,
    ):
        state = status.state_of(rung)
        colour = {
            "passed": "green",
            "failed": "red",
            "not_applicable": "dim",
            "untested": "yellow",
        }[state.value]
        console.print(f"  {rung.value:20} [{colour}]{state.value}[/{colour}]")


def _print_recipe(recipe: Recipe, recipe_dir: Path) -> None:
    console.print(f"[bold]{recipe.model.repo}[/bold] @ [cyan]{recipe.model.revision}[/cyan]")
    console.print(f"Recipe: [cyan]{recipe_dir}[/cyan]")
    console.print(f"Family: {recipe.model.family}")
    console.print(
        f"Runtime: Python {recipe.runtime.python}, candidate {recipe.resolution.candidate}"
    )
    _print_ladder(recipe.status)


@app.command()
def plan(
    repo: Annotated[
        str,
        typer.Argument(
            help="Hugging Face repo, or org/model@revision.",
            metavar="REPO",
        ),
    ],
    revision: Annotated[
        str | None,
        typer.Option("--revision", "-r", help="Revision to resolve and pin."),
    ] = None,
    recipes_dir: Annotated[
        Path,
        typer.Option("--recipes-dir", help="Directory for committed recipes."),
    ] = DEFAULT_RECIPES_DIR,
    cache_dir: Annotated[
        Path | None,
        typer.Option("--cache-dir", help="Inspection metadata cache."),
    ] = None,
    python_b: Annotated[
        str,
        typer.Option("--python-b", help="Alternative Python for candidate B."),
    ] = "3.12",
    max_python_bytes: Annotated[
        int,
        typer.Option("--max-python-bytes", min=1, help="Python scan size cap."),
    ] = 256 * 1024,
) -> None:
    """Stages 0-2: inspect, resolve candidate lockfiles, and write a recipe."""
    repo_name, resolved_revision = _split_revision(repo, revision)
    try:
        result = plan_model(
            repo_name,
            revision=resolved_revision,
            recipes_dir=recipes_dir,
            cache_dir=cache_dir,
            python_b=python_b,
            max_python_bytes=max_python_bytes,
        )
    except Exception as exc:
        console.print(f"[red]plan failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    selected = result.resolution.selected
    assert selected is not None
    console.print(
        f"[bold]Planned[/bold] {result.recipe.model.repo} @ "
        f"[cyan]{result.recipe.model.revision}[/cyan]"
    )
    console.print(
        f"Candidate {selected.candidate.key} "
        f"(Python {selected.candidate.python}) -> requirements.lock"
    )
    _print_recipe(result.recipe, result.recipe_dir)


@app.command()
def build(
    repo: Annotated[
        str,
        typer.Argument(
            help="Hugging Face repo, or org/model@revision.",
            metavar="REPO",
        ),
    ],
    revision: Annotated[
        str | None,
        typer.Option("--revision", "-r", help="Pinned recipe revision."),
    ] = None,
    recipes_dir: Annotated[
        Path,
        typer.Option("--recipes-dir", help="Directory for committed recipes."),
    ] = DEFAULT_RECIPES_DIR,
    cache_dir: Annotated[
        Path | None,
        typer.Option("--cache-dir", help="Inspection metadata cache."),
    ] = None,
    envs_dir: Annotated[
        Path | None,
        typer.Option("--envs-dir", help=f"Worker env cache (default: {DEFAULT_ENVS_DIR})."),
    ] = None,
) -> None:
    """Stages 3-4: build the worker env and run P1-P5."""
    repo_name, resolved_revision = _split_revision(repo, revision)
    try:
        result = build_model(
            repo_name,
            revision=resolved_revision,
            recipes_dir=recipes_dir,
            cache_dir=cache_dir,
            envs_dir=envs_dir,
        )
    except Exception as exc:
        console.print(f"[red]build failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[bold]Built[/bold] env {result.env.env_hash[:12]} "
        f"({'reused' if result.env.cached else 'new'}) at {result.env.env_dir}"
    )
    _print_recipe(result.recipe, result.recipe_dir)
    if result.error is not None:
        console.print(f"[red]probe failure:[/red] {result.error}")
        raise typer.Exit(code=1)


@app.command()
def verify(
    repo: Annotated[
        str,
        typer.Argument(
            help="Hugging Face repo, or org/model@revision.",
            metavar="REPO",
        ),
    ],
    audio: Annotated[
        Path,
        typer.Option(
            "--audio",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Smoke-test audio clip.",
        ),
    ],
    revision: Annotated[
        str | None,
        typer.Option("--revision", "-r", help="Pinned recipe revision."),
    ] = None,
    language: Annotated[
        str | None,
        typer.Option("--language", help="Language hint (e.g. am)."),
    ] = None,
    device: Annotated[
        str,
        typer.Option("--device", help="Worker device (cpu, cuda, mps)."),
    ] = "cpu",
    recipes_dir: Annotated[
        Path,
        typer.Option("--recipes-dir", help="Directory for committed recipes."),
    ] = DEFAULT_RECIPES_DIR,
    envs_dir: Annotated[
        Path | None,
        typer.Option("--envs-dir", help=f"Worker env cache (default: {DEFAULT_ENVS_DIR})."),
    ] = None,
    models_dir: Annotated[
        Path | None,
        typer.Option("--models-dir", help=f"Weight cache (default: {DEFAULT_MODELS_DIR})."),
    ] = None,
) -> None:
    """Stages 5-6: download weights, load, and transcribe a smoke clip."""
    repo_name, resolved_revision = _split_revision(repo, revision)
    try:
        result = verify_model(
            repo_name,
            audio_path=audio,
            revision=resolved_revision,
            recipes_dir=recipes_dir,
            envs_dir=envs_dir,
            models_dir=models_dir,
            language=language,
            device=device,
        )
    except Exception as exc:
        console.print(f"[red]verify failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[bold]Verified[/bold] {result.recipe.model.repo} with env {result.env.env_hash[:12]}"
    )
    _print_recipe(result.recipe, result.recipe_dir)
    if result.verification.error is not None:
        console.print(f"[red]verification failure:[/red] {result.verification.error}")
        raise typer.Exit(code=1)
    if result.verification.text:
        console.print(f"Transcript: {result.verification.text}")


def main() -> None:
    """Console-script entry point."""
    app()
