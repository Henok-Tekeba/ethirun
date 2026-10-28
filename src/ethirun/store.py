"""Read and write recipes and reports.

Recipes use TOML for humans and git review; reports use JSON for machines. Both
round-trip through the Pydantic schemas, so a hand-edited recipe cannot smuggle
in a verified rung without its check record.
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import tomli_w

from ethirun.schemas import (
    Recipe,
    Report,
    Rung,
    RungResult,
    RungState,
)

RECIPE_FILENAME = "recipe.toml"
REPORT_FILENAME = "report.json"


def recipe_to_toml_data(recipe: Recipe) -> dict[str, Any]:
    """Flatten a recipe into the documented TOML shape.

    ``[status]`` keeps the simple string form. Check records live alongside in
    ``[status_checks.<rung>]``, so every non-``untested`` status remains backed
    by code that ran.
    """
    data: dict[str, Any] = recipe.model_dump(
        mode="json",
        by_alias=True,
        exclude_none=True,
    )

    status: dict[str, str] = {}
    checks: dict[str, dict[str, Any]] = {}
    for rung, result in recipe.status.root.items():
        status[rung.value] = result.state.value
        if result.check is not None:
            checks[rung.value] = result.check.model_dump(
                mode="json",
                by_alias=True,
                exclude_none=True,
            )

    data["status"] = status
    if checks:
        data["status_checks"] = checks
    return data


def dumps_recipe_toml(recipe: Recipe) -> str:
    """Serialize a recipe to TOML text."""
    return tomli_w.dumps(_without_none(recipe_to_toml_data(recipe)))


def _without_none(value: Any) -> Any:
    """TOML has no null; recursively drop None values before writing."""
    if isinstance(value, dict):
        return {
            str(key): cleaned
            for key, item in value.items()
            if item is not None and (cleaned := _without_none(item)) is not None
        }
    if isinstance(value, list):
        return [_without_none(item) for item in value if item is not None]
    return value


def loads_recipe_toml(text: str) -> Recipe:
    """Parse and validate TOML recipe text."""
    data = tomllib.loads(text)

    status_data = data.pop("status", None)
    checks_data = data.pop("status_checks", {})
    if not isinstance(status_data, dict):
        raise ValueError("recipe is missing a [status] table")
    if not isinstance(checks_data, dict):
        raise ValueError("[status_checks] must be a table of check records")

    unknown_checks = set(checks_data) - set(status_data)
    if unknown_checks:
        raise ValueError(f"check records for unlisted rungs: {sorted(unknown_checks)}")

    entries: dict[Rung, Any] = {}
    for rung_name, state_name in status_data.items():
        rung = Rung(rung_name)
        state = RungState(state_name)
        check_data = checks_data.get(rung_name)

        if state is RungState.UNTESTED:
            if check_data is not None:
                raise ValueError(f"rung '{rung.value}' is untested but has a check record")
            entries[rung] = RungResult.untested()
        elif check_data is None:
            # Deliberately route through RungResult so the error names the
            # missing-check invariant rather than inventing a second message.
            entries[rung] = {"state": state.value}
        else:
            entries[rung] = {"state": state.value, "check": check_data}

    data["status"] = entries
    return Recipe.model_validate(data)


def write_recipe(recipe: Recipe, directory: Path) -> Path:
    """Write ``recipe.toml`` inside ``directory`` and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / RECIPE_FILENAME
    path.write_text(dumps_recipe_toml(recipe), encoding="utf-8")
    return path


def read_recipe(path: Path) -> Recipe:
    """Read a recipe from a TOML file."""
    return loads_recipe_toml(path.read_text(encoding="utf-8"))


def dumps_report_json(report: Report) -> str:
    """Serialize a report to stable, indented JSON.

    ``None`` values are kept on purpose: report JSON supports null, and exact
    round trips matter for evidence review.
    """
    return report.model_dump_json(by_alias=True, indent=2) + "\n"


def loads_report_json(text: str) -> Report:
    """Parse and validate report JSON text."""
    return Report.model_validate_json(text)


def write_report(report: Report, directory: Path) -> Path:
    """Write ``report.json`` inside ``directory`` and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / REPORT_FILENAME
    path.write_text(dumps_report_json(report), encoding="utf-8")
    return path


def read_report(path: Path) -> Report:
    """Read a report from a JSON file."""
    return loads_report_json(path.read_text(encoding="utf-8"))


class RecipeStore:
    """Filesystem store rooted at a recipes directory.

    Layout: ``<root>/<org>__<model>/<sha>/{recipe.toml,report.json}``.
    Recipes are addressed by SHA, never by a moving branch name.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    @staticmethod
    def slug(repo: str) -> str:
        """Turn ``org/model`` into the on-disk directory name."""
        if "/" not in repo:
            raise ValueError(f"model repo must be 'org/model', got {repo!r}")
        return repo.replace("/", "__")

    def directory_for(self, repo: str, revision: str) -> Path:
        """Return the recipe directory for one pinned revision."""
        return self.root / self.slug(repo) / revision

    def save(self, recipe: Recipe, report: Report | None = None) -> Path:
        """Write a recipe and, when supplied, its report."""
        if report is not None:
            if report.model.repo != recipe.model.repo:
                raise ValueError("report and recipe refer to different repositories")
            if report.model.revision != recipe.model.revision:
                raise ValueError("report and recipe refer to different revisions")
            if report.model.family != recipe.model.family:
                raise ValueError("report and recipe refer to different families")
            if report.status != recipe.status:
                raise ValueError("report and recipe disagree about the evidence ladder")
            write_report(report, self.directory_for(recipe.model.repo, recipe.model.revision))

        return write_recipe(
            recipe,
            self.directory_for(recipe.model.repo, recipe.model.revision),
        )

    def load(self, repo: str, revision: str) -> Recipe:
        """Load a recipe by repository and pinned SHA."""
        return read_recipe(self.directory_for(repo, revision) / RECIPE_FILENAME)

    def load_report(self, repo: str, revision: str) -> Report:
        """Load the report by repository and pinned SHA."""
        return read_report(self.directory_for(repo, revision) / REPORT_FILENAME)

    def recipes(self) -> Iterator[Recipe]:
        """Yield every stored recipe in deterministic path order."""
        for path in sorted(self.root.glob(f"*/*/{RECIPE_FILENAME}")):
            yield read_recipe(path)
