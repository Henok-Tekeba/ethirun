"""TOML/JSON round trips and store behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ethirun.schemas import (
    CheckRecord,
    Recipe,
    Report,
    Rung,
    RungState,
    StatusLadder,
)
from ethirun.store import (
    RECIPE_FILENAME,
    REPORT_FILENAME,
    RecipeStore,
    dumps_recipe_toml,
    dumps_report_json,
    loads_recipe_toml,
    loads_report_json,
    read_recipe,
    write_recipe,
)
from tests.factories import NOW, make_recipe, make_report


def test_recipe_toml_round_trip(sample_recipe: Recipe) -> None:
    text = dumps_recipe_toml(sample_recipe)

    assert "[model]" in text
    assert 'repo = "ethiotech/whisper-tiny-am"' in text
    assert 'candidate = "A"' in text
    assert 'exclude_newer = "2025-03-01"' in text
    assert "[status]" in text
    assert 'transcribes = "passed"' in text
    assert 'weights_compatible = "not_applicable"' in text
    assert "[status_checks.transcribes]" in text

    loaded = loads_recipe_toml(text)

    assert loaded == sample_recipe
    assert loaded.status.state_of(Rung.TRANSCRIBES) is RungState.PASSED
    assert loaded.status.state_of(Rung.WEIGHTS_COMPATIBLE) is RungState.NOT_APPLICABLE
    assert loaded.status.state_of(Rung.ACCURACY_CHECKED) is RungState.UNTESTED


def test_recipe_json_round_trip(sample_recipe: Recipe) -> None:
    payload = sample_recipe.model_dump_json(by_alias=True)

    loaded = Recipe.model_validate_json(payload)

    assert loaded == sample_recipe
    assert json.loads(payload)["schema"] == 1


def test_recipe_toml_drops_nested_none_values() -> None:
    ladder = StatusLadder.initial().with_check(
        CheckRecord(
            rung=Rung.INSPECTED,
            check="stage0.inspect",
            stage="stage0",
            outcome=RungState.PASSED,
            ran_at=NOW,
            detail={
                "kept": "value",
                "dropped": None,
                "nested": {"kept": 1, "dropped": None},
            },
        )
    )
    recipe = make_recipe(status=ladder)

    loaded = loads_recipe_toml(dumps_recipe_toml(recipe))

    check = loaded.status.root[Rung.INSPECTED].check
    assert check is not None
    assert check.detail == {"kept": "value", "nested": {"kept": 1}}


def test_report_json_round_trip(sample_report: Report) -> None:
    payload = dumps_report_json(sample_report)

    loaded = loads_report_json(payload)

    assert loaded == sample_report
    assert json.loads(payload)["schema"] == 1
    assert loaded.status.state_of(Rung.TRANSCRIBES) is RungState.PASSED
    assert loaded.status.root[Rung.TRANSCRIBES].check is not None
    assert len(loaded.evidence) == 2


def test_json_cannot_smuggle_in_a_passed_rung_without_check() -> None:
    payload = json.loads(dumps_report_json(make_report()))
    del payload["status"]["transcribes"]["check"]

    with pytest.raises(ValidationError, match="requires a check record"):
        loads_report_json(json.dumps(payload))


def test_untested_recipe_omits_check_tables() -> None:
    recipe = make_recipe(status=StatusLadder.initial())

    text = dumps_recipe_toml(recipe)

    assert "[status_checks" not in text
    assert 'transcribes = "untested"' in text

    loaded = loads_recipe_toml(text)
    assert loaded == recipe


def test_toml_cannot_smuggle_in_a_passed_rung_without_check() -> None:
    text = dumps_recipe_toml(make_recipe(status=StatusLadder.initial()))
    tampered = text.replace('transcribes = "untested"', 'transcribes = "passed"')

    with pytest.raises(ValidationError, match="requires a check record"):
        loads_recipe_toml(tampered)


def test_toml_rejects_check_record_for_untested_rung(sample_recipe: Recipe) -> None:
    text = dumps_recipe_toml(sample_recipe)
    tampered = text.replace('transcribes = "passed"', 'transcribes = "untested"')

    with pytest.raises(ValueError, match="untested but has a check record"):
        loads_recipe_toml(tampered)


def test_write_and_read_recipe(tmp_path: Path) -> None:
    recipe = make_recipe()
    target = tmp_path / "recipe"

    path = write_recipe(recipe, target)

    assert path == target / RECIPE_FILENAME
    assert read_recipe(path) == recipe


def test_store_saves_recipe_and_report(tmp_path: Path) -> None:
    recipe = make_recipe()
    report = make_report()
    store = RecipeStore(tmp_path / "recipes")

    path = store.save(recipe, report)

    expected_dir = tmp_path / "recipes" / "ethiotech__whisper-tiny-am" / recipe.model.revision
    assert path == expected_dir / RECIPE_FILENAME
    assert (expected_dir / REPORT_FILENAME).is_file()
    assert store.load(recipe.model.repo, recipe.model.revision) == recipe
    assert store.load_report(recipe.model.repo, recipe.model.revision) == report
    assert list(store.recipes()) == [recipe]


def test_store_rejects_report_for_a_different_model(tmp_path: Path) -> None:
    recipe = make_recipe()
    other_recipe = make_recipe(repo="other/model")
    report = make_report()
    report = Report.create(
        model=report.model.model_copy(update={"repo": other_recipe.model.repo}),
        generated_at=report.generated_at,
        evidence=report.evidence,
        status=report.status,
    )
    store = RecipeStore(tmp_path)

    with pytest.raises(ValueError, match="different repositories"):
        store.save(recipe, report)


def test_store_rejects_report_status_mismatch(tmp_path: Path) -> None:
    recipe = make_recipe()
    report = make_report(status=StatusLadder.initial())
    store = RecipeStore(tmp_path)

    with pytest.raises(ValueError, match="disagree about the evidence ladder"):
        store.save(recipe, report)


def test_store_slug_and_directory_layout(tmp_path: Path) -> None:
    store = RecipeStore(tmp_path)

    assert store.slug("org/model") == "org__model"
    assert store.directory_for("org/model", "a" * 40) == tmp_path / "org__model" / ("a" * 40)

    with pytest.raises(ValueError, match="org/model"):
        store.slug("model-without-org")
