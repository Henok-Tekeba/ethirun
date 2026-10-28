"""Workflow helper tests that do not touch the network or install anything."""

from __future__ import annotations

from ethirun.models import ALLOW_PATTERNS, SAFETENSORS_PATTERNS
from ethirun.schemas import (
    CheckRecord,
    Report,
    ResolutionSpec,
    Rung,
    RungState,
    StatusLadder,
)
from ethirun.workflow import _candidate_specs
from tests.factories import NOW, make_recipe, make_report


def _report_with_candidates() -> Report:
    ladder = StatusLadder.initial().with_check(
        CheckRecord(
            rung=Rung.RESOLVED,
            check="stage2.resolve",
            stage="stage2",
            outcome=RungState.PASSED,
            ran_at=NOW,
            detail={
                "candidate": "A",
                "attempts": [
                    {"candidate": "A", "python": "3.11", "status": "resolved"},
                    {"candidate": "B", "python": "3.12", "status": "failed"},
                    {"candidate": "C", "python": "3.11", "status": "resolved"},
                ],
            },
        )
    )
    return make_report(status=ladder)


def test_candidate_fallback_prefers_current_recipe_selection() -> None:
    report = _report_with_candidates()
    recipe_a = make_recipe()
    recipe_c = recipe_a.model_copy(update={"resolution": ResolutionSpec(candidate="C")})

    assert [item[0] for item in _candidate_specs(report, recipe_a)] == ["A", "C"]
    assert [item[0] for item in _candidate_specs(report, recipe_c)] == ["C", "A"]


def test_safetensors_preference_does_not_request_bin_duplicates() -> None:
    assert "*.safetensors" in SAFETENSORS_PATTERNS
    assert "*.bin" not in SAFETENSORS_PATTERNS
    assert "*.bin" in ALLOW_PATTERNS
