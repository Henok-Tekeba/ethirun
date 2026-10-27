"""The strict rule: no rung may claim an observation without its check."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from ethirun.schemas import (
    CheckRecord,
    Rung,
    RungResult,
    RungState,
    StatusLadder,
)
from tests.factories import make_check


def _plain_untested() -> dict[str, str]:
    return {rung.value: RungState.UNTESTED.value for rung in Rung}


@pytest.mark.parametrize("rung", list(Rung))
@pytest.mark.parametrize(
    "state",
    [RungState.PASSED, RungState.FAILED, RungState.NOT_APPLICABLE],
)
def test_observed_rung_without_check_is_rejected(rung: Rung, state: RungState) -> None:
    with pytest.raises(ValidationError, match="requires a check record"):
        RungResult(state=state)

    entries = _plain_untested()
    entries[rung.value] = state.value
    with pytest.raises(ValidationError, match="without a check record"):
        StatusLadder.model_validate(entries)


@pytest.mark.parametrize("rung", list(Rung))
def test_state_dict_without_check_is_rejected(rung: Rung) -> None:
    entries: dict[str, object] = {item.value: {"state": RungState.UNTESTED.value} for item in Rung}
    entries[rung.value] = {"state": RungState.PASSED.value}

    with pytest.raises(ValidationError, match="requires a check record"):
        StatusLadder.model_validate(entries)


def test_untested_rung_cannot_carry_check() -> None:
    check = make_check(Rung.INSPECTED)

    with pytest.raises(ValidationError, match="cannot carry a check record"):
        RungResult(state=RungState.UNTESTED, check=check)


def test_check_outcome_must_match_rung_state() -> None:
    failed_check = make_check(Rung.TRANSCRIBES, RungState.FAILED)

    with pytest.raises(ValidationError, match="does not match check outcome"):
        RungResult(state=RungState.PASSED, check=failed_check)


def test_check_must_belong_to_the_rung_it_proves() -> None:
    entries = dict(StatusLadder.initial().root)
    entries[Rung.TRANSCRIBES] = RungResult(
        state=RungState.PASSED,
        check=make_check(Rung.INSPECTED),
    )

    with pytest.raises(ValidationError, match="carries a check for 'inspected'"):
        StatusLadder(entries)


def test_ladder_requires_every_rung() -> None:
    entries = _plain_untested()
    del entries[Rung.ACCURACY_CHECKED.value]

    with pytest.raises(ValidationError, match="missing rungs: accuracy_checked"):
        StatusLadder.model_validate(entries)


def test_check_outcome_cannot_be_untested() -> None:
    with pytest.raises(ValidationError, match="cannot have outcome 'untested'"):
        CheckRecord(
            rung=Rung.INSPECTED,
            check="stage0.inspect",
            stage="stage0",
            outcome=RungState.UNTESTED,
            ran_at=datetime(2026, 10, 27, 12, 0, tzinfo=UTC),
        )


def test_check_timestamp_must_be_timezone_aware() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        CheckRecord(
            rung=Rung.INSPECTED,
            check="stage0.inspect",
            stage="stage0",
            outcome=RungState.PASSED,
            ran_at=datetime.fromisoformat("2026-10-27T12:00:00"),
        )


def test_recording_a_check_does_not_mutate_the_original_ladder() -> None:
    ladder = StatusLadder.initial()
    updated = ladder.with_check(make_check(Rung.INSPECTED))

    assert ladder.state_of(Rung.INSPECTED) is RungState.UNTESTED
    assert updated.state_of(Rung.INSPECTED) is RungState.PASSED
    assert updated.root[Rung.INSPECTED].check is not None


def test_initial_ladder_marks_every_rung_untested() -> None:
    ladder = StatusLadder.initial()

    assert set(ladder.states()) == set(Rung)
    assert all(state is RungState.UNTESTED for state in ladder.states().values())
