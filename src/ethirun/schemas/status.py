"""Evidence ladder with hard enforcement that a rung has a matching check.

A rung is never a free-floating string. Any state other than ``untested`` must
carry a :class:`CheckRecord` for the same rung, and that record's outcome must
match the state. This makes "we never tried" impossible to confuse with "it
passed" or "it failed".
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    RootModel,
    field_validator,
    model_validator,
)

from ethirun.schemas.enums import Rung, RungState

_UNVERIFIED_STATES = frozenset({RungState.UNTESTED})


class CheckRecord(BaseModel):
    """Proof that a check actually ran.

    ``check`` is a stable identifier such as ``stage4.imports``; ``stage`` says
    where in the pipeline it ran. A check can only report a state that is a
    real observation, never ``untested``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    rung: Rung
    check: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    outcome: RungState
    ran_at: datetime
    detail: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("ran_at")
    @classmethod
    def _require_timezone(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("check timestamp must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _outcome_must_be_observed(self) -> Self:
        if self.outcome is RungState.UNTESTED:
            raise ValueError("a check record cannot have outcome 'untested'")
        return self


class RungResult(BaseModel):
    """The state of one rung and, when observed, the check that proved it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state: RungState
    check: CheckRecord | None = None

    @classmethod
    def untested(cls) -> Self:
        """Return the only valid state for a rung that has not been checked."""
        return cls(state=RungState.UNTESTED, check=None)

    @model_validator(mode="after")
    def _state_requires_matching_check(self) -> Self:
        if self.state in _UNVERIFIED_STATES:
            if self.check is not None:
                raise ValueError(f"rung state '{self.state.value}' cannot carry a check record")
            return self

        if self.check is None:
            raise ValueError(f"rung state '{self.state.value}' requires a check record")

        if self.check.outcome != self.state:
            raise ValueError(
                f"rung state '{self.state.value}' does not match "
                f"check outcome '{self.check.outcome.value}'"
            )
        return self


class StatusLadder(RootModel[dict[Rung, RungResult]]):
    """Complete evidence ladder: every rung is present, with no implicit claims."""

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="before")
    @classmethod
    def _normalize_entries(cls, data: Any) -> Any:
        # Constructing from an existing ladder must not rewrap or lose checks.
        if isinstance(data, StatusLadder):
            return dict(data.root)
        if not isinstance(data, dict):
            raise ValueError("status ladder must map every rung to a result")

        normalized: dict[Any, Any] = {}
        for raw_rung, raw_result in data.items():
            rung = Rung(raw_rung)

            if isinstance(raw_result, RungResult):
                normalized[rung] = raw_result
            elif isinstance(raw_result, str):
                state = RungState(raw_result)
                if state not in _UNVERIFIED_STATES:
                    raise ValueError(
                        f"rung '{rung.value}' cannot be '{state.value}' without a check record"
                    )
                normalized[rung] = RungResult.untested()
            else:
                normalized[rung] = raw_result
        return normalized

    @model_validator(mode="after")
    def _require_complete_ladder(self) -> Self:
        missing = [rung.value for rung in Rung if rung not in self.root]
        if missing:
            raise ValueError(f"status ladder is missing rungs: {', '.join(missing)}")

        for rung, result in self.root.items():
            check = result.check
            if check is not None and check.rung != rung:
                raise ValueError(f"rung '{rung.value}' carries a check for '{check.rung.value}'")
        return self

    @classmethod
    def initial(cls) -> Self:
        """Build a ladder where every rung is explicitly untested."""
        return cls({rung: RungResult.untested() for rung in Rung})

    def states(self) -> dict[Rung, RungState]:
        """Return a rung-to-state summary suitable for recipe.toml."""
        return {rung: result.state for rung, result in self.root.items()}

    def state_of(self, rung: Rung) -> RungState:
        """Return one rung's state."""
        return self.root[rung].state

    def with_check(self, check: CheckRecord) -> Self:
        """Return a new ladder with ``check`` applied to its rung.

        The original ladder is untouched, so callers cannot mutate a recipe's
        evidence after the fact.
        """
        entries: dict[Rung, RungResult] = dict(self.root)
        entries[check.rung] = RungResult(state=check.outcome, check=check)
        return type(self)(entries)
