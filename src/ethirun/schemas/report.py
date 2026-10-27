"""Machine-readable report written by inspection and verification stages."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ethirun.schemas.evidence import Evidence
from ethirun.schemas.inspection import FileRecord, StaticScanResult
from ethirun.schemas.model_ref import ModelRef
from ethirun.schemas.status import StatusLadder


class Report(BaseModel):
    """One report for one pinned model.

    Evidence items are append-only observations. The status ladder carries the
    matching check record for every rung that is not ``untested``.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
    )

    schema_version: Literal[1] = Field(
        default=1,
        alias="schema",
        serialization_alias="schema",
    )
    model: ModelRef
    generated_at: datetime
    evidence: tuple[Evidence, ...] = ()
    inventory: tuple[FileRecord, ...] = ()
    scan: StaticScanResult | None = None
    status: StatusLadder = Field(default_factory=StatusLadder.initial)

    @field_validator("generated_at")
    @classmethod
    def _require_timezone(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("report timestamp must be timezone-aware")
        return value

    @classmethod
    def create(
        cls,
        *,
        model: ModelRef,
        generated_at: datetime,
        evidence: tuple[Evidence, ...] = (),
        inventory: tuple[FileRecord, ...] = (),
        scan: StaticScanResult | None = None,
        status: StatusLadder | None = None,
    ) -> Self:
        """Named constructor that defaults to a fully untested ladder."""
        return cls(
            model=model,
            generated_at=generated_at,
            evidence=evidence,
            inventory=inventory,
            scan=scan,
            status=status if status is not None else StatusLadder.initial(),
        )
