"""Immutable recipe schema, matching the documented ``recipe.toml`` layout."""

from __future__ import annotations

from datetime import date
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field

from ethirun.schemas.model_ref import RecipeModel
from ethirun.schemas.status import StatusLadder


class RuntimeSpec(BaseModel):
    """Worker runtime pin. Model libraries live only in this environment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    python: str = Field(pattern=r"^3\.(10|11|12)$")
    lockfile: str = Field(default="requirements.lock", pattern=r"^[A-Za-z0-9._-]+$")
    torch: Literal["cpu", "cuda"] = "cpu"
    trust_remote_code: bool = False
    adapter_version: str = Field(default="1", min_length=1)


class ResolutionSpec(BaseModel):
    """Which resolver candidate produced the lockfile, and its date bound."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate: Literal["A", "B", "C"]
    exclude_newer: date | None = None


class VerifiedRecord(BaseModel):
    """How and when the recipe was last verified end to end."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    when: date
    hardware: str = Field(default="cpu", min_length=1)


class Recipe(BaseModel):
    """An immutable, SHA-pinned recipe.

    The serialized ``schema`` key is exposed as ``schema_version`` in Python to
    avoid colliding with Pydantic internals.
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
    model: RecipeModel
    runtime: RuntimeSpec
    resolution: ResolutionSpec
    status: StatusLadder
    verified: VerifiedRecord | None = None

    @classmethod
    def create(
        cls,
        *,
        model: RecipeModel,
        runtime: RuntimeSpec,
        resolution: ResolutionSpec,
        status: StatusLadder,
        verified: VerifiedRecord | None = None,
    ) -> Self:
        """Named constructor that keeps the schema version explicit."""
        return cls(
            model=model,
            runtime=runtime,
            resolution=resolution,
            status=status,
            verified=verified,
        )
