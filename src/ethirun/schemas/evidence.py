"""Evidence items recorded during inspection and verification."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ethirun.schemas.enums import EvidenceKind


class Evidence(BaseModel):
    """One claim about a model, and where it came from.

    ``kind`` keeps the strength of the claim explicit: model-card prose is not
    the same evidence as a config file, an inference, or a check that ran.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    kind: EvidenceKind
    confidence: float = Field(ge=0.0, le=1.0)
