"""Pinned model identity shared by reports and recipes."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

REPO_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$"
REVISION_PATTERN = r"^[0-9a-f]{40}$"


class ModelRef(BaseModel):
    """A Hugging Face repository pinned to one commit.

    ``main`` is never a valid revision here: it is resolved once by the
    inspector and only the full SHA is recorded.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    repo: str = Field(pattern=REPO_PATTERN)
    revision: str = Field(pattern=REVISION_PATTERN)
    family: str | None = Field(default=None, min_length=1)


class RecipeModel(ModelRef):
    """Model section of an immutable recipe; a family is mandatory."""

    family: str = Field(min_length=1)
