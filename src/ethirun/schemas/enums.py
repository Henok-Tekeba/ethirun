"""Enumerations shared by the recipe and report schemas."""

from __future__ import annotations

from enum import StrEnum


class Rung(StrEnum):
    """Evidence-ladder rungs, in the order they must be proven."""

    INSPECTED = "inspected"
    RESOLVED = "resolved"
    INSTALLED = "installed"
    IMPORTS = "imports"
    CONSTRUCTS = "constructs"
    WEIGHTS_COMPATIBLE = "weights_compatible"
    LOADS = "loads"
    TRANSCRIBES = "transcribes"
    ACCURACY_CHECKED = "accuracy_checked"


class RungState(StrEnum):
    """State of one evidence rung."""

    PASSED = "passed"
    FAILED = "failed"
    UNTESTED = "untested"
    NOT_APPLICABLE = "not_applicable"


class EvidenceKind(StrEnum):
    """Provenance class for a single evidence item."""

    EXPLICIT = "explicit"
    REPO = "repo"
    INFERRED = "inferred"
    OBSERVED = "observed"
