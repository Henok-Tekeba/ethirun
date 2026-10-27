"""Builders for valid schema instances used across tests."""

from __future__ import annotations

from datetime import UTC, date, datetime

from ethirun.schemas import (
    CheckRecord,
    Evidence,
    EvidenceKind,
    ModelRef,
    Recipe,
    RecipeModel,
    Report,
    ResolutionSpec,
    Rung,
    RungState,
    RuntimeSpec,
    StatusLadder,
    VerifiedRecord,
)

SHA = "0123456789abcdef0123456789abcdef01234567"
REPO = "ethiotech/whisper-tiny-am"
FAMILY = "transformers_whisper"
NOW = datetime(2026, 10, 27, 12, 30, tzinfo=UTC)

CHECK_IDS: dict[Rung, str] = {
    Rung.INSPECTED: "stage0.inspect",
    Rung.RESOLVED: "stage2.resolve",
    Rung.INSTALLED: "stage3.install",
    Rung.IMPORTS: "stage4.imports",
    Rung.CONSTRUCTS: "stage4.constructs",
    Rung.WEIGHTS_COMPATIBLE: "stage4.weights",
    Rung.LOADS: "stage5.load",
    Rung.TRANSCRIBES: "stage6.smoke",
    Rung.ACCURACY_CHECKED: "stage7.accuracy",
}

STAGES: dict[Rung, str] = {
    Rung.INSPECTED: "stage0",
    Rung.RESOLVED: "stage2",
    Rung.INSTALLED: "stage3",
    Rung.IMPORTS: "stage4",
    Rung.CONSTRUCTS: "stage4",
    Rung.WEIGHTS_COMPATIBLE: "stage4",
    Rung.LOADS: "stage5",
    Rung.TRANSCRIBES: "stage6",
    Rung.ACCURACY_CHECKED: "stage7",
}


def make_evidence(
    claim: str = "config.json declares WhisperForConditionalGeneration",
    source_file: str = "config.json",
    kind: EvidenceKind = EvidenceKind.REPO,
    confidence: float = 0.95,
) -> Evidence:
    return Evidence(
        claim=claim,
        source_file=source_file,
        kind=kind,
        confidence=confidence,
    )


def make_check(
    rung: Rung,
    outcome: RungState = RungState.PASSED,
    *,
    check_id: str | None = None,
    ran_at: datetime = NOW,
) -> CheckRecord:
    return CheckRecord(
        rung=rung,
        check=check_id or CHECK_IDS[rung],
        stage=STAGES[rung],
        outcome=outcome,
        ran_at=ran_at,
        detail={"rung": rung.value},
    )


def make_ladder() -> StatusLadder:
    """The reference ladder: verified through transcribes, no accuracy claim."""
    ladder = StatusLadder.initial()
    for rung in (
        Rung.INSPECTED,
        Rung.RESOLVED,
        Rung.INSTALLED,
        Rung.IMPORTS,
        Rung.CONSTRUCTS,
        Rung.LOADS,
        Rung.TRANSCRIBES,
    ):
        ladder = ladder.with_check(make_check(rung))
    return ladder.with_check(make_check(Rung.WEIGHTS_COMPATIBLE, RungState.NOT_APPLICABLE))


def make_recipe(
    *,
    status: StatusLadder | None = None,
    repo: str = REPO,
    revision: str = SHA,
    family: str = FAMILY,
) -> Recipe:
    return Recipe.create(
        model=RecipeModel(repo=repo, revision=revision, family=family),
        runtime=RuntimeSpec(python="3.11", torch="cpu"),
        resolution=ResolutionSpec(candidate="A", exclude_newer=date(2025, 3, 1)),
        status=status if status is not None else make_ladder(),
        verified=VerifiedRecord(when=date(2026, 10, 27), hardware="cpu"),
    )


def make_report(*, status: StatusLadder | None = None) -> Report:
    return Report.create(
        model=ModelRef(repo=REPO, revision=SHA, family=FAMILY),
        generated_at=NOW,
        evidence=(
            make_evidence(),
            make_evidence(
                claim="Model card says the model was trained on Amharic",
                source_file="README.md",
                kind=EvidenceKind.EXPLICIT,
                confidence=0.8,
            ),
        ),
        status=status if status is not None else make_ladder(),
    )
