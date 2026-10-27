"""Stage 2 resolver tests: matrix, commands, failures, and a local-wheel run."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from ethirun.resolver import (
    CANDIDATE_DATE_BOUND_DAYS,
    CandidateSpec,
    Resolver,
    ResolverError,
    default_candidates,
    seeds_for_family,
)
from ethirun.static_scan import CTC_FAMILY, CUSTOM_SCRIPT_FAMILY, WHISPER_FAMILY
from ethirun.uv import Uv, UvIndexOptions
from tests.uv_helpers import FakeUv, build_wheel


def test_default_matrix_has_three_candidates_and_120_day_bound() -> None:
    reference = date(2025, 1, 1)

    candidates = default_candidates(reference)

    assert [candidate.key for candidate in candidates] == ["A", "B", "C"]
    assert candidates[0] == CandidateSpec("A", "3.11", date(2025, 5, 1))
    assert candidates[1] == CandidateSpec("B", "3.12", date(2025, 5, 1))
    assert candidates[2] == CandidateSpec("C", "3.11", None)
    excluded = candidates[0].exclude_newer
    assert excluded is not None
    assert (excluded - reference).days == CANDIDATE_DATE_BOUND_DAYS


def test_seeds_use_adapter_hints_and_transformers_version_hint() -> None:
    assert seeds_for_family(
        WHISPER_FAMILY,
        {"transformers_version": "4.40.0"},
    ) == ("transformers>=4.40.0", "torch", "numpy", "soundfile")

    assert seeds_for_family(CTC_FAMILY, {}) == (
        "transformers",
        "torch",
        "numpy",
        "soundfile",
    )

    with pytest.raises(ResolverError, match="not supported"):
        seeds_for_family(CUSTOM_SCRIPT_FAMILY, {})


def test_all_successful_candidates_write_locks_and_select_a(tmp_path: Path) -> None:
    fake = FakeUv()
    index = UvIndexOptions(find_links=tmp_path / "wheels", no_index=True)
    resolver = Resolver(fake, index=index, generate_hashes=True)
    output = tmp_path / "resolution"

    result = resolver.resolve(
        ("demo-package>=1",),
        output_dir=output,
        reference_date=date(2025, 1, 1),
        python_b="3.10",
    )

    assert result.resolved is True
    assert result.selected is not None
    assert result.selected.candidate.key == "A"
    assert result.selected.lockfile == output / "requirements-A.lock"
    assert result.recipe_lockfile == output / "requirements.lock"
    assert result.recipe_lockfile.read_text(encoding="utf-8") == "demo-package==1.0.0\n"
    assert result.requirements_in.read_text(encoding="utf-8") == "demo-package>=1\n"
    assert [attempt.status for attempt in result.attempts] == [
        "resolved",
        "resolved",
        "resolved",
    ]
    assert all((output / f"requirements-{key}.lock").is_file() for key in "ABC")

    assert len(fake.calls) == 3
    for key, call in zip("ABC", fake.calls, strict=True):
        assert "--no-index" in call
        assert "--find-links" in call
        assert "--generate-hashes" in call
        assert call[-1] == "-"
        if key == "C":
            assert "--exclude-newer" not in call
            assert call[call.index("--python-version") + 1] == "3.11"
        else:
            assert call[call.index("--exclude-newer") + 1] == "2025-05-01"
            expected_python = "3.11" if key == "A" else "3.10"
            assert call[call.index("--python-version") + 1] == expected_python


def test_date_bound_failures_are_kept_and_c_is_selected(tmp_path: Path) -> None:
    fake = FakeUv(fail_date_bound=True)
    resolver = Resolver(fake, generate_hashes=False)
    output = tmp_path / "resolution"

    result = resolver.resolve(
        ("demo-package",),
        output_dir=output,
        reference_date=date(2025, 1, 1),
    )

    assert [attempt.status for attempt in result.attempts] == [
        "failed",
        "failed",
        "resolved",
    ]
    assert result.selected is not None
    assert result.selected.candidate.key == "C"
    assert result.recipe_lockfile is not None
    assert result.recipe_lockfile.is_file()
    assert "No solution found" in result.attempts[0].message
    assert not (output / "requirements-A.lock").exists()
    assert not (output / "requirements-B.lock").exists()
    assert all("--generate-hashes" not in call for call in fake.calls)


def test_empty_seed_list_is_rejected(tmp_path: Path) -> None:
    resolver = Resolver(FakeUv())

    with pytest.raises(ResolverError, match="at least one seed"):
        resolver.resolve(
            (),
            output_dir=tmp_path,
            reference_date=date(2025, 1, 1),
        )


def test_real_uv_resolves_local_wheels_without_network(tmp_path: Path) -> None:
    wheels = tmp_path / "wheels"
    build_wheel(wheels, "ethirun-demo-dep", "1.0.0")
    build_wheel(
        wheels,
        "ethirun-demo",
        "1.0.0",
        requires=["ethirun-demo-dep==1.0.0"],
    )

    resolver = Resolver(
        Uv(timeout=120),
        index=UvIndexOptions(find_links=wheels, no_index=True),
        generate_hashes=True,
    )
    output = tmp_path / "resolution"

    result = resolver.resolve(
        ("ethirun-demo",),
        output_dir=output,
        reference_date=datetime.now(UTC).date(),
        python_b="3.11",
    )

    assert result.resolved is True
    assert [attempt.status for attempt in result.attempts] == [
        "resolved",
        "resolved",
        "resolved",
    ]
    assert result.selected is not None
    assert result.selected.candidate.key == "A"
    assert result.recipe_lockfile is not None
    lock_text = result.recipe_lockfile.read_text(encoding="utf-8")
    assert "ethirun-demo==1.0.0" in lock_text
    assert "ethirun-demo-dep==1.0.0" in lock_text
    assert "sha256:" in lock_text
