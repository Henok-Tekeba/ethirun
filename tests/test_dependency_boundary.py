"""Guard the host/worker dependency boundary.

Model runtimes belong in per-recipe worker virtualenvs. If any of these ever
appear in the host project metadata, process isolation has been compromised.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_HOST_PACKAGES = frozenset(
    {
        "accelerate",
        "ctranslate2",
        "datasets",
        "faster-whisper",
        "librosa",
        "nemo-toolkit",
        "onnxruntime",
        "sentencepiece",
        "soundfile",
        "tensorflow",
        "tokenizers",
        "torch",
        "torchaudio",
        "transformers",
        "whisper",
    }
)

ALLOWED_HOST_PACKAGES = frozenset(
    {
        "huggingface-hub",  # Stage 0 inspection and weight-header metadata
        "pydantic",  # Recipe/report schemas (Milestone 1)
        "rich",  # Human-readable rendered summaries
        "tomli-w",  # Recipe serialization (Milestone 1)
        "typer",  # CLI
    }
)


def _normalize(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _requirement_name(requirement: str) -> str:
    for separator in ("[", "<", ">", "=", "!", "~", ";", " "):
        requirement = requirement.split(separator, maxsplit=1)[0]
    return _normalize(requirement)


def _project_metadata() -> dict[str, object]:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)
    project = data["project"]
    assert isinstance(project, dict)
    return project


def test_host_dependencies_are_allowlisted() -> None:
    project = _project_metadata()
    dependencies = project.get("dependencies", [])
    assert isinstance(dependencies, list)

    unexpected = {_requirement_name(dep) for dep in dependencies} - ALLOWED_HOST_PACKAGES
    assert not unexpected, f"Host gained non-allowlisted dependencies: {sorted(unexpected)}"


def test_no_model_runtime_in_host_dependencies() -> None:
    project = _project_metadata()
    dependencies = project.get("dependencies", [])
    assert isinstance(dependencies, list)

    violations = {_requirement_name(dep) for dep in dependencies} & FORBIDDEN_HOST_PACKAGES
    assert not violations, f"Model runtimes leaked into the host environment: {sorted(violations)}"


def test_worker_runtime_is_never_a_dependency_group() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)

    groups = data.get("dependency-groups", {})
    assert isinstance(groups, dict)
    for group in groups.values():
        assert isinstance(group, list)
        violations = {_requirement_name(dep) for dep in group} & FORBIDDEN_HOST_PACKAGES
        assert not violations, f"Model runtime found in dev dependency group: {sorted(violations)}"
