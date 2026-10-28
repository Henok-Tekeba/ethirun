"""End-to-end workflow: plan, build, and verify a real model recipe."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal, cast

from ethirun.client import WorkerClient, WorkerConfig, default_worker_dir
from ethirun.envs import EnvBuildError, EnvManager, EnvResult
from ethirun.inspector import DEFAULT_CACHE_DIR, Inspector
from ethirun.models import DEFAULT_MODELS_DIR, download_model
from ethirun.resolver import ResolutionResult, Resolver, seeds_for_family
from ethirun.schemas import (
    CheckRecord,
    Evidence,
    EvidenceKind,
    Recipe,
    RecipeModel,
    Report,
    ResolutionSpec,
    Rung,
    RungState,
    RuntimeSpec,
    VerifiedRecord,
)
from ethirun.store import (
    RecipeStore,
    read_recipe,
    read_report,
    write_recipe,
    write_report,
)
from ethirun.verifier import ProbeResult, VerificationResult, run_probes, verify_worker

DEFAULT_RECIPES_DIR = Path("recipes")
CandidateKey = Literal["A", "B", "C"]


class WorkflowError(RuntimeError):
    """The requested stage could not be completed."""


@dataclass(frozen=True, slots=True)
class PlanResult:
    recipe: Recipe
    report: Report
    recipe_dir: Path
    resolution: ResolutionResult


@dataclass(frozen=True, slots=True)
class BuildResult:
    recipe: Recipe
    report: Report
    recipe_dir: Path
    env: EnvResult
    probes: dict[str, Any]
    error: str | None


@dataclass(frozen=True, slots=True)
class VerifyResult:
    recipe: Recipe
    report: Report
    recipe_dir: Path
    env: EnvResult
    verification: VerificationResult


@dataclass(frozen=True, slots=True)
class LoadedRecipe:
    recipe: Recipe
    report: Report
    directory: Path


def _save(recipe: Recipe, report: Report, directory: Path) -> tuple[Recipe, Report]:
    report = report.model_copy(update={"status": recipe.status})
    write_recipe(recipe, directory)
    write_report(report, directory)
    return recipe, report


def _candidate_specs(
    report: Report,
    recipe: Recipe,
) -> list[tuple[CandidateKey, str, date | None]]:
    """Return (key, python, exclude_newer) in probe-fallback order."""
    specs: list[tuple[CandidateKey, str, date | None]] = []
    check = report.status.root[Rung.RESOLVED].check
    attempts = check.detail.get("attempts") if check is not None else None
    if isinstance(attempts, list):
        for attempt in attempts:
            if not isinstance(attempt, dict) or attempt.get("status") != "resolved":
                continue
            key = attempt.get("candidate")
            python = attempt.get("python")
            if key not in {"A", "B", "C"} or not isinstance(python, str):
                continue
            raw_exclude = attempt.get("exclude_newer")
            exclude = (
                date.fromisoformat(raw_exclude)
                if isinstance(raw_exclude, str) and raw_exclude
                else None
            )
            specs.append((cast(CandidateKey, key), python, exclude))

    if not specs:
        specs.append(
            (
                recipe.resolution.candidate,
                recipe.runtime.python,
                recipe.resolution.exclude_newer,
            )
        )

    selected = recipe.resolution.candidate
    return sorted(specs, key=lambda item: (item[0] != selected, "ABC".find(item[0])))


def _installed_check(env: EnvResult, now: datetime) -> CheckRecord:
    return CheckRecord(
        rung=Rung.INSTALLED,
        check="stage3.install",
        stage="stage3",
        outcome=RungState.PASSED,
        ran_at=now,
        detail={
            "env_hash": env.env_hash,
            "env_dir": str(env.env_dir),
            "python": str(env.python),
            "lockfile_sha256": env.lockfile_sha256,
            "cached": env.cached,
        },
    )


def find_recipe(
    repo: str,
    *,
    recipes_dir: Path = DEFAULT_RECIPES_DIR,
    revision: str | None = None,
) -> LoadedRecipe:
    """Load a recipe by revision, or the most recently touched recipe for a repo."""
    base = recipes_dir / RecipeStore.slug(repo)
    if revision is not None:
        directory = base / revision
        if not (directory / "recipe.toml").is_file():
            raise WorkflowError(f"no recipe for {repo}@{revision}; run plan first")
    else:
        directories = sorted(
            (path for path in base.glob("*") if (path / "recipe.toml").is_file()),
            key=lambda path: path.stat().st_mtime,
        )
        if not directories:
            raise WorkflowError(f"no recipes for {repo}; run plan first")
        directory = directories[-1]

    return LoadedRecipe(
        recipe=read_recipe(directory / "recipe.toml"),
        report=read_report(directory / "report.json"),
        directory=directory,
    )


def plan_model(
    repo: str,
    *,
    revision: str | None = None,
    recipes_dir: Path = DEFAULT_RECIPES_DIR,
    cache_dir: Path | None = None,
    python_b: str = "3.12",
    max_python_bytes: int = 256 * 1024,
    resolver: Resolver | None = None,
    now: datetime | None = None,
) -> PlanResult:
    """Inspect, resolve, and write an immutable recipe plus candidate locks."""
    spec_now = now or datetime.now(UTC)
    report = Inspector(
        cache_dir=cache_dir,
        max_python_bytes=max_python_bytes,
    ).inspect(repo, revision=revision)
    scan = report.scan
    if scan is None or not scan.adapter.supported or scan.adapter.family is None:
        raise WorkflowError(
            f"{repo} has no supported adapter (family={getattr(scan, 'adapter', None)})"
        )

    reference_date = report.commit_date or report.generated_at.date()
    seeds = seeds_for_family(
        scan.adapter.family,
        {"transformers_version": scan.transformers_version},
    )
    recipe_dir = RecipeStore(recipes_dir).directory_for(repo, report.model.revision)
    candidates_dir = recipe_dir / "candidates"
    if candidates_dir.exists():
        shutil.rmtree(candidates_dir)

    resolution = (resolver or Resolver()).resolve(
        seeds,
        output_dir=candidates_dir,
        reference_date=reference_date,
        python_b=python_b,
    )
    selected = resolution.selected
    if selected is None or selected.lockfile is None:
        messages = "; ".join(
            f"{attempt.candidate.key}: {attempt.message}" for attempt in resolution.attempts
        )
        raise WorkflowError(f"all resolver candidates failed: {messages}")

    recipe_dir.mkdir(parents=True, exist_ok=True)
    lockfile = recipe_dir / "requirements.lock"
    shutil.copyfile(selected.lockfile, lockfile)
    shutil.copyfile(resolution.requirements_in, recipe_dir / "requirements.in")

    attempts_detail = []
    for attempt in resolution.attempts:
        attempt_detail: dict[str, Any] = {
            "candidate": attempt.candidate.key,
            "python": attempt.candidate.python,
            "status": attempt.status,
            "message": attempt.message[:1000],
        }
        if attempt.candidate.exclude_newer is not None:
            attempt_detail["exclude_newer"] = attempt.candidate.exclude_newer.isoformat()
        attempts_detail.append(attempt_detail)

    resolution_detail: dict[str, Any] = {
        "candidate": selected.candidate.key,
        "python": selected.candidate.python,
        "attempts": attempts_detail,
    }
    if selected.candidate.exclude_newer is not None:
        resolution_detail["exclude_newer"] = selected.candidate.exclude_newer.isoformat()
    status = report.status.with_check(
        CheckRecord(
            rung=Rung.RESOLVED,
            check="stage2.resolve",
            stage="stage2",
            outcome=RungState.PASSED,
            ran_at=spec_now,
            detail=resolution_detail,
        )
    )

    resolution_evidence = tuple(
        Evidence(
            claim=(
                f"resolver candidate {attempt.candidate.key} "
                f"(Python {attempt.candidate.python}) {attempt.status}"
            ),
            source_file="uv pip compile",
            kind=EvidenceKind.OBSERVED,
            confidence=1.0,
        )
        for attempt in resolution.attempts
    )
    report = report.model_copy(
        update={"status": status, "evidence": report.evidence + resolution_evidence}
    )

    recipe = Recipe.create(
        model=RecipeModel(
            repo=report.model.repo,
            revision=report.model.revision,
            family=scan.adapter.family,
        ),
        runtime=RuntimeSpec(
            python=selected.candidate.python,
            torch="cpu",
            trust_remote_code=False,
            adapter_version="1",
        ),
        resolution=ResolutionSpec(
            candidate=selected.candidate.key,
            exclude_newer=selected.candidate.exclude_newer,
        ),
        status=status,
    )
    recipe, report = _save(recipe, report, recipe_dir)
    return PlanResult(recipe, report, recipe_dir, resolution)


def build_model(
    repo: str,
    *,
    revision: str | None = None,
    recipes_dir: Path = DEFAULT_RECIPES_DIR,
    cache_dir: Path | None = None,
    envs_dir: Path | None = None,
    worker_dir: Path | None = None,
    now: datetime | None = None,
) -> BuildResult:
    """Build the recipe env, run P1-P5, and update the recipe/report."""
    stage_now = now or datetime.now(UTC)
    loaded = find_recipe(repo, recipes_dir=recipes_dir, revision=revision)
    recipe = loaded.recipe
    report = loaded.report

    cache_root = cache_dir or DEFAULT_CACHE_DIR
    model_dir = cache_root / RecipeStore.slug(recipe.model.repo) / recipe.model.revision
    if not (model_dir / "config.json").is_file():
        raise WorkflowError(f"inspection cache missing at {model_dir}; re-run plan for {repo}")

    candidate_specs = _candidate_specs(report, recipe)
    attempts: list[Evidence] = []
    failures: list[str] = []
    env: EnvResult | None = None
    probe_result: ProbeResult | None = None
    selected_key: CandidateKey = recipe.resolution.candidate
    selected_exclude = recipe.resolution.exclude_newer
    selected_python = recipe.runtime.python

    for key, python, exclude_newer in candidate_specs:
        lockfile = loaded.directory / "candidates" / f"requirements-{key}.lock"
        if not lockfile.is_file():
            continue
        try:
            candidate_env = EnvManager(root=envs_dir).build(lockfile, python)
        except EnvBuildError as exc:
            failures.append(f"{key}: {exc}")
            attempts.append(
                Evidence(
                    claim=f"candidate {key} env failed: {exc.message[:300]}",
                    source_file="uv",
                    kind=EvidenceKind.OBSERVED,
                    confidence=1.0,
                )
            )
            continue

        installed_status = recipe.status.with_check(_installed_check(candidate_env, stage_now))
        try:
            with WorkerClient(
                WorkerConfig(
                    python=candidate_env.python,
                    worker_dir=worker_dir or default_worker_dir(),
                    adapter=recipe.model.family or "fake",
                    startup_timeout=120.0,
                    default_timeout=900.0,
                )
            ) as client:
                candidate_probes = run_probes(
                    client,
                    model_dir=str(model_dir),
                    repo_id=recipe.model.repo,
                    revision=recipe.model.revision,
                    now=stage_now,
                    probe_timeout=600.0,
                    initial_status=installed_status,
                )
        except Exception as exc:
            failures.append(f"{key}: {type(exc).__name__}: {exc}")
            attempts.append(
                Evidence(
                    claim=f"candidate {key} worker failed: {exc}",
                    source_file="worker",
                    kind=EvidenceKind.OBSERVED,
                    confidence=1.0,
                )
            )
            continue

        if candidate_probes.error is not None:
            failures.append(f"{key}: {candidate_probes.error}")
            attempts.append(
                Evidence(
                    claim=f"candidate {key} probes failed: {candidate_probes.error[:300]}",
                    source_file="worker",
                    kind=EvidenceKind.OBSERVED,
                    confidence=1.0,
                )
            )
            continue

        env = candidate_env
        probe_result = candidate_probes
        selected_key = key
        selected_exclude = exclude_newer
        selected_python = python
        break

    if env is None or probe_result is None:
        raise WorkflowError("all resolved candidates failed: " + "; ".join(failures))

    shutil.copyfile(
        loaded.directory / "candidates" / f"requirements-{selected_key}.lock",
        loaded.directory / "requirements.lock",
    )
    runtime = RuntimeSpec(
        python=selected_python,
        lockfile=recipe.runtime.lockfile,
        torch=recipe.runtime.torch,
        trust_remote_code=recipe.runtime.trust_remote_code,
        adapter_version=recipe.runtime.adapter_version,
    )
    resolution = ResolutionSpec(
        candidate=selected_key,
        exclude_newer=selected_exclude,
    )
    status = probe_result.status
    resolved_check = status.root[Rung.RESOLVED].check
    if resolved_check is not None:
        detail = dict(resolved_check.detail)
        detail.update(
            {
                "candidate": selected_key,
                "python": selected_python,
                "selected_at_build": True,
            }
        )
        if selected_exclude is not None:
            detail["exclude_newer"] = selected_exclude.isoformat()
        else:
            detail.pop("exclude_newer", None)
        status = status.with_check(
            CheckRecord(
                rung=Rung.RESOLVED,
                check="stage2.resolve",
                stage="stage2",
                outcome=RungState.PASSED,
                ran_at=resolved_check.ran_at,
                detail=detail,
            )
        )

    env_evidence = Evidence(
        claim=(
            f"worker env {env.env_hash[:12]} "
            f"{'reused' if env.cached else 'built'} for Python {selected_python} "
            f"(candidate {selected_key})"
        ),
        source_file="uv",
        kind=EvidenceKind.OBSERVED,
        confidence=1.0,
    )
    updated_recipe = recipe.model_copy(
        update={
            "status": status,
            "runtime": runtime,
            "resolution": resolution,
        }
    )
    updated_report = report.model_copy(
        update={
            "status": status,
            "evidence": (*report.evidence, *attempts, env_evidence),
        }
    )
    updated_recipe, updated_report = _save(
        updated_recipe,
        updated_report,
        loaded.directory,
    )
    return BuildResult(
        recipe=updated_recipe,
        report=updated_report,
        recipe_dir=loaded.directory,
        env=env,
        probes=probe_result.probes,
        error=probe_result.error,
    )


def verify_model(
    repo: str,
    *,
    audio_path: Path,
    revision: str | None = None,
    recipes_dir: Path = DEFAULT_RECIPES_DIR,
    envs_dir: Path | None = None,
    models_dir: Path | None = None,
    worker_dir: Path | None = None,
    language: str | None = None,
    device: str = "cpu",
    now: datetime | None = None,
) -> VerifyResult:
    """Download weights, load, transcribe, and persist the final recipe/report."""
    stage_now = now or datetime.now(UTC)
    audio = audio_path.expanduser().resolve()
    if not audio.is_file():
        raise WorkflowError(f"audio file not found: {audio}")

    loaded = find_recipe(repo, recipes_dir=recipes_dir, revision=revision)
    recipe = loaded.recipe
    report = loaded.report

    weights_dir = download_model(
        recipe.model.repo,
        recipe.model.revision,
        root=models_dir or DEFAULT_MODELS_DIR,
        prefer_safetensors=any(record.path.endswith(".safetensors") for record in report.inventory),
    )
    env = EnvManager(root=envs_dir).build(
        loaded.directory / "requirements.lock",
        recipe.runtime.python,
    )
    status = recipe.status.with_check(_installed_check(env, stage_now))

    with WorkerClient(
        WorkerConfig(
            python=env.python,
            worker_dir=worker_dir or default_worker_dir(),
            adapter=recipe.model.family or "fake",
            startup_timeout=120.0,
            default_timeout=1800.0,
        )
    ) as client:
        verification = verify_worker(
            client,
            model_dir=str(weights_dir),
            audio_path=str(audio),
            repo_id=recipe.model.repo,
            revision=recipe.model.revision,
            language=language,
            device=device,
            now=stage_now,
            probe_timeout=600.0,
            transcribe_timeout=1800.0,
            initial_status=status,
        )

    verified = VerifiedRecord(when=stage_now.date(), hardware=device)
    updated_recipe = recipe.model_copy(update={"status": verification.status, "verified": verified})
    audio_digest = hashlib.sha256(audio.read_bytes()).hexdigest()
    audio_evidence = Evidence(
        claim=(
            f"smoke clip {audio.name} sha256={audio_digest}"
            f"{f' language={language}' if language else ''}"
        ),
        source_file="verify",
        kind=EvidenceKind.OBSERVED,
        confidence=1.0,
    )
    updated_report = report.model_copy(
        update={
            "status": verification.status,
            "evidence": (*report.evidence, audio_evidence),
        }
    )
    updated_recipe, updated_report = _save(
        updated_recipe,
        updated_report,
        loaded.directory,
    )
    return VerifyResult(
        recipe=updated_recipe,
        report=updated_report,
        recipe_dir=loaded.directory,
        env=env,
        verification=verification,
    )
