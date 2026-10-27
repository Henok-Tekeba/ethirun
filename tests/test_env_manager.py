"""Stage 3 env manager tests with real uv and local wheels only."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ethirun.envs import ENV_MARKER, EnvBuildError, EnvManager
from ethirun.resolver import Resolver
from ethirun.uv import Uv, UvIndexOptions
from tests.uv_helpers import SpyUv, build_wheel


def _host_distributions() -> tuple[tuple[str, str], ...]:
    distributions = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]
        if name:
            distributions.append((name.lower(), distribution.version or ""))
    return tuple(sorted(distributions))


@pytest.fixture
def wheelhouse(tmp_path: Path) -> Path:
    wheels = tmp_path / "wheels"
    build_wheel(wheels, "ethirun-demo-dep", "1.0.0")
    build_wheel(
        wheels,
        "ethirun-demo",
        "1.0.0",
        requires=["ethirun-demo-dep==1.0.0"],
    )
    return wheels


@pytest.fixture
def index_options(wheelhouse: Path) -> UvIndexOptions:
    return UvIndexOptions(find_links=wheelhouse, no_index=True)


@pytest.fixture
def lockfile(tmp_path: Path, index_options: UvIndexOptions) -> Path:
    resolver = Resolver(
        Uv(timeout=120),
        index=index_options,
        generate_hashes=True,
    )
    result = resolver.resolve(
        ("ethirun-demo",),
        output_dir=tmp_path / "resolution",
        reference_date=datetime.now(UTC).date(),
        python_b="3.11",
    )
    assert result.recipe_lockfile is not None
    return result.recipe_lockfile


def test_build_creates_isolated_env_and_runs_check(
    tmp_path: Path,
    lockfile: Path,
    index_options: UvIndexOptions,
) -> None:
    spy = SpyUv(Uv(timeout=120))
    manager = EnvManager(
        root=tmp_path / "envs",
        uv=spy,
        index=index_options,
        worker_version="worker-test",
    )
    before = _host_distributions()

    result = manager.build(lockfile, "3.11")

    after = _host_distributions()
    assert result.cached is False
    assert result.env_dir.is_dir()
    assert result.python.is_file()
    assert result.env_dir.name == result.env_hash
    assert result.python != Path(sys.executable)

    probe = subprocess.run(
        [
            str(result.python),
            "-c",
            "import json, sys; print(json.dumps({'prefix': sys.prefix, "
            "'purelib': __import__('sysconfig').get_path('purelib')}))",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr
    env_paths = json.loads(probe.stdout)
    assert env_paths["prefix"] == str(result.env_dir)
    assert str(result.env_dir) in env_paths["purelib"]

    assert spy.calls[0][:2] == ("venv", str(result.env_dir))
    assert any(call[:2] == ("pip", "sync") for call in spy.calls)
    assert any(call[:2] == ("pip", "check") for call in spy.calls)

    completed = subprocess.run(
        [
            str(result.python),
            "-c",
            "import ethirun_demo, ethirun_demo_dep; "
            "print(ethirun_demo.__version__, ethirun_demo_dep.__version__)",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "1.0.0 1.0.0"

    marker = json.loads((result.env_dir / ENV_MARKER).read_text(encoding="utf-8"))
    assert marker["env_hash"] == result.env_hash
    assert marker["lockfile_sha256"] == result.lockfile_sha256
    assert marker["python_version"] == "3.11"
    assert marker["worker_version"] == "worker-test"

    assert after == before
    assert importlib.util.find_spec("ethirun_demo") is None


def test_unchanged_lockfile_is_reused_without_invoking_uv(
    tmp_path: Path,
    lockfile: Path,
    index_options: UvIndexOptions,
) -> None:
    spy = SpyUv(Uv(timeout=120))
    manager = EnvManager(
        root=tmp_path / "envs",
        uv=spy,
        index=index_options,
        worker_version="worker-test",
    )
    first = manager.build(lockfile, "3.11")
    marker_path = first.env_dir / ENV_MARKER
    marker_mtime = marker_path.stat().st_mtime_ns
    calls_after_first = len(spy.calls)

    second = manager.build(lockfile, "3.11")

    assert second.cached is True
    assert second.env_hash == first.env_hash
    assert second.env_dir == first.env_dir
    assert second.python == first.python
    assert len(spy.calls) == calls_after_first
    assert marker_path.stat().st_mtime_ns == marker_mtime


def test_invalid_marker_triggers_a_clean_rebuild(
    tmp_path: Path,
    lockfile: Path,
    index_options: UvIndexOptions,
) -> None:
    spy = SpyUv(Uv(timeout=120))
    manager = EnvManager(
        root=tmp_path / "envs",
        uv=spy,
        index=index_options,
        worker_version="worker-test",
    )
    first = manager.build(lockfile, "3.11")
    (first.env_dir / ENV_MARKER).write_text("{}", encoding="utf-8")
    calls_before = len(spy.calls)

    second = manager.build(lockfile, "3.11")

    assert second.cached is False
    assert len(spy.calls) > calls_before


def test_hash_changes_with_lockfile_python_or_worker(tmp_path: Path) -> None:
    manager = EnvManager(root=tmp_path, uv=SpyUv(Uv(timeout=120)))
    baseline = manager.compute_hash(b"a==1\n", "3.11", "worker-1")

    assert baseline == manager.compute_hash(b"a==1\n", "3.11", "worker-1")
    assert baseline != manager.compute_hash(b"a==2\n", "3.11", "worker-1")
    assert baseline != manager.compute_hash(b"a==1\n", "3.12", "worker-1")
    assert baseline != manager.compute_hash(b"a==1\n", "3.11", "worker-2")


def test_lockfile_change_points_at_a_different_env(
    tmp_path: Path,
    lockfile: Path,
) -> None:
    manager = EnvManager(root=tmp_path / "envs", uv=SpyUv(Uv(timeout=120)))
    changed = tmp_path / "changed.lock"
    changed.write_bytes(lockfile.read_bytes() + b"# changed\n")

    assert manager.env_dir(lockfile, "3.11") != manager.env_dir(changed, "3.11")


def test_empty_lockfile_is_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "empty.lock"
    empty.write_text("\n", encoding="utf-8")
    manager = EnvManager(root=tmp_path / "envs", uv=SpyUv(Uv(timeout=120)))

    with pytest.raises(EnvBuildError, match="is empty"):
        manager.build(empty, "3.11")


def test_failed_sync_removes_the_partial_env(tmp_path: Path) -> None:
    lockfile = tmp_path / "broken.lock"
    lockfile.write_text(
        "missing-package==1.0.0 \\\n"
        "    --hash=sha256:"
        "0000000000000000000000000000000000000000000000000000000000000000\n",
        encoding="utf-8",
    )
    empty_wheels = tmp_path / "empty-wheels"
    empty_wheels.mkdir()
    manager = EnvManager(
        root=tmp_path / "envs",
        uv=Uv(timeout=120),
        index=UvIndexOptions(find_links=empty_wheels, no_index=True),
        worker_version="worker-test",
    )
    expected_dir = manager.env_dir(lockfile, "3.11")

    with pytest.raises(EnvBuildError) as info:
        manager.build(lockfile, "3.11")

    assert info.value.stage == "uv pip sync"
    assert not expected_dir.exists()
