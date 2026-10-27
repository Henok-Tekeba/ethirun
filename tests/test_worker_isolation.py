"""Static guard: the worker runtime must stay stdlib-only.

Reads worker sources with AST; never imports them into the host process. A
third-party import here would mean worker dependencies leaked into the host or
that the worker needs installation, both of which break isolation.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from ethirun import WORKER_VERSION
from ethirun_worker import WORKER_VERSION as RUNTIME_WORKER_VERSION
from tests.worker_harness import WORKER_DIR


def test_worker_versions_match() -> None:
    assert WORKER_VERSION == RUNTIME_WORKER_VERSION


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            if node.module:
                roots.add(node.module.split(".")[0])

    return roots


def test_worker_imports_only_stdlib_and_itself() -> None:
    allowed = set(sys.stdlib_module_names) | {"ethirun_worker"}
    sources = sorted((WORKER_DIR / "ethirun_worker").rglob("*.py"))

    assert sources, "worker package sources not found"

    for path in sources:
        unexpected = _import_roots(path) - allowed
        assert not unexpected, f"{path.relative_to(WORKER_DIR)} imports {sorted(unexpected)}"
