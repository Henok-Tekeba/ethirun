"""Static guard: the worker runtime stays importable without model libraries.

Only module-level imports are checked. Runtime libraries such as torch and
transformers are imported lazily inside probe/load functions, which is what
lets the worker source live on PYTHONPATH without installation. This guard
reads source with AST and never imports the worker into the host process.
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


def _module_level_import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()

    for node in tree.body:
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            if node.module:
                roots.add(node.module.split(".")[0])

    return roots


def test_worker_top_level_imports_are_stdlib_only() -> None:
    allowed = set(sys.stdlib_module_names) | {"ethirun_worker"}
    sources = sorted((WORKER_DIR / "ethirun_worker").rglob("*.py"))

    assert sources, "worker package sources not found"

    for path in sources:
        unexpected = _module_level_import_roots(path) - allowed
        assert not unexpected, f"{path.relative_to(WORKER_DIR)} imports {sorted(unexpected)}"
