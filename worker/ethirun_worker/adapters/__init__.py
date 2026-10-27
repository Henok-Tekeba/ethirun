"""Adapter registry. New model families add one module here."""

from __future__ import annotations

from collections.abc import Callable

from ethirun_worker.adapters.base import WorkerAdapter
from ethirun_worker.adapters.fake import FakeAdapter

_ADAPTERS: dict[str, Callable[[], WorkerAdapter]] = {
    FakeAdapter.name: FakeAdapter,
}


def create_adapter(name: str) -> WorkerAdapter:
    """Instantiate a worker adapter by name.

    Only the fake adapter exists in Milestone 2; transformer adapters arrive in
    Milestone 5 without changing this protocol.
    """
    try:
        adapter_type = _ADAPTERS[name]
    except KeyError as exc:
        known = ", ".join(sorted(_ADAPTERS))
        raise ValueError(f"unknown worker adapter {name!r}; known: {known}") from exc
    return adapter_type()
