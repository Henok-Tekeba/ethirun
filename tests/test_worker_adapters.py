"""Worker adapter registry tests that do not import torch or transformers."""

from __future__ import annotations

from typing import Any

import pytest

from ethirun_worker.adapters import create_adapter
from ethirun_worker.adapters.transformers_asr import CTCAdapter, WhisperAdapter
from ethirun_worker.probes import ProbeError


@pytest.mark.parametrize(
    ("name", "adapter_type"),
    [
        ("fake", None),
        ("transformers_whisper", WhisperAdapter),
        ("transformers_ctc", CTCAdapter),
    ],
)
def test_registry_contains_all_adapters(
    name: str,
    adapter_type: type[Any] | None,
) -> None:
    adapter = create_adapter(name)

    if adapter_type is None:
        assert adapter.name == name
    else:
        assert isinstance(adapter, adapter_type)


def test_unknown_adapter_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown worker adapter"):
        create_adapter("nemo")


def test_unknown_probe_is_rejected_without_runtime_imports() -> None:
    adapter = WhisperAdapter()

    with pytest.raises(ProbeError, match="unknown probe"):
        adapter.probe("does_not_exist", {})


def test_probe_requires_model_dir() -> None:
    adapter = CTCAdapter()

    with pytest.raises(ProbeError, match="model_dir"):
        adapter.probe("meta_build", {})
