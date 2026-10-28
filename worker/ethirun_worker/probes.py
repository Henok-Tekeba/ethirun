"""Pure probe helpers that do not import model runtimes.

The name/shape comparison is deliberately stdlib-only so the host test suite
can verify P5 logic without torch or transformers installed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

KNOWN_PREFIXES = ("", "model.", "module.")


class ProbeError(RuntimeError):
    """A probe could not run against this model."""


def _strip_prefix(name: str, prefix: str) -> str:
    return name[len(prefix) :] if prefix and name.startswith(prefix) else name


def _shape_tuple(shape: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(dimension) for dimension in shape)


def compare_state_shapes(
    local: Mapping[str, Sequence[int]],
    remote: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    """Compare a meta model state dict with safetensors metadata.

    Returns one of ``match``, ``match_with_prefix`` or ``mismatch`` plus bounded
    diagnostic samples. Tied/omitted weights are reported as a mismatch for a
    human to investigate, never as a hard failure.
    """
    if not remote:
        return {
            "verdict": "not_applicable",
            "detail": {"reason": "safetensors metadata contains no tensors"},
        }

    pairs = sorted(
        (
            (local_prefix, remote_prefix)
            for local_prefix in KNOWN_PREFIXES
            for remote_prefix in KNOWN_PREFIXES
        ),
        key=lambda pair: (bool(pair[0]) + bool(pair[1]), pair[0], pair[1]),
    )

    aligned: tuple[str, str, dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]] | None = None
    for local_prefix, remote_prefix in pairs:
        normalized_local = {
            _strip_prefix(name, local_prefix): _shape_tuple(shape) for name, shape in local.items()
        }
        normalized_remote = {
            _strip_prefix(name, remote_prefix): _shape_tuple(shape)
            for name, shape in remote.items()
        }
        if set(normalized_local) == set(normalized_remote):
            aligned = (local_prefix, remote_prefix, normalized_local, normalized_remote)
            break

    if aligned is None:
        missing_in_remote = sorted(set(local) - set(remote))
        missing_in_local = sorted(set(remote) - set(local))
        return {
            "verdict": "mismatch",
            "detail": {
                "local_tensors": len(local),
                "remote_tensors": len(remote),
                "missing_in_remote": missing_in_remote[:20],
                "missing_in_local": missing_in_local[:20],
                "missing_in_remote_count": len(missing_in_remote),
                "missing_in_local_count": len(missing_in_local),
            },
        }

    local_prefix, remote_prefix, normalized_local, normalized_remote = aligned
    mismatches = [
        {
            "name": name,
            "local_shape": list(normalized_local[name]),
            "remote_shape": list(normalized_remote[name]),
        }
        for name in sorted(normalized_local)
        if normalized_local[name] != normalized_remote[name]
    ]

    verdict = (
        "mismatch"
        if mismatches
        else ("match_with_prefix" if local_prefix or remote_prefix else "match")
    )
    return {
        "verdict": verdict,
        "detail": {
            "local_tensors": len(local),
            "remote_tensors": len(remote),
            "local_prefix": local_prefix,
            "remote_prefix": remote_prefix,
            "shape_mismatch_count": len(mismatches),
            "shape_mismatches": mismatches[:20],
        },
    }
