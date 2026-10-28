"""P5 name/shape comparison is pure and testable without torch."""

from __future__ import annotations

from ethirun_worker.probes import compare_state_shapes


def test_exact_match() -> None:
    result = compare_state_shapes(
        {"encoder.weight": (4, 8), "decoder.bias": (4,)},
        {"encoder.weight": (4, 8), "decoder.bias": (4,)},
    )

    assert result["verdict"] == "match"
    assert result["detail"]["shape_mismatch_count"] == 0
    assert result["detail"]["local_tensors"] == 2
    assert result["detail"]["remote_tensors"] == 2


def test_remote_model_prefix_is_accepted() -> None:
    result = compare_state_shapes(
        {"encoder.weight": (4, 8)},
        {"model.encoder.weight": (4, 8)},
    )

    assert result["verdict"] == "match_with_prefix"
    assert result["detail"]["remote_prefix"] == "model."


def test_local_model_prefix_is_accepted() -> None:
    result = compare_state_shapes(
        {"model.encoder.weight": (4, 8)},
        {"encoder.weight": (4, 8)},
    )

    assert result["verdict"] == "match_with_prefix"
    assert result["detail"]["local_prefix"] == "model."


def test_shape_mismatch_is_investigate_not_a_crash() -> None:
    result = compare_state_shapes(
        {"encoder.weight": (4, 8)},
        {"encoder.weight": (4, 16)},
    )

    assert result["verdict"] == "mismatch"
    assert result["detail"]["shape_mismatch_count"] == 1
    assert result["detail"]["shape_mismatches"] == [
        {
            "name": "encoder.weight",
            "local_shape": [4, 8],
            "remote_shape": [4, 16],
        }
    ]


def test_missing_and_extra_tensors_are_reported() -> None:
    result = compare_state_shapes(
        {"a": (1,), "b": (2,)},
        {"a": (1,), "c": (3,)},
    )

    assert result["verdict"] == "mismatch"
    assert result["detail"]["missing_in_remote"] == ["b"]
    assert result["detail"]["missing_in_local"] == ["c"]
    assert result["detail"]["missing_in_remote_count"] == 1
    assert result["detail"]["missing_in_local_count"] == 1


def test_empty_remote_metadata_is_not_applicable() -> None:
    result = compare_state_shapes({"a": (1,)}, {})

    assert result["verdict"] == "not_applicable"
    assert "no tensors" in result["detail"]["reason"]


def test_shape_values_may_be_lists() -> None:
    result = compare_state_shapes(
        {"encoder.weight": (4, 8)},
        {"model.encoder.weight": [4, 8]},
    )

    assert result["verdict"] == "match_with_prefix"
