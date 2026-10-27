"""Stage 1 unit tests: AST parsing, imports, declarations, adapter choice."""

from __future__ import annotations

from typing import Any

from ethirun.schemas import ImportKind, StaticScanResult
from ethirun.static_scan import (
    CTC_FAMILY,
    CUSTOM_SCRIPT_FAMILY,
    NEMO_FAMILY,
    ONNX_FAMILY,
    WHISPER_FAMILY,
    normalize_package,
    parse_declared_packages,
    scan_python,
    select_adapter,
)


def _scan(
    sources: dict[str, str],
    *,
    config: dict[str, Any] | None = None,
    requirements: str | None = None,
    file_paths: list[str] | None = None,
) -> StaticScanResult:
    return scan_python(
        sources,
        file_paths=file_paths if file_paths is not None else list(sources),
        requirements=requirements,
        config=config or {},
    )


def test_parse_declared_packages_handles_versions_extras_and_comments() -> None:
    text = """
    # runtime
    transformers>=4.40  # inline comment
    torch==2.2.*
    huggingface_hub[cli]>=0.28
    tokenizers @ git+https://example.invalid/tokenizers
    -r other-requirements.txt
    git+https://example.invalid/unknown
    """

    assert parse_declared_packages(text) == (
        "huggingface-hub",
        "tokenizers",
        "torch",
        "transformers",
    )
    assert parse_declared_packages(None) == ()
    assert normalize_package("HuggingFace_Hub") == "huggingface-hub"


def test_imports_are_classified_mapped_and_checked_against_requirements() -> None:
    source = """
import os
import numpy as np
import yaml
from PIL import Image
from sklearn.metrics import accuracy_score
from transformers import WhisperForConditionalGeneration
import local_helper
from . import sibling
"""
    scan = _scan(
        {"modeling_custom.py": source},
        file_paths=["modeling_custom.py", "local_helper.py", "sibling.py"],
        requirements="transformers\ntorch\nhuggingface-hub\nPyYAML\n",
    )

    by_module = {record.module: record for record in scan.imports}
    assert by_module["os"].kind is ImportKind.STDLIB
    assert by_module["os"].pypi_name is None
    assert by_module["local_helper"].kind is ImportKind.LOCAL
    assert by_module[".sibling"].kind is ImportKind.LOCAL

    numpy_record = by_module["numpy"]
    assert numpy_record.kind is ImportKind.THIRD_PARTY
    assert numpy_record.pypi_name == "numpy"
    assert numpy_record.declared is False

    assert by_module["yaml"].pypi_name == "pyyaml"
    assert by_module["yaml"].declared is True
    assert by_module["PIL"].pypi_name == "pillow"
    assert by_module["PIL"].declared is False
    assert by_module["sklearn.metrics"].pypi_name == "scikit-learn"
    assert by_module["sklearn.metrics"].declared is False
    assert by_module["transformers"].declared is True

    assert scan.undeclared_imports == ("numpy", "pillow", "scikit-learn")
    assert scan.declared_packages == ("huggingface-hub", "pyyaml", "torch", "transformers")


def test_repository_code_is_parsed_but_never_executed() -> None:
    source = """
import torch

raise RuntimeError("this must never run during inspection")
"""
    scan = _scan({"model.py": source}, requirements="torch")

    assert scan.python_files == ("model.py",)
    assert scan.scan_errors == ()
    assert scan.imports[0].pypi_name == "torch"


def test_syntax_errors_are_recorded_without_crashing() -> None:
    scan = _scan({"broken.py": "def broken(:\n    pass\n"})

    assert len(scan.scan_errors) == 1
    assert "SyntaxError" in scan.scan_errors[0]
    assert scan.python_files == ("broken.py",)


def test_whisper_adapter_scores_above_threshold() -> None:
    choice = select_adapter(
        {
            "architectures": ["WhisperForConditionalGeneration"],
            "model_type": "whisper",
        },
        ["config.json"],
        pipeline_tag="automatic-speech-recognition",
        library_name="transformers",
    )

    assert choice.family == WHISPER_FAMILY
    assert choice.supported is True
    assert choice.requires_trust_remote_code is False
    assert choice.confidence >= 0.9
    assert choice.reasons.count("hub pipeline tag is automatic-speech-recognition") == 1


def test_ctc_adapter_scores_above_threshold() -> None:
    choice = select_adapter(
        {
            "architectures": ["Wav2Vec2ForCTC"],
            "model_type": "wav2vec2",
        },
        ["config.json"],
    )

    assert choice.family == CTC_FAMILY
    assert choice.supported is True
    assert choice.confidence >= 0.8


def test_auto_map_forces_the_custom_script_trust_gate() -> None:
    scan = _scan(
        {"modeling_custom.py": "from transformers import PreTrainedModel\n"},
        config={
            "architectures": ["WhisperForConditionalGeneration"],
            "model_type": "whisper",
            "auto_map": {"AutoModel": "modeling_custom.CustomWhisper"},
        },
        requirements="transformers",
    )

    assert scan.remote_code is True
    assert "auto_map" in scan.remote_code_reasons[0]
    assert scan.adapter.family == CUSTOM_SCRIPT_FAMILY
    assert scan.adapter.supported is False
    assert scan.adapter.requires_trust_remote_code is True


def test_nemo_and_onnx_are_reported_as_unsupported_families() -> None:
    nemo = select_adapter({}, ["model.nemo"])
    onnx = select_adapter({}, ["model.onnx"])

    assert nemo.family == NEMO_FAMILY
    assert nemo.supported is False
    assert onnx.family == ONNX_FAMILY
    assert onnx.supported is False


def test_unknown_architecture_is_a_valid_unsupported_result() -> None:
    choice = select_adapter(
        {"architectures": ["SomeOtherSpeechModel"], "model_type": "other"},
        ["config.json"],
    )

    assert choice.family is None
    assert choice.supported is False
    assert choice.confidence < 0.6


def test_non_asr_heads_are_not_misclassified_as_ctc_or_whisper() -> None:
    classification = select_adapter(
        {
            "architectures": ["Wav2Vec2ForSequenceClassification"],
            "model_type": "wav2vec2",
        },
        ["config.json"],
    )
    whisper_classification = select_adapter(
        {
            "architectures": ["WhisperForAudioClassification"],
            "model_type": "whisper",
        },
        ["config.json"],
    )

    assert classification.family is None
    assert classification.supported is False
    assert whisper_classification.family is None
    assert whisper_classification.supported is False


def test_repository_python_presence_is_recorded_without_auto_map() -> None:
    scan = _scan(
        {"script.py": "print('metadata helper')\n"},
        config={"architectures": ["Wav2Vec2ForCTC"]},
    )

    assert scan.remote_code is False
    assert "repository contains Python source files" in scan.remote_code_reasons
