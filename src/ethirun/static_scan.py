"""Stage 1: read repository Python with AST. Nothing is imported or executed."""

from __future__ import annotations

import ast
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

from ethirun.schemas import (
    AdapterChoice,
    ImportKind,
    ImportRecord,
    StaticScanResult,
)

MAX_PYTHON_FILE_BYTES = 256 * 1024
ADAPTER_THRESHOLD = 0.6

WHISPER_FAMILY = "transformers_whisper"
CTC_FAMILY = "transformers_ctc"
NEMO_FAMILY = "nemo"
CUSTOM_SCRIPT_FAMILY = "custom_script"
ONNX_FAMILY = "onnx"

PYPI_NAMES: dict[str, str] = {
    "accelerate": "accelerate",
    "cv2": "opencv-python",
    "datasets": "datasets",
    "einops": "einops",
    "evaluate": "evaluate",
    "fairseq": "fairseq",
    "flax": "flax",
    "gradio": "gradio",
    "huggingface_hub": "huggingface-hub",
    "hydra": "hydra-core",
    "IPython": "ipython",
    "jax": "jax",
    "jiwer": "jiwer",
    "kenlm": "kenlm",
    "keras": "keras",
    "librosa": "librosa",
    "matplotlib": "matplotlib",
    "nemo": "nemo-toolkit",
    "nemo_toolkit": "nemo-toolkit",
    "numpy": "numpy",
    "omegaconf": "omegaconf",
    "onnx": "onnx",
    "onnxruntime": "onnxruntime",
    "packaging": "packaging",
    "pandas": "pandas",
    "peft": "peft",
    "PIL": "pillow",
    "protobuf": "protobuf",
    "psutil": "psutil",
    "pyctcdecode": "pyctcdecode",
    "regex": "regex",
    "requests": "requests",
    "safetensors": "safetensors",
    "scipy": "scipy",
    "sentencepiece": "sentencepiece",
    "setuptools": "setuptools",
    "sklearn": "scikit-learn",
    "soundfile": "soundfile",
    "tensorflow": "tensorflow",
    "tokenizers": "tokenizers",
    "torch": "torch",
    "torchaudio": "torchaudio",
    "tqdm": "tqdm",
    "transformers": "transformers",
    "typing_extensions": "typing-extensions",
    "whisper": "openai-whisper",
    "yaml": "pyyaml",
}

_CTC_MARKERS = (
    "wav2vec2",
    "hubert",
    "wavlm",
    "unispeech",
    "data2vec",
    "mms",
    "seamless",
    "xlsr",
)
_PACKAGE_NAME = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")


def normalize_package(name: str) -> str:
    """Normalize a package name the way PyPI does."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_declared_packages(requirements: str | None) -> tuple[str, ...]:
    """Extract package names from a simple requirements.txt."""
    if not requirements:
        return ()

    names: set[str] = set()
    for raw_line in requirements.splitlines():
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if not line or line.startswith(("-", "git+", "http:", "https:", "file:")):
            continue

        match = _PACKAGE_NAME.match(line)
        if match:
            names.add(normalize_package(match.group(1)))

    return tuple(sorted(names))


def _pypi_name(module_root: str) -> str:
    mapped = PYPI_NAMES.get(module_root)
    if mapped is not None:
        return mapped
    return normalize_package(module_root)


def _local_modules(file_paths: Sequence[str]) -> set[str]:
    modules: set[str] = set()
    for path_str in file_paths:
        path = PurePosixPath(path_str)
        if path.suffix != ".py":
            continue
        if len(path.parts) == 1:
            modules.add(path.stem)
        else:
            modules.add(path.parts[0])
    return modules


def _import_roots(source: str, filename: str) -> tuple[list[tuple[str, int]], str | None]:
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        line = exc.lineno if exc.lineno is not None else 0
        return [], f"{filename}: SyntaxError at line {line}: {exc.msg}"

    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                prefix = "." * node.level
                if node.module:
                    found.append((prefix + node.module, node.lineno))
                else:
                    found.extend((prefix + alias.name, node.lineno) for alias in node.names)
            elif node.module:
                found.append((node.module, node.lineno))
    return found, None


def _classify_import(
    module: str,
    *,
    local_modules: set[str],
    declared: set[str],
    stdlib: frozenset[str],
) -> tuple[ImportKind, str | None, bool | None]:
    if module.startswith("."):
        return ImportKind.LOCAL, None, None

    root = module.split(".")[0]
    if root in local_modules:
        return ImportKind.LOCAL, None, None
    if root in stdlib:
        return ImportKind.STDLIB, None, None

    pypi_name = _pypi_name(root)
    return ImportKind.THIRD_PARTY, pypi_name, normalize_package(pypi_name) in declared


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def select_adapter(
    config: Mapping[str, Any],
    file_paths: Sequence[str],
    *,
    pipeline_tag: str | None = None,
    library_name: str | None = None,
) -> AdapterChoice:
    """Choose an adapter family from config and file names, with confidence."""
    architectures = [item.lower() for item in _string_list(config.get("architectures"))]
    architecture_text = " ".join(architectures)
    model_type_raw = config.get("model_type")
    model_type = model_type_raw.lower() if isinstance(model_type_raw, str) else ""

    auto_map = config.get("auto_map")
    if auto_map:
        keys = ", ".join(sorted(auto_map)) if isinstance(auto_map, Mapping) else "present"
        return AdapterChoice(
            family=CUSTOM_SCRIPT_FAMILY,
            confidence=0.95,
            supported=False,
            requires_trust_remote_code=True,
            reasons=(
                f"config.json declares auto_map ({keys}); remote code needs the trust gate",
                "custom-script adapter is not implemented yet",
            ),
        )

    lower_paths = [path.lower() for path in file_paths]
    if any(path.endswith(".nemo") for path in lower_paths):
        return AdapterChoice(
            family=NEMO_FAMILY,
            confidence=0.95,
            supported=False,
            reasons=("repository contains a .nemo archive; NeMo support is not implemented yet",),
        )

    reasons: list[str] = []
    asr_pipeline = pipeline_tag == "automatic-speech-recognition"
    has_ctc_head = "ctc" in architecture_text
    whisper_signal = "conditionalgeneration" in architecture_text or asr_pipeline
    ctc_signal = has_ctc_head or asr_pipeline

    whisper_score = 0.0
    if model_type == "whisper" and whisper_signal:
        whisper_score += 0.4
        reasons.append("config model_type is whisper")
    if "whisper" in architecture_text and whisper_signal:
        whisper_score += 0.45
        reasons.append("an architecture name contains Whisper")
    if "conditionalgeneration" in architecture_text:
        whisper_score += 0.1
    if asr_pipeline:
        whisper_score += 0.05
        reasons.append("hub pipeline tag is automatic-speech-recognition")
    if library_name == "transformers":
        whisper_score += 0.02

    ctc_score = 0.0
    matched_ctc = [marker for marker in _CTC_MARKERS if marker in architecture_text]
    if matched_ctc and ctc_signal:
        ctc_score += 0.5
        reasons.append(f"CTC architecture marker: {matched_ctc[0]}")
    if has_ctc_head:
        ctc_score += 0.25
    if model_type in _CTC_MARKERS and ctc_signal:
        ctc_score += 0.35
        reasons.append(f"config model_type is {model_type}")
    if asr_pipeline:
        ctc_score += 0.05
        reasons.append("hub pipeline tag is automatic-speech-recognition")
    if library_name == "transformers":
        ctc_score += 0.02

    whisper_score = min(round(whisper_score, 2), 0.99)
    ctc_score = min(round(ctc_score, 2), 0.99)

    if max(whisper_score, ctc_score) >= ADAPTER_THRESHOLD:
        unique_reasons = tuple(dict.fromkeys(reasons))
        if whisper_score >= ctc_score:
            return AdapterChoice(
                family=WHISPER_FAMILY,
                confidence=whisper_score,
                supported=True,
                reasons=unique_reasons,
            )
        return AdapterChoice(
            family=CTC_FAMILY,
            confidence=ctc_score,
            supported=True,
            reasons=unique_reasons,
        )

    if any(path.endswith(".onnx") for path in lower_paths):
        return AdapterChoice(
            family=ONNX_FAMILY,
            confidence=0.8,
            supported=False,
            reasons=("repository contains ONNX artifacts; ONNX support is not implemented",),
        )

    return AdapterChoice(
        family=None,
        confidence=max(whisper_score, ctc_score),
        supported=False,
        reasons=(
            "no adapter scored above the support threshold",
            f"architectures: {architectures or 'none'}",
            f"model_type: {model_type or 'unknown'}",
        ),
    )


def scan_python(
    python_sources: Mapping[str, str],
    *,
    file_paths: Sequence[str],
    requirements: str | None,
    config: Mapping[str, Any],
    pipeline_tag: str | None = None,
    library_name: str | None = None,
) -> StaticScanResult:
    """Statically scan Python sources and select an adapter.

    ``python_sources`` maps repository-relative paths to text. The caller has
    already applied the size cap. No source is compiled, imported or executed.
    """
    declared = set(parse_declared_packages(requirements))
    local_modules = _local_modules(file_paths)
    stdlib = frozenset(sys.stdlib_module_names) | frozenset(sys.builtin_module_names)

    imports: list[ImportRecord] = []
    errors: list[str] = []
    undeclared: set[str] = set()

    for path in sorted(python_sources):
        roots, error = _import_roots(python_sources[path], path)
        if error is not None:
            errors.append(error)
        for module, lineno in roots:
            kind, pypi_name, is_declared = _classify_import(
                module,
                local_modules=local_modules,
                declared=declared,
                stdlib=stdlib,
            )
            imports.append(
                ImportRecord(
                    module=module,
                    pypi_name=pypi_name,
                    source_file=path,
                    lineno=lineno,
                    kind=kind,
                    declared=is_declared,
                )
            )
            if kind is ImportKind.THIRD_PARTY and is_declared is False and pypi_name:
                undeclared.add(pypi_name)

    auto_map = config.get("auto_map")
    remote_code_reasons: list[str] = []
    if auto_map:
        remote_code_reasons.append("config.json declares auto_map")
    if any(path.lower().endswith(".py") for path in file_paths):
        remote_code_reasons.append("repository contains Python source files")

    return StaticScanResult(
        python_files=tuple(sorted(python_sources)),
        remote_code=bool(auto_map),
        remote_code_reasons=tuple(remote_code_reasons),
        imports=tuple(imports),
        declared_packages=tuple(sorted(declared)),
        undeclared_imports=tuple(sorted(undeclared)),
        scan_errors=tuple(errors),
        adapter=select_adapter(
            config,
            file_paths,
            pipeline_tag=pipeline_tag,
            library_name=library_name,
        ),
    )
