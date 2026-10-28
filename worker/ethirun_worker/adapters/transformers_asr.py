"""Worker-side transformers adapters for Whisper and CTC ASR families.

All torch/transformers/numpy imports are lazy: the worker source directory stays
importable without model dependencies, and probes fail with a structured worker
error when a runtime is missing.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ethirun_worker.audio import load_audio_16k_mono
from ethirun_worker.probes import ProbeError, compare_state_shapes


class TransformersASRAdapter:
    """Shared probes and pipeline operations for transformers ASR models."""

    name = "transformers_asr"
    language_aware = False
    architecture_markers: tuple[str, ...] = ()

    def __init__(self) -> None:
        self._pipeline: Any = None
        self._device = "cpu"
        self._metadata_files: dict[str, Any] = {}

    def probe(self, name: str, args: Mapping[str, Any]) -> dict[str, Any]:
        if name in {"imports", "p1"}:
            return self._probe_imports()
        if name in {"architecture", "p2"}:
            return self._probe_architecture(self._model_dir(args))
        if name in {"meta_build", "p3"}:
            return self._probe_meta_build(self._model_dir(args))
        if name in {"processor", "p4"}:
            return self._probe_processor(self._model_dir(args))
        if name in {"safetensors", "p5"}:
            return self._probe_safetensors(self._model_dir(args), args)
        raise ProbeError(f"unknown probe {name!r}")

    def load(self, model_dir: str, device: str) -> dict[str, Any]:
        from transformers import pipeline

        path = Path(model_dir)
        if not path.is_dir():
            raise ProbeError(f"model directory not found: {model_dir}")

        device_arg: Any = -1 if device == "cpu" else device
        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=str(path),
            device=device_arg,
            trust_remote_code=False,
        )
        self._device = device
        return {
            "loaded": True,
            "device": device,
            "pipeline": type(self._pipeline).__name__,
        }

    def transcribe(self, audio_path: str, language: str | None) -> str:
        if self._pipeline is None:
            raise ProbeError("model must be loaded before transcribe")

        audio = load_audio_16k_mono(audio_path)
        kwargs: dict[str, Any] = {}
        if self.language_aware and language:
            kwargs["generate_kwargs"] = {"language": language}

        result = self._pipeline(audio, **kwargs)
        text = result.get("text") if isinstance(result, Mapping) else result
        if not isinstance(text, str):
            raise ProbeError("ASR pipeline returned no text field")
        return text

    # ----------------------------- probes ------------------------------

    def _probe_imports(self) -> dict[str, Any]:
        import torch
        import transformers

        return {
            "verdict": "pass",
            "detail": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "cuda": bool(torch.cuda.is_available()),
                "mps": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
            },
        }

    def _probe_architecture(self, model_dir: str) -> dict[str, Any]:
        import transformers

        config = _load_config(model_dir)
        class_name = _architecture_name(config)
        if config.get("auto_map"):
            raise ProbeError("config auto_map requires trust-remote-code; refusing to load")

        model_class = getattr(transformers, class_name, None)
        if model_class is None:
            raise ProbeError(f"transformers has no architecture class {class_name!r}")
        if self.architecture_markers and not any(
            marker in class_name.lower() for marker in self.architecture_markers
        ):
            raise ProbeError(f"{class_name!r} does not match the {self.name} adapter")

        return {
            "verdict": "pass",
            "detail": {
                "class": class_name,
                "adapter": self.name,
                "available": True,
            },
        }

    def _probe_meta_build(self, model_dir: str) -> dict[str, Any]:
        import torch
        from transformers import AutoConfig

        raw_config = _load_config(model_dir)
        if raw_config.get("auto_map"):
            raise ProbeError("config auto_map requires trust-remote-code; refusing to load")

        config = AutoConfig.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=False,
        )
        class_name = _architecture_name(config.to_dict())

        import transformers

        model_class = getattr(transformers, class_name, None)
        if model_class is None:
            raise ProbeError(f"transformers has no architecture class {class_name!r}")

        with torch.device("meta"):
            model = model_class(config)
        model = model.to("meta")

        state = model.state_dict()
        parameters = sum(parameter.numel() for parameter in model.parameters())
        return {
            "verdict": "pass",
            "detail": {
                "class": class_name,
                "parameters": int(parameters),
                "state_dict_entries": len(state),
                "meta_device": all(tensor.device.type == "meta" for tensor in state.values()),
            },
        }

    def _probe_processor(self, model_dir: str) -> dict[str, Any]:
        from transformers import AutoProcessor, AutoTokenizer

        processor = AutoProcessor.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=False,
        )
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                local_files_only=True,
                trust_remote_code=False,
            )

        return {
            "verdict": "pass",
            "detail": {
                "processor": type(processor).__name__,
                "tokenizer": type(tokenizer).__name__,
            },
        }

    def _probe_safetensors(
        self,
        model_dir: str,
        args: Mapping[str, Any],
    ) -> dict[str, Any]:
        repo_id = args.get("repo_id")
        revision = args.get("revision")
        if not isinstance(repo_id, str) or not repo_id:
            return {
                "verdict": "not_applicable",
                "detail": {"reason": "no remote repo id supplied (local model)"},
            }
        if not isinstance(revision, str) or not revision:
            return {
                "verdict": "not_applicable",
                "detail": {"reason": "no pinned revision supplied"},
            }

        remote_shapes = self._fetch_safetensors_shapes(repo_id, revision)
        if not remote_shapes:
            return {
                "verdict": "not_applicable",
                "detail": {"reason": "repository has no safetensors metadata"},
            }

        local_shapes = self._meta_state_shapes(model_dir)
        comparison = compare_state_shapes(local_shapes, remote_shapes)
        detail = comparison.get("detail")
        if not isinstance(detail, dict):
            detail = {}
        detail.update(
            {
                "repo_id": repo_id,
                "revision": revision,
                "safetensors_files": len(self._metadata_files),
            }
        )
        return {"verdict": comparison["verdict"], "detail": detail}

    def _fetch_safetensors_shapes(
        self,
        repo_id: str,
        revision: str,
    ) -> dict[str, tuple[int, ...]]:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import EntryNotFoundError

        try:
            metadata = HfApi().get_safetensors_metadata(repo_id, revision=revision)
        except EntryNotFoundError:
            self._metadata_files = {}
            return {}
        except Exception as exc:
            raise ProbeError(
                f"could not fetch safetensors metadata for {repo_id}@{revision}: {exc}"
            ) from exc

        self._metadata_files = dict(metadata.files_metadata)
        shapes: dict[str, tuple[int, ...]] = {}
        for file_metadata in metadata.files_metadata.values():
            for tensor_name, tensor_info in file_metadata.tensors.items():
                shapes[tensor_name] = tuple(tensor_info.shape)
        return shapes

    def _meta_state_shapes(self, model_dir: str) -> dict[str, tuple[int, ...]]:
        import torch
        import transformers
        from transformers import AutoConfig

        raw_config = _load_config(model_dir)
        if raw_config.get("auto_map"):
            raise ProbeError("config auto_map requires trust-remote-code; refusing to load")

        config = AutoConfig.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=False,
        )
        class_name = _architecture_name(config.to_dict())
        model_class = getattr(transformers, class_name, None)
        if model_class is None:
            raise ProbeError(f"transformers has no architecture class {class_name!r}")

        with torch.device("meta"):
            model = model_class(config)
        return {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}

    @staticmethod
    def _model_dir(args: Mapping[str, Any]) -> str:
        model_dir = args.get("model_dir")
        if not isinstance(model_dir, str) or not model_dir:
            raise ProbeError("probe requires a string 'model_dir'")
        return model_dir


class WhisperAdapter(TransformersASRAdapter):
    """Whisper conditional-generation models via the transformers ASR pipeline."""

    name = "transformers_whisper"
    language_aware = True
    architecture_markers = ("whisper",)


class CTCAdapter(TransformersASRAdapter):
    """Wav2Vec2/HuBERT/WavLM-style CTC models via the ASR pipeline."""

    name = "transformers_ctc"
    language_aware = False
    architecture_markers = (
        "ctc",
        "wav2vec2",
        "hubert",
        "wavlm",
        "mms",
        "unispeech",
        "data2vec",
        "seamless",
    )


def _load_config(model_dir: str) -> dict[str, Any]:
    path = Path(model_dir) / "config.json"
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProbeError(f"could not read {path}: {exc}") from exc
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProbeError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(config, dict):
        raise ProbeError(f"{path} must contain a JSON object")
    return config


def _architecture_name(config: Mapping[str, Any]) -> str:
    architectures = config.get("architectures")
    if (
        not isinstance(architectures, list)
        or not architectures
        or not isinstance(architectures[0], str)
    ):
        raise ProbeError("config.json has no usable architectures[0]")
    return architectures[0]
