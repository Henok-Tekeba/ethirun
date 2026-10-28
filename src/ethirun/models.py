"""Download model weights for Stage 5, never during inspection.

Only the files needed to load a standard transformers ASR model are fetched:
configs, tokenizer/processor files, and safetensors/PyTorch weights. Alternate
frameworks and optimizer states are deliberately excluded. When a repo
publishes safetensors, PyTorch ``.bin`` duplicates are skipped.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_MODELS_DIR = Path.home() / ".ethirun" / "models"

COMMON_PATTERNS = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "feature_extractor_config.json",
    "normalizer.json",
    "chat_template.json",
    "chat_template.jinja",
    "special_tokens_map.json",
    "added_tokens.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "spiece.model",
    "sentencepiece.bpe.model",
    "tokenizer.model",
)
SAFETENSORS_PATTERNS = (
    *COMMON_PATTERNS,
    "*.safetensors",
    "*.safetensors.index.json",
)
ALLOW_PATTERNS = (*SAFETENSORS_PATTERNS, "*.bin")


class ModelDownloadError(RuntimeError):
    """Weights could not be downloaded or staged."""


def download_model(
    repo: str,
    revision: str,
    *,
    root: Path = DEFAULT_MODELS_DIR,
    prefer_safetensors: bool = False,
) -> Path:
    """Download loadable weights at a pinned revision and return the directory."""
    destination = root / repo.replace("/", "__") / revision
    destination.mkdir(parents=True, exist_ok=True)
    patterns = SAFETENSORS_PATTERNS if prefer_safetensors else ALLOW_PATTERNS
    try:
        path = snapshot_download(
            repo_id=repo,
            revision=revision,
            local_dir=str(destination),
            allow_patterns=list(patterns),
        )
    except Exception as exc:
        raise ModelDownloadError(f"could not download {repo}@{revision}: {exc}") from exc
    return Path(path)
