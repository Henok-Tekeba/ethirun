"""Build tiny random-weight Whisper and CTC models inside a worker venv.

Run with the worker Python, never the host interpreter:

    python tests/scripts/make_tiny_models.py <output-dir>

The models are deliberately rigged to emit a short text token: Whisper's final
projection and CTC's lm_head are biased, and Whisper gets forced decoder ids.
This lets CI verify the real pipeline reaches a non-empty transcription without
downloading any model.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

SAMPLE_RATE = 16_000


def _make_audio(path: Path) -> None:
    duration = 0.5
    timeline = np.arange(int(duration * SAMPLE_RATE)) / SAMPLE_RATE
    samples = (0.1 * np.sin(2 * np.pi * 440 * timeline)).astype(np.float32)
    sf.write(path, samples, SAMPLE_RATE)


def _whisper_tokenizer(model_dir: Path, vocab: dict[str, int]) -> WhisperTokenizer:
    parameters = inspect.signature(WhisperTokenizer).parameters
    if "vocab_file" in parameters:
        vocab_path = model_dir / "vocab.json"
        merges_path = model_dir / "merges.txt"
        vocab_path.write_text(json.dumps(vocab), encoding="utf-8")
        merges_path.write_text("#version: 0.2\n", encoding="utf-8")
        return WhisperTokenizer(
            vocab_file=str(vocab_path),
            merges_file=str(merges_path),
            bos_token="<|startoftranscript|>",
            eos_token="<|endoftext|>",
            unk_token="<|endoftext|>",
            pad_token="<|endoftext|>",
        )

    return WhisperTokenizer(
        vocab=vocab,
        merges=[],
        bos_token="<|startoftranscript|>",
        eos_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        pad_token="<|endoftext|>",
    )


def _make_whisper(root: Path) -> None:
    model_dir = root / "whisper"
    model_dir.mkdir()
    # Text ids stay below <|notimestamps|> + 1 so the ASR decoder treats them
    # as text rather than timestamp tokens.
    vocab = {
        "<|endoftext|>": 0,
        "<|startoftranscript|>": 1,
        "a": 2,
        "b": 3,
        "<|am|>": 4,
        "<|transcribe|>": 5,
        "<|notimestamps|>": 6,
        " ": 7,
        "c": 8,
    }
    tokenizer = _whisper_tokenizer(model_dir, vocab)
    feature_extractor = WhisperFeatureExtractor(
        feature_size=80,
        sampling_rate=SAMPLE_RATE,
        hop_length=160,
        chunk_length=1,
        n_fft=400,
    )
    processor = WhisperProcessor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )
    config = WhisperConfig(
        vocab_size=len(vocab),
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        num_mel_bins=80,
        max_source_positions=100,
        max_target_positions=20,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=0,
        decoder_start_token_id=1,
    )
    model = WhisperForConditionalGeneration(config)
    with torch.no_grad():
        model.proj_out.weight.zero_()
        model.proj_out.weight[vocab["a"]].fill_(1.0)
    model.generation_config.forced_decoder_ids = [[1, vocab["a"]]]
    model.generation_config.max_length = 4
    model.generation_config.suppress_tokens = []
    model.generation_config.begin_suppress_tokens = []

    model.save_pretrained(model_dir)
    processor.save_pretrained(model_dir)


def _make_ctc(root: Path) -> None:
    model_dir = root / "ctc"
    model_dir.mkdir()
    vocab = {
        "<pad>": 0,
        "<s>": 1,
        "</s>": 2,
        "<unk>": 3,
        "a": 4,
        "b": 5,
        "|": 6,
    }
    vocab_path = model_dir / "vocab.json"
    vocab_path.write_text(json.dumps(vocab), encoding="utf-8")
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=str(vocab_path),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=SAMPLE_RATE,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False,
    )
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )
    config = Wav2Vec2Config(
        vocab_size=len(vocab),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        conv_dim=(8,),
        conv_stride=(4,),
        conv_kernel=(8,),
        num_conv_pos_embeddings=8,
        num_conv_pos_embedding_groups=2,
        feat_extract_norm="group",
        do_stable_layer_norm=False,
        pad_token_id=0,
        ctc_loss_reduction="sum",
        ctc_zero_infinity=True,
    )
    model = Wav2Vec2ForCTC(config)
    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.lm_head.bias.zero_()
        model.lm_head.bias[vocab["a"]] = 20.0

    model.save_pretrained(model_dir)
    processor.save_pretrained(model_dir)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: make_tiny_models.py <output-dir>", file=sys.stderr)
        return 2

    torch.manual_seed(0)
    root = Path(argv[1])
    root.mkdir(parents=True, exist_ok=True)
    _make_audio(root / "clip.wav")
    _make_whisper(root)
    _make_ctc(root)
    print(json.dumps({"whisper": str(root / "whisper"), "ctc": str(root / "ctc")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
