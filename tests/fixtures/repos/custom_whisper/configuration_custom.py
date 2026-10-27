from transformers import PretrainedConfig


class CustomConfig(PretrainedConfig):
    model_type = "custom_whisper"
