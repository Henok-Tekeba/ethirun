# Custom Whisper fixture with remote code.
import numpy as np
import torch
from PIL import Image
from transformers import WhisperForConditionalGeneration

from .configuration_custom import CustomConfig

raise RuntimeError("inspection must never execute model repository code")
