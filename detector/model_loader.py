import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, PreTrainedModel, ProcessorMixin
import os
from typing import Tuple

_CACHE = {}

def get_model_and_processor() -> Tuple[PreTrainedModel, ProcessorMixin]:
    """
    Model loader

        Used model: Qwen2.5-VL-3B-Instruct

    Returns:
        model, processor
    """

    if "qwen" in _CACHE:
        return _CACHE["qwen"], _CACHE["processor"]

    local_model_path = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen2.5-VL-3B-Instruct")
    local_model_path = os.path.abspath(local_model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        local_model_path,
        dtype="auto",
        low_cpu_mem_usage=True,
        device_map="auto" if device == "cuda" else None,
    )

    processor = AutoProcessor.from_pretrained(
        local_model_path,
        min_pixels=224*224,
        max_pixels=1024*1024,
    )

    _CACHE["qwen"], _CACHE["processor"] = qwen, processor
    return qwen, processor


