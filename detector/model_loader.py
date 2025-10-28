import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

_CACHE = {}

def get_model_and_processor():
    """Model loader"""

    if "qwen" in _CACHE:
        return _CACHE["qwen"], _CACHE["processor"]

    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto" if device == "cuda" else None
    )

    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=224*224,
        max_pixels=1024*1024
    )

    _CACHE["qwen"], _CACHE["processor"] = qwen, processor
    return qwen, processor


