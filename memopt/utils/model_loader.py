"""
memopt/utils/model_loader.py

Safe model loading for large models (up to 30B on single GPU).
Call this before agent.run() for models >7B params.
"""
import re
import logging
from typing import Optional

import torch

log = logging.getLogger("memopt.utils")


def load_large_model(
    model_name_or_path: str,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.float16,
) -> "torch.nn.Module":
    """
    Load a large model safely for memopt optimization.

    Handles:
    - FP16 loading (mandatory for 13B+ to fit in 80GB)
    - Low CPU memory usage (stream weights, don't double-load)
    - Memory check before loading
    - Clear error if model won't fit

    Usage:
        model = load_large_model("mistralai/Mistral-7B-v0.1")
        report = agent.run(model, inputs)

    Supports up to ~30B params on single 80GB GPU in FP16.
    For >30B: use 2+ GPUs or INT4 quantization (not memopt's scope).
    """
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        raise ImportError(
            "transformers not installed. "
            "Run: pip install transformers accelerate"
        )

    # Estimate model size before loading
    estimated_gb = _estimate_model_gb(model_name_or_path, dtype)

    if estimated_gb is not None and torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        free_vram = (
            torch.cuda.get_device_properties(0).total_memory
            - torch.cuda.memory_reserved(0)
        ) / 1e9

        log.info(
            "Loading %s: estimated=%.0fGB free_vram=%.1fGB total_vram=%.1fGB",
            model_name_or_path, estimated_gb, free_vram, total_vram,
        )

        # Hard stop: model won't fit
        if estimated_gb > total_vram * 0.85:
            raise RuntimeError(
                f"Model too large for single GPU: "
                f"estimated={estimated_gb:.0f}GB "
                f"GPU has {total_vram:.0f}GB. "
                f"For 30B+ models: use 2+ GPUs or INT4 quantization."
            )

        # Warning: tight fit, optimization headroom limited
        if estimated_gb > total_vram * 0.70:
            log.warning(
                "Model uses %.0fGB of %.0fGB VRAM. "
                "Only %.0fGB free for optimization. "
                "INT8 quantization pass may OOM — agent will handle gracefully.",
                estimated_gb, total_vram, total_vram - estimated_gb,
            )

    log.info("Loading %s in %s...", model_name_or_path, dtype)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map=device,
        low_cpu_mem_usage=True,   # stream weights — don't double RAM
    )
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    actual_gb = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / 1e9

    log.info(
        "Loaded: %.1fB params, %.1fGB VRAM",
        param_count / 1e9, actual_gb,
    )

    return model


def _estimate_model_gb(
    model_name: str,
    dtype: torch.dtype = torch.float16,
) -> Optional[float]:
    """
    Estimate model VRAM from name. Returns None if cannot estimate.
    FP16 = 2 bytes/param. FP32 = 4 bytes/param.
    """
    bytes_per_param = 2 if dtype == torch.float16 else 4

    # Extract param count from common naming patterns
    patterns = [
        r"(\d+\.?\d*)b",   # 7b, 13b, 30b, 1.5b
        r"(\d+)B",          # 7B, 13B
    ]

    name_lower = model_name.lower()
    for pattern in patterns:
        match = re.search(pattern, name_lower)
        if match:
            billions = float(match.group(1))
            gb = billions * 1e9 * bytes_per_param / 1e9
            return gb

    return None  # Cannot estimate
