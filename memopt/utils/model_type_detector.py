"""
memopt.utils.model_type_detector
=================================

Detects the model family from class name / module structure and returns the
ordered list of optimization candidates applicable to that family.

Design goals:
  - Zero false positives: never apply a GPU-only optimization to a CPU model
  - Graceful degradation: unknown families fall back to a safe default set
  - Hardware-aware: caller provides HardwareProfile; candidates are filtered
    against capability flags before being returned
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch.nn as nn


# ---------------------------------------------------------------------------
# Model family → ordered candidate list
# ---------------------------------------------------------------------------

# Priority order within each row matters:
#   layout optimizations (channels_last)      → before compile
#   attention optimizations (fa2/fa3/sdpa)    → before qkv_fusion / int8
#   int8 after attention (preserves act range)
#   compile always last (fuses all above)
#
# "flash_attn2" and "flash_attn3" are separate candidates so the agent can
# try FA3 (Hopper) first, then fall back to FA2 (Ampere), then SDPA.

MODEL_FAMILY_OPTIMIZATIONS: Dict[str, List[str]] = {
    # ── Transformer (encoder-only: BERT, RoBERTa, DistilBERT) ────────────────
    # No 4-bit quantization for encoders — accuracy degrades too much at 4-bit
    "transformer_encoder": [
        "flash_attn3",    # Hopper only — gated by HardwareProfile
        "flash_attn2",    # Ampere+ — gated by HardwareProfile
        "sdpa",           # PyTorch native SDPA (always available)
        "qkv_fusion",     # fuse Q/K/V projections into one batched matmul
        "int8",           # regime-gated (seq>=1024, batch×seq<=4096)
        "fp8",            # Hopper only — gated by HardwareProfile
        "compile",
    ],

    # ── Transformer (decoder-only: GPT-2, LLaMA, Mistral) ────────────────────
    # awq_4bit/gptq_4bit: reduce weight bits → converts compute-bound → memory-bound
    "transformer_decoder": [
        "flash_attn3",
        "flash_attn2",
        "sdpa",
        "qkv_fusion",
        "awq_4bit",       # AWQ 4-bit — best quality, requires calibration data
        "gptq_4bit",      # GPTQ 4-bit — fallback if AWQ unavailable
        "int8",
        "fp8",
        "compile",
    ],

    # ── MoE (Mixtral-8x7B) — decoder + expert routing ────────────────────────
    "moe": [
        "moe_optimize",   # compile router + expert prefetch
        "flash_attn3",
        "flash_attn2",
        "sdpa",
        "awq_4bit",
        "gptq_4bit",
        "compile",
    ],

    # ── CNN (ResNet, VGG, EfficientNet, ConvNeXt) ─────────────────────────────
    "cnn": [
        "channels_last",  # NHWC layout: reduces memory traffic for conv
        "compile",
    ],

    # ── Vision Transformer (ViT, DeiT, Swin) ─────────────────────────────────
    "vision_transformer": [
        "flash_attn3",
        "flash_attn2",
        "sdpa",
        "qkv_fusion",
        "channels_last",  # patch embedding is a Conv2d
        "compile",
    ],

    # ── MLP (linear-only: no attention, no conv) ──────────────────────────────
    "mlp": [
        "int8",
        "fp8",
        "compile",
    ],

    # ── Recurrent (LSTM, GRU) ─────────────────────────────────────────────────
    "recurrent": [
        "compile",
    ],

    # ── Fallback: generic model (unknown structure) ───────────────────────────
    "unknown": [
        "channels_last",
        "sdpa",
        "int8",
        "compile",
    ],
}


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def detect_model_type(model: nn.Module) -> str:
    """
    Infer the model family string from class name + module structure.

    Returns one of the keys in MODEL_FAMILY_OPTIMIZATIONS:
      "transformer_encoder" | "transformer_decoder" | "cnn" |
      "vision_transformer"  | "mlp"                 | "recurrent" |
      "unknown"
    """
    # 1. Try known class names first (fast path, unambiguous)
    family = _detect_from_class_name(model)
    if family != "unknown":
        return family

    # 2. Fall back to structural analysis
    return _detect_from_structure(model)


def get_applicable_optimizations(
    model_family: str,
    hw: Any,                              # HardwareProfile
    tried: Optional[set] = None,
) -> List[str]:
    """
    Return the ordered candidate list for (model_family, hardware), minus any
    candidates in `tried`.

    Hardware filtering:
      flash_attn3 → hw.supports_flash_attn3
      flash_attn2 → hw.supports_flash_attn2
      fp8         → hw.supports_fp8
      int8        → hw.supports_int8
      channels_last, sdpa, qkv_fusion, compile → always included

    Args:
        model_family: Return value of detect_model_type().
        hw:           HardwareProfile from detect_hardware().
        tried:        Set of candidate names already attempted this session.

    Returns:
        Ordered list of applicable, untried candidate names.
    """
    if tried is None:
        tried = set()

    all_candidates = MODEL_FAMILY_OPTIMIZATIONS.get(
        model_family,
        MODEL_FAMILY_OPTIMIZATIONS["unknown"],
    )

    result = []
    for c in all_candidates:
        if c in tried:
            continue
        if not _is_supported(c, hw):
            continue
        result.append(c)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_supported(candidate: str, hw: Any) -> bool:
    """Check whether the hardware supports this optimization candidate."""
    if candidate == "flash_attn3":
        return getattr(hw, "supports_flash_attn3", False)
    if candidate == "flash_attn2":
        return getattr(hw, "supports_flash_attn2", False)
    if candidate == "fp8":
        return getattr(hw, "supports_fp8", False)
    if candidate == "int8":
        return getattr(hw, "supports_int8", True)
    if candidate == "channels_last":
        return getattr(hw, "supports_channels_last", True)
    # awq_4bit, gptq_4bit, moe_optimize, sdpa, qkv_fusion, compile — always supported
    # (AWQ/GPTQ gate themselves at runtime if package not installed)
    return True


def _detect_from_class_name(model: nn.Module) -> str:
    """
    Map common HuggingFace / torchvision class names to family strings.
    Returns "unknown" if name isn't recognized.
    """
    cls_name = type(model).__name__.lower()

    # MoE (Mixtral) — check before generic decoder keywords
    moe_keywords = ["mixtral", "moe", "sparsemoe", "switchformer"]
    for kw in moe_keywords:
        if kw in cls_name:
            return "moe"

    # Decoder-only transformers
    decoder_keywords = [
        "gpt2", "gptj", "gptneo", "gptneox",
        "llama", "mistral", "falcon", "mpt", "bloom",
        "opt", "codegen", "phi", "gemma", "qwen",
        "starcoder", "replit", "causal",
    ]
    for kw in decoder_keywords:
        if kw in cls_name:
            return "transformer_decoder"

    # Encoder-only transformers
    encoder_keywords = [
        "bert", "roberta", "distilbert", "albert", "electra",
        "deberta", "xlm", "camembert", "flaubert",
    ]
    for kw in encoder_keywords:
        if kw in cls_name:
            return "transformer_encoder"

    # Vision transformers
    vit_keywords = ["vit", "deit", "swin", "beit", "dino"]
    for kw in vit_keywords:
        if kw in cls_name:
            return "vision_transformer"

    # CNNs
    cnn_keywords = [
        "resnet", "vgg", "alexnet", "squeezenet", "densenet",
        "efficientnet", "mobilenet", "regnet", "convnext",
        "mnasnet", "shufflenet",
    ]
    for kw in cnn_keywords:
        if kw in cls_name:
            return "cnn"

    # Recurrent
    if "lstm" in cls_name or "gru" in cls_name or "rnn" in cls_name:
        return "recurrent"

    return "unknown"


def _detect_from_structure(model: nn.Module) -> str:
    """
    Fallback: inspect submodule types / names to infer family.
    Called only when class name doesn't match any known pattern.
    """
    has_attention  = False
    has_conv       = False
    has_causal_mask = False
    has_cross_attn = False
    has_rnn        = False
    module_names   = []

    for name, mod in model.named_modules():
        mod_type = type(mod).__name__.lower()
        mod_name = name.lower()

        module_names.append(mod_name)

        # Attention detection
        if "attention" in mod_name or "attention" in mod_type:
            has_attention = True
        if "multiheadattention" in mod_type:
            has_attention = True

        # Causal (decoder) detection
        if "causal" in mod_name or "causal" in mod_type:
            has_causal_mask = True
        if "is_decoder" in mod_name:
            has_causal_mask = True

        # Cross attention → likely seq2seq or encoder-decoder
        if "crossattention" in mod_type or "cross_attention" in mod_name:
            has_cross_attn = True

        # Conv detection
        if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            has_conv = True

        # RNN detection
        if isinstance(mod, (nn.LSTM, nn.GRU, nn.RNN)):
            has_rnn = True

    if has_rnn:
        return "recurrent"

    if has_attention:
        # Disambiguate decoder vs encoder: look for causal mask clues or
        # "masked_self_attn", "causal", "is_decoder" in module names
        full_name_str = " ".join(module_names)
        if has_causal_mask or "causal" in full_name_str or "masked" in full_name_str:
            return "transformer_decoder"

        # Conv + attention → vision transformer (patch embed is Conv2d)
        if has_conv:
            return "vision_transformer"

        return "transformer_encoder"

    if has_conv:
        return "cnn"

    # No attention, no conv, no rnn → assume MLP
    return "mlp"
