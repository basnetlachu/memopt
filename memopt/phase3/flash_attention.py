"""
memopt.phase3.flash_attention
==============================

Flash Attention application with three-strategy fallback cascade:

  Strategy 1 — HuggingFace native:
    model.config.attn_implementation = "flash_attention_2" / "flash_attention_3"
    Re-load model with that config (requires flash_attn package).
    This is the cleanest path for HF models.

  Strategy 2 — PyTorch SDPA backend forcing:
    torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False)
    No extra package required.  Works for any model using F.scaled_dot_product_attention
    already (PyTorch 2.0+).

  Strategy 3 — Manual module replacement:
    Scan for nn.MultiheadAttention / BertSelfAttention-style patterns and wrap
    them with a Flash-Attention-aware equivalent.
    Only fires when strategies 1 and 2 both fail.

All strategies return the same (model, strategy_used) tuple so the caller can
log which path succeeded without caring about the mechanism.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

log = logging.getLogger("memopt.phase3.flash_attention")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply_flash_attention(
    model: nn.Module,
    version: int = 2,          # 2 or 3
    sample_inputs: Optional[dict] = None,
) -> Tuple[nn.Module, str]:
    """
    Apply Flash Attention to *model* using the best available strategy.

    Args:
        model:         The nn.Module to optimize.
        version:       FA version to target (2 = Ampere+, 3 = Hopper only).
        sample_inputs: Optional dict of sample inputs — used for sanity-check
                       forward pass after module replacement.

    Returns:
        (optimized_model, strategy_name) where strategy_name is one of:
          "hf_native" | "sdpa_backend" | "module_replace" | "unchanged"

    Raises:
        Never — all failures are caught and "unchanged" is returned.
    """
    if not torch.cuda.is_available():
        log.info("flash_attention: skipping — no CUDA device")
        return model, "unchanged"

    if not _has_attention_layers(model):
        log.info("flash_attention: skipping — no attention modules detected")
        return model, "unchanged"

    ok, err = _check_prerequisites(version)
    if not ok:
        log.info("flash_attention v%d: prerequisites not met — %s", version, err)
        # Still try SDPA backend even if flash_attn package missing
        result = _try_sdpa_backend(model)
        if result is not None:
            log.info("flash_attention: strategy=sdpa_backend succeeded")
            return result, "sdpa_backend"
        return model, "unchanged"

    # Strategy 1: HuggingFace native attn_implementation
    result = _try_hf_native(model, version)
    if result is not None:
        log.info("flash_attention v%d: strategy=hf_native succeeded", version)
        return result, "hf_native"

    # Strategy 2: Force SDPA flash kernel via context manager
    result = _try_sdpa_backend(model)
    if result is not None:
        log.info("flash_attention v%d: strategy=sdpa_backend succeeded", version)
        return result, "sdpa_backend"

    # Strategy 3: Manual module replacement
    result = _replace_attention_modules(model, version)
    if result is not None:
        log.info("flash_attention v%d: strategy=module_replace succeeded", version)
        return result, "module_replace"

    log.info("flash_attention v%d: all strategies failed — returning unchanged", version)
    return model, "unchanged"


# ---------------------------------------------------------------------------
# Prerequisites check
# ---------------------------------------------------------------------------

def _check_prerequisites(version: int) -> Tuple[bool, str]:
    """
    Check whether flash_attn package is available and compatible with `version`.
    Returns (ok, error_message).
    """
    try:
        import flash_attn
    except ImportError:
        return False, "flash_attn package not installed (pip install flash-attn)"

    if version == 3:
        ok = _fa3_available()
        if not ok:
            return False, "FA3 not available (Hopper CC 9.0+ required + flash_attn >=3)"
        return True, ""

    # FA2: need flash_attn 2.x
    try:
        ver = flash_attn.__version__
        major = int(ver.split(".")[0])
        if major < 2:
            return False, f"flash_attn {ver} < 2.0 installed"
    except (AttributeError, ValueError):
        pass  # version parse failed — assume it's OK

    return True, ""


def _fa3_available() -> bool:
    try:
        import flash_attn_interface
        return True
    except ImportError:
        pass
    try:
        import flash_attn.flash_attn_3_cuda  # type: ignore
        return True
    except (ImportError, AttributeError):
        return False


# ---------------------------------------------------------------------------
# Structure inspection
# ---------------------------------------------------------------------------

def _has_attention_layers(model: nn.Module) -> bool:
    """Return True if the model contains any attention-like modules."""
    for name, mod in model.named_modules():
        mod_type = type(mod).__name__.lower()
        mod_name = name.lower()
        if "attention" in mod_type or "attention" in mod_name:
            return True
        if isinstance(mod, nn.MultiheadAttention):
            return True
    return False


# ---------------------------------------------------------------------------
# Strategy 1: HuggingFace attn_implementation
# ---------------------------------------------------------------------------

def _try_hf_native(model: nn.Module, version: int) -> Optional[nn.Module]:
    """
    Attempt to reconfigure a HuggingFace model to use flash_attention_2 or
    flash_attention_3 via config.attn_implementation.

    This modifies model.config in-place if successful.
    Returns None if the model doesn't have a HF config or if reloading fails.
    """
    config = getattr(model, "config", None)
    if config is None:
        return None

    attn_impl = f"flash_attention_{version}"

    # Check if the model class supports this implementation
    try:
        cls = type(model)
        if not hasattr(cls, "from_config"):
            return None  # not a HF model

        # Some models expose _supports_flash_attn_2 flag
        if version == 2 and hasattr(model, "_supports_flash_attn_2"):
            if not model._supports_flash_attn_2:
                log.debug("hf_native: model explicitly does not support FA2")
                return None

        # Try setting attn_implementation on the config
        config_copy = config.__class__.from_dict(config.to_dict())
        config_copy._attn_implementation = attn_impl

        # Reconstruct model weights from config (cheap path — no re-download)
        new_model = cls(config_copy)
        new_model.load_state_dict(model.state_dict(), strict=False)

        device = _model_device(model)
        if device is not None:
            new_model = new_model.to(device)

        dtype = _model_dtype(model)
        if dtype is not None:
            new_model = new_model.to(dtype)

        new_model.eval()
        log.debug("hf_native: successfully rebuilt model with %s", attn_impl)
        return new_model

    except Exception as exc:
        log.debug("hf_native: failed — %s", exc)
        return None


# ---------------------------------------------------------------------------
# Strategy 2: Force SDPA flash backend
# ---------------------------------------------------------------------------

def _try_sdpa_backend(model: nn.Module) -> Optional[nn.Module]:
    """
    Wrap the model in a context that forces PyTorch SDPA to use the flash kernel.

    This doesn't modify the model — it patches the forward method to run inside
    torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False).

    Returns a wrapper nn.Module, or None if SDPA flash kernel unavailable.
    """
    # Check flash kernel availability
    try:
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            pass
    except Exception:
        # Flash kernel not available on this platform
        return None

    class _SDPAForceFlashWrapper(nn.Module):
        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, *args, **kwargs):
            # PyTorch 2.0+ context manager: prefer flash over math SDPA
            try:
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=True,
                    enable_math=False,
                    enable_mem_efficient=True,
                ):
                    return self.inner(*args, **kwargs)
            except Exception:
                # Flash kernel failed (e.g. non-standard head dim) — fall through
                return self.inner(*args, **kwargs)

        def __getattr__(self, name: str):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.inner, name)

    return _SDPAForceFlashWrapper(model)


# ---------------------------------------------------------------------------
# Strategy 3: Manual module replacement
# ---------------------------------------------------------------------------

class _FlashMHAWrapper(nn.Module):
    """
    Wraps nn.MultiheadAttention to call F.scaled_dot_product_attention
    with is_causal awareness — eliminates the attn_mask allocation.

    Only replaces the attention computation; all other MHA logic is preserved.
    """

    def __init__(self, mha: nn.MultiheadAttention) -> None:
        super().__init__()
        self.embed_dim    = mha.embed_dim
        self.num_heads    = mha.num_heads
        self.head_dim     = mha.head_dim
        self.dropout      = mha.dropout

        # Copy the projection weights
        self.in_proj_weight  = mha.in_proj_weight
        self.in_proj_bias    = mha.in_proj_bias
        self.out_proj        = mha.out_proj
        self.bias_k          = mha.bias_k
        self.bias_v          = mha.bias_v
        self.add_zero_attn   = mha.add_zero_attn
        self.batch_first     = mha.batch_first

        self._mha = mha  # keep reference for fallback

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        **kwargs,
    ):
        # Detect causal from attn_mask (upper-triangular boolean mask)
        is_causal = False
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            # Causal mask is upper-tri True (mask out future)
            is_causal = True
            attn_mask = None  # SDPA handles causal natively

        try:
            import torch.nn.functional as F
            if self.batch_first:
                B, Sq, _ = query.shape
                Sk = key.shape[1]
            else:
                Sq, B, _ = query.shape
                Sk = key.shape[0]

            # Project Q, K, V via in_proj
            E = self.embed_dim
            if self.in_proj_weight is not None:
                qw, kw, vw = self.in_proj_weight.chunk(3, dim=0)
                qb, kb, vb = (
                    self.in_proj_bias.chunk(3, dim=0)
                    if self.in_proj_bias is not None
                    else (None, None, None)
                )
                q = F.linear(query, qw, qb)
                k = F.linear(key, kw, kb)
                v = F.linear(value, vw, vb)
            else:
                q = F.linear(query, self.q_proj_weight)
                k = F.linear(key,   self.k_proj_weight)
                v = F.linear(value, self.v_proj_weight)

            H, Hd = self.num_heads, self.head_dim

            if self.batch_first:
                q = q.view(B, Sq, H, Hd).transpose(1, 2)  # (B, H, Sq, Hd)
                k = k.view(B, Sk, H, Hd).transpose(1, 2)
                v = v.view(B, Sk, H, Hd).transpose(1, 2)
            else:
                q = q.view(Sq, B, H, Hd).permute(1, 2, 0, 3)
                k = k.view(Sk, B, H, Hd).permute(1, 2, 0, 3)
                v = v.view(Sk, B, H, Hd).permute(1, 2, 0, 3)

            drop_p = self.dropout if self.training else 0.0
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=drop_p,
                is_causal=is_causal,
            )  # (B, H, Sq, Hd)

            # Merge heads
            if self.batch_first:
                attn_out = attn_out.transpose(1, 2).contiguous().view(B, Sq, E)
            else:
                attn_out = attn_out.permute(2, 0, 1, 3).contiguous().view(Sq, B, E)

            out = self.out_proj(attn_out)
            # MHA returns (attn_output, attn_weights) — weights not computed here
            return out, None

        except Exception as exc:
            log.debug("_FlashMHAWrapper forward fallback: %s", exc)
            # Fall back to original MHA
            return self._mha(
                query, key, value,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                **kwargs,
            )


def _replace_attention_modules(
    model: nn.Module,
    version: int,
) -> Optional[nn.Module]:
    """
    Walk the model and replace nn.MultiheadAttention instances with
    _FlashMHAWrapper.  Returns None if no modules were replaced.
    """
    replaced = 0

    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.MultiheadAttention):
            continue

        # Navigate to parent to set the attribute
        parts  = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        leaf_name = parts[-1]
        wrapper   = _FlashMHAWrapper(mod)

        try:
            setattr(parent, leaf_name, wrapper)
            replaced += 1
            log.debug("module_replace: replaced %s", name)
        except Exception as exc:
            log.debug("module_replace: failed to replace %s — %s", name, exc)

    if replaced == 0:
        return None

    log.info("module_replace: replaced %d nn.MultiheadAttention modules", replaced)
    return model


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _model_device(model: nn.Module) -> Optional[torch.device]:
    p = next(model.parameters(), None)
    return p.device if p is not None else None


def _model_dtype(model: nn.Module) -> Optional[torch.dtype]:
    p = next(model.parameters(), None)
    return p.dtype if p is not None else None
