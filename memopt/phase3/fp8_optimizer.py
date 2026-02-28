"""
memopt.phase3.fp8_optimizer
=============================

FP8 quantization via NVIDIA Transformer Engine (te).

Hardware gate: Hopper (CC 9.0+) ONLY.
FP8 requires H100 / H200 hardware-native 8-bit floating point compute units.
Attempting FP8 on Ampere or older silently falls back to FP16 in TE, defeating the
purpose — so we guard explicitly.

Two modes:
  apply_fp8(model)           — static quantization: replace nn.Linear with
                               te.Linear (Transformer Engine linear layer with
                               FP8 weight storage + FP8 matmul).
  apply_fp8_autocast(model)  — wrap model.forward() in te.fp8_autocast() context.
                               Non-destructive: model architecture unchanged;
                               TE intercepts the GEMM calls at runtime.
                               This is the safer fallback when te.Linear
                               replacement is not applicable.

Both return (optimized_model, mode_used) where mode_used is one of:
  "te_linear" | "te_autocast" | "unchanged"
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

log = logging.getLogger("memopt.phase3.fp8_optimizer")


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def apply_fp8(
    model: nn.Module,
    prefer_autocast: bool = False,
) -> Tuple[nn.Module, str]:
    """
    Apply FP8 optimization to *model* on Hopper hardware.

    Strategy:
      1. Try te.Linear replacement (prefer_autocast=False default).
      2. Fall back to te.fp8_autocast wrapper if replacement fails.
      3. Return unchanged if Transformer Engine unavailable or non-Hopper.

    Args:
        model:           Model to optimize.
        prefer_autocast: If True, skip te.Linear replacement and go straight
                         to the autocast wrapper (lighter, less invasive).

    Returns:
        (optimized_model, mode_used)
    """
    if not _is_hopper():
        log.info("fp8: skipping — not Hopper hardware (CC 9.0+ required)")
        return model, "unchanged"

    te = _import_transformer_engine()
    if te is None:
        log.info("fp8: skipping — transformer-engine not installed")
        return model, "unchanged"

    if not prefer_autocast:
        result = _replace_linear_with_fp8(model, te)
        if result is not None:
            log.info("fp8: mode=te_linear — replaced %d linear layers", result[1])
            return result[0], "te_linear"

    # Fallback: autocast wrapper
    result = apply_fp8_autocast(model)
    if result[1] != "unchanged":
        return result

    log.info("fp8: all strategies failed — returning unchanged")
    return model, "unchanged"


def apply_fp8_autocast(model: nn.Module) -> Tuple[nn.Module, str]:
    """
    Wrap *model* in a Transformer Engine fp8_autocast context.

    Returns (wrapper, "te_autocast") or (model, "unchanged").
    """
    if not _is_hopper():
        return model, "unchanged"

    te = _import_transformer_engine()
    if te is None:
        return model, "unchanged"

    try:
        fp8_autocast_fn = getattr(te, "fp8_autocast", None)
        if fp8_autocast_fn is None:
            # Older TE API: transformer_engine.pytorch.fp8_autocast
            import transformer_engine.pytorch as te_torch
            fp8_autocast_fn = te_torch.fp8_autocast
    except Exception as exc:
        log.debug("fp8_autocast: cannot locate fp8_autocast function — %s", exc)
        return model, "unchanged"

    class _FP8AutocastWrapper(nn.Module):
        """Thin wrapper: runs model.forward() inside te.fp8_autocast()."""

        def __init__(self, inner: nn.Module, _autocast_fn) -> None:
            super().__init__()
            self.inner         = inner
            self._autocast_fn  = _autocast_fn
            self._fp8_enabled  = True

        def forward(self, *args, **kwargs):
            if self._fp8_enabled:
                try:
                    with self._autocast_fn(enabled=True):
                        return self.inner(*args, **kwargs)
                except Exception as exc:
                    log.warning(
                        "_FP8AutocastWrapper: fp8_autocast failed, disabling — %s", exc
                    )
                    self._fp8_enabled = False

            # FP8 disabled (after failure): plain forward
            return self.inner(*args, **kwargs)

        def __getattr__(self, name: str):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.inner, name)

    wrapper = _FP8AutocastWrapper(model, fp8_autocast_fn)
    log.info("fp8: autocast wrapper applied")
    return wrapper, "te_autocast"


# ---------------------------------------------------------------------------
# Internal: te.Linear replacement
# ---------------------------------------------------------------------------

def _replace_linear_with_fp8(
    model: nn.Module,
    te,
) -> Optional[Tuple[nn.Module, int]]:
    """
    Replace every nn.Linear with transformer_engine.pytorch.Linear.

    TE Linear stores weights in FP8 and executes GEMM in FP8.

    Returns (modified_model, num_replaced), or None if replacement fails.
    """
    try:
        import transformer_engine.pytorch as te_torch
        TELinear = te_torch.Linear
    except Exception as exc:
        log.debug("_replace_linear_with_fp8: cannot import te.Linear — %s", exc)
        return None

    replaced = 0

    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue

        # Navigate to parent
        parts  = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        leaf_name = parts[-1]

        try:
            # Build equivalent TE Linear
            te_linear = TELinear(
                in_features  = mod.in_features,
                out_features = mod.out_features,
                bias         = mod.bias is not None,
            )

            # Copy weights (TE stores weights in a QuantizedTensor internally)
            with torch.no_grad():
                te_linear.weight.data.copy_(mod.weight.data)
                if mod.bias is not None:
                    te_linear.bias.data.copy_(mod.bias.data)

            # Move to same device as original
            device = mod.weight.device
            te_linear = te_linear.to(device)

            setattr(parent, leaf_name, te_linear)
            replaced += 1
            log.debug("te_linear: replaced %s (%dx%d)", name, mod.in_features, mod.out_features)

        except Exception as exc:
            log.debug("te_linear: failed to replace %s — %s", name, exc)

    if replaced == 0:
        return None

    return model, replaced


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _is_hopper() -> bool:
    """Return True only when running on Hopper (CC 9.0+) hardware."""
    if not torch.cuda.is_available():
        return False
    try:
        props = torch.cuda.get_device_properties(0)
        return props.major >= 9
    except Exception:
        return False


def _import_transformer_engine():
    """
    Try to import transformer_engine.  Returns the module or None.
    We accept both the top-level package and the pytorch sub-package.
    """
    try:
        import transformer_engine as te
        return te
    except ImportError:
        pass
    try:
        import transformer_engine.pytorch as te
        return te
    except ImportError:
        pass
    return None
