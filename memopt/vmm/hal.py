"""
Hardware Abstraction Layer (HAL).

Single responsibility: detect which hardware is present and return the
correct backend singleton. Every other VMM module imports `backend` from
here. No other VMM file may touch torch.cuda, torch.hip, or any
hardware-specific import directly.

Detection order: CUDA → ROCm → Unified (CPU / Apple Silicon fallback).
"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


def _detect_backend():
    try:
        import torch

        if torch.cuda.is_available():
            # Distinguish ROCm from CUDA before choosing backend
            if getattr(torch.version, "hip", None) is not None:
                from .backends.rocm_backend import ROCmBackend
                b = ROCmBackend()
                logger.info("VMM HAL: ROCm backend (AMD GPU detected)")
                return b

            from .backends.cuda_backend import CUDABackend
            b = CUDABackend()
            logger.info("VMM HAL: CUDA backend (NVIDIA GPU detected)")
            return b

    except ImportError:
        pass

    from .backends.unified_backend import UnifiedBackend
    b = UnifiedBackend()
    logger.info("VMM HAL: Unified backend (CPU-only or Apple Silicon)")
    return b


# Module-level singleton — instantiated once at first import
backend = _detect_backend()
tiers = backend.detect_tiers()
tier_names: list[str] = [t.name for t in tiers]


def get_backend():
    """Return the active backend. Prefer importing `backend` directly."""
    return backend


__all__ = ["backend", "tiers", "tier_names", "get_backend"]
