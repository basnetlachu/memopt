"""
memopt.utils.gpu_info
=====================

Lightweight shared CUDA/ROCm detection helper.

Returns a plain dataclass with the raw hardware identity so callers
don't each duplicate torch.cuda API calls.  Higher-level modules
(hardware_detector, portability_layer) derive their own richer
structs from this raw data.

Always succeeds — never raises.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CudaInfo:
    """Raw hardware identity snapshot."""
    device_name:    str    # e.g. "NVIDIA A100-SXM4-80GB"
    compute_cap:    str    # e.g. "8.0"  (empty string on non-CUDA)
    total_memory_gb: float # device total memory in GB (0.0 if unknown)
    is_rocm:        bool   # True when running on AMD ROCm
    is_available:   bool   # True when a usable GPU was detected


def get_cuda_info(device: int = 0) -> CudaInfo:
    """
    Return a CudaInfo for *device* (default 0).

    Detection order:
      1. NVIDIA CUDA
      2. AMD ROCm (torch.version.hip is set)
      3. CPU / no GPU  →  CudaInfo with is_available=False
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return CudaInfo(
                device_name="", compute_cap="",
                total_memory_gb=0.0, is_rocm=False, is_available=False,
            )

        name  = torch.cuda.get_device_name(device)
        props = torch.cuda.get_device_properties(device)
        total_gb = props.total_memory / (1024 ** 3)

        is_rocm = bool(getattr(torch.version, "hip", None))

        if is_rocm:
            cap = ""
        else:
            major, minor = torch.cuda.get_device_capability(device)
            cap = f"{major}.{minor}"

        return CudaInfo(
            device_name=name,
            compute_cap=cap,
            total_memory_gb=round(total_gb, 1),
            is_rocm=is_rocm,
            is_available=True,
        )

    except Exception:
        return CudaInfo(
            device_name="", compute_cap="",
            total_memory_gb=0.0, is_rocm=False, is_available=False,
        )
