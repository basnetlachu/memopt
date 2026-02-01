"""
GPU-Specific Tuning Profiles - Phase 4

Provides optimized default settings for different GPU architectures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch

logger = logging.getLogger("memopt")


@dataclass
class GPUProfile:
    """Tuning profile for a specific GPU architecture."""
    name: str
    compute_capability: tuple  # (major, minor)

    # Memory characteristics
    l2_cache_mb: float
    memory_bandwidth_gbps: float
    shared_memory_per_sm_kb: float

    # Optimization thresholds
    min_tensor_size_for_compile: int  # Skip compile for tiny tensors
    preferred_compile_mode: str
    enable_tf32: bool
    enable_cudnn_benchmark: bool

    # Attribution thresholds (adjusted for GPU)
    cache_thrashing_threshold: float  # Working set / L2 ratio
    memory_bound_threshold: float  # Stall ratio to consider memory-bound


# Pre-defined profiles for common GPUs
_GPU_PROFILES: Dict[str, GPUProfile] = {
    # Ampere
    "A100": GPUProfile(
        name="A100",
        compute_capability=(8, 0),
        l2_cache_mb=40,
        memory_bandwidth_gbps=2039,
        shared_memory_per_sm_kb=164,
        min_tensor_size_for_compile=1024 * 1024,  # 1MB
        preferred_compile_mode="reduce-overhead",
        enable_tf32=True,
        enable_cudnn_benchmark=True,
        cache_thrashing_threshold=1.5,  # A100 has big L2, higher threshold
        memory_bound_threshold=0.5,  # A100 is bandwidth-rich
    ),
    "A10": GPUProfile(
        name="A10",
        compute_capability=(8, 6),
        l2_cache_mb=24,
        memory_bandwidth_gbps=600,
        shared_memory_per_sm_kb=100,
        min_tensor_size_for_compile=512 * 1024,
        preferred_compile_mode="default",
        enable_tf32=True,
        enable_cudnn_benchmark=True,
        cache_thrashing_threshold=1.2,
        memory_bound_threshold=0.6,
    ),

    # Hopper
    "H100": GPUProfile(
        name="H100",
        compute_capability=(9, 0),
        l2_cache_mb=50,
        memory_bandwidth_gbps=3350,
        shared_memory_per_sm_kb=228,
        min_tensor_size_for_compile=2 * 1024 * 1024,
        preferred_compile_mode="max-autotune",
        enable_tf32=True,
        enable_cudnn_benchmark=True,
        cache_thrashing_threshold=2.0,
        memory_bound_threshold=0.4,
    ),

    # Ada Lovelace
    "RTX4090": GPUProfile(
        name="RTX4090",
        compute_capability=(8, 9),
        l2_cache_mb=72,
        memory_bandwidth_gbps=1008,
        shared_memory_per_sm_kb=100,
        min_tensor_size_for_compile=1024 * 1024,
        preferred_compile_mode="reduce-overhead",
        enable_tf32=True,
        enable_cudnn_benchmark=True,
        cache_thrashing_threshold=1.8,
        memory_bound_threshold=0.55,
    ),
    "RTX4080": GPUProfile(
        name="RTX4080",
        compute_capability=(8, 9),
        l2_cache_mb=64,
        memory_bandwidth_gbps=717,
        shared_memory_per_sm_kb=100,
        min_tensor_size_for_compile=512 * 1024,
        preferred_compile_mode="default",
        enable_tf32=True,
        enable_cudnn_benchmark=True,
        cache_thrashing_threshold=1.5,
        memory_bound_threshold=0.6,
    ),

    # Turing
    "T4": GPUProfile(
        name="T4",
        compute_capability=(7, 5),
        l2_cache_mb=4,
        memory_bandwidth_gbps=320,
        shared_memory_per_sm_kb=64,
        min_tensor_size_for_compile=256 * 1024,
        preferred_compile_mode="default",
        enable_tf32=False,  # Turing doesn't have TF32
        enable_cudnn_benchmark=True,
        cache_thrashing_threshold=0.8,  # Small L2
        memory_bound_threshold=0.7,
    ),

    # Volta
    "V100": GPUProfile(
        name="V100",
        compute_capability=(7, 0),
        l2_cache_mb=6,
        memory_bandwidth_gbps=900,
        shared_memory_per_sm_kb=96,
        min_tensor_size_for_compile=512 * 1024,
        preferred_compile_mode="default",
        enable_tf32=False,
        enable_cudnn_benchmark=True,
        cache_thrashing_threshold=0.9,
        memory_bound_threshold=0.6,
    ),
}


def _create_default_profile(gpu_name: str, props) -> GPUProfile:
    """Create a default profile based on compute capability."""
    major, minor = props.major, props.minor

    # Estimate L2 cache
    if major >= 9:
        l2_mb = 50
    elif major >= 8:
        l2_mb = 40
    else:
        l2_mb = 6

    # Try to get actual L2 if available
    if hasattr(props, 'l2_cache_size') and props.l2_cache_size > 0:
        l2_mb = props.l2_cache_size / (1024 * 1024)

    return GPUProfile(
        name=gpu_name,
        compute_capability=(major, minor),
        l2_cache_mb=l2_mb,
        memory_bandwidth_gbps=1000,  # Conservative default
        shared_memory_per_sm_kb=props.max_shared_memory_per_block // 1024,
        min_tensor_size_for_compile=512 * 1024,
        preferred_compile_mode="default",
        enable_tf32=major >= 8,
        enable_cudnn_benchmark=True,
        cache_thrashing_threshold=1.0,
        memory_bound_threshold=0.6,
    )


def get_gpu_profile(device: int = 0) -> GPUProfile:
    """
    Get the tuning profile for the current GPU.

    Args:
        device: CUDA device index

    Returns:
        GPUProfile for the device
    """
    if not torch.cuda.is_available():
        # Return a conservative CPU-like profile
        return GPUProfile(
            name="CPU",
            compute_capability=(0, 0),
            l2_cache_mb=32,
            memory_bandwidth_gbps=50,
            shared_memory_per_sm_kb=0,
            min_tensor_size_for_compile=0,
            preferred_compile_mode="default",
            enable_tf32=False,
            enable_cudnn_benchmark=False,
            cache_thrashing_threshold=1.0,
            memory_bound_threshold=0.7,
        )

    props = torch.cuda.get_device_properties(device)
    gpu_name = props.name

    # Try to match a known profile
    for key, profile in _GPU_PROFILES.items():
        if key in gpu_name:
            logger.debug(f"Using GPU profile: {key}")
            return profile

    # Create a default profile
    logger.debug(f"Creating default profile for: {gpu_name}")
    return _create_default_profile(gpu_name, props)


def apply_gpu_profile(profile: GPUProfile):
    """
    Apply GPU profile settings to PyTorch.

    Args:
        profile: GPUProfile to apply
    """
    if not torch.cuda.is_available():
        return

    # TF32 settings
    torch.backends.cuda.matmul.allow_tf32 = profile.enable_tf32
    torch.backends.cudnn.allow_tf32 = profile.enable_tf32

    # cuDNN benchmark
    torch.backends.cudnn.benchmark = profile.enable_cudnn_benchmark

    logger.info(f"Applied GPU profile: {profile.name} "
               f"(TF32={profile.enable_tf32}, cudnn_bench={profile.enable_cudnn_benchmark})")


def get_profile_summary() -> str:
    """Get a summary of the current GPU profile."""
    profile = get_gpu_profile()

    lines = [
        f"GPU Profile: {profile.name}",
        f"  Compute: {profile.compute_capability[0]}.{profile.compute_capability[1]}",
        f"  L2 Cache: {profile.l2_cache_mb:.1f} MB",
        f"  Bandwidth: {profile.memory_bandwidth_gbps:.0f} GB/s",
        f"  TF32: {profile.enable_tf32}",
        f"  Compile Mode: {profile.preferred_compile_mode}",
    ]
    return "\n".join(lines)
