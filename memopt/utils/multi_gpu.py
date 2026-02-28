"""
memopt.utils.multi_gpu
========================

Multi-GPU support via PyTorch FSDP (Fully Sharded Data Parallel).

Design:
  - Single-GPU path is COMPLETELY UNCHANGED (zero-impact when num_gpus=1)
  - Multi-GPU only activates when: multiple GPUs detected AND multi_gpu=True
    AND model > 70% of single-GPU VRAM
  - FSDP shards parameters across GPUs so each holds 1/N of the weights
  - All-gather happens automatically before each forward layer

Target use case: LLaMA-70B (140GB) across 2x A100 80GB (160GB total).

Not supported:
  - Training (use DDP/FSDP directly for training)
  - Models that don't support FSDP wrapping (checked at runtime)
  - Gradient accumulation
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn

log = logging.getLogger("memopt.utils.multi_gpu")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_available_gpus() -> int:
    """Return count of available CUDA GPUs. Returns 0 if CUDA unavailable."""
    try:
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.device_count()
    except Exception:
        return 0


def get_multi_gpu_memory_gb() -> float:
    """
    Return total VRAM in GB across ALL available CUDA GPUs.
    Returns 0.0 if CUDA unavailable.
    """
    try:
        if not torch.cuda.is_available():
            return 0.0
        total = 0
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total += props.total_memory
        return total / 1e9
    except Exception:
        return 0.0


def is_multi_gpu_needed(
    model: nn.Module,
    hardware=None,    # HardwareProfile from detect_hardware()
    threshold: float = 0.70,
) -> bool:
    """
    Check if model requires multi-GPU to run.

    Returns True if model parameter size > threshold × single-GPU VRAM.

    Args:
        model:     nn.Module to check.
        hardware:  HardwareProfile (for VRAM info). If None, uses torch API.
        threshold: Fraction of single-GPU VRAM above which multi-GPU is needed
                   (default 0.70 = 70%).

    Returns:
        True if model is too large for a single GPU.
    """
    try:
        # Model size in bytes
        model_bytes = sum(
            p.numel() * p.element_size() for p in model.parameters()
        )

        # Single GPU VRAM
        if hardware is not None and hasattr(hardware, "vram_gb"):
            vram_bytes = hardware.vram_gb * 1e9
        elif torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_bytes = props.total_memory
        else:
            return False  # No CUDA — irrelevant

        needed = model_bytes > threshold * vram_bytes
        log.debug(
            "is_multi_gpu_needed: model=%.1fGB, single_gpu=%.1fGB, needed=%s",
            model_bytes / 1e9, vram_bytes / 1e9, needed,
        )
        return needed

    except Exception as exc:
        log.debug("is_multi_gpu_needed failed: %s", exc)
        return False


def apply_fsdp(
    model: nn.Module,
    num_gpus: Optional[int] = None,
    cpu_offload: bool = False,
) -> Tuple[nn.Module, bool]:
    """
    Wrap model in FSDP for multi-GPU inference.

    Requirements (all must be true; returns (model, False) if any fail):
      1. Multiple CUDA GPUs available (or num_gpus specified > 1)
      2. torch.distributed initialized (or can be initialized here)
      3. FSDP available (PyTorch >= 1.12)

    Args:
        model:       nn.Module (should be on CPU before FSDP wrapping).
        num_gpus:    Override GPU count (None = auto-detect).
        cpu_offload: Enable CPU offload for parameters not in use.
                     Reduces VRAM further but adds PCIe transfer overhead.

    Returns:
        (fsdp_model, applied: bool)
        fsdp_model = original model if applied=False.
    """
    available_gpus = detect_available_gpus() if num_gpus is None else num_gpus

    if available_gpus <= 1:
        log.info("apply_fsdp: only %d GPU(s) available — skipping", available_gpus)
        return model, False

    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import CPUOffload, ShardingStrategy
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    except ImportError:
        log.warning("apply_fsdp: FSDP not available (requires PyTorch >= 1.12)")
        return model, False

    try:
        # Initialize distributed if not already done
        if not torch.distributed.is_initialized():
            _init_distributed(available_gpus)

        if not torch.distributed.is_initialized():
            log.warning("apply_fsdp: could not initialize torch.distributed")
            return model, False

        rank = torch.distributed.get_rank()

        # Configure FSDP
        fsdp_kwargs = {
            "sharding_strategy": ShardingStrategy.FULL_SHARD,
        }

        if cpu_offload:
            fsdp_kwargs["cpu_offload"] = CPUOffload(offload_params=True)

        # Try to auto-wrap transformer layers
        auto_wrap_cls = _detect_transformer_block_class(model)
        if auto_wrap_cls is not None:
            import functools
            fsdp_kwargs["auto_wrap_policy"] = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={auto_wrap_cls},
            )

        fsdp_model = FSDP(model, **fsdp_kwargs)
        fsdp_model.eval()

        log.info(
            "apply_fsdp: FSDP applied across %d GPUs (rank=%d, cpu_offload=%s)",
            available_gpus, rank, cpu_offload,
        )
        return fsdp_model, True

    except Exception as exc:
        log.warning("apply_fsdp failed: %s", exc)
        return model, False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _init_distributed(world_size: int) -> None:
    """
    Initialize torch.distributed for single-node multi-GPU.
    Uses nccl backend (GPU-to-GPU); gloo fallback for CPU.
    """
    try:
        # Single-node: each process gets one GPU via rank
        # For inference we typically run as a single process with FSDP
        # using the 'spawn' approach or environment variables
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", str(world_size))
        os.environ.setdefault("LOCAL_RANK", "0")

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=world_size,
            rank=0,
        )
        log.info("torch.distributed initialized: backend=%s world_size=%d", backend, world_size)
    except Exception as exc:
        log.debug("_init_distributed failed: %s", exc)


def _detect_transformer_block_class(model: nn.Module) -> Optional[type]:
    """
    Try to find the repeating transformer block class for auto_wrap_policy.

    Looks for common naming patterns: 'DecoderLayer', 'EncoderLayer',
    'TransformerBlock', 'Block', 'Layer'.

    Returns the class object if found, None otherwise.
    """
    target_names = [
        "decoderlayer", "encoderlayer", "transformerblock",
        "block", "llamadecoderlayer", "mistraldecoderlayer",
        "gpt2block", "bertlayer",
    ]

    seen_classes: dict = {}
    for name, module in model.named_modules():
        cls_name = type(module).__name__.lower()
        for target in target_names:
            if target in cls_name:
                seen_classes[type(module)] = seen_classes.get(type(module), 0) + 1

    if not seen_classes:
        return None

    # Return most frequent candidate (most likely the repeating block)
    return max(seen_classes, key=seen_classes.get)
