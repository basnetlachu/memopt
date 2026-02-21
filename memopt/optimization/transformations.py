"""
Model transformations with regression guard.

``safe_compile`` wraps ``torch.compile`` (via ``selective_compile``) with a
benchmark-before/after harness that automatically rolls back to the original
model if the compiled version regresses beyond a configurable threshold.

Typical usage::

    from memopt.optimization.transformations import safe_compile

    # bottleneck_map: {module_name: BottleneckType}  (from classify_and_rank)
    opt_model, speedup, rolled_back = safe_compile(model, bottleneck_map)
    print(f"speedup={speedup:.2f}x  rolled_back={rolled_back}")
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger("memopt")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_WARMUP_RUNS = 5
_TIMED_RUNS = 20


def _make_dummy_input(model: nn.Module) -> Tuple:
    """
    Create a minimal dummy input for *any* model so we can benchmark it.

    Tries, in order:
    1. ``model.dummy_input`` attribute (user-provided)
    2. Single random float32 tensor (batch=1, size=64) on the model's device
    """
    if hasattr(model, "dummy_input"):
        dummy = model.dummy_input
        if not isinstance(dummy, tuple):
            dummy = (dummy,)
        return dummy

    device = next(model.parameters(), torch.tensor(0.0)).device
    return (torch.randn(1, 64, device=device),)


def _bench(model: nn.Module, inputs: Tuple, use_cuda: bool) -> float:
    """
    Return median latency in milliseconds over *_TIMED_RUNS* runs.

    Uses CUDA events when a GPU is available; falls back to wall-clock time.
    """
    model.eval()

    def _run():
        with torch.no_grad():
            try:
                model(*inputs)
            except Exception:
                # If inputs don't match, fall through — caller handles it
                raise

    if use_cuda:
        # Warmup
        for _ in range(_WARMUP_RUNS):
            _run()
        torch.cuda.synchronize()

        times = []
        for _ in range(_TIMED_RUNS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _run()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))  # ms
    else:
        for _ in range(_WARMUP_RUNS):
            _run()

        times = []
        for _ in range(_TIMED_RUNS):
            t0 = time.perf_counter()
            _run()
            times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return times[len(times) // 2]  # median


def selective_compile(
    model: nn.Module,
    bottleneck_map: Dict[str, object],
) -> nn.Module:
    """
    Compile only MEMORY_BOUND_DRAM sub-modules (avoids wasteful re-compilation
    of compute-bound layers where torch.compile overhead ≥ gain).

    If *bottleneck_map* is empty or no sub-module is MEMORY_BOUND_DRAM, the
    entire model is compiled as a fallback.

    Args:
        model:          Original PyTorch model.
        bottleneck_map: Mapping of qualified module name → BottleneckType
                        (or any object whose ``.value`` or ``str()`` contains
                        ``"memory_bound_dram"``).

    Returns:
        Model with selected sub-modules (or whole model) compiled.
    """
    # Identify sub-modules to compile
    target_names = set()
    for name, btype in bottleneck_map.items():
        btype_str = getattr(btype, "value", str(btype)).lower()
        if "memory_bound_dram" in btype_str:
            target_names.add(name)

    if not target_names:
        logger.info("[selective_compile] No MEMORY_BOUND_DRAM modules; compiling whole model.")
        return torch.compile(model)

    # Compile targeted sub-modules only
    compiled_any = False
    for name, submodule in list(model.named_modules()):
        if name in target_names:
            try:
                compiled = torch.compile(submodule)
                # Walk the parent chain and replace
                parts = name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], compiled)
                compiled_any = True
                logger.info("[selective_compile] Compiled sub-module: %s", name)
            except Exception as exc:
                logger.warning("[selective_compile] Could not compile %s: %s", name, exc)

    if not compiled_any:
        logger.info("[selective_compile] No sub-modules compiled; falling back to whole-model compile.")
        return torch.compile(model)

    return model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def safe_compile(
    model: nn.Module,
    bottleneck_map: Dict[str, object],
    regression_threshold: float = 0.95,
) -> Tuple[nn.Module, float, bool]:
    """
    Compile *model* with a regression guard.

    Steps:
    1. Benchmark baseline latency (5 warmup + 20 timed runs, median).
    2. Deep-copy the model; apply ``selective_compile``.
    3. Benchmark compiled latency.
    4. Compute ``speedup = baseline / compiled``.
    5. If ``speedup < regression_threshold`` → roll back, return original model.
    6. Otherwise return compiled model.

    Args:
        model:                Original PyTorch model (never mutated).
        bottleneck_map:       Sub-module → BottleneckType mapping.
        regression_threshold: Minimum acceptable speedup ratio (default 0.95).
                              A value of 0.95 means "allow up to 5 % regression."

    Returns:
        Tuple of:
        - optimized_model : compiled model if kept, original if rolled back.
        - speedup_ratio   : baseline_ms / compiled_ms  (>1 = faster).
        - rolled_back     : True if regression guard triggered.

    Raises:
        RuntimeError: If benchmarking fails for both baseline and compiled model.
    """
    use_cuda = torch.cuda.is_available() and next(
        (True for p in model.parameters() if p.is_cuda), False
    )

    # Build dummy inputs once
    try:
        inputs = _make_dummy_input(model)
    except StopIteration:
        # No parameters — cannot benchmark; return model unchanged
        logger.warning("[safe_compile] Model has no parameters; skipping compilation.")
        return model, 1.0, False

    # --- Baseline benchmark ---
    try:
        baseline_ms = _bench(model, inputs, use_cuda)
    except Exception as exc:
        logger.error("[safe_compile] Baseline benchmark failed: %s", exc)
        raise RuntimeError(f"safe_compile: baseline benchmark failed: {exc}") from exc

    # --- Compile a copy ---
    compiled_model = selective_compile(copy.deepcopy(model), bottleneck_map)

    # --- Compiled benchmark ---
    try:
        compiled_ms = _bench(compiled_model, inputs, use_cuda)
    except Exception as exc:
        logger.warning("[safe_compile] Compiled benchmark failed (%s); rolling back.", exc)
        return model, 0.0, True

    speedup = baseline_ms / compiled_ms if compiled_ms > 0 else 0.0

    if speedup < regression_threshold:
        logger.warning(
            "[safe_compile] Regression detected: speedup=%.3fx < threshold=%.2f; rolling back.",
            speedup,
            regression_threshold,
        )
        return model, speedup, True

    logger.info(
        "[safe_compile] Compilation accepted: baseline=%.2fms compiled=%.2fms speedup=%.3fx",
        baseline_ms,
        compiled_ms,
        speedup,
    )
    return compiled_model, speedup, False
