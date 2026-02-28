"""
Universal Optimizer — any nn.Module, any NVIDIA GPU.

Three public functions:
  get_ridge_point_flops_per_byte()  → float
  select_optimizations(model, sample_input)  → UniversalPlan
  safe_compile(module, sample_input, ...)  → nn.Module

No hardcoded GPU specs. No flash_attn (ABI broken on PyTorch 2.6+cu124).
No compile without regression guard.
"""

from __future__ import annotations

import copy
import logging
import signal
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger("memopt")

# Lazy import to avoid circular dependency (input_handler imports torch, not universal_optimizer)
def _get_input_handler():
    from memopt.utils.input_handler import (
        detect_input_format, forward, get_sequence_length, get_batch_size, InputFormat,
    )
    return detect_input_format, forward, get_sequence_length, get_batch_size, InputFormat

# ---------------------------------------------------------------------------
# 1. Runtime ridge-point detection
# ---------------------------------------------------------------------------

def get_ridge_point_flops_per_byte(device: int = 0) -> float:
    """
    Return the roofline ridge point (FLOPS/byte) for the current GPU.

    Below ridge  → memory-bound  → SDPA / channels_last help.
    Above ridge  → compute-bound → compile helps (cautiously).

    Uses get_gpu_spec() which already covers 30+ GPUs and falls back to
    architecture-based estimates (Volta/Turing/Ampere/Ada/Hopper) for
    unknown devices.  As a final safety net, falls back to raw device
    properties so it never crashes on a completely unknown GPU.

    Expected values (FP32):
      V100  ≈ 80   T4  ≈ 57   RTX 3090 ≈ 90
      A100  ≈ 156  RTX 4090 ≈ 165  H100 ≈ 222
    """
    try:
        from ..profiler.hardware_counters import get_gpu_spec
        spec = get_gpu_spec()
        # Use FP16 ridge: PyTorch uses tensor cores (TF32/FP16) for all
        # matmul, not scalar FP32 cores.  FP16 peak is the operationally
        # correct ridge point for transformer workloads.
        ridge = spec.ridge_point_fp16
        logger.info(
            "GPU: %s | Ridge: %.0f FLOPS/byte (FP16 tensor-core) | "
            "Peak FP16: %.0f TFLOPS | BW: %.0f GB/s",
            spec.name, ridge,
            spec.peak_fp16_tflops,
            spec.peak_memory_bandwidth_gbps,
        )
        return ridge
    except Exception:
        pass

    # Pure compute-capability fallback — no dict lookup, never fails.
    # memory_clock_rate / memory_bus_width were removed from PyTorch 2.6
    # get_device_properties(), so we use architecture-based estimates.
    props = torch.cuda.get_device_properties(device)
    cc = (props.major, props.minor)
    # Approximate ridge points per architecture (FLOPS/byte, conservative):
    #   Hopper (9.x): ~222  Ampere-DC (8.0): ~156  Ada (8.9): ~165
    #   Ampere-consumer (8.6): ~90  Turing (7.5): ~57  Volta (7.0): ~80
    if cc >= (9, 0):
        ridge = 200.0
    elif cc >= (8, 9):
        ridge = 165.0
    elif cc >= (8, 0):
        ridge = 140.0
    elif cc >= (7, 5):
        ridge = 57.0
    elif cc >= (7, 0):
        ridge = 80.0
    else:
        ridge = 50.0
    logger.info(
        "GPU (cc-fallback): %s SM%d.%d | Ridge: %.0f FLOPS/byte (arch estimate)",
        props.name, cc[0], cc[1], ridge,
    )
    return ridge


# ---------------------------------------------------------------------------
# 2. Regime-aware optimization plan
# ---------------------------------------------------------------------------

@dataclass
class UniversalPlan:
    """Lightweight optimization plan for a specific model + input config."""
    use_sdpa: bool = False
    use_channels_last: bool = False
    compile_mode: Optional[str] = None   # "reduce-overhead", "max-autotune", or None
    ai: float = 0.0
    ridge: float = 0.0
    is_memory_bound: bool = False
    has_attention: bool = False
    is_cnn: bool = False
    skip_reason: str = ""

    def __str__(self) -> str:
        parts = []
        if self.use_sdpa:
            parts.append("SDPA")
        if self.use_channels_last:
            parts.append("channels_last")
        if self.compile_mode:
            parts.append(f"compile({self.compile_mode})")
        regime = "MEM" if self.is_memory_bound else "COM"
        return (
            f"[{regime} AI={self.ai:.0f}/ridge={self.ridge:.0f}] "
            + (", ".join(parts) if parts else "no-op")
            + (f" [skip: {self.skip_reason}]" if self.skip_reason else "")
        )


def _detect_attention(model: nn.Module) -> bool:
    """True if model contains any attention-like module."""
    attn_types = (nn.MultiheadAttention,)
    attn_names = {"attn", "attention", "self_attn", "cross_attn"}
    attn_attr  = {"q_proj", "k_proj", "v_proj", "query", "key", "value"}
    for name, mod in model.named_modules():
        if isinstance(mod, attn_types):
            return True
        leaf = name.split(".")[-1].lower()
        if leaf in attn_names:
            return True
        if any(hasattr(mod, a) for a in attn_attr):
            return True
    return False


def _detect_conv(model: nn.Module) -> bool:
    """True if model contains Conv2d layers."""
    return any(isinstance(m, nn.Conv2d) for _, m in model.named_modules())


def _estimate_ai(model: nn.Module, sample_input) -> float:
    """
    Estimate arithmetic intensity = theoretical_flops / measured_dram_bytes.

    FLOPs:  sum over Linear/Conv2d/MultiheadAttention layers (theoretical).
    DRAM:   allocator peak-delta × 2 (each byte is read once + written once).

    sample_input may be a Dict[str, Any] or an InputFormat.
    Returns inf if DRAM traffic cannot be measured (CPU-only or no allocation).
    """
    detect_input_format, fwd, get_sequence_length, get_batch_size, InputFormat = _get_input_handler()

    if not torch.cuda.is_available():
        return float("inf")

    # Normalise to InputFormat so we can call forward() universally
    if isinstance(sample_input, InputFormat):
        fmt = sample_input
    else:
        try:
            fmt = detect_input_format(model, sample_input)
        except Exception:
            return float("inf")

    # ── DRAM via allocator ──────────────────────────────────────────────────
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    try:
        with torch.no_grad():
            fwd(model, fmt)
        torch.cuda.synchronize()
    except Exception:
        return float("inf")
    peak = torch.cuda.max_memory_allocated()
    dram_bytes = max(0, peak - mem_before) * 2   # read + write lower bound

    # ── FLOPs via module walk ───────────────────────────────────────────────
    total_flops = 0
    B = get_batch_size(fmt) or 1
    S = get_sequence_length(fmt) or 1

    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            in_f, out_f = mod.in_features, mod.out_features
            total_flops += 2 * B * S * in_f * out_f
        elif isinstance(mod, nn.Conv2d):
            # rough: 2 × B × out_channels × (H×W estimate) × in_channels × kH × kW
            kH, kW = mod.kernel_size if isinstance(mod.kernel_size, tuple) else (mod.kernel_size, mod.kernel_size)
            total_flops += 2 * B * mod.out_channels * S * mod.in_channels * kH * kW
        elif isinstance(mod, nn.MultiheadAttention):
            d = mod.embed_dim
            # QKV proj + attn scores + context + out proj (simplified)
            total_flops += (6 * B * S * d * d) + (4 * B * S * S * d)

    if dram_bytes == 0 or total_flops == 0:
        return float("inf")
    return total_flops / dram_bytes


def select_optimizations(
    model: nn.Module,
    sample_input,           # Dict[str, Any] | InputFormat | Any batch
    ridge: Optional[float] = None,
) -> UniversalPlan:
    """
    Profile model with sample_input, classify compute regime, return plan.

    sample_input may be a dict, raw tensor, tuple, or InputFormat.

    Rules (from benchmark evidence on A100):
    - CNN  → channels_last + compile(reduce-overhead)       always safe
    - Memory-bound transformer → SDPA + compile(reduce-overhead)
    - Compute-bound transformer → compile(reduce-overhead)  only (regression guard required)
    - Skip compile for tiny models (< 1ms base) where overhead > gain
    """
    if ridge is None:
        ridge = get_ridge_point_flops_per_byte()

    has_attn = _detect_attention(model)
    is_cnn   = _detect_conv(model)
    ai       = _estimate_ai(model, sample_input)  # accepts dict or InputFormat
    is_mem   = (ai < ridge)

    plan = UniversalPlan(
        ai=ai, ridge=ridge,
        is_memory_bound=is_mem,
        has_attention=has_attn,
        is_cnn=is_cnn,
    )

    if is_cnn:
        plan.use_channels_last = True
        plan.compile_mode = "reduce-overhead"

    elif has_attn:
        if is_mem:
            # Memory-bound attention: SDPA removes O(N²) HBM writes
            plan.use_sdpa = True
            plan.compile_mode = "reduce-overhead"
        else:
            # Compute-bound: only compile (with regression guard)
            plan.compile_mode = "reduce-overhead"

    else:
        # Generic MLP / other — compile if not trivially small
        plan.compile_mode = "reduce-overhead"

    logger.info("select_optimizations: %s", plan)
    return plan


# ---------------------------------------------------------------------------
# 3. safe_compile — compile with regression guard
# ---------------------------------------------------------------------------

def _bench_ms(
    fn,
    warmup: int = 5,
    iters: int = 20,
    use_cuda: bool = True,
) -> float:
    """Benchmark fn() in milliseconds using CUDA events when available."""
    with torch.no_grad():
        for _ in range(warmup):
            fn()
    if use_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        with torch.no_grad():
            for _ in range(iters):
                fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters
    else:
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(iters):
                fn()
        return (time.perf_counter() - t0) * 1000 / iters


def _compile_with_timeout(
    module: nn.Module,
    mode: str,
    timeout_seconds: int = 300,
) -> nn.Module:
    """
    Fix E: torch.compile with SIGALRM timeout (Linux main-thread only).
    Falls back to no-timeout compile in worker threads or on non-Linux.
    Returns original module on timeout.
    """
    can_alarm = (
        hasattr(signal, "SIGALRM")
        and threading.current_thread() is threading.main_thread()
    )
    if not can_alarm:
        return torch.compile(module, mode=mode, fullgraph=False, backend="inductor")

    def _handler(signum, frame):
        raise TimeoutError(f"torch.compile timed out after {timeout_seconds}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_seconds)
    try:
        return torch.compile(module, mode=mode, fullgraph=False, backend="inductor")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def safe_compile(
    module: nn.Module,
    sample_input,           # Dict[str, Any] | InputFormat | Any batch
    mode: str = "reduce-overhead",
    regression_threshold: float = 0.95,
    warmup_iters: int = 10,
    bench_iters: int = 50,
    compile_timeout: int = 300,
) -> nn.Module:
    """
    Apply torch.compile only if it does not regress performance.

    If compiled version is slower than regression_threshold × baseline,
    the original module is returned unchanged.

    Args:
        module:               The nn.Module to (potentially) compile.
        sample_input:         Dict, tensor, tuple, or InputFormat.
        mode:                 torch.compile mode string.
        regression_threshold: Roll back if speedup < this value (0.95 = 5% margin).
        warmup_iters:         Warmup iterations before benchmarking.
        bench_iters:          Benchmark iterations.
        compile_timeout:      Max seconds for torch.compile call (Fix E). 0 = no limit.

    Returns:
        Compiled module (or original if compile regressed / failed / timed out).
    """
    detect_input_format, fwd, _, __, InputFormat = _get_input_handler()
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    # Normalise sample_input to InputFormat once
    if isinstance(sample_input, InputFormat):
        fmt = sample_input
    else:
        try:
            fmt = detect_input_format(module, sample_input, device=device)
        except Exception as exc:
            logger.warning("safe_compile: input detection failed (%s) — using original", exc)
            return module

    def run_original():
        return fwd(module, fmt)

    baseline_ms = _bench_ms(run_original, warmup=warmup_iters, iters=bench_iters, use_cuda=use_cuda)

    # Fix A: skip inner deepcopy for large models — torch.compile wraps without
    # modifying the module, so the original is safe to pass directly.
    # For small models, deepcopy preserves compiled state if safe_compile is re-called.
    param_count = sum(p.numel() for p in module.parameters())
    module_to_compile = module if param_count > 1e9 else copy.deepcopy(module)

    try:
        if compile_timeout > 0:
            compiled = _compile_with_timeout(module_to_compile, mode, compile_timeout)
        else:
            compiled = torch.compile(
                module_to_compile,
                mode=mode,
                fullgraph=False,
                backend="inductor",
            )
    except TimeoutError as exc:
        logger.warning("safe_compile: %s — using original", exc)
        return module
    except Exception as exc:
        logger.warning("safe_compile: torch.compile failed (%s) — using original", exc)
        return module

    def run_compiled():
        return fwd(compiled, fmt)

    # Warmup the compiled version (first call triggers JIT — always slow)
    try:
        with torch.no_grad():
            for _ in range(max(warmup_iters, 3)):
                run_compiled()
    except Exception as exc:
        logger.warning("safe_compile: compiled warmup failed (%s) — rolling back", exc)
        return module

    compiled_ms = _bench_ms(run_compiled, warmup=warmup_iters, iters=bench_iters, use_cuda=use_cuda)
    speedup = baseline_ms / compiled_ms if compiled_ms > 0 else 1.0

    if speedup < regression_threshold:
        logger.warning(
            "safe_compile: compile(%s) regressed to %.2fx "
            "(%.2fms → %.2fms) — rolling back to original",
            mode, speedup, baseline_ms, compiled_ms,
        )
        return module

    logger.info(
        "safe_compile: compile(%s) %.2fx speedup (%.2fms → %.2fms) — committed",
        mode, speedup, baseline_ms, compiled_ms,
    )
    return compiled


# ---------------------------------------------------------------------------
# 4. apply_universal_plan — wires plan into a model
# ---------------------------------------------------------------------------

def _apply_sdpa(model: nn.Module) -> nn.Module:
    """
    Replace nn.MultiheadAttention modules with SDPA-based equivalents.
    Uses PyTorch's native scaled_dot_product_attention (flash-attn v2 backend).
    Does NOT attempt to import flash_attn (ABI broken on PyTorch 2.6+cu124).
    """
    if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        logger.warning("_apply_sdpa: SDPA not available (requires PyTorch >= 2.0)")
        return model

    class _SDPAWrapper(nn.Module):
        def __init__(self, mha: nn.MultiheadAttention):
            super().__init__()
            self.in_proj_weight  = mha.in_proj_weight
            self.in_proj_bias    = mha.in_proj_bias
            self.out_proj        = mha.out_proj
            self.num_heads       = mha.num_heads
            self.embed_dim       = mha.embed_dim
            self.batch_first     = mha.batch_first

        def forward(self, query, key=None, value=None):
            if key is None:
                key = query
            if value is None:
                value = query
            x = query if self.batch_first else query.transpose(0, 1)
            B, S, d = x.shape
            qkv = nn.functional.linear(x, self.in_proj_weight, self.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
            dh = d // self.num_heads
            def split(t):
                return t.view(B, S, self.num_heads, dh).transpose(1, 2)
            out = nn.functional.scaled_dot_product_attention(
                split(q), split(k), split(v), is_causal=False
            )
            out = out.transpose(1, 2).contiguous().view(B, S, d)
            out = nn.functional.linear(out, self.out_proj.weight, self.out_proj.bias)
            if not self.batch_first:
                out = out.transpose(0, 1)
            return out, None   # match MHA's (output, weights) signature

    replaced = 0
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.MultiheadAttention):
            continue
        wrapper = _SDPAWrapper(mod)
        # navigate to parent and setattr
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        try:
            setattr(parent, parts[-1], wrapper)
            replaced += 1
            logger.info("_apply_sdpa: replaced %s with SDPA wrapper", name)
        except Exception as exc:
            logger.warning("_apply_sdpa: could not replace %s: %s", name, exc)

    if replaced == 0:
        logger.debug("_apply_sdpa: no nn.MultiheadAttention found")
    return model


def apply_universal_plan(
    model: nn.Module,
    sample_input: Dict[str, Any],
    plan: UniversalPlan,
) -> nn.Module:
    """
    Execute an UniversalPlan on model in-place.  Returns the (possibly compiled)
    model.  All steps are guarded — a failure in one step does not prevent
    subsequent steps from running.
    """
    result = model

    # channels_last conversion
    if plan.use_channels_last:
        try:
            result = result.to(memory_format=torch.channels_last)
            # convert sample_input tensors too for consistency check
            logger.info("apply_universal_plan: converted to channels_last")
        except Exception as exc:
            logger.warning("apply_universal_plan: channels_last failed: %s", exc)

    # SDPA replacement
    if plan.use_sdpa:
        try:
            result = _apply_sdpa(result)
        except Exception as exc:
            logger.warning("apply_universal_plan: SDPA failed: %s", exc)

    # Compile with regression guard
    if plan.compile_mode:
        # For channels_last models, convert sample_input too
        compile_input = sample_input
        if plan.use_channels_last:
            compile_input = {
                k: (v.to(memory_format=torch.channels_last)
                    if isinstance(v, torch.Tensor) and v.ndim == 4
                    else v)
                for k, v in sample_input.items()
            }
        result = safe_compile(result, compile_input, mode=plan.compile_mode)

    return result


# ---------------------------------------------------------------------------
# 5. UniversalOptimizer — stateless helper with testable gate methods
# ---------------------------------------------------------------------------

class UniversalOptimizer:
    """
    Stateless helper class that wraps the module-level optimizer functions.
    Provides individually testable gate methods (e.g., _should_apply_int8).
    Instantiate with UniversalOptimizer() — no state, safe to share.
    """

    def _should_apply_int8(
        self,
        inputs,                         # Dict[str, Any] | InputFormat
        bottleneck_type: Any,
        model: Optional[nn.Module] = None,
    ) -> bool:
        """
        INT8 regime gate — three tiers based on model size.

        inputs may be a plain dict or an InputFormat (from detect_input_format).

        Physics:
        - Small models (<7B): need seq>=1024 to be memory-bound
          (GEMM matrices too small at short seq for INT8 to win)
        - Large models (7B-20B): memory-bound at seq>=512
          (larger weight matrices = more memory traffic per token)
        - Very large models (>20B): memory-bound at seq>=128
          (30B weight matrices dominate bandwidth at any seq length)

        Thresholds are physics-based estimates. The test-measure-commit loop
        catches any cases where INT8 doesn't win in practice.
        """
        _, __, get_sequence_length, get_batch_size, InputFormat = _get_input_handler()

        # Resolve COMPUTE_BOUND — support both string and enum
        bottleneck_str = (
            bottleneck_type.value
            if hasattr(bottleneck_type, "value")
            else str(bottleneck_type)
        )
        if "compute_bound" in bottleneck_str.lower():
            logger.info("INT8 gate: SKIP — compute-bound")
            return False

        # Extract seq_len and batch via InputFormat helpers (or legacy dict path)
        if isinstance(inputs, InputFormat):
            seq_len = get_sequence_length(inputs) or 1
            batch   = get_batch_size(inputs) or 1
        else:
            # Legacy dict path — try known key names, then first 2D+ tensor
            seq_len = 1
            batch   = 1
            for key in ("input_ids", "inputs_embeds", "x", "input"):
                if key in inputs and isinstance(inputs[key], torch.Tensor):
                    t = inputs[key]
                    batch   = t.shape[0]
                    seq_len = t.shape[1] if t.ndim >= 2 else 1
                    break
            if seq_len == 1 and batch == 1:
                for v in inputs.values():
                    if isinstance(v, torch.Tensor) and v.ndim >= 2:
                        batch   = v.shape[0]
                        seq_len = v.shape[1]
                        break

        param_count = (
            sum(p.numel() for p in model.parameters()) if model is not None else 0
        )

        # Tier 1: Very large models >=20B — always memory-bound
        if param_count >= 20e9:
            result = seq_len >= 128
            logger.info(
                "INT8 gate (>=20B, %.0fB params): seq=%d >= 128 → %s",
                param_count / 1e9, seq_len, result,
            )
            return result

        # Tier 2: Large models >=7B-<20B
        elif param_count >= 7e9:
            result = seq_len >= 512
            logger.info(
                "INT8 gate (>=7B-<20B, %.0fB params): seq=%d >= 512 → %s",
                param_count / 1e9, seq_len, result,
            )
            return result

        # Tier 3: Small models <7B — original calibrated gate
        else:
            result = (seq_len >= 1024) and (batch * seq_len <= 4096)
            logger.info(
                "INT8 gate (<7B, %.1fB params): seq=%d>=1024 AND batch*seq=%d<=4096 → %s",
                param_count / 1e9, seq_len, batch * seq_len, result,
            )
            return result
