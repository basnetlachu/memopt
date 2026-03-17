"""
Kernel hooks — drop-in replacements for three ops in the serving layer.

Each hook checks the kernel cache first. On cache hit it runs the
synthesised fused kernel. On cache miss it runs the standard PyTorch op.
The caller never needs to know which path ran.

Thread safety: all hooks are stateless. The KernelCache and
AutoOptimizer they reference are module-level singletons initialised
once at server startup via init_hooks().

Design rule: hooks must never raise. Any error in the fused path
falls back to PyTorch silently. Inference correctness is never at risk.
"""
from __future__ import annotations
import logging
import threading
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # never executed at runtime; satisfies Pylance for annotations
    import torch

logger = logging.getLogger(__name__)

# Module-level singletons — written once at startup, read on every request.
# _hook_lock guards only the writes in init_hooks().
# The reads in apply_* do NOT hold the lock — they rely on Python's
# guaranteed atomic reference reads for module-level variables.
# Module objects returned by cache.get() are captured in local variables
# before run_kernel is called — this keeps the module alive (reference
# count > 0) for the duration of the call even if the cache replaces
# the entry concurrently.
_cache:     Optional[object] = None
_optimizer: Optional[object] = None
_hook_lock  = threading.Lock()   # guards init_hooks() writes only

# Fallback telemetry — counts how many times each op fell back to PyTorch
# due to a fused kernel error. A spike here is a canary that the synthesised
# kernel is hitting an edge case (unexpected tensor striding, memory layout
# mismatch) not present during validation. Exposed via stats() for Grafana.
_fallback_counts: dict = {}
_fallback_lock    = threading.Lock()


def _record_fallback(op_name: str) -> None:
    """Increment the fallback counter for op_name. Thread-safe."""
    with _fallback_lock:
        _fallback_counts[op_name] = _fallback_counts.get(op_name, 0) + 1
        count = _fallback_counts[op_name]   # read inside lock — no TOCTOU

    # Warning fired outside lock — logging is slow, never hold locks while logging
    if count == 1 or count % 100 == 0:
        logger.warning(
            f"Kernel fallback [{op_name}]: {count} total. "
            f"Fused kernel hit an unexpected tensor layout. "
            f"Check stats()['fallback_counts'] for details."
        )


def _log_cache_miss(op_name: str, reason: str) -> None:
    """
    Log why a hook fell back to unfused PyTorch.
    Reasons: 'no_cache' | 'no_module' | 'no_run_kernel' | 'warmup'
    At debug level — not noisy in production, visible when debugging.
    """
    logger.debug(f"Hook fallback [{op_name}]: reason={reason}")


def init_hooks(cache, optimizer) -> None:
    """
    Called once at server startup to wire in the kernel cache
    and auto-optimizer. Safe to call multiple times — idempotent.
    """
    global _cache, _optimizer
    with _hook_lock:
        _cache     = cache
        _optimizer = optimizer
    logger.info("Kernel hooks initialised")


def apply_rope(
    xq: "torch.Tensor",
    xk: "torch.Tensor",
    cos: "torch.Tensor",
    sin: "torch.Tensor",
) -> "Tuple[torch.Tensor, torch.Tensor]":
    """
    Rotary Position Embedding — fused or unfused depending on cache.

    Fused path (cache hit):    5.18x faster at seq_len=16384
    Unfused path (cache miss): standard PyTorch, identical output
    # AUDIT P1: 5.18x figure is a single-device benchmark (Blackwell GB200,
    # seq_len=16384). Actual speedup varies by GPU architecture, seq length,
    # and HBM bandwidth. Do not treat this as a guaranteed production number.

    The auto-optimizer synthesises the fused kernel on first call and
    registers it in the cache. All subsequent calls use the fused path.
    """
    # Notify optimizer — it will synthesise a kernel if stall rate is high
    if _optimizer is not None:
        _optimizer.notify(
            op_name="memopt.rope_fused",
            args=(xq, xk, cos, sin),
        )

    # Check cache
    if _cache is None:
        _log_cache_miss("memopt.rope_fused", "no_cache")
    else:
        from memopt.kernels.kernel_cache import cache_key
        key    = cache_key("memopt.rope_fused",
                           [list(xq.shape), list(xk.shape)],
                           _get_hardware())
        module = _cache.get(key)   # local ref — keeps module alive (RCU pattern)
        if module is None:
            _log_cache_miss("memopt.rope_fused", "warmup")
        else:
            run = getattr(module, "run_kernel", None)
            if run is None:
                _log_cache_miss("memopt.rope_fused", "no_run_kernel")
            else:
                try:
                    # module is referenced locally — safe from GC even if
                    # cache replaces the entry concurrently during this call
                    return run(xq, xk, cos, sin)
                except Exception as e:
                    _record_fallback("memopt.rope_fused")
                    logger.debug(f"RoPE fused kernel error: {e} — falling back")

    # Unfused fallback
    return _rope_unfused(xq, xk, cos, sin)


def apply_layer_norm_residual(
    x:        "torch.Tensor",
    residual: "torch.Tensor",
    weight:   "torch.Tensor",
    bias:     "torch.Tensor",
    eps:      float = 1e-5,
) -> "torch.Tensor":
    """
    Fused residual add + layer norm — fused or unfused depending on cache.

    Fused path (cache hit):    1.30x faster (measured: seq_len=8192,
                               hidden=4096, fp16, RTX PRO 6000 Blackwell)
    Unfused path (cache miss): standard PyTorch, identical output
    """
    if _optimizer is not None:
        _optimizer.notify(
            op_name="memopt.ln_residual_fused",
            args=(x, residual, weight, bias),
        )

    if _cache is None:
        _log_cache_miss("memopt.ln_residual_fused", "no_cache")
    else:
        from memopt.kernels.kernel_cache import cache_key
        key    = cache_key("memopt.ln_residual_fused",
                           [list(x.shape), list(residual.shape)],
                           _get_hardware())
        module = _cache.get(key)   # local ref — keeps module alive (RCU pattern)
        if module is None:
            _log_cache_miss("memopt.ln_residual_fused", "warmup")
        else:
            run = getattr(module, "run_kernel", None)
            if run is None:
                _log_cache_miss("memopt.ln_residual_fused", "no_run_kernel")
            else:
                try:
                    # module is referenced locally — safe from GC even if
                    # cache replaces the entry concurrently during this call
                    return run(x, residual, weight, bias)
                except Exception as e:
                    _record_fallback("memopt.ln_residual_fused")
                    logger.debug(f"LN+Residual fused kernel error: {e} — falling back")

    import torch.nn.functional as F  # type: ignore[import-untyped]
    return F.layer_norm(x + residual, (x.shape[-1],), weight, bias, eps)


def apply_scaled_softmax(
    scores: "torch.Tensor",
    scale:  float,
) -> "torch.Tensor":
    """
    Fused scale + softmax for attention scores — fused or unfused.

    Fused path (cache hit):    1.32x faster (measured: batch=4,
                               n_heads=32, seq_len=2048, fp16,
                               RTX PRO 6000 Blackwell)
    Unfused path (cache miss): standard PyTorch, identical output
    """
    if _optimizer is not None:
        _optimizer.notify(
            op_name="memopt.scaled_softmax_fused",
            args=(scores, scale),
        )

    if _cache is None:
        _log_cache_miss("memopt.scaled_softmax_fused", "no_cache")
    else:
        from memopt.kernels.kernel_cache import cache_key
        key    = cache_key("memopt.scaled_softmax_fused",
                           [list(scores.shape)],
                           _get_hardware())
        module = _cache.get(key)   # local ref — keeps module alive (RCU pattern)
        if module is None:
            _log_cache_miss("memopt.scaled_softmax_fused", "warmup")
        else:
            run = getattr(module, "run_kernel", None)
            if run is None:
                _log_cache_miss("memopt.scaled_softmax_fused", "no_run_kernel")
            else:
                try:
                    # module is referenced locally — safe from GC even if
                    # cache replaces the entry concurrently during this call
                    return run(scores, scale)
                except Exception as e:
                    _record_fallback("memopt.scaled_softmax_fused")
                    logger.debug(f"Softmax fused kernel error: {e} — falling back")

    import torch.nn.functional as F  # type: ignore[import-untyped]
    return F.softmax(scores * scale, dim=-1)


def stats() -> dict:
    """Return cache, optimizer, and fallback telemetry for Grafana."""
    with _fallback_lock:
        fallbacks = dict(_fallback_counts)
    result = {
        "hooks_initialised": _cache is not None,
        "fallback_counts":   fallbacks,   # canary — spikes = edge case hits
    }
    if _cache is not None:
        result["cache"] = _cache.stats()
    if _optimizer is not None:
        result["optimizer"] = _optimizer.stats()
    return result


# ── Internal helpers ───────────────────────────────────────────────────

def _rope_unfused(xq, xk, cos, sin):
    import torch  # type: ignore[import-untyped]
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    cos_ = cos.unsqueeze(1) if cos.dim() == 2 and xq.dim() == 3 else cos
    sin_ = sin.unsqueeze(1) if sin.dim() == 2 and xq.dim() == 3 else sin
    return (
        xq * cos_ + rotate_half(xq) * sin_,
        xk * cos_ + rotate_half(xk) * sin_,
    )


def _get_hardware() -> str:
    """
    Return a hardware identifier string for cache key generation.

    Format: "cuda:{arch}:{device_name}"
    Examples:
      "cuda:blackwell:NVIDIA RTX PRO 6000 Blackwell Server Edition"
      "cuda:hopper:NVIDIA H100 SXM5 80GB"
      "cuda:ampere:NVIDIA A100-SXM4-80GB"

    Including the architecture prevents collisions on heterogeneous fleets —
    a kernel compiled for Blackwell (sm120) will not be served to a Hopper
    (sm90) that happens to be running the same op shapes.
    """
    try:
        import torch  # type: ignore[import-untyped]
        if torch.cuda.is_available():
            name  = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            arch  = _cuda_arch_name(major, minor)
            return f"cuda:{arch}:{name}"
        if hasattr(torch.version, "hip") and torch.version.hip:
            name = torch.cuda.get_device_name(0) \
                   if torch.cuda.is_available() else "unknown"
            return f"rocm:cdna:{name}"
    except Exception:
        pass
    return "cpu"


def _cuda_arch_name(major: int, minor: int) -> str:
    """Map CUDA compute capability to architecture name."""
    if major >= 12:
        return "blackwell"       # RTX PRO 6000, B100, B200: sm120+
    if major >= 10:
        return "blackwell_next"  # reserved for future Blackwell variants
    if major == 9:
        return "hopper"          # H100, H200: sm90
    if major == 8 and minor == 9:
        return "ada_lovelace"    # RTX 4090: sm89
    if major == 8:
        return "ampere"          # A100, RTX 3090: sm80
    if major == 7 and minor == 5:
        return "turing"          # T4, RTX 2080: sm75
    if major == 7:
        return "volta"           # V100: sm70
    if major == 6:
        return "pascal"          # P100: sm60
    return f"sm{major}{minor}"
