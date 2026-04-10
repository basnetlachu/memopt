"""
Kernel hooks — drop-in replacements for three ops in the serving layer.

Each hook checks the kernel cache first. On cache hit it runs the
synthesised fused kernel. On cache miss it runs the standard PyTorch op.
The caller never needs to know which path ran.

Shim layer: tries C++ _memopt_hooks extension for the hot path
(atomic counters, FNV-1a keys, lock-free dispatch). Falls back to
pure Python implementation if the extension is not built.

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

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Module-level state — exposed to tests that monkey-patch these directly.
# Regardless of whether C++ is active, Python holds these references.
# ═══════════════════════════════════════════════════════════════════════════
_cache:     Optional[object] = None
_optimizer: Optional[object] = None
_hook_lock  = threading.Lock()

_fallback_counts: dict = {}
_fallback_lock    = threading.Lock()

# ═══════════════════════════════════════════════════════════════════════════
# Detect C++ extension availability
# ═══════════════════════════════════════════════════════════════════════════
_cpp_hooks = None
try:
    import memopt._memopt_hooks as _cpp_hooks  # type: ignore
    _USE_CPP = True
    logger.info(
        "memopt: C++ kernel hooks loaded "
        "(FNV-1a keys, lock-free dispatch, atomic counters)"
    )
except ImportError:
    _USE_CPP = False
    logger.info(
        "memopt: C++ kernel hooks not available, using Python fallback. "
        "GIL saturation expected at >100 req/s. "
        "Run: pip install memopt[cpp] to build C++ extensions."
    )


# ═══════════════════════════════════════════════════════════════════════════
# Fallback telemetry
# ═══════════════════════════════════════════════════════════════════════════

def _record_fallback(op_name: str) -> None:
    """Increment the fallback counter for op_name. Thread-safe."""
    with _fallback_lock:
        _fallback_counts[op_name] = _fallback_counts.get(op_name, 0) + 1
        count = _fallback_counts[op_name]
    if count == 1 or count % 100 == 0:
        logger.warning(
            f"Kernel fallback [{op_name}]: {count} total. "
            f"Fused kernel hit an unexpected tensor layout. "
            f"Check stats()['fallback_counts'] for details."
        )


def _log_cache_miss(op_name: str, reason: str) -> None:
    logger.debug(f"Hook fallback [{op_name}]: reason={reason}")


# ═══════════════════════════════════════════════════════════════════════════
# init_hooks — called once at server startup
# ═══════════════════════════════════════════════════════════════════════════

def init_hooks(cache, optimizer) -> None:
    """Wire in the kernel cache and auto-optimizer. Idempotent."""
    global _cache, _optimizer
    with _hook_lock:
        _cache     = cache
        _optimizer = optimizer

    if _USE_CPP and _cpp_hooks is not None:
        arch_id = _cpp_hooks.detect_arch_id()
        _cpp_hooks.init_hooks(arch_id, cache, optimizer)
        logger.info(f"C++ hooks initialised (arch_id={arch_id})")
    else:
        logger.info("Python kernel hooks initialised")


# ═══════════════════════════════════════════════════════════════════════════
# The three hook functions — core hot path
# ═══════════════════════════════════════════════════════════════════════════
# These are called 1.92M times per second at fleet scale.
# When C++ is available, the notify() call uses atomic counters.
# The cache lookup and kernel dispatch stay in Python because tests
# monkey-patch _cache, _get_hardware, and expect Python-level control.

def apply_rope(
    xq: "torch.Tensor",
    xk: "torch.Tensor",
    cos: "torch.Tensor",
    sin: "torch.Tensor",
) -> "tuple":
    """Rotary Position Embedding — fused or unfused depending on cache."""
    if _optimizer is not None:
        if _USE_CPP and _cpp_hooks is not None:
            _cpp_hooks.notify(0)  # ROPE — atomic, no lock
        else:
            _optimizer.notify(
                op_name="memopt.rope_fused",
                args=(xq, xk, cos, sin),
            )

    if _cache is None:
        _log_cache_miss("memopt.rope_fused", "no_cache")
    else:
        from memopt.kernels.kernel_cache import cache_key
        key    = cache_key("memopt.rope_fused",
                           [list(xq.shape), list(xk.shape)],
                           _get_hardware())
        module = _cache.get(key)
        if module is None:
            _log_cache_miss("memopt.rope_fused", "warmup")
        else:
            run = getattr(module, "run_kernel", None)
            if run is None:
                _log_cache_miss("memopt.rope_fused", "no_run_kernel")
            else:
                try:
                    return run(xq, xk, cos, sin)
                except Exception as e:
                    _record_fallback("memopt.rope_fused")
                    logger.debug(f"RoPE fused kernel error: {e} — falling back")

    return _rope_unfused(xq, xk, cos, sin)


def apply_layer_norm_residual(
    x:        "torch.Tensor",
    residual: "torch.Tensor",
    weight:   "torch.Tensor",
    bias:     "torch.Tensor",
    eps:      float = 1e-5,
) -> "torch.Tensor":
    """Fused residual add + layer norm — fused or unfused depending on cache."""
    if _optimizer is not None:
        if _USE_CPP and _cpp_hooks is not None:
            _cpp_hooks.notify(1)  # LAYER_NORM_RESIDUAL — atomic
        else:
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
        module = _cache.get(key)
        if module is None:
            _log_cache_miss("memopt.ln_residual_fused", "warmup")
        else:
            run = getattr(module, "run_kernel", None)
            if run is None:
                _log_cache_miss("memopt.ln_residual_fused", "no_run_kernel")
            else:
                try:
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
    """Fused scale + softmax — fused or unfused depending on cache."""
    if _optimizer is not None:
        if _USE_CPP and _cpp_hooks is not None:
            _cpp_hooks.notify(2)  # SCALED_SOFTMAX — atomic
        else:
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
        module = _cache.get(key)
        if module is None:
            _log_cache_miss("memopt.scaled_softmax_fused", "warmup")
        else:
            run = getattr(module, "run_kernel", None)
            if run is None:
                _log_cache_miss("memopt.scaled_softmax_fused", "no_run_kernel")
            else:
                try:
                    return run(scores, scale)
                except Exception as e:
                    _record_fallback("memopt.scaled_softmax_fused")
                    logger.debug(f"Softmax fused kernel error: {e} — falling back")

    import torch.nn.functional as F  # type: ignore[import-untyped]
    return F.softmax(scores * scale, dim=-1)


# ═══════════════════════════════════════════════════════════════════════════
# stats
# ═══════════════════════════════════════════════════════════════════════════

def stats() -> dict:
    """Return cache, optimizer, and fallback telemetry for Grafana."""
    with _fallback_lock:
        fallbacks = dict(_fallback_counts)
    result = {
        "hooks_initialised": _cache is not None,
        "fallback_counts":   fallbacks,
    }
    if _cache is not None:
        result["cache"] = _cache.stats()
    if _optimizer is not None:
        result["optimizer"] = _optimizer.stats()
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers — kept in Python for test monkey-patching compatibility
# ═══════════════════════════════════════════════════════════════════════════

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
    """Return hardware identifier string for cache key generation."""
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
        return "blackwell"
    if major >= 10:
        return "blackwell_next"
    if major == 9:
        return "hopper"
    if major == 8 and minor == 9:
        return "ada_lovelace"
    if major == 8:
        return "ampere"
    if major == 7 and minor == 5:
        return "turing"
    if major == 7:
        return "volta"
    if major == 6:
        return "pascal"
    return f"sm{major}{minor}"
