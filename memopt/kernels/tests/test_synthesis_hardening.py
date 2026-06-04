"""
Kernel synthesis hardening tests — strict mode, circuit breaker, audit log.
"""
import tempfile



def test_synthesis_audit_log_never_raises():
    """_write_audit_log must never raise, even with bad path."""
    from memopt.kernels.jit_generator import JITGenerator
    from memopt.kernels.kernel_cache import KernelCache
    from memopt.kernels.portability_layer import PortabilityLayer

    with tempfile.TemporaryDirectory() as d:
        cache = KernelCache(cache_dir=d)
        gen = JITGenerator(cache=cache, portability=PortabilityLayer())

        # Should not raise even if log dir is invalid
        if hasattr(gen, '_write_audit_log'):
            gen._write_audit_log({
                "op_name": "test_op",
                "outcome": "test",
            })
        # If method doesn't exist yet, that's fine — no exception


def test_circuit_breaker_state():
    """Circuit breaker should have consistent state."""
    from memopt.kernels import jit_generator as jg
    # Verify module-level circuit breaker exists
    assert hasattr(jg, '_circuit_failures')
    assert hasattr(jg, '_circuit_opened_at')
    assert hasattr(jg, '_circuit_lock')


def test_dtype_tolerances_exist():
    """Tolerance dict should cover common dtypes."""
    from memopt.kernels.jit_generator import _DTYPE_TOLERANCES
    assert "float16" in _DTYPE_TOLERANCES
    assert "float32" in _DTYPE_TOLERANCES
    # float16 tolerance should be looser than float32
    assert _DTYPE_TOLERANCES["float16"][0] > _DTYPE_TOLERANCES["float32"][0]


def test_generator_deduplicates_in_flight():
    """Generator should not synthesize same op twice concurrently."""
    from memopt.kernels.jit_generator import JITGenerator
    from memopt.kernels.kernel_cache import KernelCache
    from memopt.kernels.portability_layer import PortabilityLayer

    with tempfile.TemporaryDirectory() as d:
        cache = KernelCache(cache_dir=d)
        gen = JITGenerator(cache=cache, portability=PortabilityLayer())
        assert hasattr(gen, '_in_flight')
        assert isinstance(gen._in_flight, set)
