"""
Tests for kernel hooks and auto-optimizer integration.
All tests pass with no GPU, no Triton, no API key.
"""
import types
import threading
import tempfile
import pytest

from memopt.serving import kernel_hooks
from memopt.serving.auto_optimizer import AutoOptimizer
from memopt.kernels.kernel_cache import KernelCache, cache_key


def _make_mock_module(output_fn):
    """Create a fake compiled module with run_kernel."""
    m = types.ModuleType("mock_kernel")
    m.run_kernel = output_fn
    return m


# ── kernel_hooks ──────────────────────────────────────────────────────

def test_rope_hook_fallback_no_cache():
    """Without init_hooks(), apply_rope must use unfused path without error."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")

    kernel_hooks._cache     = None
    kernel_hooks._optimizer = None

    xq  = torch.randn(64, 8, 32)
    xk  = torch.randn(64, 8, 32)
    cos = torch.randn(64, 32)
    sin = torch.randn(64, 32)

    q_out, k_out = kernel_hooks.apply_rope(xq, xk, cos, sin)
    assert q_out.shape == xq.shape
    assert k_out.shape == xk.shape


def test_rope_hook_uses_cached_kernel():
    """apply_rope must use the cached kernel when one exists."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")

    with tempfile.TemporaryDirectory() as d:
        cache = KernelCache(cache_dir=d)

        xq  = torch.randn(64, 8, 32)
        xk  = torch.randn(64, 8, 32)
        cos = torch.randn(64, 32)
        sin = torch.randn(64, 32)

        sentinel = torch.zeros(1)
        called   = {"count": 0}

        def mock_run(q, k, c, s):
            called["count"] += 1
            return sentinel, sentinel

        mock_mod = _make_mock_module(mock_run)
        mock_mod.__source__ = ""

        key   = cache_key("memopt.rope_fused",
                          [list(xq.shape), list(xk.shape)],
                          "cpu")
        event = type("E", (), {"op_name": "memopt.rope_fused",
                               "hardware": "cpu"})()
        cache.put(key, mock_mod, event)

        original_hw = kernel_hooks._get_hardware
        kernel_hooks._get_hardware = lambda: "cpu"
        kernel_hooks.init_hooks(cache=cache, optimizer=None)

        try:
            kernel_hooks.apply_rope(xq, xk, cos, sin)
            assert called["count"] == 1, "Cached kernel was not called"
        finally:
            kernel_hooks._get_hardware = original_hw
            kernel_hooks._cache        = None
            kernel_hooks._optimizer    = None


def test_ln_hook_fallback_no_cache():
    """apply_layer_norm_residual must not raise without cache."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")

    kernel_hooks._cache     = None
    kernel_hooks._optimizer = None

    x        = torch.randn(32, 64)
    residual = torch.randn(32, 64)
    weight   = torch.ones(64)
    bias     = torch.zeros(64)

    out = kernel_hooks.apply_layer_norm_residual(x, residual, weight, bias)
    assert out.shape == x.shape


def test_softmax_hook_fallback_no_cache():
    """apply_scaled_softmax must not raise without cache."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")

    kernel_hooks._cache     = None
    kernel_hooks._optimizer = None

    scores = torch.randn(2, 4, 32, 32)
    out    = kernel_hooks.apply_scaled_softmax(scores, 0.125)
    assert out.shape == scores.shape


def test_hooks_stats_no_cache():
    kernel_hooks._cache     = None
    kernel_hooks._optimizer = None
    s = kernel_hooks.stats()
    assert s["hooks_initialised"] is False
    assert "fallback_counts" in s


def test_fallback_counter_increments_on_error():
    """Fallback counter must increment when fused kernel raises."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")

    with tempfile.TemporaryDirectory() as d:
        cache = KernelCache(cache_dir=d)

        # Plant a broken module that always raises
        def broken_run(q, k, c, s):
            raise RuntimeError("simulated striding error")

        broken_mod = types.ModuleType("broken")
        broken_mod.run_kernel = broken_run
        broken_mod.__source__ = ""

        xq  = torch.randn(8, 4, 16)
        xk  = torch.randn(8, 4, 16)
        cos = torch.randn(8, 16)
        sin = torch.randn(8, 16)

        key   = cache_key("memopt.rope_fused",
                          [list(xq.shape), list(xk.shape)], "cpu")
        event = type("E", (), {"op_name": "memopt.rope_fused",
                               "hardware": "cpu"})()
        cache.put(key, broken_mod, event)

        original_hw = kernel_hooks._get_hardware
        kernel_hooks._get_hardware = lambda: "cpu"
        kernel_hooks._fallback_counts.clear()
        kernel_hooks.init_hooks(cache=cache, optimizer=None)

        try:
            # This must not raise — fallback to PyTorch unfused
            result = kernel_hooks.apply_rope(xq, xk, cos, sin)
            assert result is not None

            s = kernel_hooks.stats()
            assert s["fallback_counts"].get("memopt.rope_fused", 0) == 1, \
                "Fallback counter should be 1 after one broken kernel call"
        finally:
            kernel_hooks._get_hardware = original_hw
            kernel_hooks._cache        = None
            kernel_hooks._optimizer    = None
            kernel_hooks._fallback_counts.clear()


def test_hardware_aware_cache_key_differs_by_arch():
    """Cache keys for the same op on different GPU architectures must differ."""
    from memopt.serving.kernel_hooks import _cuda_arch_name
    ampere    = _cuda_arch_name(8, 0)
    blackwell = _cuda_arch_name(9, 0)
    hopper    = _cuda_arch_name(9, 0)   # same compute capability → same arch
    ada       = _cuda_arch_name(8, 9)

    assert ampere    != blackwell
    assert ampere    != ada
    assert blackwell == hopper   # both sm90 → both same name

    k_ampere    = cache_key("memopt.rope_fused", [[64, 8, 32]], f"cuda:{ampere}:A100")
    k_blackwell = cache_key("memopt.rope_fused", [[64, 8, 32]], f"cuda:{blackwell}:B200")
    assert k_ampere != k_blackwell, \
        "Same op+shapes on different GPU architectures must have different cache keys"


# ── AutoOptimizer ──────────────────────────────────────────────────────

def test_optimizer_does_not_fire_before_warmup():
    """Synthesis must not fire until WARM_UP_CALLS reached."""
    with tempfile.TemporaryDirectory() as d:
        cache = KernelCache(cache_dir=d)
        gen   = type("G", (), {
            "handle": lambda self, e: None,
            "stats":  lambda self: {},
            "_build_prompt": lambda self, e: "",
        })()
        opt = AutoOptimizer(generator=gen, cache=cache)
        opt.start()

        try:
            import torch
            args = (torch.randn(4, 8, 16),)
        except ImportError:
            args = ()

        for _ in range(10):
            opt.notify("memopt.rope_fused", args)

        assert opt.stats()["total_fired"] == 0


def test_optimizer_fires_after_warmup():
    """Synthesis fires once WARM_UP_CALLS is reached and no cache entry."""
    import time
    from memopt.serving.auto_optimizer import WARM_UP_CALLS

    fired = threading.Event()

    class MockGen:
        def handle(self, event):
            fired.set()
        def stats(self):
            return {}
        def _build_prompt(self, e):
            return ""

    with tempfile.TemporaryDirectory() as d:
        cache = KernelCache(cache_dir=d)
        opt   = AutoOptimizer(generator=MockGen(), cache=cache)
        opt.start()

        try:
            import torch
            args = (torch.randn(64, 8, 32), torch.randn(64, 8, 32))
        except ImportError:
            args = ()

        for _ in range(WARM_UP_CALLS + 1):
            opt.notify("memopt.rope_fused", args)

        # Generous timeout — slow CI runners (Linux containers under
        # contention) sometimes take 10s+ to schedule the synthesis
        # thread. We also poll opt.stats() because the threading.Event
        # sometimes fires after the stats counter increments.
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if opt.stats().get("total_fired", 0) >= 1:
                break
            fired.wait(timeout=0.5)
        assert opt.stats()["total_fired"] >= 1


def test_auto_optimizer_unknown_op_no_recursion():
    """
    _build_prompt must not recurse for op names outside the three
    known ops. Previously raised RecursionError via monkey-patch loop.
    """
    import tempfile
    from memopt.kernels.kernel_cache import KernelCache
    from memopt.kernels.portability_layer import PortabilityLayer
    from memopt.kernels.jit_generator import JITGenerator
    from memopt.serving.auto_optimizer import AutoOptimizer
    from memopt.kernels.bottleneck_detector import BottleneckEvent

    with tempfile.TemporaryDirectory() as d:
        cache = KernelCache(cache_dir=d)
        gen   = JITGenerator(cache=cache, portability=PortabilityLayer())
        opt   = AutoOptimizer(generator=gen, cache=cache)

        event = BottleneckEvent(
            op_name="memopt.unknown_custom_op",
            input_shapes=[[64, 64]],
            dtype="float16",
            access_pattern="sequential",
            stall_rate=0.5,
            hardware="cpu",
        )

        try:
            prompt = opt._build_prompt(event)
            assert isinstance(prompt, str) and len(prompt) > 0
        except RecursionError:
            pytest.fail(
                "_build_prompt raised RecursionError for unknown op. "
                "Fix: use JITGenerator._build_prompt(self._gen, event) "
                "in the fallback, not self._gen._build_prompt(event)."
            )


def test_optimizer_does_not_refire_within_gap():
    """Synthesis must not re-fire within SYNTHESIS_GAP_S seconds."""
    import time
    from memopt.serving.auto_optimizer import WARM_UP_CALLS

    fire_count = {"n": 0}

    class MockGen:
        def handle(self, event):
            fire_count["n"] += 1
        def stats(self):
            return {}
        def _build_prompt(self, e):
            return ""

    with tempfile.TemporaryDirectory() as d:
        cache = KernelCache(cache_dir=d)
        opt   = AutoOptimizer(generator=MockGen(), cache=cache)
        opt.start()

        try:
            import torch
            args = (torch.randn(64, 8, 32),)
        except ImportError:
            args = ()

        for _ in range(WARM_UP_CALLS * 3):
            opt.notify("memopt.rope_fused", args)

        time.sleep(0.2)
        assert fire_count["n"] <= 1, \
            f"Synthesis fired {fire_count['n']} times — should fire at most once"


# ── CUDA gather kernel availability ───────────────────────────────────

def test_has_cuda_gather_reflects_build():
    """has_cuda_gather() returns a bool reflecting CUDA compilation."""
    try:
        import memopt._memopt_paged as p
    except ImportError:
        pytest.skip("C++ paged extension not built")

    result = p.has_cuda_gather()
    assert isinstance(result, bool)
    # On this machine: just verify it doesn't crash
    # and returns a consistent value
    assert p.has_cuda_gather() == result


# ── Kubernetes health check endpoints ─────────────────────────────────

def test_healthz_returns_200_when_running():
    """Liveness probe returns 200 with status and checks."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi not installed")
    from memopt.serving.server import app
    if app is None:
        pytest.skip("FastAPI not available")
    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "node_id" in data
    assert "checks" in data
    assert "uptime_seconds" in data


def test_readyz_returns_200_or_503():
    """Readiness probe returns 200 or 503, never 500."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi not installed")
    from memopt.serving.server import app
    if app is None:
        pytest.skip("FastAPI not available")
    client = TestClient(app)
    response = client.get("/readyz")
    assert response.status_code in [200, 503]
    data = response.json()
    assert "status" in data
    assert "node_id" in data


def test_healthz_never_returns_500():
    """Liveness probe must return 200 or 503, never 500."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi not installed")
    from memopt.serving.server import app
    if app is None:
        pytest.skip("FastAPI not available")
    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code != 500


def test_health_endpoint_unchanged():
    """Original /health endpoint still works."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi not installed")
    from memopt.serving.server import app
    if app is None:
        pytest.skip("FastAPI not available")
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
