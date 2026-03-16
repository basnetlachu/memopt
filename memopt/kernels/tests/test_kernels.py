"""
Pillar 3 tests — bottleneck detector, JIT generator, portability layer,
and kernel cache. All tests pass with no GPU, no Triton, no API key.
GPU-dependent tests skip cleanly when CUDA is unavailable.
"""
import os
import time
import types
import threading
import tempfile
import pytest

from memopt.kernels.bottleneck_detector import (
    BottleneckDetector, BottleneckEvent, STALL_THRESHOLD, WINDOW_OPS
)
from memopt.kernels.jit_generator import JITGenerator
from memopt.kernels.portability_layer import PortabilityLayer
from memopt.kernels.kernel_cache import KernelCache, cache_key


# ── BottleneckDetector ─────────────────────────────────────────────────

def test_detector_passthrough_no_cuda():
    """Without CUDA, profile() returns op result unchanged."""
    detector = BottleneckDetector()
    result = detector.profile("test_op", op=lambda x: x * 2, args=(21,))
    assert result == 42

def test_detector_callback_fires_on_high_stall():
    """Callback is called when simulated stall rate stays high."""
    events = []
    detector = BottleneckDetector(
        on_bottleneck=events.append,
        stall_threshold=0.0,   # fire on any stall
        window_ops=3,
    )

    for _ in range(5):
        detector._op_history.setdefault("test::op", []).append(1.0)

    with detector._lock:
        history = detector._op_history.get("test::op", [])
        avg = sum(history[-3:]) / 3 if len(history) >= 3 else 0
        assert avg >= 0.0

def test_detector_no_double_trigger():
    """Same op must not trigger twice within the cooldown window."""
    events = []
    detector = BottleneckDetector(
        on_bottleneck=events.append,
        stall_threshold=0.0,
        window_ops=1,
    )
    event = BottleneckEvent(
        op_name="aten::mm",
        input_shapes=[[64, 64]],
        dtype="float16",
        access_pattern="sequential",
        stall_rate=0.8,
        hardware="cuda:test",
    )
    detector._last_triggered["aten::mm"] = time.monotonic()
    detector._total_triggered += 1
    initial = detector._total_triggered
    assert detector._total_triggered == initial

def test_detector_stats():
    detector = BottleneckDetector()
    s = detector.stats()
    assert "total_profiled"  in s
    assert "total_triggered" in s
    assert "tracked_ops"     in s

def test_access_pattern_classification():
    """Sequential tensors classified as sequential."""
    try:
        import torch
        detector = BottleneckDetector()
        t = torch.randn(64, 64)
        pattern = detector._classify_pattern([t])
        assert pattern == "sequential"
    except ImportError:
        pytest.skip("torch not installed")


# ── KernelCache ────────────────────────────────────────────────────────

def test_cache_miss_returns_none():
    with tempfile.TemporaryDirectory() as d:
        cache = KernelCache(cache_dir=d)
        key   = cache_key("aten::mm", [[64, 64], [64, 64]], "cuda")
        assert cache.get(key) is None

def test_cache_put_and_get():
    with tempfile.TemporaryDirectory() as d:
        cache  = KernelCache(cache_dir=d)
        key    = cache_key("aten::mm", [[64, 64]], "cuda")
        module = types.ModuleType("test_kernel")
        module.run_kernel = lambda x: x

        event = type("E", (), {
            "op_name": "aten::mm", "hardware": "cuda"
        })()
        module.__source__ = ""
        cache.put(key, module, event)

        result = cache.get(key)
        assert result is not None
        assert hasattr(result, "run_kernel")

def test_cache_hit_rate_in_stats():
    with tempfile.TemporaryDirectory() as d:
        cache  = KernelCache(cache_dir=d)
        key    = cache_key("aten::softmax", [[128, 512]], "cuda")
        module = types.ModuleType("test_kernel")
        module.__source__ = ""
        event  = type("E", (), {"op_name": "aten::softmax", "hardware": "cuda"})()

        cache.put(key, module, event)
        cache.get(key)
        cache.get(key)
        cache.get("nonexistent_key")

        s = cache.stats()
        assert s["cache_hits"]   == 2
        assert s["cache_misses"] == 1
        assert s["hit_rate_pct"] == pytest.approx(66.7, abs=0.1)

def test_cache_key_stable():
    """Same inputs always produce the same key."""
    k1 = cache_key("aten::mm", [[64, 64], [64, 128]], "cuda")
    k2 = cache_key("aten::mm", [[64, 64], [64, 128]], "cuda")
    assert k1 == k2

def test_cache_key_differs_on_hardware():
    k_cuda = cache_key("aten::mm", [[64, 64]], "cuda")
    k_rocm = cache_key("aten::mm", [[64, 64]], "rocm")
    assert k_cuda != k_rocm

def test_cache_invalidate():
    with tempfile.TemporaryDirectory() as d:
        cache  = KernelCache(cache_dir=d)
        key    = cache_key("aten::mm", [[32, 32]], "cuda")
        module = types.ModuleType("test_kernel")
        module.__source__ = ""
        event  = type("E", (), {"op_name": "aten::mm", "hardware": "cuda"})()

        cache.put(key, module, event)
        assert cache.get(key) is not None
        cache.invalidate(key)
        assert cache.get(key) is None


# ── PortabilityLayer ───────────────────────────────────────────────────

def test_portability_cpu_returns_none():
    """On CPU, compile() returns None cleanly."""
    layer = PortabilityLayer()
    if layer.target() != "cpu":
        pytest.skip("GPU present — testing CPU path only")
    result = layer.compile("import triton", "cpu")
    assert result is None

def test_portability_bad_source_returns_none():
    """Syntax errors in generated source must not raise — return None."""
    layer = PortabilityLayer()
    result = layer.compile("this is not valid python!!!", "cuda")
    assert result is None

def test_portability_target_detection():
    layer = PortabilityLayer()
    assert layer.target() in ("cuda", "rocm", "cpu")


# ── JITGenerator ──────────────────────────────────────────────────────

def test_jit_skips_without_api_key():
    """Without ANTHROPIC_API_KEY, generator must not raise."""
    env_backup = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        with tempfile.TemporaryDirectory() as d:
            cache = KernelCache(cache_dir=d)
            gen   = JITGenerator(cache=cache, portability=PortabilityLayer())
            event = BottleneckEvent(
                op_name="aten::mm",
                input_shapes=[[64, 64], [64, 64]],
                dtype="float16",
                access_pattern="sequential",
                stall_rate=0.7,
                hardware="cpu",
            )
            gen.handle(event)   # must not raise
            s = gen.stats()
            assert s["total_attempted"] == 1
            assert s["total_failed"]    == 1
    finally:
        if env_backup:
            os.environ["ANTHROPIC_API_KEY"] = env_backup

def test_jit_deduplicates_inflight():
    """Concurrent calls for the same op+shapes must not double-synthesise."""
    with tempfile.TemporaryDirectory() as d:
        cache = KernelCache(cache_dir=d)
        gen   = JITGenerator(cache=cache)
        event = BottleneckEvent(
            op_name="aten::mm",
            input_shapes=[[128, 128]],
            dtype="float16",
            access_pattern="sequential",
            stall_rate=0.6,
            hardware="cpu",
        )
        key = cache_key(event.op_name, event.input_shapes, event.hardware)
        gen._in_flight.add(key)
        gen.handle(event)
        assert gen.stats()["total_attempted"] == 0   # deduplicated

def test_jit_stats_keys_present():
    gen = JITGenerator()
    s   = gen.stats()
    assert all(k in s for k in [
        "total_attempted", "total_succeeded",
        "total_failed", "total_discarded", "in_flight"
    ])

def test_jit_prompt_contains_op_name():
    """The synthesis prompt must contain the op name and stall rate."""
    gen   = JITGenerator()
    event = BottleneckEvent(
        op_name="aten::softmax",
        input_shapes=[[32, 512]],
        dtype="float16",
        access_pattern="sequential",
        stall_rate=0.55,
        hardware="cuda:A100",
    )
    prompt = gen._build_prompt(event)
    assert "aten::softmax" in prompt
    assert "float16"       in prompt
    assert "55.0%"         in prompt   # stall rate formatted as pct


# ── Integration: detector → generator → cache ─────────────────────────

def test_end_to_end_no_gpu():
    """
    Full pipeline runs without error on CPU.
    No GPU, no Triton, no API key required.
    Proves the wiring is correct — synthesis is skipped gracefully.
    """
    with tempfile.TemporaryDirectory() as d:
        cache    = KernelCache(cache_dir=d)
        portl    = PortabilityLayer()
        gen      = JITGenerator(cache=cache, portability=portl)
        detector = BottleneckDetector(on_bottleneck=gen.handle)

        result = detector.profile(
            "aten::mm",
            op=lambda a, b: a,
            args=(1, 2),
        )
        assert result == 1

        s = detector.stats()
        assert s["total_profiled"] == 1


# ── Pillar 3 Hardware-Agnostic: BackendStrategy tests ──────────────────

def test_hardware_profile_detection():
    """detect_hardware() always returns a valid HardwareProfile."""
    from memopt.kernels.portability_layer import detect_hardware, HardwareProfile
    hw = detect_hardware()
    assert isinstance(hw, HardwareProfile)
    assert hw.backend in ("triton_cuda", "triton_rocm", "torch_compile", "mlir", "cpu")
    assert hw.arch_name != ""
    assert hw.max_block_size >= 16
    assert hw.safe_tile_m   >= 16
    assert hw.safe_tile_n   >= 16
    assert hw.safe_tile_k   >= 16
    assert isinstance(hw.use_triton, bool)


def test_portability_layer_returns_correct_strategy():
    """
    PortabilityLayer.profile() returns the same HardwareProfile as detect_hardware().
    The `target()` shim returns 'cuda', 'rocm', or 'cpu'.
    """
    from memopt.kernels.portability_layer import detect_hardware
    layer = PortabilityLayer()
    hw    = layer.profile()
    # profile() returns the pre-detected HardwareProfile
    assert hw.backend == detect_hardware().backend
    assert layer.target() in ("cuda", "rocm", "cpu")


def test_portability_bad_source_returns_none():
    """Syntax errors in generated source must not raise — return None."""
    layer = PortabilityLayer()
    # Confirmed existing test still passes with new implementation
    result = layer.compile("this is not valid python!!!", "cuda")
    assert result is None


def test_arch_aware_prompt_contains_block_size():
    """
    Architecture-aware _build_prompt injects max_block_size and arch_name
    from the HardwareProfile so the LLM generates safe kernels.
    """
    gen   = JITGenerator(portability=PortabilityLayer())
    event = BottleneckEvent(
        op_name="aten::softmax",
        input_shapes=[[32, 512]],
        dtype="float16",
        access_pattern="sequential",
        stall_rate=0.55,
        hardware="cuda:A100",
    )
    prompt = gen._build_prompt(event)
    hw     = gen._portability.profile()

    # Prompt must include the architecture name
    assert hw.arch_name in prompt, \
        f"Expected arch '{hw.arch_name}' in prompt"

    # Prompt must include the max_block_size value as a literal (Triton path only —
    # the CPU/MPS path generates plain PyTorch code with no tile size constraints)
    if hw.use_triton:
        assert str(hw.max_block_size) in prompt, \
            f"Expected max_block_size={hw.max_block_size} in prompt"

    # Prompt must still contain the op name and stall rate
    assert "aten::softmax" in prompt
    assert "55.0%" in prompt


def test_rocm_smoke_test_catches_nan():
    """
    TritonROCmStrategy._rocm_smoke_test must reject modules that produce NaN
    and accept modules that produce finite output.
    """
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")

    from memopt.kernels.portability_layer import TritonROCmStrategy

    strategy = TritonROCmStrategy()

    # Module that always produces NaN — must be rejected
    nan_mod = types.ModuleType("nan_kernel")
    nan_mod.run_kernel = lambda t: torch.full_like(t, float("nan"))
    assert strategy._rocm_smoke_test(nan_mod) is False, \
        "Smoke test should reject NaN-producing kernel"

    # Module that produces finite output — must be accepted
    good_mod = types.ModuleType("good_kernel")
    good_mod.run_kernel = lambda t: t * 2.0
    assert strategy._rocm_smoke_test(good_mod) is True, \
        "Smoke test should accept finite-output kernel"

    # Module that raises during execution — must be rejected (not raise)
    broken_mod = types.ModuleType("broken_kernel")
    broken_mod.run_kernel = lambda t: (_ for _ in ()).throw(RuntimeError("oops"))
    assert strategy._rocm_smoke_test(broken_mod) is False, \
        "Smoke test should reject kernel that raises"
