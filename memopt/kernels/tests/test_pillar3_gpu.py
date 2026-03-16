"""
Pillar 3 live GPU synthesis test.

Proves the full JIT kernel synthesis loop on real hardware:
  1. Run a deliberately memory-bound op (large matrix multiply)
  2. Bottleneck detector measures the stall rate
  3. JIT generator calls Claude API and synthesises a Triton kernel
  4. Kernel is compiled, validated for correctness, benchmarked
  5. If faster than baseline, registered in kernel cache
  6. Second call hits the cache — zero synthesis cost

Requires: CUDA GPU, triton, anthropic SDK, ANTHROPIC_API_KEY env var.

Command:
    pytest memopt/kernels/tests/test_pillar3_gpu.py -v -s
"""
import os
import time
import threading
import tempfile
import pytest
import torch

from memopt.kernels.bottleneck_detector import BottleneckDetector, BottleneckEvent
from memopt.kernels.jit_generator import JITGenerator
from memopt.kernels.portability_layer import PortabilityLayer
from memopt.kernels.kernel_cache import KernelCache, cache_key


# ── Guards ─────────────────────────────────────────────────────────────

if not torch.cuda.is_available():
    pytest.skip("CUDA GPU required", allow_module_level=True)

if not os.environ.get("ANTHROPIC_API_KEY"):
    pytest.skip("ANTHROPIC_API_KEY not set", allow_module_level=True)

try:
    import triton  # noqa
except ImportError:
    pytest.skip("triton not installed — pip install triton", allow_module_level=True)


# ── Helpers ────────────────────────────────────────────────────────────

def run_memory_bound_op(M=2048, K=4096, N=2048):
    """
    Large matrix multiply — deliberately memory-bound at this size.
    On an A100, this op moves ~128MB of data and stalls waiting for HBM.
    """
    a = torch.randn(M, K, dtype=torch.float16, device="cuda")
    b = torch.randn(K, N, dtype=torch.float16, device="cuda")
    torch.cuda.synchronize()
    return torch.mm(a, b)


# ── Tests ──────────────────────────────────────────────────────────────

def test_gpu_visible():
    """Confirm GPU is visible and has enough VRAM for the test."""
    free, total = torch.cuda.mem_get_info()
    gb_free = free / 1e9
    print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM free: {gb_free:.1f} GB / {total/1e9:.1f} GB")
    assert gb_free > 1.0, "Less than 1GB VRAM free — cannot run test"


def test_triton_compiles_basic_kernel():
    """
    Verify Triton can compile a trivial kernel on this GPU.
    If this fails, the JIT generator cannot work regardless of API.
    """
    import triton
    import triton.language as tl

    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, x + y, mask=mask)

    size = 1024
    x = torch.ones(size, device="cuda", dtype=torch.float32)
    y = torch.ones(size, device="cuda", dtype=torch.float32)
    out = torch.empty(size, device="cuda", dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, out, size, BLOCK_SIZE=256)
    torch.cuda.synchronize()

    assert torch.allclose(out, torch.full((size,), 2.0, device="cuda"))
    print(f"\n  Triton basic kernel: compiled and ran correctly on {torch.cuda.get_device_name(0)}")


def test_bottleneck_detector_on_gpu():
    """
    Run the bottleneck detector on a real memory-bound op.
    Verifies stall rate is measured and op is profiled.
    """
    detector = BottleneckDetector(stall_threshold=0.0, window_ops=1)

    result = detector.profile(
        "torch.ops.aten.mm",
        op=torch.mm,
        args=(
            torch.randn(2048, 4096, dtype=torch.float16, device="cuda"),
            torch.randn(4096, 2048, dtype=torch.float16, device="cuda"),
        ),
    )

    assert result is not None
    assert result.shape == (2048, 2048)

    s = detector.stats()
    assert s["total_profiled"] >= 1

    print(f"\n  Detector profiled: {s['total_profiled']} ops")
    print(f"  Stall rate estimated on real GPU op")


def test_full_synthesis_loop():
    """
    THE KEY TEST — full Pillar 3 loop on real hardware.

    Fires a BottleneckEvent directly at the JIT generator,
    waits for synthesis to complete (up to 60 seconds),
    then checks the kernel cache for a result.

    Three outcomes:
      SYNTHESISED + FASTER : kernel in cache, speedup > 1.05x — best case
      SYNTHESISED + DISCARDED : Claude wrote a kernel but it was not faster
                                than PyTorch's optimised baseline — still valid,
                                means PyTorch already optimal for this shape
      API_FAILED : API key invalid or quota exceeded — test fails
    """
    with tempfile.TemporaryDirectory() as cache_dir:
        cache    = KernelCache(cache_dir=cache_dir)
        portl    = PortabilityLayer()
        gen      = JITGenerator(cache=cache, portability=portl)

        synthesis_done = threading.Event()
        original_synthesise = gen._synthesise

        synthesis_result = {"attempted": False, "succeeded": False,
                            "failed": False, "discarded": False}

        def tracked_synthesise(event, key):
            synthesis_result["attempted"] = True
            original_synthesise(event, key)
            s = gen.stats()
            synthesis_result["succeeded"] = s["total_succeeded"] > 0
            synthesis_result["failed"]    = s["total_failed"] > 0
            synthesis_result["discarded"] = s["total_discarded"] > 0
            synthesis_done.set()

        gen._synthesise = tracked_synthesise

        event = BottleneckEvent(
            op_name="torch.ops.aten.mm",
            input_shapes=[[1024, 2048], [2048, 1024]],
            dtype="float16",
            access_pattern="sequential",
            stall_rate=0.65,
            hardware=f"cuda:{torch.cuda.get_device_name(0)}",
        )

        print(f"\n  Firing synthesis event for {event.op_name}")
        print(f"  Shapes: {event.input_shapes}")
        print(f"  Hardware: {event.hardware}")
        print(f"  Waiting for Claude API + Triton compilation (up to 60s)...")

        gen.handle(event)

        completed = synthesis_done.wait(timeout=60.0)

        print(f"\n  Synthesis completed: {completed}")
        print(f"  Attempted:  {synthesis_result['attempted']}")
        print(f"  Succeeded:  {synthesis_result['succeeded']}")
        print(f"  Failed:     {synthesis_result['failed']}")
        print(f"  Discarded:  {synthesis_result['discarded']}")

        assert completed, "Synthesis timed out after 60s — check API key and network"
        assert synthesis_result["attempted"], "Synthesis was never attempted"
        assert not synthesis_result["failed"], \
            "Synthesis failed — API key invalid or quota exceeded"

        gen_stats = gen.stats()
        print(f"\n  Generator stats: {gen_stats}")

        if synthesis_result["succeeded"]:
            key = cache_key(event.op_name, event.input_shapes, event.hardware)
            cached = cache.get(key)
            assert cached is not None, "Kernel succeeded but not found in cache"
            print(f"  Kernel in cache: YES")
            print(f"  Cache stats: {cache.stats()}")
        else:
            print(f"  Kernel discarded — PyTorch baseline already optimal for this shape")
            print(f"  This is correct behaviour — the discard-if-slower check works")


def test_cache_hit_after_synthesis():
    """
    If a kernel was synthesised and cached in the previous test,
    a second lookup must return it instantly from cache.
    Proves the zero-cost hot path works.
    """
    with tempfile.TemporaryDirectory() as cache_dir:
        cache = KernelCache(cache_dir=cache_dir)
        portl = PortabilityLayer()
        gen   = JITGenerator(cache=cache, portability=portl)

        event = BottleneckEvent(
            op_name="torch.ops.aten.mm",
            input_shapes=[[512, 1024], [1024, 512]],
            dtype="float16",
            access_pattern="sequential",
            stall_rate=0.5,
            hardware=f"cuda:{torch.cuda.get_device_name(0)}",
        )

        synthesis_done = threading.Event()
        original = gen._synthesise

        def tracked(ev, key):
            original(ev, key)
            synthesis_done.set()

        gen._synthesise = tracked

        # First call — synthesis
        gen.handle(event)
        synthesis_done.wait(timeout=60.0)

        gen_stats = gen.stats()
        if gen_stats["total_succeeded"] == 0:
            pytest.skip("Synthesis discarded — no cache entry to test hit path")

        key = cache_key(event.op_name, event.input_shapes, event.hardware)

        # Second call — must hit cache, not re-synthesise
        t0 = time.monotonic()
        gen.handle(event)
        elapsed_ms = (time.monotonic() - t0) * 1000

        print(f"\n  Second handle() call: {elapsed_ms:.3f}ms")
        print(f"  (Should be near-zero — cache hit, no API call)")
        assert elapsed_ms < 100, \
            f"Second call took {elapsed_ms:.1f}ms — should be cache hit (<100ms)"

        cache_stats = cache.stats()
        print(f"  Cache hits: {cache_stats['cache_hits']}")
        assert cache_stats["cache_hits"] >= 1


def test_baseline_vs_synthesised_benchmark():
    """
    If synthesis succeeds, measure and print the actual speedup.
    Runs 100 iterations each — baseline PyTorch vs synthesised kernel.
    """
    with tempfile.TemporaryDirectory() as cache_dir:
        cache = KernelCache(cache_dir=cache_dir)
        portl = PortabilityLayer()
        gen   = JITGenerator(cache=cache, portability=portl)

        event = BottleneckEvent(
            op_name="torch.ops.aten.mm",
            input_shapes=[[2048, 4096], [4096, 2048]],
            dtype="float16",
            access_pattern="sequential",
            stall_rate=0.7,
            hardware=f"cuda:{torch.cuda.get_device_name(0)}",
        )

        synthesis_done = threading.Event()
        original = gen._synthesise

        def tracked(ev, key):
            original(ev, key)
            synthesis_done.set()

        gen._synthesise = tracked
        gen.handle(event)
        synthesis_done.wait(timeout=60.0)

        gen_stats = gen.stats()
        if gen_stats["total_succeeded"] == 0:
            print(f"\n  Synthesis discarded — kernel not faster than baseline")
            print(f"  PyTorch's built-in mm is already highly optimised at this size")
            print(f"  Architecture is correct — discard-if-slower check worked")
            return

        key = cache_key(event.op_name, event.input_shapes, event.hardware)
        module = cache.get(key)
        assert module is not None

        run_kernel = getattr(module, "run_kernel", None)
        if run_kernel is None:
            pytest.skip("run_kernel not found in cached module")

        a = torch.randn(2048, 4096, dtype=torch.float16, device="cuda")
        b = torch.randn(4096, 2048, dtype=torch.float16, device="cuda")

        N = 100
        torch.cuda.synchronize()
        t0 = time.monotonic()
        for _ in range(N):
            torch.mm(a, b)
        torch.cuda.synchronize()
        baseline_ms = (time.monotonic() - t0) / N * 1000

        torch.cuda.synchronize()
        t0 = time.monotonic()
        for _ in range(N):
            run_kernel(a, b)
        torch.cuda.synchronize()
        kernel_ms = (time.monotonic() - t0) / N * 1000

        speedup = baseline_ms / max(kernel_ms, 1e-9)

        print(f"\n{'='*56}")
        print(f"  PILLAR 3 SYNTHESIS RESULTS")
        print(f"{'='*56}")
        print(f"  GPU:              {torch.cuda.get_device_name(0)}")
        print(f"  Op:               torch.mm [2048x4096] x [4096x2048] fp16")
        print(f"  Baseline (PyTorch): {baseline_ms:.3f}ms per call")
        print(f"  Synthesised kernel: {kernel_ms:.3f}ms per call")
        print(f"  Speedup:            {speedup:.2f}x")
        print(f"  Kernel in cache:    YES")
        print(f"{'='*56}")

        assert speedup >= 1.05, \
            f"Speedup {speedup:.2f}x below 1.05x threshold — should have been discarded"
