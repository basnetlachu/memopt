"""
Tests for Pillar 3 enhancements:
  - Flash attention synthesis prompt
  - Custom op spec and synthesis API
  - Kernel library persistence with SHA-256 integrity
  - Kernel library endpoint
  - Benchmark script syntax

All tests run without GPU, Triton, or Anthropic API key.
"""
import os
import sys
import tempfile
import threading
from unittest import mock

import pytest


# ── CustomOpSpec ────────────────────────────────────────────────────


def test_custom_op_spec_valid():
    from memopt.kernels.jit_generator import CustomOpSpec
    spec = CustomOpSpec(
        name="test_op",
        description="Element-wise GELU",
        input_shapes=[(4, 512, 768)],
        input_dtypes=["float16"],
        output_shape=(4, 512, 768))
    assert spec.name == "test_op"
    assert spec.atol == 1e-2


def test_custom_op_spec_empty_name_raises():
    from memopt.kernels.jit_generator import CustomOpSpec
    with pytest.raises(ValueError):
        CustomOpSpec(
            name="",
            description="test",
            input_shapes=[(4, 512)],
            input_dtypes=["float16"],
            output_shape=(4, 512))


def test_custom_op_spec_empty_desc_raises():
    from memopt.kernels.jit_generator import CustomOpSpec
    with pytest.raises(ValueError):
        CustomOpSpec(
            name="op",
            description="",
            input_shapes=[(4, 512)],
            input_dtypes=["float16"],
            output_shape=(4, 512))


def test_custom_op_spec_with_reference():
    from memopt.kernels.jit_generator import CustomOpSpec
    spec = CustomOpSpec(
        name="gelu",
        description="GELU activation",
        input_shapes=[(4, 512)],
        input_dtypes=["float16"],
        output_shape=(4, 512),
        reference_fn=lambda x: x,
        atol=1e-3,
        rtol=1e-3)
    assert spec.reference_fn is not None
    assert spec.atol == 1e-3


# ── Flash attention prompt ──────────────────────────────────────────


def test_flash_attention_prompt_has_requirements():
    from memopt.kernels.jit_generator import FLASH_ATTENTION_PROMPT
    assert "online softmax" in FLASH_ATTENTION_PROMPT
    assert "causal" in FLASH_ATTENTION_PROMPT
    assert "run_kernel" in FLASH_ATTENTION_PROMPT
    assert "tiling" in FLASH_ATTENTION_PROMPT


def test_flash_attention_prompt_format():
    from memopt.kernels.jit_generator import FLASH_ATTENTION_PROMPT
    # Should be formattable with the expected keys
    result = FLASH_ATTENTION_PROMPT.format(
        hardware="A100",
        q_shape=(2, 8, 512, 64),
        k_shape=(2, 8, 512, 64),
        v_shape=(2, 8, 512, 64),
        dtype="float16",
        causal=False)
    assert "A100" in result
    assert "(2, 8, 512, 64)" in result


# ── Custom op prompt ────────────────────────────────────────────────


def test_custom_op_prompt_has_requirements():
    from memopt.kernels.jit_generator import CUSTOM_OP_PROMPT
    assert "run_kernel" in CUSTOM_OP_PROMPT
    assert "@triton.jit" in CUSTOM_OP_PROMPT
    assert "{description}" in CUSTOM_OP_PROMPT


# ── Synthesize without API key ──────────────────────────────────────


def test_synthesize_flash_attention_no_key():
    """Without API key returns None cleanly."""
    from memopt.kernels.jit_generator import JITGenerator
    from memopt.kernels.kernel_cache import KernelCache
    from memopt.kernels.portability_layer import PortabilityLayer

    env = {k: v for k, v in os.environ.items()
           if k != "ANTHROPIC_API_KEY"}

    with mock.patch.dict(os.environ, env, clear=True):
        gen = JITGenerator(
            cache=KernelCache(),
            portability=PortabilityLayer())

        result = gen.synthesize_flash_attention(
            q_shape=(2, 8, 512, 64),
            k_shape=(2, 8, 512, 64),
            v_shape=(2, 8, 512, 64))

        assert result is None


def test_synthesize_custom_op_no_key():
    """Without API key returns None cleanly."""
    from memopt.kernels.jit_generator import JITGenerator, CustomOpSpec
    from memopt.kernels.kernel_cache import KernelCache
    from memopt.kernels.portability_layer import PortabilityLayer

    env = {k: v for k, v in os.environ.items()
           if k != "ANTHROPIC_API_KEY"}

    spec = CustomOpSpec(
        name="test_fused",
        description="GELU activation",
        input_shapes=[(4, 512, 768)],
        input_dtypes=["float16"],
        output_shape=(4, 512, 768))

    with mock.patch.dict(os.environ, env, clear=True):
        gen = JITGenerator(
            cache=KernelCache(),
            portability=PortabilityLayer())
        result = gen.synthesize_custom_op(spec)
        assert result is None


# ── Hardware description ────────────────────────────────────────────


def test_get_hardware_description_cpu():
    """On CPU returns a valid string."""
    from memopt.kernels.jit_generator import JITGenerator
    from memopt.kernels.kernel_cache import KernelCache
    from memopt.kernels.portability_layer import PortabilityLayer

    gen = JITGenerator(
        cache=KernelCache(),
        portability=PortabilityLayer())
    desc = gen._get_hardware_description()
    assert isinstance(desc, str)
    assert len(desc) > 0


# ── KernelLibrary ──────────────────────────────────────────────────


def test_kernel_library_save_and_get():
    from memopt.kernels.kernel_cache import KernelLibrary

    with tempfile.TemporaryDirectory() as d:
        lib = KernelLibrary(library_path=d)

        source = "def run_kernel(x): return x"
        metadata = {"op_name": "test_op", "speedup": 1.5}

        ok = lib.save(
            op_name="test_op",
            hardware_hash="abc123def456",
            source=source,
            metadata=metadata)
        assert ok is True

        entry = lib.get("test_op", "abc123def456")
        assert entry is not None
        assert entry["source"] == source
        assert entry["metadata"]["speedup"] == 1.5


def test_kernel_library_missing_returns_none():
    from memopt.kernels.kernel_cache import KernelLibrary

    with tempfile.TemporaryDirectory() as d:
        lib = KernelLibrary(library_path=d)
        result = lib.get("nonexistent_op", "abc123")
        assert result is None


def test_kernel_library_integrity_check():
    from memopt.kernels.kernel_cache import KernelLibrary

    with tempfile.TemporaryDirectory() as d:
        lib = KernelLibrary(library_path=d)
        source = "def run_kernel(x): return x"
        lib.save("tamper_op", "hashxyz12345", source, {"speedup": 1.5})

        # Tamper with source file
        entry_dir = lib._catalog.get("tamper_op_hashxyz12345")
        if entry_dir:
            src_path = os.path.join(entry_dir, "kernel.py")
            with open(src_path, "w") as f:
                f.write("TAMPERED CODE")

            # Get should detect tamper and return None
            result = lib.get("tamper_op", "hashxyz12345")
            assert result is None
            assert lib._stats["corrupted"] >= 1


def test_kernel_library_list():
    from memopt.kernels.kernel_cache import KernelLibrary

    with tempfile.TemporaryDirectory() as d:
        lib = KernelLibrary(library_path=d)

        lib.save("op_a", "hw_aaa_123456",
                 "def run_kernel(x): return x",
                 {"speedup": 1.2, "op_name": "op_a"})
        lib.save("op_b", "hw_bbb_123456",
                 "def run_kernel(x): return x",
                 {"speedup": 1.8, "op_name": "op_b"})

        kernels = lib.list_kernels()
        assert len(kernels) == 2
        op_names = [k["op_name"] for k in kernels]
        assert "op_a" in op_names
        assert "op_b" in op_names


def test_kernel_library_stats_keys():
    from memopt.kernels.kernel_cache import KernelLibrary

    with tempfile.TemporaryDirectory() as d:
        lib = KernelLibrary(library_path=d)
        stats = lib.stats()
        required = ["entries", "loads", "load_hits", "saves",
                    "corrupted", "library_path", "catalog_size"]
        for key in required:
            assert key in stats, f"Missing key: {key}"


def test_kernel_library_reload_from_disk():
    """Library reloads catalog after restart (new instance same path)."""
    from memopt.kernels.kernel_cache import KernelLibrary

    with tempfile.TemporaryDirectory() as d:
        lib1 = KernelLibrary(library_path=d)
        lib1.save("persist_op", "hw_persist123",
                  "def run_kernel(x): return x",
                  {"speedup": 2.0, "op_name": "persist_op"})

        # New instance same path — should find the entry
        lib2 = KernelLibrary(library_path=d)
        entry = lib2.get("persist_op", "hw_persist123")
        assert entry is not None
        assert entry["metadata"]["speedup"] == 2.0


def test_kernel_library_thread_safe():
    from memopt.kernels.kernel_cache import KernelLibrary

    with tempfile.TemporaryDirectory() as d:
        lib = KernelLibrary(library_path=d)
        errors = []

        def writer(n):
            try:
                for i in range(10):
                    lib.save(
                        f"op_{n}_{i}", f"hw_{n}_{i}_abcdef",
                        f"def run_kernel(x): return x  # {n}_{i}",
                        {"speedup": 1.1 + i * 0.1, "op_name": f"op_{n}_{i}"})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,))
                   for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert lib.stats()["saves"] == 40


# ── Kernel library endpoint ─────────────────────────────────────────


def test_kernel_library_endpoint():
    from fastapi.testclient import TestClient
    from memopt.serving.server import app
    client = TestClient(app)

    response = client.get("/kernels/library")
    assert response.status_code == 200
    data = response.json()

    required = ["total", "kernels", "note"]
    for key in required:
        assert key in data, f"Missing key: {key}"

    assert isinstance(data["kernels"], list)
    # Source code must not be in response
    for k in data["kernels"]:
        assert "source" not in k


# ── Benchmark script syntax ─────────────────────────────────────────


def test_benchmark_kernels_syntax():
    import ast
    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts", "benchmark_kernels.py")
    with open(script_path) as f:
        ast.parse(f.read())


def test_benchmark_op_gpu_required():
    """benchmark_op returns gpu_required on CPU."""
    import torch
    if torch.cuda.is_available():
        pytest.skip("GPU available")

    sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(sys_path, "scripts"))
    try:
        from benchmark_kernels import benchmark_op

        result = benchmark_op(
            op_name="test",
            shapes={"inputs": []},
            reference_fn=lambda: None,
            synthesize_fn=lambda: None)

        assert result["status"] == "gpu_required"
    finally:
        sys.path.pop(0)


# ── KernelCache.make_key ────────────────────────────────────────────


def test_kernel_cache_make_key():
    from memopt.kernels.kernel_cache import KernelCache
    cache = KernelCache()
    key = cache.make_key("test_op", [(4, 512)], "cuda:A100")
    assert isinstance(key, str)
    assert len(key) == 64  # SHA-256 hex


# ── get_kernel_library singleton ────────────────────────────────────


def test_get_kernel_library_singleton():
    from memopt.kernels.kernel_cache import get_kernel_library
    lib1 = get_kernel_library()
    lib2 = get_kernel_library()
    assert lib1 is lib2
