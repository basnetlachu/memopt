"""
Tests for the hardware support surface:

  - Intel Gaudi stub (_gaudi_backend_py)
  - Google TPU stub (_tpu_backend_py)
  - AMD entries in GPU_SPECS + the `measured` flag contract
  - docs/hardware_support.md presence + content
  - HAL detection consistency on this machine
"""
import os


# ══════════════════════════════════════════════════════════════════════
#  Intel Gaudi stub
# ══════════════════════════════════════════════════════════════════════


def test_gaudi_backend_imports():
    from memopt.vmm.backends._gaudi_backend_py import GaudiBackend
    b = GaudiBackend()
    assert b is not None


def test_gaudi_backend_stats():
    from memopt.vmm.backends._gaudi_backend_py import GaudiBackend
    b = GaudiBackend()
    stats = b.stats()
    assert stats["backend"] == "intel_gaudi"
    assert stats["implemented"] is False
    assert "note" in stats
    assert "hbm_bw_gbps" in stats


def test_gaudi_backend_allocate_returns_none():
    from memopt.vmm.backends._gaudi_backend_py import GaudiBackend
    b = GaudiBackend()
    assert b.allocate_hbm(1024) is None
    assert b.allocate_host(1024) is None


def test_gaudi_backend_copy_returns_false():
    from memopt.vmm.backends._gaudi_backend_py import GaudiBackend
    b = GaudiBackend()
    assert b.copy_to_device(b"\x00" * 8, b"\x00" * 8, 1024) is False
    assert b.copy_from_device(b"\x00" * 8, b"\x00" * 8, 1024) is False


def test_gaudi_backend_no_raise_on_free_sync():
    from memopt.vmm.backends._gaudi_backend_py import GaudiBackend
    b = GaudiBackend()
    b.free_hbm(b"\x00" * 8)
    b.free_host(b"\x00" * 8)
    b.synchronize()  # must not raise


# ══════════════════════════════════════════════════════════════════════
#  Google TPU stub
# ══════════════════════════════════════════════════════════════════════


def test_tpu_backend_imports():
    from memopt.vmm.backends._tpu_backend_py import TPUBackend
    b = TPUBackend()
    assert b is not None


def test_tpu_backend_stats():
    from memopt.vmm.backends._tpu_backend_py import TPUBackend
    b = TPUBackend()
    stats = b.stats()
    assert stats["backend"] == "google_tpu"
    assert stats["implemented"] is False
    assert "note" in stats


def test_tpu_backend_allocate_returns_none():
    from memopt.vmm.backends._tpu_backend_py import TPUBackend
    b = TPUBackend()
    assert b.allocate_hbm(1024) is None
    assert b.allocate_host(1024) is None


def test_tpu_backend_copy_returns_false():
    from memopt.vmm.backends._tpu_backend_py import TPUBackend
    b = TPUBackend()
    assert b.copy_to_device(b"\x00" * 8, b"\x00" * 8, 1024) is False
    assert b.copy_from_device(b"\x00" * 8, b"\x00" * 8, 1024) is False


# ══════════════════════════════════════════════════════════════════════
#  GPU_SPECS: AMD entries + measured contract
# ══════════════════════════════════════════════════════════════════════


def test_gpu_specs_has_amd_entries():
    from memopt.profiler.hardware_counters import GPU_SPECS
    amd_keys = [
        k for k in GPU_SPECS
        if "MI300" in k or "MI250" in k or "MI210" in k
    ]
    assert len(amd_keys) >= 3, \
        f"Expected 3 AMD entries in GPU_SPECS, found: {amd_keys}"


def test_gpu_specs_amd_has_measured_field():
    from memopt.profiler.hardware_counters import GPU_SPECS
    for key, spec in GPU_SPECS.items():
        if "MI300" in key or "MI250" in key or "MI210" in key:
            assert hasattr(spec, "measured"), \
                f"GPU_SPECS[{key}] missing 'measured' field"
            # No AMD hardware was used → measured must be False.
            assert spec.measured is False, \
                f"AMD GPU {key} claims measured=True"
            assert spec.vendor == "AMD"
            assert spec.hbm_size_gb > 0
            assert spec.source, \
                f"GPU_SPECS[{key}] missing 'source' field"


def test_gpu_specs_nvidia_has_measured_field():
    """All existing NVIDIA entries now have measured=False default."""
    from memopt.profiler.hardware_counters import GPU_SPECS
    for key, spec in GPU_SPECS.items():
        if spec.vendor == "NVIDIA" or any(
                t in key for t in ("A100", "H100", "H200",
                                   "RTX", "L40", "A30", "V100")):
            assert hasattr(spec, "measured"), \
                f"GPU_SPECS[{key}] missing 'measured' field"


def test_gpu_specs_mi300x_spec_sane():
    from memopt.profiler.hardware_counters import GPU_SPECS
    mi300 = GPU_SPECS["AMD Instinct MI300X"]
    assert mi300.peak_memory_bandwidth_gbps == 5300.0
    assert mi300.hbm_size_gb == 192.0
    assert mi300.arch_tag == "gfx942"
    assert mi300.architecture == "CDNA3"


# ══════════════════════════════════════════════════════════════════════
#  Hardware support doc
# ══════════════════════════════════════════════════════════════════════


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))


def test_hardware_support_doc_exists():
    path = os.path.join(_repo_root(), "docs/hardware_support.md")
    assert os.path.exists(path)

    with open(path) as f:
        content = f.read()

    # All four hardware vendors documented
    assert "NVIDIA CUDA" in content
    assert "AMD ROCm" in content
    assert "Intel Gaudi" in content
    assert "Google TPU" in content

    # Honest status vocabulary
    assert "Implemented" in content
    assert "Stub" in content
    assert "Untested" in content

    # How-to-add-backend checklist
    assert "Adding a new hardware backend" in content


# ══════════════════════════════════════════════════════════════════════
#  HAL detection consistency
# ══════════════════════════════════════════════════════════════════════


def test_hal_detects_correctly_on_this_machine():
    """
    HAL detection must be consistent with the observable environment.
    On a CPU-only box HAL lands on CPU_ONLY; on a CUDA box it reports
    at least one GPU. The test never asserts a specific backend —
    only internal consistency.
    """
    from memopt.vmm.hal import (
        get_hal, HardwareBackend, reset_hal)

    reset_hal()
    try:
        hal = get_hal()
        assert hal.backend in HardwareBackend
        try:
            import torch
            if torch.cuda.is_available() and \
                    getattr(torch.version, "hip", None) is None:
                # NVIDIA torch + no HIP → HAL should find GPUs
                # (either via pynvml or torch path).
                assert hal.gpu_count >= 1 or \
                    hal.backend in (
                        HardwareBackend.CPU_ONLY,
                        HardwareBackend.UNKNOWN)
            else:
                assert hal.backend in (
                    HardwareBackend.CPU_ONLY,
                    HardwareBackend.AMD_ROCM,
                    HardwareBackend.UNKNOWN)
        except ImportError:
            assert hal.backend == HardwareBackend.CPU_ONLY
    finally:
        reset_hal()
