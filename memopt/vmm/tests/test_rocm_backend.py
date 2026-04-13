"""
Tests for the AMD ROCm backend.

Runs on every platform. Tests that would exercise real HIP API
calls auto-skip unless the C++ extension is built with
`-DMEMOPT_ROCM_AVAILABLE` on actual AMD hardware.
"""
import os

import pytest


def _rocm_cpp_available() -> bool:
    try:
        import memopt._memopt_rocm as r
        return bool(getattr(r, "ROCM_AVAILABLE", False))
    except ImportError:
        return False


requires_rocm_hw = pytest.mark.skipif(
    not _rocm_cpp_available(),
    reason="ROCm C++ not built (requires AMD hardware + ROCm)")


# ══════════════════════════════════════════════════════════════════════
#  Python shim — works on any platform
# ══════════════════════════════════════════════════════════════════════


def test_rocm_backend_imports():
    from memopt.vmm.backends._rocm_backend_py import ROCmBackend
    backend = ROCmBackend()
    assert backend is not None


def test_rocm_backend_init_no_crash():
    from memopt.vmm.backends._rocm_backend_py import ROCmBackend
    backend = ROCmBackend(device_id=0)
    stats = backend.stats()
    assert stats["backend"] == "amd_rocm"
    assert stats["device_id"] == 0


def test_rocm_backend_stats_keys():
    from memopt.vmm.backends._rocm_backend_py import ROCmBackend
    backend = ROCmBackend()
    stats = backend.stats()
    for key in (
            "backend", "device_id", "cpp_available",
            "unified_memory", "gcn_arch",
            "total_gb", "free_gb",
            "hbm_bw_gbps", "dram_bw_gbps"):
        assert key in stats, f"Missing stats key: {key}"


def test_rocm_backend_never_raises_on_alloc():
    from memopt.vmm.backends._rocm_backend_py import ROCmBackend
    backend = ROCmBackend()
    # Allocate+free round-trip should never raise whether or not
    # the C++ extension is present.
    ptr = backend.allocate_hbm(1024)
    assert ptr is not None
    backend.free_hbm(ptr)


def test_rocm_backend_fallback_allocate_shape():
    """Without C++: returns an 8-byte fake pointer."""
    from memopt.vmm.backends._rocm_backend_py import ROCmBackend
    backend = ROCmBackend()
    if not backend._cpp_available:
        ptr = backend.allocate_hbm(1024)
        assert ptr is not None
        assert len(ptr) == 8


def test_rocm_backend_fallback_copy_returns_true():
    """Without C++: copy methods return True (simulated success)."""
    from memopt.vmm.backends._rocm_backend_py import ROCmBackend
    backend = ROCmBackend()
    if not backend._cpp_available:
        fake_ptr = b"\x00" * 8
        assert backend.copy_to_device(
            fake_ptr, fake_ptr, 1024) is True
        assert backend.copy_from_device(
            fake_ptr, fake_ptr, 1024) is True


def test_rocm_backend_synchronize_never_raises():
    from memopt.vmm.backends._rocm_backend_py import ROCmBackend
    backend = ROCmBackend()
    backend.synchronize()


def test_rocm_backend_bandwidth_env_override():
    """Bandwidth constants are env-configurable for calibration."""
    old = os.environ.get("MEMOPT_ROCM_HBM_BW_GBPS")
    os.environ["MEMOPT_ROCM_HBM_BW_GBPS"] = "9999.0"
    try:
        # Module re-import picks up the new env var; we do a fresh
        # import to exercise the class-level constant evaluation.
        import importlib
        import memopt.vmm.backends._rocm_backend_py as mod
        importlib.reload(mod)
        assert mod.ROCmBackend._HBM_BANDWIDTH_GBPS == 9999.0
    finally:
        if old is not None:
            os.environ["MEMOPT_ROCM_HBM_BW_GBPS"] = old
        else:
            os.environ.pop("MEMOPT_ROCM_HBM_BW_GBPS", None)
        # Restore original default for downstream tests
        import importlib
        import memopt.vmm.backends._rocm_backend_py as mod
        importlib.reload(mod)


# ══════════════════════════════════════════════════════════════════════
#  Legacy detect_tiers() contract (consumed by VMM HAL)
# ══════════════════════════════════════════════════════════════════════


def test_rocm_backend_detect_tiers_returns_three():
    from memopt.vmm.backends._rocm_backend_py import ROCmBackend
    backend = ROCmBackend()
    tiers = backend.detect_tiers()
    assert len(tiers) == 3
    names = [t.name for t in tiers]
    assert names == ["hbm", "dram", "nvme"]


def test_rocm_backend_tiers_have_bandwidth():
    from memopt.vmm.backends._rocm_backend_py import ROCmBackend
    backend = ROCmBackend()
    tiers = backend.detect_tiers()
    for t in tiers:
        assert t.bandwidth_gbps > 0
        assert t.latency_us > 0


# ══════════════════════════════════════════════════════════════════════
#  Stub module imports cleanly on non-ROCm hosts
# ══════════════════════════════════════════════════════════════════════


def test_rocm_stub_import():
    """_memopt_rocm either imports (real or stub) or fails cleanly."""
    try:
        import memopt._memopt_rocm as r
        # If imported, it has ROCM_AVAILABLE flag
        assert hasattr(r, "ROCM_AVAILABLE")
        assert isinstance(r.ROCM_AVAILABLE, bool)
    except ImportError:
        # C++ extension not built yet — acceptable.
        pass


# ══════════════════════════════════════════════════════════════════════
#  Hardware-only integration tests (auto-skip without AMD)
# ══════════════════════════════════════════════════════════════════════


@requires_rocm_hw
def test_rocm_device_info():
    import memopt._memopt_rocm as r
    info = r.get_device_info(0)
    assert info.total_memory > 0
    assert len(info.name) > 0
    assert len(info.gcn_arch) > 0


@requires_rocm_hw
def test_rocm_alloc_free():
    import memopt._memopt_rocm as r
    ptr = r.hip_alloc(1024 * 1024)
    assert ptr is not None
    assert len(ptr) == 8  # pointer-size bytes
    r.hip_free(ptr)


@requires_rocm_hw
def test_rocm_unified_memory_mi300x():
    import memopt._memopt_rocm as r
    info = r.get_device_info(0)
    if "gfx94" in info.gcn_arch:
        assert info.has_unified_memory is True


@requires_rocm_hw
def test_rocm_is_unified_memory_device_matches_info():
    import memopt._memopt_rocm as r
    info = r.get_device_info(0)
    assert r.is_unified_memory_device(0) == \
        info.has_unified_memory
