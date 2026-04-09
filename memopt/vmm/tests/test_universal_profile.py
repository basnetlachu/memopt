"""Tests for vmm.universal_profile — runs on CPU with no GPU."""
import os
import tempfile

import pytest

from memopt.vmm.universal_profile import (
    MemoryTier,
    UniversalMemoryProfile,
    detect_universal_profile,
)

_VALID_ARCHITECTURES = {
    "cuda_ampere", "cuda_hopper", "cuda_blackwell",
    "cuda_unknown", "rocm", "cpu", "unknown",
}

_cuda_available = False
try:
    import torch
    _cuda_available = torch.cuda.is_available()
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────


def test_profile_returns_universal_memory_profile():
    p = detect_universal_profile()
    assert isinstance(p, UniversalMemoryProfile)


def test_profile_never_raises():
    # Call twice (idempotent) — should never raise regardless of env
    p1 = detect_universal_profile()
    p2 = detect_universal_profile()
    assert p1 is not None and p2 is not None


def test_dram_tier_always_present():
    p = detect_universal_profile()
    names = [t.name for t in p.tiers]
    assert "dram" in names


@pytest.mark.skipif(not _cuda_available, reason="No CUDA GPU available")
def test_hbm_tier_present_on_cuda():
    p = detect_universal_profile()
    names = [t.name for t in p.tiers]
    assert "hbm" in names


def test_nvme_tier_present_when_env_set():
    with tempfile.TemporaryDirectory() as td:
        old = os.environ.get("MEMOPT_NVME_DIR")
        try:
            os.environ["MEMOPT_NVME_DIR"] = td
            p = detect_universal_profile()
            names = [t.name for t in p.tiers]
            assert "nvme" in names
        finally:
            if old is None:
                os.environ.pop("MEMOPT_NVME_DIR", None)
            else:
                os.environ["MEMOPT_NVME_DIR"] = old


def test_nvme_tier_absent_when_env_not_set():
    old = os.environ.get("MEMOPT_NVME_DIR")
    try:
        os.environ.pop("MEMOPT_NVME_DIR", None)
        p = detect_universal_profile()
        names = [t.name for t in p.tiers]
        assert "nvme" not in names
    finally:
        if old is not None:
            os.environ["MEMOPT_NVME_DIR"] = old


def test_tiers_ordered_fastest_first():
    p = detect_universal_profile()
    latencies = [t.latency_us for t in p.tiers]
    assert latencies == sorted(latencies), "tiers should be ordered by ascending latency"


def test_total_capacity_is_sum_of_tiers():
    p = detect_universal_profile()
    expected = sum(t.capacity_gb for t in p.tiers)
    assert abs(p.total_capacity_gb - expected) < 0.01


def test_architecture_is_valid_string():
    p = detect_universal_profile()
    assert p.architecture in _VALID_ARCHITECTURES


def test_cxl_detection_never_raises():
    # Just ensure the full detection path doesn't blow up
    p = detect_universal_profile()
    assert isinstance(p.supports_cxl, bool)
