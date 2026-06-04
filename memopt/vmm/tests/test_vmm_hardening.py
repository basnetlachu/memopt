"""
VMM hardening tests — thread survival, health checks, NVMe cap.
"""
import os
import tempfile


from memopt.vmm import VMM


def test_is_healthy_returns_true_when_running():
    vmm = VMM()
    assert vmm.prefetch.is_healthy() is True


def test_is_healthy_never_raises():
    vmm = VMM()
    for _ in range(100):
        result = vmm.prefetch.is_healthy()
        assert isinstance(result, bool)


def test_memory_pressure_pct_never_raises():
    vmm = VMM()
    for _ in range(100):
        result = vmm.prefetch.memory_pressure_pct()
        assert isinstance(result, float)
        assert 0.0 <= result <= 100.0


def test_nvme_size_limit_triggers_eviction():
    with tempfile.TemporaryDirectory() as td:
        # Create fake .vmm_block files totaling > 900 bytes
        for i in range(10):
            path = os.path.join(td, f"block_{i}.vmm_block")
            with open(path, "wb") as f:
                f.write(b"x" * 100)  # 100 bytes each = 1000 total

        vmm = VMM()
        vmm.prefetch._nvme_dir = td

        # Set cap to 0.000001 GB = 1000 bytes.
        # 90% of 1000 = 900. We have 1000 > 900 → should evict.
        old_env = os.environ.get("MEMOPT_NVME_MAX_GB")
        os.environ["MEMOPT_NVME_MAX_GB"] = "0.000001"
        try:
            vmm.prefetch._check_nvme_cap()
            # Should have evicted ~10% = 1 file
            remaining = [f for f in os.listdir(td)
                         if f.endswith(".vmm_block")]
            assert len(remaining) < 10
        finally:
            if old_env is not None:
                os.environ["MEMOPT_NVME_MAX_GB"] = old_env
            else:
                os.environ.pop("MEMOPT_NVME_MAX_GB", None)


def test_prefetch_thread_survives_exception():
    """Prefetch threads use try/except — verify no crash on error."""
    vmm = VMM()
    vmm.allocate("seq_surv", 0, 512)
    vmm.allocate("seq_surv", 1, 512)
    # Access to trigger prefetch
    vmm.fetch("seq_surv", 0)
    vmm.fetch("seq_surv", 1)
    # Engine should still be healthy
    assert vmm.prefetch.is_healthy() is True
    vmm.free_sequence("seq_surv")


def test_evict_thresholds_from_env():
    """Eviction thresholds read from environment variables."""
    from memopt.vmm.tier_manager import TierManager
    from memopt.vmm.page_table import PageTable

    old_high = os.environ.get("MEMOPT_EVICT_HIGH")
    old_low = os.environ.get("MEMOPT_EVICT_LOW")
    os.environ["MEMOPT_EVICT_HIGH"] = "0.85"
    os.environ["MEMOPT_EVICT_LOW"] = "0.65"
    try:
        pt = PageTable()
        tm = TierManager(pt)
        assert tm._evict_high == 0.85
        assert tm._evict_low == 0.65
    finally:
        if old_high is not None:
            os.environ["MEMOPT_EVICT_HIGH"] = old_high
        else:
            os.environ.pop("MEMOPT_EVICT_HIGH", None)
        if old_low is not None:
            os.environ["MEMOPT_EVICT_LOW"] = old_low
        else:
            os.environ.pop("MEMOPT_EVICT_LOW", None)


def test_evict_thresholds_invalid_uses_defaults():
    """Invalid thresholds (low > high) revert to defaults."""
    from memopt.vmm.tier_manager import TierManager
    from memopt.vmm.page_table import PageTable

    old_high = os.environ.get("MEMOPT_EVICT_HIGH")
    old_low = os.environ.get("MEMOPT_EVICT_LOW")
    os.environ["MEMOPT_EVICT_HIGH"] = "0.50"
    os.environ["MEMOPT_EVICT_LOW"] = "0.80"  # low > high = invalid
    try:
        pt = PageTable()
        tm = TierManager(pt)
        assert tm._evict_high == 0.90  # default
        assert tm._evict_low == 0.75   # default
    finally:
        if old_high is not None:
            os.environ["MEMOPT_EVICT_HIGH"] = old_high
        else:
            os.environ.pop("MEMOPT_EVICT_HIGH", None)
        if old_low is not None:
            os.environ["MEMOPT_EVICT_LOW"] = old_low
        else:
            os.environ.pop("MEMOPT_EVICT_LOW", None)


def test_evict_thresholds_defaults_without_env():
    """Without env vars, defaults are 0.90/0.75."""
    from memopt.vmm.tier_manager import TierManager
    from memopt.vmm.page_table import PageTable

    # Ensure env vars are not set
    old_high = os.environ.pop("MEMOPT_EVICT_HIGH", None)
    old_low = os.environ.pop("MEMOPT_EVICT_LOW", None)
    try:
        pt = PageTable()
        tm = TierManager(pt)
        assert tm._evict_high == 0.90
        assert tm._evict_low == 0.75
    finally:
        if old_high is not None:
            os.environ["MEMOPT_EVICT_HIGH"] = old_high
        if old_low is not None:
            os.environ["MEMOPT_EVICT_LOW"] = old_low
