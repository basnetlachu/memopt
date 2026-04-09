"""
Smoke test — runs on any hardware without a GPU.
Tests the full stack: allocate → fetch → promote → evict → free.
"""
import json
import tempfile

import pytest
from memopt.vmm import VMM
from memopt.vmm.access_log import AccessLog
from memopt.vmm.prefetch_engine import PrefetchEngine


def test_basic_allocate_fetch_free():
    vmm = VMM()
    vmm.allocate("seq_test", block_index=0, size_bytes=1024)
    entry = vmm.fetch("seq_test", block_index=0)
    assert entry is not None
    assert entry.size_bytes == 1024
    vmm.free_sequence("seq_test")


def test_stats():
    vmm = VMM()
    vmm.allocate("seq_a", 0, 512)
    vmm.allocate("seq_a", 1, 512)
    s = vmm.stats()
    assert s["total_blocks"] == 2
    vmm.free_sequence("seq_a")
    assert vmm.stats()["total_blocks"] == 0


def test_prefetch_records_access():
    vmm = VMM()
    vmm.allocate("seq_b", 0, 512)
    vmm.allocate("seq_b", 1, 512)
    vmm.fetch("seq_b", 0)
    vmm.fetch("seq_b", 1)
    ps = vmm.stats()["prefetch"]
    assert ps["total_transitions_recorded"] >= 1
    vmm.free_sequence("seq_b")


def test_multiple_sequences_isolated():
    vmm = VMM()
    vmm.allocate("s1", 0, 256)
    vmm.allocate("s2", 0, 256)
    assert vmm.stats()["total_blocks"] == 2
    vmm.free_sequence("s1")
    assert vmm.stats()["total_blocks"] == 1
    vmm.free_sequence("s2")
    assert vmm.stats()["total_blocks"] == 0


def test_fetch_unknown_block_raises():
    vmm = VMM()
    with pytest.raises(KeyError):
        vmm.fetch("nonexistent", block_index=0)


def test_backend_and_tiers_present():
    from memopt.vmm import backend, tiers, tier_names
    assert backend is not None
    assert len(tiers) >= 2          # at minimum DRAM + NVMe
    assert len(tier_names) == len(tiers)
    assert all(isinstance(n, str) for n in tier_names)


# ── Tenant isolation ───────────────────────────────────────────────────

def test_tenant_owns_sequence_after_allocate():
    """The first tenant to allocate a sequence_id becomes its owner."""
    vmm = VMM()
    vmm.allocate("seq_alpha", 0, 256, tenant_id="alice")
    # Same tenant can allocate more blocks
    vmm.allocate("seq_alpha", 1, 256, tenant_id="alice")
    entry = vmm.fetch("seq_alpha", 0, tenant_id="alice")
    assert entry is not None
    vmm.free_sequence("seq_alpha", tenant_id="alice")


def test_cross_tenant_allocate_raises():
    """A second tenant must not be able to allocate into an existing sequence."""
    vmm = VMM()
    vmm.allocate("seq_shared", 0, 256, tenant_id="alice")
    with pytest.raises(PermissionError):
        vmm.allocate("seq_shared", 1, 256, tenant_id="bob")
    vmm.free_sequence("seq_shared", tenant_id="alice")


def test_cross_tenant_fetch_raises():
    """A tenant must not fetch a sequence owned by another tenant."""
    vmm = VMM()
    vmm.allocate("seq_private", 0, 256, tenant_id="alice")
    with pytest.raises(PermissionError):
        vmm.fetch("seq_private", 0, tenant_id="bob")
    vmm.free_sequence("seq_private", tenant_id="alice")


def test_cross_tenant_free_raises():
    """A tenant must not free a sequence owned by another tenant."""
    vmm = VMM()
    vmm.allocate("seq_locked", 0, 256, tenant_id="alice")
    with pytest.raises(PermissionError):
        vmm.free_sequence("seq_locked", tenant_id="bob")
    vmm.free_sequence("seq_locked", tenant_id="alice")


# ── Layer 1: access log + universal profile integration ──────────────

def test_prefetch_engine_logs_access_event():
    """PrefetchEngine emits BlockAccessEvent to AccessLog when provided."""
    with tempfile.TemporaryDirectory() as td:
        log = AccessLog(log_dir=td)
        vmm = VMM()
        vmm.prefetch = PrefetchEngine(vmm.tier_manager, access_log=log)

        vmm.prefetch.record_access(
            "seq_log", 0,
            token_position=42, attention_layer=3,
        )
        log.shutdown()

        path = log.stats()["log_file_path"]
        with open(path) as f:
            obj = json.loads(f.readline())
        assert obj["token_position"] == 42
        assert obj["attention_layer"] == 3
        assert obj["sequence_id"] == "seq_log"


def test_prefetch_engine_hw_profile_not_none():
    """PrefetchEngine always detects a hardware profile."""
    vmm = VMM()
    profile = vmm.prefetch.get_hw_profile()
    assert profile is not None
    assert isinstance(profile.architecture, str) and profile.architecture != ""
    assert len(profile.tiers) >= 1


def test_prefetch_engine_works_without_access_log():
    """Backward compat: PrefetchEngine works with no access_log argument."""
    vmm = VMM()
    vmm.allocate("seq_compat", 0, 512)
    vmm.allocate("seq_compat", 1, 512)
    vmm.fetch("seq_compat", 0)
    vmm.fetch("seq_compat", 1)
    vmm.free_sequence("seq_compat")
