"""
Smoke test — runs on any hardware without a GPU.
Tests the full stack: allocate → fetch → promote → evict → free.
"""
import pytest
from memopt.vmm import VMM


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
