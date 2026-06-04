"""
GUM hardening tests — lease expiry, fetch retry.
"""


from memopt.cluster.block_directory import BlockEntry, LocalBlockDirectory


def test_expired_lease_logic():
    """Leases should be releasable after acquire."""
    d = LocalBlockDirectory()
    e = BlockEntry("hash_exp", "node-a", "nvme", "/tmp/e.bin", 256)
    d.register(e)
    d.acquire_lease("hash_exp", "remote-node")
    assert d.can_evict("hash_exp") is False
    d.release_lease("hash_exp", "remote-node")
    assert d.can_evict("hash_exp") is True


def test_lease_does_not_affect_non_nvme():
    """Only NVMe blocks should be leasable."""
    d = LocalBlockDirectory()
    e = BlockEntry("hash_hbm", "node-a", "hbm", "/tmp/h.bin", 256)
    d.register(e)
    result = d.acquire_lease("hash_hbm", "remote-node")
    assert result is False
    assert d.can_evict("hash_hbm") is True


def test_fetch_block_returns_none_on_missing():
    """RemoteBlockClient returns None for non-existent blocks."""
    from memopt.cluster.remote_block import RemoteBlockClient
    client = RemoteBlockClient(node_id="test-node")
    # Fetch from unreachable host
    result = client.fetch_block("nonexistent_hash", "127.0.0.1", 19999)
    assert result is None


def test_fetch_block_increments_failure_counter():
    """Each failed fetch should increment the failure counter."""
    from memopt.cluster.remote_block import RemoteBlockClient
    client = RemoteBlockClient(node_id="test-node")
    initial = client.stats().get("failures", 0)
    client.fetch_block("hash1", "127.0.0.1", 19998)
    client.fetch_block("hash2", "127.0.0.1", 19998)
    after = client.stats().get("failures", 0)
    assert after >= initial + 2


def test_release_lease_on_missing_block_is_safe():
    """Releasing a lease on a non-existent block should not raise."""
    d = LocalBlockDirectory()
    # Should not raise
    d.release_lease("nonexistent_hash", "some-node")


def test_can_evict_missing_returns_true():
    """can_evict on missing block returns True (safe to evict)."""
    d = LocalBlockDirectory()
    assert d.can_evict("nonexistent") is True


def test_fetch_block_retries_on_failure():
    """fetch_block should retry before returning None."""
    import os
    from unittest.mock import patch
    from memopt.cluster.remote_block import RemoteBlockClient

    client = RemoteBlockClient(node_id="test-node")

    call_count = {"n": 0}
    original = client._try_fetch_once

    def mock_fetch(h, host, port):
        call_count["n"] += 1
        if call_count["n"] <= 2:
            return None  # fail first 2 attempts
        return b"success_data"  # succeed on 3rd

    # Set retries=2, delay=0 for fast test
    os.environ["MEMOPT_FETCH_RETRIES"] = "2"
    os.environ["MEMOPT_FETCH_RETRY_DELAY_S"] = "0"
    try:
        with patch.object(client, '_try_fetch_once',
                          side_effect=mock_fetch):
            result = client.fetch_block("hash123", "127.0.0.1", 19999)
            assert result == b"success_data"
            assert call_count["n"] == 3  # 1 initial + 2 retries
    finally:
        os.environ.pop("MEMOPT_FETCH_RETRIES", None)
        os.environ.pop("MEMOPT_FETCH_RETRY_DELAY_S", None)
