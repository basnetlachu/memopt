"""
Pillar 5 — Global Unified Memory tests.
All pass without RDMA hardware, without Redis, without GPU.
Uses loopback TCP and in-process block directory.
"""
import os
import time
import socket
import tempfile
import threading
import hashlib

from memopt.cluster.block_directory import (
    LocalBlockDirectory, BlockEntry, make_directory
)
from memopt.cluster.remote_block import (
    RemoteBlockServer, RemoteBlockClient,
)


# ── BlockEntry ─────────────────────────────────────────────────────────

def test_block_entry_not_expired_immediately():
    e = BlockEntry(
        content_hash="abc123", node_id="node-a", tier="nvme",
        path="/tmp/block.bin", size_bytes=1024
    )
    assert not e.is_expired()
    assert e.is_leasable()


def test_block_entry_expired_after_ttl():
    e = BlockEntry(
        content_hash="abc", node_id="node-a", tier="nvme",
        path="/tmp/b.bin", size_bytes=64,
        registered_at=time.time() - 9999
    )
    assert e.is_expired()
    assert not e.is_leasable()


def test_block_entry_hbm_not_leasable():
    """Only NVMe blocks can be leased — HBM blocks are already in memory."""
    e = BlockEntry(
        content_hash="x", node_id="n", tier="hbm",
        path="", size_bytes=128
    )
    assert not e.is_leasable()


# ── LocalBlockDirectory ────────────────────────────────────────────────

def test_directory_register_and_lookup():
    d = LocalBlockDirectory()
    e = BlockEntry("hash1", "node-a", "nvme", "/tmp/b1.bin", 512)
    d.register(e)
    found = d.lookup("hash1")
    assert found is not None
    assert found.node_id == "node-a"


def test_directory_lookup_missing_returns_none():
    d = LocalBlockDirectory()
    assert d.lookup("nonexistent") is None


def test_directory_lookup_expired_returns_none():
    d = LocalBlockDirectory()
    e = BlockEntry(
        "hash_old", "node-a", "nvme", "/tmp/old.bin", 64,
        registered_at=time.time() - 9999
    )
    d.register(e)
    assert d.lookup("hash_old") is None


def test_directory_lease_lifecycle():
    d = LocalBlockDirectory()
    e = BlockEntry("hashX", "node-a", "nvme", "/tmp/x.bin", 256)
    d.register(e)

    assert d.can_evict("hashX")
    granted = d.acquire_lease("hashX", "node-b")
    assert granted
    assert not d.can_evict("hashX")

    d.release_lease("hashX", "node-b")
    assert d.can_evict("hashX")


def test_directory_lease_not_granted_for_missing():
    d = LocalBlockDirectory()
    granted = d.acquire_lease("nonexistent", "node-b")
    assert not granted


def test_directory_release_lease_on_missing_is_safe():
    """release_lease must not raise if entry was evicted after lease."""
    d = LocalBlockDirectory()
    d.release_lease("nonexistent", "node-b")   # must not raise


def test_directory_multiple_leases():
    """Multiple nodes can hold leases simultaneously."""
    d = LocalBlockDirectory()
    e = BlockEntry("hashM", "node-a", "nvme", "/tmp/m.bin", 128)
    d.register(e)

    d.acquire_lease("hashM", "node-b")
    d.acquire_lease("hashM", "node-c")
    assert not d.can_evict("hashM")

    d.release_lease("hashM", "node-b")
    assert not d.can_evict("hashM")   # node-c still holding

    d.release_lease("hashM", "node-c")
    assert d.can_evict("hashM")


def test_directory_deregister():
    d = LocalBlockDirectory()
    e = BlockEntry("hashD", "node-a", "nvme", "/tmp/d.bin", 64)
    d.register(e)
    d.deregister("hashD")
    assert d.lookup("hashD") is None


def test_directory_stats():
    d = LocalBlockDirectory()
    for i in range(3):
        d.register(BlockEntry(f"h{i}", "node-a", "nvme",
                               f"/tmp/b{i}.bin", 128))
    d.acquire_lease("h0", "node-b")
    s = d.stats()
    assert s["total_entries"] == 3
    assert s["active_entries"] == 3
    assert s["leased_entries"] == 1


def test_directory_list_node_blocks():
    d = LocalBlockDirectory()
    d.register(BlockEntry("ha", "node-a", "nvme", "/tmp/a.bin", 64))
    d.register(BlockEntry("hb", "node-b", "nvme", "/tmp/b.bin", 64))
    d.register(BlockEntry("hc", "node-a", "nvme", "/tmp/c.bin", 64))
    blocks_a = d.list_node_blocks("node-a")
    assert len(blocks_a) == 2
    assert all(b.node_id == "node-a" for b in blocks_a)


def test_make_directory_returns_local_without_redis():
    os.environ.pop("REDIS_URL", None)
    d = make_directory("test-node")
    assert isinstance(d, LocalBlockDirectory)


# ── RemoteBlockServer + RemoteBlockClient ─────────────────────────────

def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_block_data(size: int = 1024) -> bytes:
    return os.urandom(size)


def _content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_remote_block_round_trip():
    """
    Full round-trip: client requests a block from server,
    server reads from 'NVMe' (temp file), client receives it.
    Data integrity verified with SHA-256.
    """
    data  = _make_block_data(4096)
    chash = _content_hash(data)

    with tempfile.TemporaryDirectory() as tmpdir:
        block_path = os.path.join(tmpdir, "block.bin")
        with open(block_path, "wb") as f:
            f.write(data)

        directory = LocalBlockDirectory()
        entry     = BlockEntry(chash, "node-server", "nvme",
                               block_path, len(data))
        directory.register(entry)

        def read_block(path):
            try:
                with open(path, "rb") as f:
                    return f.read()
            except OSError:
                return None

        port   = _find_free_port()
        server = RemoteBlockServer(
            node_id="node-server",
            block_directory=directory,
            read_block_fn=read_block,
            port=port,
        )
        server.start()
        time.sleep(0.1)   # let server start

        client   = RemoteBlockClient(node_id="node-client", timeout_s=3.0)
        received = client.fetch_block(chash, "127.0.0.1", port)

        server.stop()

        assert received is not None, "fetch_block returned None"
        assert received == data, "Data integrity check failed"
        assert _content_hash(received) == chash

        sstats = server.stats()
        cstats = client.stats()
        assert sstats["requests_served"] == 1
        assert cstats["fetches"] == 1
        assert cstats["bytes_fetched"] == len(data)


def test_remote_block_not_found_returns_none():
    """Requesting a block that does not exist returns None, never raises."""
    directory = LocalBlockDirectory()

    def read_block(path):
        return None

    port   = _find_free_port()
    server = RemoteBlockServer(
        node_id="node-server",
        block_directory=directory,
        read_block_fn=read_block,
        port=port,
    )
    server.start()
    time.sleep(0.1)

    client   = RemoteBlockClient(node_id="node-client", timeout_s=2.0)
    received = client.fetch_block("nonexistent_hash", "127.0.0.1", port)

    server.stop()
    assert received is None


def test_remote_block_server_unreachable_returns_none():
    """Client returns None when server is not running. Never raises."""
    client   = RemoteBlockClient(node_id="node-client", timeout_s=0.3)
    received = client.fetch_block(
        "some_hash", "127.0.0.1", _find_free_port()
    )
    assert received is None
    assert client.stats()["failures"] == 1


def test_lease_round_trip():
    """acquire_lease and release_lease work correctly over the network."""
    data  = _make_block_data(512)
    chash = _content_hash(data)

    with tempfile.TemporaryDirectory() as tmpdir:
        block_path = os.path.join(tmpdir, "leased.bin")
        with open(block_path, "wb") as f:
            f.write(data)

        directory = LocalBlockDirectory()
        entry     = BlockEntry(chash, "node-server", "nvme",
                               block_path, len(data))
        directory.register(entry)

        def read_block(path):
            with open(path, "rb") as f:
                return f.read()

        port   = _find_free_port()
        server = RemoteBlockServer(
            node_id="node-server",
            block_directory=directory,
            read_block_fn=read_block,
            port=port,
        )
        server.start()
        time.sleep(0.1)

        client  = RemoteBlockClient(node_id="node-client", timeout_s=2.0)
        granted = client.acquire_lease(chash, "127.0.0.1", port)
        assert granted, "Lease should be granted for valid block"
        assert not directory.can_evict(chash), \
            "Block must not be evictable while leased"

        client.release_lease(chash, "127.0.0.1", port)
        time.sleep(0.05)
        assert directory.can_evict(chash), \
            "Block must be evictable after lease released"

        server.stop()


def test_concurrent_block_requests():
    """Multiple clients requesting the same block concurrently."""
    data  = _make_block_data(2048)
    chash = _content_hash(data)

    with tempfile.TemporaryDirectory() as tmpdir:
        block_path = os.path.join(tmpdir, "concurrent.bin")
        with open(block_path, "wb") as f:
            f.write(data)

        directory = LocalBlockDirectory()
        directory.register(BlockEntry(
            chash, "node-server", "nvme", block_path, len(data)
        ))

        def read_block(path):
            with open(path, "rb") as f:
                return f.read()

        port   = _find_free_port()
        server = RemoteBlockServer(
            node_id="node-server",
            block_directory=directory,
            read_block_fn=read_block,
            port=port,
        )
        server.start()
        time.sleep(0.1)

        results   = []
        lock      = threading.Lock()
        n_clients = 5

        def fetch():
            c = RemoteBlockClient(node_id="node-c", timeout_s=3.0)
            r = c.fetch_block(chash, "127.0.0.1", port)
            with lock:
                results.append(r)

        threads = [threading.Thread(target=fetch) for _ in range(n_clients)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        server.stop()

        assert len(results) == n_clients
        for r in results:
            assert r == data, "All concurrent fetches must return correct data"


# ── TierManager GUM integration ──────────────────────────────────────

def test_tier_manager_accepts_gum_params():
    """TierManager constructor accepts GUM parameters without error."""
    from memopt.vmm.tier_manager import TierManager
    from memopt.vmm.page_table import PageTable
    from unittest.mock import MagicMock

    pt = PageTable()
    mock_client = MagicMock()
    mock_dir = MagicMock()

    tm = TierManager(
        pt,
        remote_client=mock_client,
        block_directory=mock_dir,
        node_id="test-node")

    assert tm._remote_client is mock_client
    assert tm._block_directory is mock_dir
    assert tm._node_id == "test-node"


def test_tier_manager_stats_shows_gum_enabled():
    """stats() should report gum_enabled=True when configured."""
    from memopt.vmm.tier_manager import TierManager
    from memopt.vmm.page_table import PageTable
    from unittest.mock import MagicMock

    pt = PageTable()
    tm = TierManager(pt,
                     remote_client=MagicMock(),
                     block_directory=MagicMock(),
                     node_id="n1")
    s = tm.stats()
    assert s["gum_enabled"] is True


def test_tier_manager_stats_shows_gum_disabled():
    """stats() should report gum_enabled=False by default."""
    from memopt.vmm.tier_manager import TierManager
    from memopt.vmm.page_table import PageTable

    pt = PageTable()
    tm = TierManager(pt)
    s = tm.stats()
    assert s["gum_enabled"] is False
