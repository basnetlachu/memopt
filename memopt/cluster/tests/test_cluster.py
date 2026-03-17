"""
Cluster layer tests — hypervisor routing, transport, and borrow protocol.
All tests use TCPTransport and LocalGKDBackend. No hardware required.
"""
import time
import threading
import pytest
from memopt.cluster.transport import TCPTransport, RemoteRegion
from memopt.cluster.hypervisor import MemoryHypervisor, ClusterMap, _estimate_latency_us
from memopt.cluster.gkd_store import GKDStore


# ── ClusterMap tests ───────────────────────────────────────────────────

def test_register_and_candidate_selection():
    """Nodes with enough free memory appear as candidates."""
    cm = ClusterMap()
    cm.update("node-a", "10.0.0.1", {"dram": 100 * 1024**3})
    cm.update("node-b", "10.0.0.2", {"dram": 200 * 1024**3})

    candidates = cm.candidates(
        size_bytes=50 * 1024**3,
        tier="dram",
        exclude_node="node-c",
    )
    assert len(candidates) == 2

def test_candidate_excludes_requesting_node():
    """A node must not borrow from itself."""
    cm = ClusterMap()
    cm.update("node-a", "10.0.0.1", {"dram": 100 * 1024**3})
    candidates = cm.candidates(50 * 1024**3, "dram", exclude_node="node-a")
    assert len(candidates) == 0

def test_candidate_excludes_insufficient_nodes():
    """A node with less free memory than requested must not be a candidate."""
    cm = ClusterMap()
    cm.update("node-b", "10.0.0.2", {"dram": 10})   # only 10 bytes free
    candidates = cm.candidates(1024**3, "dram", exclude_node="node-a")
    assert len(candidates) == 0

def test_stale_node_marked_unreachable():
    """A node that misses heartbeats must be excluded from candidates."""
    cm = ClusterMap()
    cm.update("node-b", "10.0.0.2", {"dram": 200 * 1024**3})
    # Manually set last_seen to the past to simulate timeout
    cm._nodes["node-b"].last_seen = time.monotonic() - 10.0
    cm.reap_stale()
    candidates = cm.candidates(1024**3, "dram", exclude_node="node-a")
    assert len(candidates) == 0


# ── Latency estimation tests ────────────────────────────────────────────

def test_same_rack_lower_latency():
    assert _estimate_latency_us("rack1-node1", "rack1-node2") < \
           _estimate_latency_us("rack1-node1", "rack2-node1")

def test_unknown_topology_conservative():
    lat = _estimate_latency_us("nodeA", "nodeB")
    assert lat >= 5.0   # unknown → at least 5 µs estimate


# ── MemoryHypervisor tests ─────────────────────────────────────────────

def test_borrow_returns_offer():
    """borrow_memory must return a valid offer when a donor exists."""
    h = MemoryHypervisor(node_id="rack1-node1", enable_network=False)
    h.register_local_node({"dram": 10 * 1024**3})
    h.register_peer("rack1-node2", "10.0.0.2", {"dram": 200 * 1024**3})

    offer = h.borrow_memory(50 * 1024**3, tier="dram")
    assert offer is not None
    assert offer.node_id == "rack1-node2"
    assert offer.size_bytes == 50 * 1024**3
    assert offer.tier == "dram"

def test_borrow_prefers_rack_local():
    """When two donors are available, the rack-local one wins."""
    h = MemoryHypervisor(node_id="rack1-node1", enable_network=False)
    h.register_local_node({"dram": 1})
    h.register_peer("rack1-node2", "10.0.0.2", {"dram": 200 * 1024**3})  # same rack
    h.register_peer("rack2-node1", "10.0.1.1", {"dram": 200 * 1024**3})  # diff rack

    offer = h.borrow_memory(50 * 1024**3, tier="dram")
    assert offer.node_id == "rack1-node2", "Should prefer same-rack donor"

def test_borrow_returns_none_when_no_donors():
    """Returns None gracefully when no node has enough free memory."""
    h = MemoryHypervisor(node_id="node-a", enable_network=False)
    h.register_local_node({"dram": 0})
    # No peers registered
    offer = h.borrow_memory(1024**3, tier="dram")
    assert offer is None

def test_stats_tracks_borrows():
    h = MemoryHypervisor(node_id="rack1-node1", enable_network=False)
    h.register_local_node({"dram": 1})
    h.register_peer("rack1-node2", "10.0.0.2", {"dram": 200 * 1024**3})

    h.borrow_memory(1024**3)
    h.borrow_memory(1024**3)

    s = h.stats()
    assert s["borrows_total"]    == 2
    assert s["borrows_failed"]   == 0
    assert s["bytes_borrowed_gb"] > 0

def test_failed_borrow_tracked_in_stats():
    h = MemoryHypervisor(node_id="node-a", enable_network=False)
    h.register_local_node({"dram": 0})
    h.borrow_memory(1024**3)   # will fail — no peers
    s = h.stats()
    assert s["borrows_failed"] == 1


# ── TCPTransport tests ─────────────────────────────────────────────────

def test_tcp_write_and_read():
    """TCPTransport must transfer data correctly between two endpoints."""
    server = TCPTransport(listen_port=19001)
    server.start_server()
    time.sleep(0.05)   # let server thread start

    client = TCPTransport(listen_port=19002)
    connected = client.connect("server", "127.0.0.1", port=19001)
    assert connected, "TCP connection must succeed on localhost"

    # Register a buffer on the server side
    data = bytearray(b"hello from server " + b"x" * 110)
    region = server.register_memory(data)

    remote = RemoteRegion(
        node_id="server",
        addr=region.addr,
        rkey=region.rkey,
        length=len(data),
    )

    dst = bytearray(len(data))
    elapsed = client.read(remote, dst)

    assert dst == data, "Read data must exactly match written data"
    assert elapsed < 1.0, f"TCP read on localhost must be < 1s, got {elapsed:.3f}s"

    server.deregister_memory(region)
    client.close()
    server.close()

def test_tcp_write_from_client_to_server():
    """TCPTransport write must deposit data on the remote side."""
    server = TCPTransport(listen_port=19003)
    server.start_server()
    time.sleep(0.05)

    client = TCPTransport(listen_port=19004)
    client.connect("server", "127.0.0.1", port=19003)

    src = bytearray(b"payload_" * 16)
    remote = RemoteRegion(
        node_id="server", addr=99999, rkey=99999, length=len(src)
    )
    client.write(remote, src)
    time.sleep(0.05)   # let server process

    with server._lock:
        stored = server._memory.get(99999)
    assert stored is not None
    assert bytes(stored[:len(src)]) == bytes(src)

    client.close()
    server.close()


# ── Integration: hypervisor + GKD ─────────────────────────────────────

def test_hypervisor_exposes_gkd_stats():
    """Hypervisor stats must include GKD hit/miss counts."""
    gkd = GKDStore()
    h   = MemoryHypervisor(node_id="node-a", gkd=gkd, enable_network=False)
    h.register_local_node({"dram": 10 * 1024**3})

    tokens = list(range(64))
    gkd.lookup(tokens, 64)                                      # miss
    gkd.register(tokens, 64, "blk:0", "node-a", 131_072)
    gkd.lookup(tokens, 64)                                      # hit

    s = h.stats()
    assert s["gkd_stats"]["cache_hits"]   == 1
    assert s["gkd_stats"]["cache_misses"] == 1
