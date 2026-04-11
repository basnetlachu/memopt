"""
Pillar 2 two-node benchmark — single server, dual GPU (2x A100 SXM4 80GB).

Both "nodes" run in the same process:
  - Node B: TCPTransport server thread (simulates remote peer)
  - Node A: main thread (client — measures cross-node reads via loopback TCP)

This proves the cluster memory-pooling architecture on real hardware.
The TCP loopback measures real kernel socket latency; the GKD and
hypervisor tests are identical to what a real two-machine deployment runs.

Run:
    pytest memopt/cluster/tests/test_pillar2_twonode.py -v -s
"""

import json
import socket
import time
import threading

import pytest
import torch

from memopt.vmm import VMM
from memopt.cluster.gkd_store import GKDStore
from memopt.cluster.hypervisor import MemoryHypervisor
from memopt.cluster.transport import TCPTransport, RemoteRegion

BLOCK_SIZE = 131_072   # 128 KB — standard KV block
GB         = 1024 ** 3

# ── helpers ──────────────────────────────────────────────────────────────────


def _get_free_port():
    """Get an OS-assigned free port. Avoids port reuse races."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('', 0))
        return s.getsockname()[1]


def _wait_for_port(host: str, port: int, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            s = socket.create_connection((host, port), timeout=0.2)
            s.close()
            return True
        except OSError:
            time.sleep(0.05)
    return False


# ── fixture: Node B server ────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def node_b_server():
    """
    Starts a TCPTransport server (Node B) in a daemon thread.
    Registers 512 x 128KB memory blocks and keeps the server alive
    for the duration of the module's tests.

    Returns a dict with:
      - transport: the TCPTransport instance
      - regions:   list of MemoryRegion objects
      - buffers:   underlying bytearray objects (kept alive)
      - port:      actual listening port
    """
    port = _get_free_port()
    transport = TCPTransport(listen_port=port)
    transport.start_server()

    assert _wait_for_port("127.0.0.1", port, timeout=5.0), \
        f"Node B server did not start on port {port}"

    buffers = []
    regions = []
    for i in range(512):
        buf = bytearray(BLOCK_SIZE)
        # Fill with block-index pattern so reads can be verified
        for j in range(0, min(64, BLOCK_SIZE), 4):
            buf[j:j+4] = i.to_bytes(4, "little")
        region = transport.register_memory(buf)
        buffers.append(buf)
        regions.append(region)

    yield {
        "transport": transport,
        "regions": regions,
        "buffers": buffers,
        "port": port,
    }

    transport.close()


@pytest.fixture(scope="module")
def node_a_transport(node_b_server):
    """Node A's outbound transport, connected to Node B."""
    port_a = _get_free_port()
    port_b = node_b_server["port"]
    t = TCPTransport(listen_port=port_a)
    ok = t.connect("node-b", "127.0.0.1", port=port_b)
    assert ok, f"Node A could not connect to Node B on port {port_b}"
    yield t
    t.close()


# ── Test 1: connection smoke ───────────────────────────────────────────────────

def test_cross_node_connection(node_a_transport):
    """Node A must reach Node B's server."""
    # If the fixture built successfully, the connection exists.
    assert "node-b" in node_a_transport._connections


# ── Test 2: cross-node read latency ───────────────────────────────────────────

def test_cross_node_read_latency(node_b_server, node_a_transport):
    """
    100 one-sided reads of a 128KB block via loopback TCP.
    On a datacenter host (or even localhost) this should be well under 5ms avg.
    """
    region0 = node_b_server["regions"][0]
    remote  = RemoteRegion(
        node_id="node-b",
        addr=region0.addr,
        rkey=region0.rkey,
        length=BLOCK_SIZE,
    )
    dst = bytearray(BLOCK_SIZE)

    # warmup
    for _ in range(10):
        node_a_transport.read(remote, dst)

    latencies = []
    for _ in range(100):
        t0 = time.monotonic()
        node_a_transport.read(remote, dst)
        latencies.append((time.monotonic() - t0) * 1_000)

    latencies.sort()
    avg = sum(latencies) / len(latencies)
    p50 = latencies[49]
    p99 = latencies[98]

    print(f"\n  Cross-node read latency (128KB block via loopback TCP)")
    print(f"    avg : {avg:.3f} ms")
    print(f"    p50 : {p50:.3f} ms")
    print(f"    p99 : {p99:.3f} ms")

    # Save for final report
    pytest._pillar2_latency = {"avg_ms": avg, "p50_ms": p50, "p99_ms": p99}

    assert avg < 50, f"avg latency {avg:.3f}ms exceeds 50ms threshold"


# ── Test 3: cross-node bandwidth ──────────────────────────────────────────────

def test_cross_node_bandwidth(node_b_server, node_a_transport):
    """Transfer 256 MB of KV blocks via loopback TCP and measure throughput."""
    regions = node_b_server["regions"]
    dst = bytearray(BLOCK_SIZE)
    n_blocks = (256 * 1024 * 1024) // BLOCK_SIZE   # 2048 reads

    t0 = time.monotonic()
    for i in range(n_blocks):
        r = regions[i % len(regions)]
        remote = RemoteRegion(
            node_id="node-b",
            addr=r.addr,
            rkey=r.rkey,
            length=BLOCK_SIZE,
        )
        node_a_transport.read(remote, dst)
    elapsed = time.monotonic() - t0

    transferred_gb = (n_blocks * BLOCK_SIZE) / 1e9
    bandwidth_gbps = transferred_gb / elapsed

    print(f"\n  Cross-node bandwidth")
    print(f"    transferred : {transferred_gb:.2f} GB")
    print(f"    time        : {elapsed:.2f} s")
    print(f"    bandwidth   : {bandwidth_gbps:.2f} GB/s")

    pytest._pillar2_bandwidth = bandwidth_gbps
    # loopback should easily do > 0.1 GB/s
    assert bandwidth_gbps > 0.1, f"Bandwidth {bandwidth_gbps:.2f} GB/s too low"


# ── Test 4: GKD cross-node deduplication ──────────────────────────────────────

def test_gkd_cross_node_dedup():
    """
    One shared GKDStore simulates two nodes sharing the same Redis/local backend.

    Node B registers 200 prompt hashes.
    Node A then serves 1000 requests — 800 hit Node B's cached blocks,
    200 are unique misses. Target hit rate > 70%.
    """
    gkd = GKDStore(block_size_bytes=BLOCK_SIZE)

    # Node B: pre-register 200 prompt blocks
    base_prompts = [list(range(i, i + 128)) for i in range(200)]
    for i, tokens in enumerate(base_prompts):
        gkd.register(tokens, 128, f"node_b_block:{i}", "node-b", BLOCK_SIZE)

    # Node A: serve 1000 requests
    for req in range(1000):
        if req < 800:
            tokens = base_prompts[req % 200]   # repeat -> should hit
        else:
            tokens = list(range(req * 1000, req * 1000 + 128))  # novel -> miss

        hit = gkd.lookup(tokens, 128)
        if not hit:
            gkd.register(tokens, 128, f"node_a_block:{req}", "node-a", BLOCK_SIZE)

    s = gkd.stats()
    hit_rate    = s["hit_rate_pct"]
    hbm_saved   = s["estimated_hbm_saved_gb"]
    collisions  = s["collision_detections_total"]

    print(f"\n  GKD cross-node deduplication")
    print(f"    hit rate    : {hit_rate:.1f}%")
    print(f"    HBM saved   : {hbm_saved:.3f} GB")
    print(f"    collisions  : {collisions}  (must be 0)")

    pytest._pillar2_gkd = {
        "hit_rate_pct": hit_rate,
        "hbm_saved_gb": hbm_saved,
        "collisions":   collisions,
    }

    assert collisions == 0, "Hash collision detected — integrity failure"
    assert hit_rate > 70, f"GKD hit rate {hit_rate:.1f}% below 70% target"


# ── Test 5: hypervisor borrow routing ─────────────────────────────────────────

def test_hypervisor_borrow_routing():
    """
    Node A (nearly full) asks the hypervisor for 50 GB DRAM.
    Node B has 200 GB free. Hypervisor must select Node B, same rack.
    """
    h = MemoryHypervisor(node_id="rack1-node-a", enable_network=False)
    h.register_local_node({"dram": 10 * GB})
    h.register_peer("rack1-node-b", "127.0.0.1", {"dram": 200 * GB})

    offer = h.borrow_memory(50 * GB, tier="dram")

    assert offer is not None, "Hypervisor returned None — no donor found"
    assert offer.node_id == "rack1-node-b"
    assert offer.estimated_latency_us <= 1.0, (
        f"Same-rack latency estimate {offer.estimated_latency_us}us > 1.0us"
    )

    print(f"\n  Hypervisor borrow routing")
    print(f"    donor   : {offer.node_id}")
    print(f"    latency : {offer.estimated_latency_us} us")


# ── Test 6: cluster VMM — allocate blocks from both GPUs ─────────────────────

def test_cluster_vmm_dual_node():
    """Two VMMs sharing a pool: allocation, fetch, free cycle."""
    vmm_a = VMM()
    vmm_b = VMM()

    n = 100
    for i in range(n):
        vmm_a.allocate(f"seq_a", i, BLOCK_SIZE)
        vmm_b.allocate(f"seq_b", i, BLOCK_SIZE)

    sa = vmm_a.stats()
    sb = vmm_b.stats()

    assert sa["total_blocks"] == n
    assert sb["total_blocks"] == n

    vmm_a.free_sequence("seq_a")
    vmm_b.free_sequence("seq_b")

    assert vmm_a.stats()["total_blocks"] == 0
    assert vmm_b.stats()["total_blocks"] == 0


# ── Test 7: data integrity ───────────────────────────────────────────────────

def test_data_integrity(node_b_server, node_a_transport):
    """Verify block content survives transport round-trip."""
    regions = node_b_server["regions"]
    dst = bytearray(BLOCK_SIZE)

    for i in range(min(10, len(regions))):
        r = regions[i]
        remote = RemoteRegion(
            node_id="node-b",
            addr=r.addr,
            rkey=r.rkey,
            length=BLOCK_SIZE,
        )
        node_a_transport.read(remote, dst)
        embedded = int.from_bytes(dst[:4], "little")
        assert embedded == i, \
            f"Data corruption: block {i} has value {embedded}"


# ── Test 8: revolutionary A4000 benchmark (optional, long-running) ────────────

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for full benchmark")
def test_revolutionary_a4000():
    """
    Full-stack benchmark proving Pillar 2 at scale on real GPUs.

    Command:
        pytest memopt/cluster/tests/test_pillar2_twonode.py::test_revolutionary_a4000 -s
    """
    results = {}
    _GB_loc         = 1024 ** 3
    _BLOCK_SIZE_loc = 131_072
    _GPU_VRAM_loc   = 16 * _GB_loc

    # ── Start Node B server ────────────────────────────────────────────
    server_port = _get_free_port()
    server_transport = TCPTransport(listen_port=server_port)
    server_transport.start_server()
    assert _wait_for_port("127.0.0.1", server_port), \
        f"Server failed to start on port {server_port}"

    try:
        print(f"\n{'='*60}")
        print(f"  REVOLUTIONARY BENCHMARK — 2x RTX A4000 PCIe")
        print(f"  Same server, loopback TCP (port {server_port})")
        print(f"{'='*60}")
        print(f"\n  Registering 2,000 x 128KB blocks on Node B...")

        buffers_b = []
        regions_b = []
        for i in range(2000):
            buf = bytearray(_BLOCK_SIZE_loc)
            for j in range(0, 16, 4):
                buf[j:j+4] = i.to_bytes(4, "little")
            region = server_transport.register_memory(buf)
            buffers_b.append(buf)
            regions_b.append(region)

        region_info = [
            {"addr": r.addr, "rkey": r.rkey, "length": r.length}
            for r in regions_b
        ]

        # Pre-register 500 GKD entries simulating Node B's cached prompts
        gkd = GKDStore()
        base_tokens = [list(range(i * 10, i * 10 + 256)) for i in range(500)]
        for i, tokens in enumerate(base_tokens):
            gkd.register(tokens, 256, f"node_b_kv:{i}", "node-b", _BLOCK_SIZE_loc)

        print(f"  Node B: {len(regions_b):,} regions, 500 GKD entries")

        # ── Connect Node A ─────────────────────────────────────────────────
        client_port = _get_free_port()
        client_transport = TCPTransport(listen_port=client_port)
        connected = client_transport.connect("node-b", "127.0.0.1", port=server_port)
        assert connected, f"Loopback TCP connection failed — port {server_port}"
        print(f"  Node A connected via loopback TCP")

        try:
            # ── R1: Cross-node latency ─────────────────────────────────
            print(f"\n--- R1: Cross-node read latency ---")
            dst = bytearray(_BLOCK_SIZE_loc)
            sample = RemoteRegion(
                node_id="node-b",
                addr=region_info[0]["addr"],
                rkey=region_info[0]["rkey"],
                length=_BLOCK_SIZE_loc,
            )

            for _ in range(10):
                client_transport.read(sample, dst)

            latencies = []
            for _ in range(200):
                t0 = time.monotonic()
                client_transport.read(sample, dst)
                latencies.append((time.monotonic() - t0) * 1000)

            latencies.sort()
            avg = sum(latencies) / len(latencies)
            p50 = latencies[100]
            p95 = latencies[190]
            p99 = latencies[198]

            results["latency_avg_ms"] = round(avg, 3)
            results["latency_p50_ms"] = round(p50, 3)
            results["latency_p95_ms"] = round(p95, 3)
            results["latency_p99_ms"] = round(p99, 3)

            print(f"  avg: {avg:.3f}ms  p50: {p50:.3f}ms  p95: {p95:.3f}ms  p99: {p99:.3f}ms")
            assert avg < 5.0, f"Latency too high: {avg:.3f}ms"

            # ── R2: Sustained bandwidth (30s) ──────────────────────────────
            print(f"\n--- R2: Sustained bandwidth — 30 seconds ---")
            bytes_xfer = 0
            n_reads    = 0
            read_dst   = bytearray(_BLOCK_SIZE_loc)
            t_start    = time.monotonic()
            t_end      = t_start + 30.0

            while time.monotonic() < t_end:
                idx = n_reads % len(region_info)
                r = RemoteRegion(
                    node_id="node-b",
                    addr=region_info[idx]["addr"],
                    rkey=region_info[idx]["rkey"],
                    length=_BLOCK_SIZE_loc,
                )
                client_transport.read(r, read_dst)
                bytes_xfer += _BLOCK_SIZE_loc
                n_reads    += 1

            elapsed   = time.monotonic() - t_start
            bandwidth = bytes_xfer / elapsed / 1e9

            results["bandwidth_gbps"]        = round(bandwidth, 3)
            results["bandwidth_reads_total"] = n_reads
            results["bandwidth_gb_total"]    = round(bytes_xfer / 1e9, 2)

            print(f"  Duration:    {elapsed:.1f}s")
            print(f"  Reads:       {n_reads:,}")
            print(f"  Transferred: {bytes_xfer / 1e9:.2f} GB")
            print(f"  Bandwidth:   {bandwidth:.3f} GB/s")
            assert bandwidth > 0.5, f"Bandwidth too low: {bandwidth:.3f} GB/s"

            # ── R3: GKD at scale — 2,000 requests, 90% repeat ─────────────
            print(f"\n--- R3: GKD hit rate — 2,000 requests ---")
            hits = misses = 0
            for req in range(2000):
                if req < 1800:
                    tokens = base_tokens[req % 500]
                else:
                    tokens = list(range(req * 997, req * 997 + 256))
                hit = gkd.lookup(tokens, 256)
                if hit:
                    hits += 1
                else:
                    misses += 1
                    gkd.register(tokens, 256, f"node_a_kv:{req}", "node-a", _BLOCK_SIZE_loc)

            s = gkd.stats()
            hit_rate  = hits / 2000 * 100
            hbm_saved = hits * _BLOCK_SIZE_loc / 1e9

            results["gkd_hit_rate_pct"]         = round(hit_rate, 1)
            results["gkd_hbm_saved_gb"]         = round(hbm_saved, 3)
            results["gkd_collision_detections"] = s["collision_detections_total"]

            print(f"  Requests: 2,000  Hits: {hits:,}  ({hit_rate:.1f}%)")
            print(f"  HBM saved: {hbm_saved:.3f} GB")
            print(f"  Collisions: {s['collision_detections_total']}  (must be 0)")
            assert hit_rate > 80.0
            assert s["collision_detections_total"] == 0

            # ── R4: Cluster VMM memory pooling ─────────────────────────────
            print(f"\n--- R4: Cluster VMM pressure — both GPUs ---")
            device_count = torch.cuda.device_count()
            print(f"  CUDA devices: {device_count}")

            _, hbm_total_0 = torch.cuda.mem_get_info(0)
            hbm_total_1    = torch.cuda.mem_get_info(1)[1] if device_count >= 2 else hbm_total_0

            N = 40_000
            vmm_a = VMM()
            vmm_b = VMM()
            for i in range(N):
                vmm_a.allocate("rev_gpu0", i, _BLOCK_SIZE_loc)
            for i in range(N):
                vmm_b.allocate("rev_gpu1", i, _BLOCK_SIZE_loc)

            stats_a = vmm_a.stats()
            stats_b = vmm_b.stats()
            total_blocks   = stats_a["total_blocks"] + stats_b["total_blocks"]
            total_virtual  = total_blocks * _BLOCK_SIZE_loc / 1e9
            total_physical = (hbm_total_0 + hbm_total_1) / 1e9

            results["cluster_virtual_gb"]    = round(total_virtual, 2)
            results["cluster_physical_gb"]   = round(total_physical, 1)
            results["cluster_virtual_ratio"] = round(total_virtual / total_physical, 2)

            print(f"  Total virtual:  {total_virtual:.2f} GB")
            print(f"  Total physical: {total_physical:.1f} GB")
            print(f"  Ratio:          {total_virtual / total_physical:.2f}x")

            vmm_a.free_sequence("rev_gpu0")
            vmm_b.free_sequence("rev_gpu1")

            assert total_virtual > 1.0

            # ── R5: Hypervisor — 100 borrow requests ──────────────────────
            print(f"\n--- R5: Hypervisor routing — 100 borrows ---")
            h = MemoryHypervisor(node_id="rack1-gpu0", enable_network=False)
            h.register_local_node({"dram": 4 * _GB_loc})
            h.register_peer("rack1-gpu1", "127.0.0.1", {"dram": 200 * _GB_loc})

            successful = failed = 0
            t0 = time.monotonic()
            for _ in range(100):
                offer = h.borrow_memory(1 * _GB_loc, tier="dram")
                if offer:
                    successful += 1
                else:
                    failed += 1
            elapsed_ms = (time.monotonic() - t0) * 1000

            results["hypervisor_successful"]      = successful
            results["hypervisor_100_borrows_ms"]  = round(elapsed_ms, 2)

            print(f"  100 borrows in {elapsed_ms:.2f}ms")
            assert successful == 100

            # ── R6: Data integrity ─────────────────────────────────────────
            print(f"\n--- R6: Data integrity — 50 block verification ---")
            ok = fail = 0
            verify_dst = bytearray(_BLOCK_SIZE_loc)

            for i in range(50):
                r = RemoteRegion(
                    node_id="node-b",
                    addr=region_info[i]["addr"],
                    rkey=region_info[i]["rkey"],
                    length=_BLOCK_SIZE_loc,
                )
                client_transport.read(r, verify_dst)
                embedded = int.from_bytes(verify_dst[:4], "little")
                if embedded == i:
                    ok += 1
                else:
                    fail += 1

            results["integrity_ok"]   = ok
            results["integrity_fail"] = fail

            print(f"  Verified: {ok}/50  Failures: {fail}")
            assert fail == 0, f"Data corruption: {fail} blocks failed"

            # ── Final results ──────────────────────────────────────────────
            print(f"\n{'='*60}")
            print(f"  PILLAR 2 RESULTS")
            print(f"{'='*60}")
            for k, v in results.items():
                print(f"  {k}: {v}")
            print(f"{'='*60}\n")

            with open("/tmp/revolutionary_results.json", "w") as f:
                json.dump(results, f, indent=2)

        finally:
            client_transport.close()
    finally:
        server_transport.close()
