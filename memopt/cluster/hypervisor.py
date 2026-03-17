"""
Memory hypervisor — cluster-wide memory map and borrow routing.

Responsibilities:
  1. Maintain a live map of free memory per tier per node.
  2. Answer "which node should I borrow N bytes from?" in < 1ms.
  3. Integrate GKDStore for deduplication lookups.
  4. Detect node failures via heartbeat timeout.

Gossip protocol:
  Each node broadcasts a UDP heartbeat every HEARTBEAT_INTERVAL_S seconds.
  Payload: JSON with node_id and free_bytes per tier.
  Any node not heard from in HEARTBEAT_TIMEOUT_S is marked unreachable.
  Unreachable nodes are excluded from borrow candidates immediately.

This is an intentionally simple first implementation.
Phase 2 adds: consistent hashing for GKD entries, Raft-based leader
election for hypervisor coordination, and actual latency probing.
"""
from __future__ import annotations
import json
import socket
import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List

from .transport import TCPTransport, RemoteRegion, make_transport
from .gkd_store import GKDStore

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL_S = 0.100    # 100ms between broadcasts
HEARTBEAT_TIMEOUT_S  = 0.500    # 500ms — 5 missed heartbeats = unreachable
HEARTBEAT_PORT       = 18516
BORROW_PORT_BASE     = 18520    # each node listens on 18520 + node_index


@dataclass
class NodeState:
    """Live state of one cluster node as seen by the hypervisor."""
    node_id:       str
    host:          str
    free_bytes:    Dict[str, int]   # tier_name → free bytes
    last_seen:     float = field(default_factory=time.monotonic)
    reachable:     bool = True


@dataclass
class BorrowOffer:
    """
    Result of a successful borrow_memory() call.

    The caller (TierManager) uses this to:
      1. Connect transport to offer.node_id at offer.host
      2. Call transport.read(offer.remote_region, dst_buffer)
      3. Update the VMM page table to mark the block as "remote DRAM"
    """
    node_id:       str
    host:          str
    tier:          str           # which tier on the donor node
    size_bytes:    int
    remote_region: RemoteRegion  # addressing info for the RDMA/TCP read
    estimated_latency_us: float  # for logging and stats


class ClusterMap:
    """
    Thread-safe map of all known cluster nodes and their free memory.

    Updated by:
      - Incoming heartbeats (remote nodes announcing their state)
      - Local VMM calls (this node updating its own free memory after alloc/free)

    Read by:
      - borrow_memory() to find donors
      - stats() to report cluster utilisation
    """

    def __init__(self):
        self._nodes: Dict[str, NodeState] = {}
        self._lock  = threading.RLock()

    def update(self, node_id: str, host: str, free_bytes: Dict[str, int]):
        """
        Insert or refresh a node's state in the map.

        Resets last_seen and reachable=True. Safe to call from any thread.
        """
        with self._lock:
            if node_id in self._nodes:
                self._nodes[node_id].free_bytes = free_bytes
                self._nodes[node_id].last_seen  = time.monotonic()
                self._nodes[node_id].reachable  = True
            else:
                self._nodes[node_id] = NodeState(
                    node_id=node_id, host=host, free_bytes=free_bytes
                )

    def mark_unreachable(self, node_id: str):
        """
        Explicitly mark a node as unreachable.

        Called when a connection attempt fails or when reap_stale() fires.
        Does not remove the node — it stays in the map for observability.
        """
        with self._lock:
            if node_id in self._nodes:
                self._nodes[node_id].reachable = False

    def reap_stale(self):
        """
        Mark nodes that have not sent a heartbeat recently as unreachable.

        Called periodically by the reaper thread. A node missing
        HEARTBEAT_TIMEOUT_S seconds of heartbeats is considered failed.
        """
        now = time.monotonic()
        with self._lock:
            for node in self._nodes.values():
                if node.reachable and (now - node.last_seen) > HEARTBEAT_TIMEOUT_S:
                    node.reachable = False
                    logger.warning(f"Node {node.node_id} heartbeat timeout — marked unreachable")

    def candidates(
        self,
        size_bytes: int,
        tier: str,
        exclude_node: str,
    ) -> List[NodeState]:
        """
        Return nodes that could donate size_bytes on the given tier,
        sorted by estimated latency (rack-local first).

        Filters out: unreachable nodes, the requesting node itself, and
        nodes with insufficient free memory on the requested tier.
        """
        with self._lock:
            result = [
                n for n in self._nodes.values()
                if n.reachable
                and n.node_id != exclude_node
                and n.free_bytes.get(tier, 0) >= size_bytes
            ]
        return result

    def all_nodes(self) -> List[NodeState]:
        """Return a snapshot of all known nodes (reachable and unreachable)."""
        with self._lock:
            return list(self._nodes.values())

    def total_free(self, tier: str) -> int:
        """Sum of free_bytes[tier] across all reachable nodes."""
        with self._lock:
            return sum(n.free_bytes.get(tier, 0) for n in self._nodes.values() if n.reachable)


def _estimate_latency_us(local_node: str, remote_node: str) -> float:
    """
    Rough latency estimate based on node_id prefix convention.

    Convention:  rack{N}-node{M}
    Same rack  → 1.0 µs  (InfiniBand within a rack)
    Diff rack  → 5.0 µs  (InfiniBand cross-rack)
    Unknown    → 10.0 µs (conservative)

    Phase 2: replace with active ICMP/RDMA ping measurement.
    """
    try:
        local_rack  = local_node.split("-")[0]
        remote_rack = remote_node.split("-")[0]
        if local_rack == remote_rack:
            return 1.0
        return 5.0
    except Exception:
        return 10.0


class MemoryHypervisor:
    """
    Cluster memory hypervisor.

    Single entry point for all cross-node memory operations.
    Instantiate one per process. Pass to VMM as vmm = VMM(hypervisor=h).

    Usage (single node, no network, for testing):
        h = MemoryHypervisor(node_id="node-a", host="127.0.0.1", enable_network=False)
        h.register_local_node(free_bytes={"dram": 512*1024**3, "nvme": 8*1024**4})

    Usage (cluster):
        h = MemoryHypervisor(node_id="rack1-node2", host="10.0.0.2")
        h.start()   # begins heartbeat broadcast and listener
    """

    def __init__(
        self,
        node_id:        str  = "local",
        host:           str  = "127.0.0.1",
        gkd:            Optional[GKDStore] = None,
        enable_network: bool = False,
        heartbeat_port: int  = HEARTBEAT_PORT,
        borrow_port:    int  = BORROW_PORT_BASE,
    ):
        self.node_id        = node_id
        self.host           = host
        self.gkd            = gkd or GKDStore()
        self._cluster_map   = ClusterMap()
        self._transport     = make_transport(listen_port=borrow_port)
        self._enable_net    = enable_network
        self._hb_port       = heartbeat_port
        self._borrow_port   = borrow_port
        self._running       = False

        # Stats
        self._lock              = threading.Lock()
        self._borrows_total     = 0
        self._borrows_failed    = 0
        self._bytes_borrowed    = 0

    def register_local_node(self, free_bytes: Dict[str, int]):
        """
        Register this node's own memory in the cluster map.

        Must be called before borrow_memory() will work in no-network mode.
        free_bytes maps tier names to available bytes, e.g. {"dram": 64*1024**3}.
        """
        self._cluster_map.update(self.node_id, self.host, free_bytes)

    def register_peer(self, node_id: str, host: str, free_bytes: Dict[str, int]):
        """
        Manually register a peer node.

        Used in tests and single-datacenter deployments where
        the node list is static (no gossip needed).
        """
        self._cluster_map.update(node_id, host, free_bytes)

    def start(self):
        """
        Start heartbeat broadcast and listener threads.

        Only needed for dynamic cluster membership (enable_network=True).
        No-op when enable_network=False. Safe to call multiple times.
        """
        if not self._enable_net:
            return
        self._running = True
        self._transport.start_server()
        threading.Thread(target=self._heartbeat_sender, daemon=True).start()
        threading.Thread(target=self._heartbeat_listener, daemon=True).start()
        threading.Thread(target=self._reaper, daemon=True).start()
        logger.info(f"Hypervisor started: node_id={self.node_id} host={self.host}")

    def stop(self):
        """
        Stop all background threads and close transport connections.

        After stop(), start() must not be called again on this instance.
        """
        self._running = False
        self._transport.close()

    # ── Core API ───────────────────────────────────────────────────────

    def borrow_memory(
        self,
        size_bytes: int,
        tier: str = "dram",
    ) -> Optional[BorrowOffer]:
        """
        Find a donor node and return addressing info for a remote read.

        Returns None if no donor is available. The caller must handle
        this gracefully — fall back to NVMe or return OOM to the user.

        Does NOT move any data. The caller uses the returned BorrowOffer
        to call transport.read() and pull the data.

        Selection algorithm:
          1. Filter: reachable nodes != self with free_bytes[tier] >= size_bytes
          2. Sort: ascending estimated latency (rack-local first)
          3. Return: BorrowOffer for top candidate
        """
        candidates = self._cluster_map.candidates(
            size_bytes=size_bytes,
            tier=tier,
            exclude_node=self.node_id,
        )

        if not candidates:
            with self._lock:
                self._borrows_failed += 1
            logger.debug(f"borrow_memory: no candidates for {size_bytes} bytes on {tier}")
            return None

        # Sort by estimated latency — prefer rack-local nodes
        candidates.sort(
            key=lambda n: _estimate_latency_us(self.node_id, n.node_id)
        )
        donor = candidates[0]
        lat   = _estimate_latency_us(self.node_id, donor.node_id)

        # Construct a RemoteRegion. In TCP mode addr=0 and rkey=0 are
        # placeholders — the TCPTransport server looks up by addr.
        # In RDMA mode (Phase 2) these are the real ibv_mr values.
        remote_region = RemoteRegion(
            node_id=donor.node_id,
            addr=0,
            rkey=0,
            length=size_bytes,
        )

        with self._lock:
            self._borrows_total += 1
            self._bytes_borrowed += size_bytes

        logger.debug(
            f"borrow_memory: donor={donor.node_id} "
            f"size={size_bytes} tier={tier} est_lat={lat}µs"
        )

        return BorrowOffer(
            node_id=donor.node_id,
            host=donor.host,
            tier=tier,
            size_bytes=size_bytes,
            remote_region=remote_region,
            estimated_latency_us=lat,
        )

    def update_local_free(self, tier: str, free_bytes: int):
        """
        Called by the local VMM after every alloc/free to keep the
        cluster map current. Other nodes see this via heartbeat.
        """
        existing = self._cluster_map._nodes.get(self.node_id)
        if existing:
            existing.free_bytes[tier] = free_bytes
        else:
            self._cluster_map.update(self.node_id, self.host, {tier: free_bytes})

    def stats(self) -> dict:
        """
        Return hypervisor and cluster statistics.

        Includes borrow counters, cluster node counts, and GKD stats.
        Safe to call from any thread at any time.
        """
        with self._lock:
            b = self._borrows_total
            f = self._borrows_failed
            by = self._bytes_borrowed
        nodes = self._cluster_map.all_nodes()
        return {
            "node_id":              self.node_id,
            "cluster_nodes_total":  len(nodes),
            "cluster_nodes_reachable": sum(1 for n in nodes if n.reachable),
            "borrows_total":        b,
            "borrows_failed":       f,
            "bytes_borrowed_gb":    round(by / 1e9, 3),
            "gkd_stats":            self.gkd.stats(),
            "cluster_free_dram_gb": round(
                self._cluster_map.total_free("dram") / 1e9, 3
            ),
        }

    # ── Heartbeat (network mode only) ──────────────────────────────────

    def _heartbeat_sender(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        while self._running:
            try:
                node_state = self._cluster_map._nodes.get(self.node_id)
                payload = json.dumps({
                    "node_id":   self.node_id,
                    "host":      self.host,
                    "free_bytes": node_state.free_bytes if node_state else {},
                }).encode()
                sock.sendto(payload, ("<broadcast>", self._hb_port))
            except Exception as e:
                logger.debug(f"Heartbeat send error: {e}")
            time.sleep(HEARTBEAT_INTERVAL_S)
        sock.close()

    def _heartbeat_listener(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", self._hb_port))
        sock.settimeout(1.0)
        while self._running:
            try:
                data, addr = sock.recvfrom(4096)
                msg = json.loads(data.decode())
                if msg["node_id"] != self.node_id:
                    self._cluster_map.update(
                        msg["node_id"], msg["host"], msg["free_bytes"]
                    )
            except socket.timeout:
                continue
            except Exception as e:
                logger.debug(f"Heartbeat recv error: {e}")
        sock.close()

    def _reaper(self):
        while self._running:
            time.sleep(HEARTBEAT_INTERVAL_S)
            self._cluster_map.reap_stale()
