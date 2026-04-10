"""
Federation — TCP gossip for sharing Oracle transitions across cluster nodes.

Each node periodically broadcasts its Markov transitions to a bounded
set of random peers. Peers merge incoming transitions into their local
Oracle, accelerating convergence without centralised coordination.

Peer discovery:
  1. Redis (preferred): nodes register in Redis, discover each other
     automatically. Set REDIS_URL to enable.
  2. Static list (fallback): MEMOPT_NODE_HOSTS env var.
  3. None: node runs alone, no gossip (graceful degradation).

Gossip protocol:
  - Bounded fan-out: each round gossips to K random peers
    (MEMOPT_GOSSIP_FANOUT, default 5). At 1M nodes, each node
    sends exactly 5 messages per interval.
  - Delta-only: only transitions that changed since last round
    are sent. Reduces gossip bandwidth from O(transitions)
    to O(new_transitions).
  - Frame format: 4-byte big-endian length + JSON payload.

Environment variables:
  MEMOPT_NODE_ID           str   default "node_0"
  MEMOPT_NODE_HOSTS        str   comma-separated host:port pairs (fallback)
  MEMOPT_GOSSIP_FANOUT     int   peers per gossip round (default 5)
  REDIS_URL                str   Redis for peer discovery (optional)
"""
from __future__ import annotations

import json
import logging
import os
import random
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .oracle import MemoryOracle

logger = logging.getLogger(__name__)

_DEFAULT_NODE_ID = os.environ.get("MEMOPT_NODE_ID", "node_0")
_DEFAULT_GOSSIP_PORT = 18600
_DEFAULT_GOSSIP_FANOUT = 5


@dataclass(frozen=True)
class TransitionGossip:
    from_block: int
    to_block: int
    count: int


@dataclass(frozen=True)
class GossipBatch:
    node_id: str
    epoch: int
    transitions: List[TransitionGossip]


@dataclass
class FederationStats:
    node_id: str = ""
    peers_known: int = 0
    batches_sent: int = 0
    batches_received: int = 0
    transitions_merged: int = 0
    last_gossip_epoch: int = 0


def _send_frame(sock: socket.socket, payload: bytes) -> None:
    """Send a length-prefixed frame: 4-byte big-endian length + payload."""
    sock.sendall(struct.pack(">I", len(payload)) + payload)


def _recv_frame(sock: socket.socket, timeout_s: float = 2.0) -> bytes:
    """Receive a length-prefixed frame."""
    sock.settimeout(timeout_s)
    header = _recv_exactly(sock, 4)
    length = struct.unpack(">I", header)[0]
    if length > 10 * 1024 * 1024:
        raise RuntimeError(f"Gossip frame too large: {length}")
    return _recv_exactly(sock, length)


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise RuntimeError("Connection closed mid-frame")
        buf += chunk
    return buf


class FederationManager:

    def __init__(
        self,
        oracle: "MemoryOracle",
        node_id: str = "",
        peers: Optional[List[str]] = None,
        gossip_port: int = _DEFAULT_GOSSIP_PORT,
        gossip_interval_s: float = 5.0,
        host: str = "",
    ) -> None:
        self._oracle = oracle
        self._node_id = node_id or _DEFAULT_NODE_ID
        self._host = host or socket.gethostname()
        self._gossip_port = gossip_port
        self._gossip_interval = gossip_interval_s
        self._epoch = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Static peer list (fallback)
        if peers is not None:
            self._peers = list(peers)
        else:
            raw = os.environ.get("MEMOPT_NODE_HOSTS", "")
            self._peers = [p.strip() for p in raw.split(",") if p.strip()]

        # Gossip fan-out: bounded number of peers per round
        self._fanout = int(os.environ.get(
            "MEMOPT_GOSSIP_FANOUT", str(_DEFAULT_GOSSIP_FANOUT)))

        # Delta tracking: only send changed transitions
        self._last_gossiped: Dict[Tuple[int, int], int] = {}

        # Redis peer discovery (optional)
        self._redis_client = None
        self._peer_registry_key = "memopt:federation:nodes"
        self._peer_ttl_s = 30  # node considered dead after 30s no refresh
        self._init_redis_discovery()

        self._stats = FederationStats(
            node_id=self._node_id,
            peers_known=len(self._peers),
        )

        self._sender_thread: Optional[threading.Thread] = None
        self._receiver_thread: Optional[threading.Thread] = None

    def _init_redis_discovery(self) -> None:
        """Try to connect to Redis for peer discovery."""
        redis_url = os.environ.get("REDIS_URL")
        if not redis_url:
            return
        try:
            import redis
            self._redis_client = redis.Redis.from_url(
                redis_url,
                socket_timeout=1.0,
                socket_connect_timeout=1.0,
                decode_responses=True,
            )
            self._redis_client.ping()
            logger.info(
                "memopt: federation using Redis peer discovery")
        except Exception as e:
            logger.warning(
                "memopt: Redis peer discovery unavailable: %s, "
                "using MEMOPT_NODE_HOSTS fallback", e)
            self._redis_client = None

    # ── Peer discovery ────────────────────────────────────────────────

    def _register_self(self) -> None:
        """Register this node in Redis peer registry with TTL."""
        if not self._redis_client:
            return
        try:
            key = f"{self._peer_registry_key}:{self._node_id}"
            value = f"{self._host}:{self._gossip_port}"
            self._redis_client.setex(key, self._peer_ttl_s, value)
        except Exception:
            pass  # non-fatal — node still works without registration

    def _discover_peers(self) -> List[str]:
        """
        Get current peer list.
        Redis (if available) → dynamic discovery.
        Fallback → static MEMOPT_NODE_HOSTS list.
        """
        if self._redis_client:
            try:
                pattern = f"{self._peer_registry_key}:*"
                keys = self._redis_client.keys(pattern)
                peers = []
                for key in keys:
                    key_str = key if isinstance(key, str) else key.decode()
                    node_id = key_str.split(":")[-1]
                    if node_id == self._node_id:
                        continue  # skip self
                    value = self._redis_client.get(key)
                    if value:
                        val = value if isinstance(value, str) \
                            else value.decode()
                        peers.append(val)
                if peers:
                    return peers
            except Exception as e:
                logger.debug("Redis peer discovery failed: %s", e)
        # Fallback: static peer list
        return list(self._peers)

    # ── Public API ────────────────────────────────────────────────────

    def start(self) -> None:
        self._stop_event.clear()
        self._register_self()
        self._sender_thread = threading.Thread(
            target=self._sender_loop, daemon=True,
            name="federation-sender",
        )
        self._receiver_thread = threading.Thread(
            target=self._receiver_loop, daemon=True,
            name="federation-receiver",
        )
        self._sender_thread.start()
        self._receiver_thread.start()
        logger.info(
            "FederationManager started: node=%s port=%d peers=%d "
            "fanout=%d",
            self._node_id, self._gossip_port,
            len(self._peers), self._fanout,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._sender_thread is not None:
            self._sender_thread.join(timeout=5.0)
        if self._receiver_thread is not None:
            self._receiver_thread.join(timeout=5.0)

    def stats(self) -> FederationStats:
        with self._lock:
            return FederationStats(
                node_id=self._stats.node_id,
                peers_known=len(self._discover_peers()),
                batches_sent=self._stats.batches_sent,
                batches_received=self._stats.batches_received,
                transitions_merged=self._stats.transitions_merged,
                last_gossip_epoch=self._stats.last_gossip_epoch,
            )

    # ── Gossip round ──────────────────────────────────────────────────

    def _do_gossip_round(self) -> None:
        """
        One gossip round: build delta batch, send to K random peers.
        Called by _sender_loop every gossip_interval.
        Also callable directly for testing.
        """
        # Build delta batch (only changed transitions)
        batch = self._build_delta_batch()
        if not batch.transitions:
            return  # nothing changed since last round

        payload = self._serialize_batch(batch)

        # Discover peers (Redis or static)
        available = self._discover_peers()
        if not available:
            return

        # Bounded fan-out: pick K random peers
        fanout = min(self._fanout, len(available))
        selected = random.sample(available, fanout)

        for peer in selected:
            try:
                self._send_to_peer(peer, payload)
                with self._lock:
                    self._stats.batches_sent += 1
            except Exception as exc:
                logger.debug(
                    "Federation send to %s failed: %s", peer, exc)

        # Refresh own registration in Redis
        self._register_self()

    def _build_delta_batch(self) -> GossipBatch:
        """
        Build a batch containing only transitions that changed
        since the last gossip round. Snapshot taken under oracle lock
        to prevent missing any transition.
        """
        # Snapshot current transitions under lock
        current: Dict[Tuple[int, int], int] = {}
        with self._oracle._lock:
            for from_block, counter in self._oracle._transitions.items():
                for to_block, count in counter.items():
                    current[(from_block, to_block)] = count

        # Compute delta: transitions with changed counts
        delta_transitions = []
        for key, count in current.items():
            if self._last_gossiped.get(key, 0) != count:
                delta_transitions.append(
                    TransitionGossip(key[0], key[1], count))

        # Update last_gossiped snapshot
        self._last_gossiped = current

        with self._lock:
            self._epoch += 1
            epoch = self._epoch
            self._stats.last_gossip_epoch = epoch

        return GossipBatch(
            node_id=self._node_id,
            epoch=epoch,
            transitions=delta_transitions,
        )

    # ── Internal — kept for backward compatibility with tests ─────────

    def _build_batch(self) -> GossipBatch:
        """
        Build a full batch of ALL transitions (not delta).
        Kept for backward compatibility — existing tests call this.
        Production code uses _build_delta_batch via _do_gossip_round.
        """
        with self._oracle._lock:
            transitions = []
            for from_block, counter in self._oracle._transitions.items():
                for to_block, count in counter.items():
                    transitions.append(
                        TransitionGossip(from_block, to_block, count),
                    )
        with self._lock:
            self._epoch += 1
            epoch = self._epoch
            self._stats.last_gossip_epoch = epoch
        return GossipBatch(
            node_id=self._node_id,
            epoch=epoch,
            transitions=transitions,
        )

    def _sender_loop(self) -> None:
        while not self._stop_event.wait(self._gossip_interval):
            self._do_gossip_round()

    def _receiver_loop(self) -> None:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", self._gossip_port))
            sock.listen(16)
            sock.settimeout(1.0)

            while not self._stop_event.is_set():
                try:
                    conn, addr = sock.accept()
                    threading.Thread(
                        target=self._handle_incoming,
                        args=(conn,),
                        daemon=True,
                    ).start()
                except socket.timeout:
                    continue
                except Exception as exc:
                    if not self._stop_event.is_set():
                        logger.debug(
                            "Federation accept error: %s", exc)
        except Exception as exc:
            logger.debug(
                "Federation receiver failed to start: %s", exc)

    def _merge_batch(self, batch: GossipBatch) -> None:
        if batch.node_id == self._node_id:
            return
        merged = 0
        with self._oracle._lock:
            for t in batch.transitions:
                existing = self._oracle._transitions[t.from_block][t.to_block]
                if t.count > existing:
                    self._oracle._transitions[t.from_block][t.to_block] = \
                        t.count
                    merged += 1
        with self._lock:
            self._stats.batches_received += 1
            self._stats.transitions_merged += merged

    def _serialize_batch(self, batch: GossipBatch) -> bytes:
        data = {
            "node_id": batch.node_id,
            "epoch": batch.epoch,
            "transitions": [
                {"f": t.from_block, "t": t.to_block, "c": t.count}
                for t in batch.transitions
            ],
        }
        return json.dumps(data).encode()

    def _deserialize_batch(self, raw: bytes) -> GossipBatch:
        data = json.loads(raw.decode())
        transitions = [
            TransitionGossip(t["f"], t["t"], t["c"])
            for t in data["transitions"]
        ]
        return GossipBatch(
            node_id=data["node_id"],
            epoch=data["epoch"],
            transitions=transitions,
        )

    def _send_to_peer(self, peer: str, payload: bytes) -> None:
        if ":" in peer:
            host, port_str = peer.rsplit(":", 1)
            port = int(port_str)
        else:
            host = peer
            port = self._gossip_port
        sock = socket.create_connection((host, port), timeout=2.0)
        try:
            _send_frame(sock, payload)
        finally:
            sock.close()

    def _handle_incoming(self, conn: socket.socket) -> None:
        try:
            raw = _recv_frame(conn)
            batch = self._deserialize_batch(raw)
            self._merge_batch(batch)
        except Exception as exc:
            logger.debug(
                "Federation handle incoming error: %s", exc)
        finally:
            try:
                conn.close()
            except Exception:
                pass

    # ── Helper for getting transition snapshot (used by tests) ────────

    def _get_current_transitions(self) -> Dict[Tuple[int, int], int]:
        """Return current transition counts as {(from, to): count}."""
        result = {}
        with self._oracle._lock:
            for from_b, counter in self._oracle._transitions.items():
                for to_b, count in counter.items():
                    result[(from_b, to_b)] = count
        return result
