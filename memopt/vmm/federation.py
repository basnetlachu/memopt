"""
Federation — TCP gossip for sharing Oracle transitions across cluster nodes.

Each node periodically broadcasts its Markov transitions to all known peers.
Peers merge incoming transitions into their local Oracle, accelerating
convergence without centralised coordination.

Protocol: length-prefixed JSON frames over TCP (same framing as remote_block.py).
Frame = 4-byte big-endian length + JSON payload.

Environment variables:
  MEMOPT_NODE_ID       str   default "node_0"
  MEMOPT_NODE_HOSTS    str   comma-separated host:port pairs
"""
from __future__ import annotations

import json
import logging
import os
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .oracle import MemoryOracle

logger = logging.getLogger(__name__)

_DEFAULT_NODE_ID = os.environ.get("MEMOPT_NODE_ID", "node_0")
_DEFAULT_GOSSIP_PORT = 18600


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
    ) -> None:
        self._oracle = oracle
        self._node_id = node_id or _DEFAULT_NODE_ID
        self._gossip_port = gossip_port
        self._gossip_interval = gossip_interval_s
        self._epoch = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        if peers is not None:
            self._peers = list(peers)
        else:
            raw = os.environ.get("MEMOPT_NODE_HOSTS", "")
            self._peers = [p.strip() for p in raw.split(",") if p.strip()]

        self._stats = FederationStats(
            node_id=self._node_id,
            peers_known=len(self._peers),
        )

        self._sender_thread: Optional[threading.Thread] = None
        self._receiver_thread: Optional[threading.Thread] = None

    # ── Public API ────────────────────────────────────────────────────────

    def start(self) -> None:
        self._stop_event.clear()
        self._sender_thread = threading.Thread(
            target=self._sender_loop, daemon=True, name="federation-sender",
        )
        self._receiver_thread = threading.Thread(
            target=self._receiver_loop, daemon=True, name="federation-receiver",
        )
        self._sender_thread.start()
        self._receiver_thread.start()
        logger.info(
            "FederationManager started: node=%s port=%d peers=%d",
            self._node_id, self._gossip_port, len(self._peers),
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
                peers_known=len(self._peers),
                batches_sent=self._stats.batches_sent,
                batches_received=self._stats.batches_received,
                transitions_merged=self._stats.transitions_merged,
                last_gossip_epoch=self._stats.last_gossip_epoch,
            )

    # ── Internal ──────────────────────────────────────────────────────────

    def _sender_loop(self) -> None:
        while not self._stop_event.wait(self._gossip_interval):
            batch = self._build_batch()
            if not batch.transitions:
                continue
            payload = self._serialize_batch(batch)
            for peer in self._peers:
                try:
                    self._send_to_peer(peer, payload)
                    with self._lock:
                        self._stats.batches_sent += 1
                except Exception as exc:
                    logger.debug("Federation send to %s failed: %s", peer, exc)

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
                        logger.debug("Federation accept error: %s", exc)
        except Exception as exc:
            logger.debug("Federation receiver failed to start: %s", exc)

    def _build_batch(self) -> GossipBatch:
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

    def _merge_batch(self, batch: GossipBatch) -> None:
        if batch.node_id == self._node_id:
            return
        merged = 0
        with self._oracle._lock:
            for t in batch.transitions:
                existing = self._oracle._transitions[t.from_block][t.to_block]
                if t.count > existing:
                    self._oracle._transitions[t.from_block][t.to_block] = t.count
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
            logger.debug("Federation handle incoming error: %s", exc)
        finally:
            try:
                conn.close()
            except Exception:
                pass
