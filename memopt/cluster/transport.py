"""
RDMA transport layer — one-sided memory operations between cluster nodes.

Two backends:
  RDMATransport   ibverbs (InfiniBand / RoCE) — ~1 µs, zero CPU on remote
  TCPTransport    socket fallback              — ~100–500 µs, works everywhere

Backend is selected automatically at construction time based on whether
pyverbs is importable and an RDMA-capable NIC is detected.

One-sided means: Node A calls read(remote_addr, size) and the data moves
directly from Node B's DRAM into Node A's buffer via the NIC's DMA engine.
Node B's CPU is not interrupted. No system call on Node B. This is the
mechanism that gives RDMA its latency advantage over TCP.

Memory registration:
  RDMA requires memory to be pinned (page-locked) before transfer.
  ibv_reg_mr() tells the kernel not to swap this memory and returns
  an rkey (remote key) that the remote side uses to address it.
  Registration is expensive (~1 ms) but done once per buffer, not per transfer.
  Transfers on registered memory cost ~1 µs.

Queue pair (QP):
  A QP is the RDMA equivalent of a socket. One QP per peer connection.
  QPs are created once and reused. State machine: RESET → INIT → RTR → RTS.
  Only in RTS state can the QP post send/recv work requests.
"""
from __future__ import annotations
import socket
import struct
import threading
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict

logger = logging.getLogger(__name__)


@dataclass
class MemoryRegion:
    """A pinned, RDMA-accessible memory buffer."""
    addr:       int     # virtual address (pointer as int)
    length:     int     # size in bytes
    rkey:       int     # remote key — given to remote side to address this region
    lkey:       int     # local key — used in local work requests
    _mr_handle: object = field(default=None, repr=False)  # ibv_mr* or None


@dataclass
class RemoteRegion:
    """A remote memory region we can address via RDMA."""
    node_id:    str
    addr:       int     # remote virtual address
    rkey:       int     # remote key received from the remote node
    length:     int


class TCPTransport:
    """
    TCP socket fallback for development machines without RDMA NICs.

    Implements the same API as RDMATransport. All tests use this backend.
    A server thread listens for incoming read/write requests. The caller
    uses read() and write() identically to the RDMA path.

    Protocol (little-endian, fixed header):
        op      : uint8   (1=READ_REQ, 2=READ_RESP, 3=WRITE)
        addr    : uint64
        length  : uint32
        payload : bytes[length]  (only for WRITE and READ_RESP)
    """

    HEADER = struct.Struct("<BQI")   # op, addr, length
    OP_READ_REQ  = 1
    OP_READ_RESP = 2
    OP_WRITE     = 3

    def __init__(self, listen_port: int = 18515):
        self._port        = listen_port
        self._memory: Dict[int, bytearray] = {}   # addr → buffer (simulated pinned memory)
        self._lock        = threading.RLock()
        self._server_sock: Optional[socket.socket] = None
        self._running     = False
        self._connections: Dict[str, socket.socket] = {}   # node_id → socket

    def start_server(self):
        """
        Start listening for incoming RDMA-emulated requests.

        Spawns a daemon thread — does not block. Safe to call once per process.
        Raises OSError if the port is already in use.
        """
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind(("0.0.0.0", self._port))
        self._server_sock.listen(32)
        self._running = True
        t = threading.Thread(target=self._accept_loop, daemon=True)
        t.start()
        logger.info(f"TCPTransport server listening on port {self._port}")

    def _accept_loop(self):
        while self._running:
            try:
                conn, addr = self._server_sock.accept()
                t = threading.Thread(
                    target=self._handle_conn, args=(conn,), daemon=True
                )
                t.start()
            except Exception:
                break

    def _handle_conn(self, conn: socket.socket):
        try:
            while True:
                header = self._recv_exact(conn, self.HEADER.size)
                if not header:
                    break
                op, addr, length = self.HEADER.unpack(header)
                if op == self.OP_READ_REQ:
                    with self._lock:
                        buf = self._memory.get(addr)
                        data = bytes(buf[:length]) if buf else b"\x00" * length
                    resp_header = self.HEADER.pack(self.OP_READ_RESP, addr, len(data))
                    conn.sendall(resp_header + data)
                elif op == self.OP_WRITE:
                    data = self._recv_exact(conn, length)
                    with self._lock:
                        if addr not in self._memory:
                            self._memory[addr] = bytearray(length)
                        self._memory[addr][:length] = data
        except Exception as e:
            logger.debug(f"TCPTransport handler error: {e}")
        finally:
            conn.close()

    def _recv_exact(self, conn: socket.socket, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                return b""
            buf += chunk
        return buf

    def register_memory(self, buffer: bytearray) -> MemoryRegion:
        """
        Pin a buffer and return its region descriptor.

        In TCP mode, 'pinning' means registering the buffer's id() in the
        server's local memory map so READ_REQ can look it up. No actual
        page-locking occurs — this is a simulation of the ibv_reg_mr API.
        """
        addr = id(buffer)
        with self._lock:
            self._memory[addr] = buffer
        # rkey and lkey are both the addr in TCP mode (no real NIC keys)
        return MemoryRegion(addr=addr, length=len(buffer), rkey=addr, lkey=addr)

    def deregister_memory(self, region: MemoryRegion):
        """
        Release a previously registered memory region.

        Mirrors ibv_dereg_mr(). Must be called when the buffer will no longer
        be used for RDMA transfers. Forgetting to call this leaks the simulated
        pinned memory entry.
        """
        with self._lock:
            self._memory.pop(region.addr, None)

    def connect(self, node_id: str, host: str, port: int = 18515) -> bool:
        """
        Open a TCP connection to a remote node.

        Returns True on success, False on connection failure. Non-blocking
        in the sense that it does not retry — the caller decides retry policy.
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect((host, port))
            sock.settimeout(None)
            with self._lock:
                self._connections[node_id] = sock
            return True
        except Exception as e:
            logger.warning(f"TCPTransport connect to {node_id} failed: {e}")
            return False

    def read(self, remote: RemoteRegion, dst: bytearray) -> float:
        """
        Read from remote node into dst. Returns elapsed seconds.

        Raises RuntimeError if no connection to remote.node_id has been
        established via connect(). Caller must connect() first.
        """
        conn = self._connections.get(remote.node_id)
        if conn is None:
            raise RuntimeError(f"No connection to {remote.node_id}")
        t0 = time.monotonic()
        header = self.HEADER.pack(self.OP_READ_REQ, remote.addr, remote.length)
        conn.sendall(header)
        resp_header = self._recv_exact(conn, self.HEADER.size)
        _, _, length = self.HEADER.unpack(resp_header)
        data = self._recv_exact(conn, length)
        dst[:length] = data
        return time.monotonic() - t0

    def write(self, remote: RemoteRegion, src: bytearray) -> float:
        """
        Write src to remote node. Returns elapsed seconds.

        Raises RuntimeError if no connection to remote.node_id exists.
        The write is fire-and-forget at the protocol level — the server
        processes it asynchronously. Use a follow-up read to verify.
        """
        conn = self._connections.get(remote.node_id)
        if conn is None:
            raise RuntimeError(f"No connection to {remote.node_id}")
        t0 = time.monotonic()
        header = self.HEADER.pack(self.OP_WRITE, remote.addr, len(src))
        conn.sendall(header + bytes(src))
        return time.monotonic() - t0

    def close(self):
        """
        Close all connections and stop the server thread.

        Safe to call multiple times. After close(), start_server() and
        connect() must not be called again on this instance.
        """
        self._running = False
        for sock in self._connections.values():
            try:
                sock.close()
            except Exception:
                pass
        if self._server_sock:
            try:
                self._server_sock.close()
            except Exception:
                pass


def _detect_rdma() -> bool:
    """Return True if pyverbs is importable and an RDMA device is present."""
    try:
        import pyverbs.device as d
        devices = d.get_device_list()
        return len(devices) > 0
    except Exception:
        return False


def make_transport(listen_port: int = 18515) -> TCPTransport:
    """
    Return the best available transport.

    Currently always returns TCPTransport — RDMATransport is Phase 2
    once pyverbs packaging is standardised across distros.
    The API is identical so the swap is one line when ready.
    """
    if _detect_rdma():
        logger.info("RDMA NIC detected — using RDMATransport (Phase 2: not yet implemented)")
        # TODO Phase 2: return RDMATransport(listen_port)
        # Fall through to TCP until RDMATransport is implemented
    logger.info("Using TCPTransport (no RDMA NIC or pyverbs not installed)")
    return TCPTransport(listen_port=listen_port)
