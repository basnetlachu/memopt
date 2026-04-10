"""
SHM Transport client — communicates with the C++ sidecar daemon
via shared memory ring buffers.

No C++ extension needed on the Python side. Uses mmap + struct
for direct shared memory access.

The ring buffer protocol matches csrc/transport/ring_buffer.h:
  - SPSC (single producer, single consumer)
  - SlotHeader layout: ready(1B) + size(4B) + msg_type(1B) + pad(2B) = 8B
  - Slot: header(8B) + payload(64KB)
  - RingBufferHeader: write_head(8B) + pad(56B) + read_head(8B) + pad(56B) +
                      producer_pid(4B) + pad(60B) + consumer_pid(4B) + pad(60B) = 256B

Python is the producer for the request buffer and consumer for the response buffer.
A threading.Lock serializes writes from multiple Python threads into the SPSC req buffer.
"""
from __future__ import annotations

import logging
import mmap
import os
import struct as _struct
import threading
import time
from typing import Optional

from memopt.cluster._transport_py import (
    AbstractTransport,
    MemoryRegion,
    RemoteRegion,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Constants — must match C++ ring_buffer.h and protocol.h
# ═══════════════════════════════════════════════════════════════════════════

RB_CAPACITY   = 4096
RB_SLOT_SIZE  = 65536   # 64KB max payload
SLOT_HDR_SIZE = 8       # ready(1) + size(4) + msg_type(1) + pad(2)
SLOT_TOTAL    = SLOT_HDR_SIZE + RB_SLOT_SIZE
HEADER_SIZE   = 256     # RingBufferHeader (cache-line padded)

# Message types — must match protocol.h MessageType enum
class MsgType:
    CONNECT        = 1
    READ           = 2
    WRITE          = 3
    REGISTER_MEM   = 4
    DEREGISTER_MEM = 5
    CLOSE          = 6
    SHUTDOWN       = 7
    PING           = 8
    CONNECT_ACK    = 64
    READ_COMPLETE  = 65
    WRITE_COMPLETE = 66
    REGISTER_ACK   = 67
    ERROR          = 68
    PONG           = 69

# ═══════════════════════════════════════════════════════════════════════════
# Struct format strings — match protocol.h struct layouts exactly.
# All little-endian (<). Verified by size assertions below.
# ═══════════════════════════════════════════════════════════════════════════

# ConnectRequest: node_id(64s) + host(256s) + port(H) + prefer_rdma(B) + pad(5B)
CONNECT_REQ_FMT = "<64s256sHB5x"
assert _struct.calcsize(CONNECT_REQ_FMT) == 328

# ReadRequest: node_id(64s) + remote_addr(Q) + rkey(I) + pad(I) +
#              local_addr(Q) + size(I) + pad(I) + request_id(Q)
READ_REQ_FMT = "<64sQIIQIIQ"
assert _struct.calcsize(READ_REQ_FMT) == 112

# WriteRequest: same layout as ReadRequest
WRITE_REQ_FMT = READ_REQ_FMT

# RegisterMemRequest: addr(Q) + size(Q) + request_id(Q)
REG_MEM_FMT = "<QQQ"
assert _struct.calcsize(REG_MEM_FMT) == 24

# ConnectAck: node_id(64s) + success(B) + used_rdma(B) + pad(6B) + error_msg(128s)
CONNECT_ACK_FMT = "<64sBB6x128s"
assert _struct.calcsize(CONNECT_ACK_FMT) == 200

# ReadComplete: request_id(Q) + success(B) + pad(3B) + bytes_read(I) +
#               latency_us(f) + pad(I)
READ_DONE_FMT = "<QB3xIfI"
assert _struct.calcsize(READ_DONE_FMT) == 24

# WriteComplete: same layout as ReadComplete
WRITE_DONE_FMT = READ_DONE_FMT

# RegisterAck: request_id(Q) + addr(Q) + lkey(I) + rkey(I) +
#              success(B) + pad(7B)
REG_ACK_FMT = "<QQIIB7x"
assert _struct.calcsize(REG_ACK_FMT) == 32

# ErrorResponse: request_id(Q) + error_code(B) + pad(7B) + message(128s)
ERROR_FMT = "<QB7x128s"
assert _struct.calcsize(ERROR_FMT) == 144


# ═══════════════════════════════════════════════════════════════════════════
# SHMTransport — full data plane implementation
# ═══════════════════════════════════════════════════════════════════════════

class SHMTransport(AbstractTransport):
    """
    Transport client that communicates with the C++ sidecar daemon
    via shared memory ring buffers.
    """

    def __init__(self, node_id: str, req_path: str, resp_path: str):
        self._node_id = node_id
        self._push_lock = threading.Lock()  # serialize SPSC producer writes

        # mmap request ring buffer (Python writes, daemon reads)
        self._req_fd = os.open(req_path, os.O_RDWR)
        total_size = HEADER_SIZE + RB_CAPACITY * SLOT_TOTAL
        self._req_mm = mmap.mmap(self._req_fd, total_size)

        # mmap response ring buffer (daemon writes, Python reads)
        self._resp_fd = os.open(resp_path, os.O_RDWR)
        self._resp_mm = mmap.mmap(self._resp_fd, total_size)

        self._req_id = 0
        self._bytes_sent = 0
        self._bytes_recv = 0

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    # ── AbstractTransport interface ───────────────────────────────────

    def start_server(self) -> None:
        pass  # daemon is already running

    def connect(self, node_id: str, host: str, port: int = 18515) -> bool:
        """Send CONNECT request to daemon. Returns True on success."""
        payload = _struct.pack(CONNECT_REQ_FMT,
            node_id.encode()[:64].ljust(64, b'\x00'),
            host.encode()[:256].ljust(256, b'\x00'),
            port, 1)  # prefer_rdma=1

        if not self._push(MsgType.CONNECT, payload):
            return False

        result = self._wait_response(MsgType.CONNECT_ACK, timeout=10.0)
        if result is None:
            return False

        ack = _struct.unpack(CONNECT_ACK_FMT, result)
        # ack: (node_id, success, used_rdma, error_msg)
        return bool(ack[1])

    def register_memory(self, buffer: bytearray) -> MemoryRegion:
        """Register buffer with daemon for RDMA access."""
        addr = id(buffer)
        req_id = self._next_id()
        payload = _struct.pack(REG_MEM_FMT, addr, len(buffer), req_id)

        if not self._push(MsgType.REGISTER_MEM, payload):
            # Fallback: return local-only region
            return MemoryRegion(addr=addr, length=len(buffer),
                                rkey=addr, lkey=addr)

        result = self._wait_response(MsgType.REGISTER_ACK, timeout=2.0)
        if result is None:
            return MemoryRegion(addr=addr, length=len(buffer),
                                rkey=addr, lkey=addr)

        ack = _struct.unpack(REG_ACK_FMT, result)
        # ack: (request_id, addr, lkey, rkey, success)
        if ack[4]:
            return MemoryRegion(addr=ack[1], length=len(buffer),
                                rkey=ack[3], lkey=ack[2])
        return MemoryRegion(addr=addr, length=len(buffer),
                            rkey=addr, lkey=addr)

    def deregister_memory(self, region) -> None:
        """Deregister a previously registered memory region."""
        payload = _struct.pack(REG_MEM_FMT,
                               region.addr, region.length,
                               self._next_id())
        self._push(MsgType.DEREGISTER_MEM, payload)

    def read(self, remote: RemoteRegion, dst: bytearray) -> float:
        """Issue RDMA/TCP read via daemon. Returns elapsed seconds."""
        req_id = self._next_id()
        payload = _struct.pack(READ_REQ_FMT,
            remote.node_id.encode()[:64].ljust(64, b'\x00'),
            remote.addr,
            remote.rkey,
            0,  # _pad1
            id(dst),
            len(dst),
            0,  # _pad2
            req_id)

        if not self._push(MsgType.READ, payload):
            return -1.0

        timeout = float(os.getenv("MEMOPT_RBP_TIMEOUT_S", "2.0"))
        result = self._wait_response(MsgType.READ_COMPLETE, timeout=timeout)
        if result is None:
            return -1.0

        ack = _struct.unpack(READ_DONE_FMT, result)
        # ack: (request_id, success, bytes_read, latency_us, _pad)
        if ack[1]:
            self._bytes_recv += ack[2]
            return ack[3] / 1e6  # convert µs to seconds
        return -1.0

    def write(self, remote: RemoteRegion, src: bytearray) -> float:
        """Issue RDMA/TCP write via daemon. Returns elapsed seconds."""
        req_id = self._next_id()
        payload = _struct.pack(WRITE_REQ_FMT,
            remote.node_id.encode()[:64].ljust(64, b'\x00'),
            remote.addr,
            remote.rkey,
            0,  # _pad1
            id(src),
            len(src),
            0,  # _pad2
            req_id)

        if not self._push(MsgType.WRITE, payload):
            return -1.0

        timeout = float(os.getenv("MEMOPT_RBP_TIMEOUT_S", "2.0"))
        result = self._wait_response(MsgType.WRITE_COMPLETE, timeout=timeout)
        if result is None:
            return -1.0

        ack = _struct.unpack(WRITE_DONE_FMT, result)
        # ack: (request_id, success, bytes_written, latency_us, _pad)
        if ack[1]:
            self._bytes_sent += ack[2]
            return ack[3] / 1e6
        return -1.0

    def close(self) -> None:
        """Close transport. Send CLOSE message to daemon."""
        try:
            self._push(MsgType.CLOSE, b"")
        except Exception:
            pass
        try:
            self._req_mm.close()
            os.close(self._req_fd)
        except Exception:
            pass
        try:
            self._resp_mm.close()
            os.close(self._resp_fd)
        except Exception:
            pass

    def stats(self) -> dict:
        """Return transport stats. Pings daemon for liveness check."""
        alive = self._ping(timeout=1.0)
        return {
            "transport":      "rdma_sidecar",
            "daemon_alive":   alive,
            "node_id":        self._node_id,
            "bytes_sent":     self._bytes_sent,
            "bytes_recv":     self._bytes_recv,
            "latency_us_p50": 0,
            "latency_us_p99": 0,
        }

    # ── Ring buffer operations ────────────────────────────────────────

    def _push(self, msg_type: int, payload: bytes = b"") -> bool:
        """Write one message to the request ring buffer. Thread-safe."""
        if len(payload) > RB_SLOT_SIZE:
            return False

        with self._push_lock:  # SPSC: serialize multiple Python threads
            # Read write_head (offset 0 in header)
            wh = _struct.unpack_from("<Q", self._req_mm, 0)[0]
            slot_idx = wh % RB_CAPACITY
            slot_off = HEADER_SIZE + slot_idx * SLOT_TOTAL

            # Check if slot is consumed (ready == 0)
            if self._req_mm[slot_off] != 0:
                return False  # buffer full

            # Write payload data FIRST (before setting ready flag)
            data_off = slot_off + SLOT_HDR_SIZE
            if payload:
                self._req_mm[data_off:data_off + len(payload)] = payload

            # Write slot header fields: size + msg_type
            _struct.pack_into("<I", self._req_mm,
                              slot_off + 1, len(payload))
            self._req_mm[slot_off + 5] = msg_type

            # Set ready = 1 LAST (release semantics via sequential writes)
            self._req_mm[slot_off] = 1

            # Advance write_head
            _struct.pack_into("<Q", self._req_mm, 0, wh + 1)

        return True

    def _try_pop(self) -> Optional[tuple]:
        """Read one message from the response ring buffer."""
        # Read read_head (offset 64 in header)
        rh = _struct.unpack_from("<Q", self._resp_mm, 64)[0]
        slot_idx = rh % RB_CAPACITY
        slot_off = HEADER_SIZE + slot_idx * SLOT_TOTAL

        # Check ready flag
        if self._resp_mm[slot_off] == 0:
            return None  # empty

        # Read header fields
        size = _struct.unpack_from("<I", self._resp_mm, slot_off + 1)[0]
        msg_type = self._resp_mm[slot_off + 5]

        # Read payload
        data_off = slot_off + SLOT_HDR_SIZE
        payload = bytes(self._resp_mm[data_off:data_off + size])

        # Mark consumed: ready = 0 (after reading payload)
        self._resp_mm[slot_off] = 0

        # Advance read_head
        _struct.pack_into("<Q", self._resp_mm, 64, rh + 1)

        return (msg_type, payload)

    def _wait_response(self, expected_type: int,
                       timeout: float = 2.0) -> Optional[bytes]:
        """Poll response ring buffer for a specific message type."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = self._try_pop()
            if result is not None:
                msg_type, payload = result
                if msg_type == expected_type:
                    return payload
                # Not our message — could be a response to a different
                # request. In production, use request_id matching.
                # For now, discard non-matching messages.
                logger.debug(
                    "SHMTransport: discarded msg type %d "
                    "(expected %d)", msg_type, expected_type)
            time.sleep(0.0001)  # 100µs poll interval
        return None

    def _ping(self, timeout: float = 1.0) -> bool:
        """Send PING, wait for PONG."""
        if not self._push(MsgType.PING, b""):
            return False
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = self._try_pop()
            if result and result[0] == MsgType.PONG:
                return True
            time.sleep(0.001)
        return False
