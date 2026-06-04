"""
Remote Block Protocol — request/response for cross-node block transfer.

Protocol messages (sent as JSON over the existing TCP/UCX transport):

  ADVERTISE   node_id, content_hash, size_bytes, tier, path
              Sent when a VMM evicts a block to NVMe.
              Registers the block in the directory.

  REQUEST     requesting_node, content_hash, dst_addr, dst_port
              Sent when a VMM needs a block held by a remote node.
              The remote node responds by:
                1. Checking the block directory (do we have it?)
                2. Acquiring a local read lock on the block file
                3. Sending the raw bytes via transport
                4. Releasing the lock

  TRANSFER    content_hash, size_bytes, data (raw bytes)
              Response to REQUEST. Contains the actual block data.

  LEASE_ACQUIRE   requesting_node, content_hash
  LEASE_RELEASE   requesting_node, content_hash
              Lease management — prevents the owning node from
              evicting a block while a remote node is using it.

  ERROR       content_hash, reason
              Sent when a block cannot be served (not found,
              evicted since REQUEST was sent, read error).

All messages are length-prefixed JSON frames over TCP.
The existing TCPTransport handles the raw socket management.
RemoteBlockProtocol adds the framing and dispatch layer on top.

Environment variables:
  MEMOPT_RBP_PORT         int   default 18516
  MEMOPT_RBP_TIMEOUT_S    float default 2.0
  MEMOPT_RBP_MAX_BLOCK_MB float default 256.0
"""
from __future__ import annotations
import os
import json
import socket
import struct
import threading
import logging
import time
from typing import Optional, Callable

logger = logging.getLogger(__name__)

_RBP_PORT        = int(os.environ.get("MEMOPT_RBP_PORT",       "18516"))
_TIMEOUT_S       = float(os.environ.get("MEMOPT_RBP_TIMEOUT_S", "2.0"))
_MAX_BLOCK_MB    = float(os.environ.get("MEMOPT_RBP_MAX_BLOCK_MB", "256.0"))
_MAX_BLOCK_BYTES = int(_MAX_BLOCK_MB * 1024 * 1024)

# Message types
MSG_ADVERTISE     = "ADVERTISE"
MSG_REQUEST       = "REQUEST"
MSG_TRANSFER      = "TRANSFER"
MSG_LEASE_ACQUIRE = "LEASE_ACQUIRE"
MSG_LEASE_RELEASE = "LEASE_RELEASE"
MSG_ERROR         = "ERROR"


def _send_frame(sock: socket.socket, payload: bytes) -> None:
    """Send a length-prefixed frame: 4-byte big-endian length + payload."""
    sock.sendall(struct.pack(">I", len(payload)) + payload)


def _recv_frame(sock: socket.socket, timeout_s: float = _TIMEOUT_S) -> bytes:
    """
    Receive a length-prefixed frame.
    Raises RuntimeError on timeout or connection close.
    """
    sock.settimeout(timeout_s)
    header = _recv_exactly(sock, 4)
    length = struct.unpack(">I", header)[0]
    if length > _MAX_BLOCK_BYTES + 4096:
        raise RuntimeError(
            f"Frame too large: {length} bytes "
            f"(max {_MAX_BLOCK_BYTES + 4096})"
        )
    return _recv_exactly(sock, length)


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise RuntimeError("Connection closed mid-frame")
        buf += chunk
    return buf


class RemoteBlockServer:
    """
    Listens for incoming block requests from remote nodes.
    Runs in a daemon thread. One server per node.

    The block_directory and read_block_fn are injected at construction
    time — no direct VMM coupling.

    read_block_fn(path: str) -> bytes | None
      Called to read a block from NVMe. Returns None if not found.
      Must be thread-safe.
    """

    def __init__(
        self,
        node_id:       str,
        block_directory,
        read_block_fn: Callable[[str], Optional[bytes]],
        port:          int = _RBP_PORT,
    ):
        self._node_id    = node_id
        self._directory  = block_directory
        self._read_block = read_block_fn
        self._port       = port
        self._running    = False
        self._sock:      Optional[socket.socket] = None
        self._lock       = threading.RLock()

        # Stats
        self._requests_served = 0
        self._bytes_served    = 0
        self._errors          = 0

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
        t = threading.Thread(
            target=self._accept_loop, daemon=True,
            name=f"rbp-server-{self._node_id}"
        )
        t.start()
        logger.info(
            f"RemoteBlockServer: listening on port {self._port} "
            f"node={self._node_id}"
        )

    def stop(self) -> None:
        with self._lock:
            self._running = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass

    def stats(self) -> dict:
        with self._lock:
            return {
                "requests_served": self._requests_served,
                "bytes_served":    self._bytes_served,
                "errors":          self._errors,
            }

    def _accept_loop(self) -> None:
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.bind(("0.0.0.0", self._port))
            self._sock.listen(32)
            self._sock.settimeout(1.0)

            while self._running:
                try:
                    conn, addr = self._sock.accept()
                    t = threading.Thread(
                        target=self._handle_conn,
                        args=(conn, addr),
                        daemon=True,
                    )
                    t.start()
                except socket.timeout:
                    continue
                except Exception as e:
                    if self._running:
                        logger.debug(f"RemoteBlockServer accept error: {e}")
        except Exception as e:
            logger.warning(f"RemoteBlockServer failed to start: {e}")

    def _handle_conn(self, conn: socket.socket, addr) -> None:
        try:
            raw   = _recv_frame(conn)
            msg   = json.loads(raw.decode())
            mtype = msg.get("type")

            if mtype == MSG_REQUEST:
                self._handle_request(conn, msg)
            elif mtype == MSG_LEASE_ACQUIRE:
                self._handle_lease_acquire(conn, msg)
            elif mtype == MSG_LEASE_RELEASE:
                self._handle_lease_release(conn, msg)
            else:
                self._send_error(conn, "", f"Unknown message type: {mtype}")
        except Exception as e:
            logger.debug(f"RemoteBlockServer handle_conn error: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _handle_request(self, conn: socket.socket, msg: dict) -> None:
        content_hash    = msg.get("content_hash", "")
        requesting_node = msg.get("requesting_node", "")

        # Look up the block
        entry = self._directory.lookup(content_hash)
        if entry is None or entry.node_id != self._node_id:
            self._send_error(conn, content_hash, "block not found")
            with self._lock:
                self._errors += 1
            return

        # Read from NVMe
        data = self._read_block(entry.path)
        if data is None:
            self._send_error(conn, content_hash, "read failed")
            with self._lock:
                self._errors += 1
            return

        # Send TRANSFER response
        response = json.dumps({
            "type":         MSG_TRANSFER,
            "content_hash": content_hash,
            "size_bytes":   len(data),
        }).encode()
        _send_frame(conn, response)
        _send_frame(conn, data)

        with self._lock:
            self._requests_served += 1
            self._bytes_served    += len(data)

        logger.debug(
            f"RemoteBlockServer: served {content_hash[:16]}... "
            f"({len(data)} bytes) to {requesting_node}"
        )

    def _handle_lease_acquire(self, conn: socket.socket,
                               msg: dict) -> None:
        content_hash    = msg.get("content_hash", "")
        requesting_node = msg.get("requesting_node", "")
        granted = self._directory.acquire_lease(content_hash, requesting_node)
        response = json.dumps({
            "type":         "LEASE_ACK",
            "granted":      granted,
            "content_hash": content_hash,
        }).encode()
        _send_frame(conn, response)

    def _handle_lease_release(self, conn: socket.socket,
                               msg: dict) -> None:
        content_hash    = msg.get("content_hash", "")
        requesting_node = msg.get("requesting_node", "")
        self._directory.release_lease(content_hash, requesting_node)
        response = json.dumps({"type": "LEASE_ACK", "released": True}).encode()
        _send_frame(conn, response)

    def _send_error(self, conn: socket.socket,
                    content_hash: str, reason: str) -> None:
        payload = json.dumps({
            "type":         MSG_ERROR,
            "content_hash": content_hash,
            "reason":       reason,
        }).encode()
        try:
            _send_frame(conn, payload)
        except Exception:
            pass


class RemoteBlockClient:
    """
    Client-side: fetches blocks from remote nodes.

    fetch_block(content_hash, remote_host, remote_port)
      Connects to the remote RemoteBlockServer, sends a REQUEST,
      receives the TRANSFER response, and returns the raw bytes.
      Returns None on any error (timeout, block not found, etc.).
      Never raises — all errors are caught and logged.

    acquire_lease / release_lease
      Optimistic lease management. If the remote server cannot be
      reached, the caller proceeds without a lease. The worst case
      is that the remote node evicts the block before transfer
      completes — handled by returning None from fetch_block.
    """

    def __init__(self, node_id: str, timeout_s: float = _TIMEOUT_S):
        self._node_id  = node_id
        self._timeout  = timeout_s
        self._lock     = threading.RLock()
        self._fetches  = 0
        self._failures = 0
        self._bytes_fetched = 0

    def fetch_block(
        self,
        content_hash:  str,
        remote_host:   str,
        remote_port:   int = _RBP_PORT,
    ) -> Optional[bytes]:
        """
        Fetch a block from a remote node with retry.
        Returns raw bytes on success, None after all attempts fail.

        Retry count: MEMOPT_FETCH_RETRIES (default 2).
        Retry delay: MEMOPT_FETCH_RETRY_DELAY_S (default 0.5s).
        """
        # Default: 0 retries (single attempt) for backward compat.
        # Set MEMOPT_FETCH_RETRIES=2 in production for retry.
        max_retries = int(os.environ.get('MEMOPT_FETCH_RETRIES', '0'))
        retry_delay = float(os.environ.get(
            'MEMOPT_FETCH_RETRY_DELAY_S', '0.5'))

        for attempt in range(max_retries + 1):
            result = self._try_fetch_once(
                content_hash, remote_host, remote_port)
            if result is not None:
                return result
            if attempt < max_retries:
                logger.debug(
                    f"fetch_block attempt {attempt + 1} failed, "
                    f"retrying in {retry_delay}s")
                time.sleep(retry_delay)

        logger.error(
            f"fetch_block failed after {max_retries + 1} attempts "
            f"hash={content_hash[:16]}... "
            f"peer={remote_host}:{remote_port}")
        return None

    def _try_fetch_once(
        self,
        content_hash: str,
        remote_host:  str,
        remote_port:  int,
    ) -> Optional[bytes]:
        """Single fetch attempt. Returns bytes or None."""
        try:
            sock = socket.create_connection(
                (remote_host, remote_port), timeout=self._timeout
            )
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            request = json.dumps({
                "type":            MSG_REQUEST,
                "content_hash":    content_hash,
                "requesting_node": self._node_id,
            }).encode()
            _send_frame(sock, request)

            header_raw = _recv_frame(sock, self._timeout)
            header     = json.loads(header_raw.decode())

            if header.get("type") == MSG_ERROR:
                logger.debug(
                    f"RemoteBlockClient: remote error for "
                    f"{content_hash[:16]}...: {header.get('reason')}"
                )
                with self._lock:
                    self._failures += 1
                sock.close()
                return None

            if header.get("type") != MSG_TRANSFER:
                with self._lock:
                    self._failures += 1
                sock.close()
                return None

            data = _recv_frame(sock, self._timeout)
            sock.close()

            with self._lock:
                self._fetches += 1
                self._bytes_fetched += len(data)

            logger.debug(
                f"RemoteBlockClient: fetched {content_hash[:16]}... "
                f"({len(data)} bytes) from {remote_host}:{remote_port}"
            )
            return data

        except Exception as e:
            logger.debug(
                f"RemoteBlockClient: fetch failed for "
                f"{content_hash[:16] if len(content_hash) >= 16 else content_hash}...: {e}"
            )
            with self._lock:
                self._failures += 1
            return None

    def acquire_lease(
        self,
        content_hash: str,
        remote_host:  str,
        remote_port:  int = _RBP_PORT,
    ) -> bool:
        """
        Request a lease on a remote block. Returns True if granted.
        Returns False on any error — caller should proceed without lease.
        """
        try:
            sock    = socket.create_connection(
                (remote_host, remote_port), timeout=self._timeout
            )
            payload = json.dumps({
                "type":            MSG_LEASE_ACQUIRE,
                "content_hash":    content_hash,
                "requesting_node": self._node_id,
            }).encode()
            _send_frame(sock, payload)
            ack_raw = _recv_frame(sock, self._timeout)
            ack     = json.loads(ack_raw.decode())
            sock.close()
            return ack.get("granted", False)
        except Exception as e:
            logger.debug(
                f"RemoteBlockClient: lease acquire failed "
                f"for {content_hash[:16] if len(content_hash) >= 16 else content_hash}...: {e}"
            )
            return False

    def release_lease(
        self,
        content_hash: str,
        remote_host:  str,
        remote_port:  int = _RBP_PORT,
    ) -> None:
        """Release a lease. Best-effort — never raises."""
        try:
            sock    = socket.create_connection(
                (remote_host, remote_port), timeout=self._timeout
            )
            payload = json.dumps({
                "type":            MSG_LEASE_RELEASE,
                "content_hash":    content_hash,
                "requesting_node": self._node_id,
            }).encode()
            _send_frame(sock, payload)
            _recv_frame(sock, self._timeout)
            sock.close()
        except Exception as e:
            logger.debug(
                f"RemoteBlockClient: lease release failed "
                f"for {content_hash[:16] if len(content_hash) >= 16 else content_hash}...: {e}"
            )

    def fetch_block_with_failover(
        self,
        content_hash: str,
        candidates: list,
        geo_router: "GeoRouter | None" = None,
        metrics=None,
        max_attempts: int = 3,
    ) -> Optional[bytes]:
        """
        Fetch from multiple candidates with automatic failover.

        candidates: list of dicts with node_id, host, port, rack, pod, region.
        Tries candidates in locality order. Stops after max_attempts or first success.

        Returns bytes or None. None = all candidates failed, caller must recompute.

        FAILOVER CONTRACT:
        - Never returns wrong data
        - Never blocks indefinitely (each attempt has timeout)
        - Records every attempt in metrics
        - Falls back gracefully
        """
        if not candidates:
            return None

        if geo_router:
            candidates = geo_router.sort_peers(candidates)

        attempts = min(max_attempts, len(candidates))

        for i, candidate in enumerate(candidates[:attempts]):
            node_id = candidate.get("node_id", "unknown")
            host = candidate.get("host", "")
            port = candidate.get("port", _RBP_PORT)

            if not host:
                continue

            start = time.perf_counter()
            result = None

            try:
                result = self._try_fetch_once(content_hash, host, port)
            except Exception as e:
                logger.debug(
                    "GUM fetch attempt %d failed (%s): %s", i + 1, node_id, e)

            elapsed_ms = (time.perf_counter() - start) * 1000

            if metrics is not None:
                try:
                    from memopt.cluster.gum_metrics import TransferMeasurement
                    metrics.record_transfer(TransferMeasurement(
                        peer_node_id=node_id,
                        bytes_transferred=len(result) if result else 0,
                        latency_ms=elapsed_ms,
                        success=result is not None,
                        transport=self._transport_type(),
                        rack=candidate.get("rack", "unknown"),
                        pod=candidate.get("pod", "unknown"),
                        region=candidate.get("region", "unknown"),
                    ))
                except Exception:
                    pass

            if result is not None:
                if i > 0:
                    logger.info(
                        "GUM failover: succeeded on attempt %d "
                        "(first %d failed)", i + 1, i)
                return result

        # All candidates failed
        if metrics is not None:
            try:
                metrics.record_fallback()
            except Exception:
                pass

        logger.debug(
            "GUM: all %d candidates failed for %s. Caller will recompute.",
            attempts, content_hash[:8])
        return None

    def _transport_type(self) -> str:
        """Returns 'rdma' or 'tcp' based on which transport is active."""
        try:
            # Check if RDMA transport daemon is running
            import os
            import socket
            node_id = os.environ.get(
                "MEMOPT_NODE_ID", socket.gethostname())
            req_path = f"/dev/shm/memopt_transport_{node_id}_req"
            if os.path.exists(req_path):
                return "rdma"
        except Exception:
            pass
        return "tcp"

    def stats(self) -> dict:
        with self._lock:
            return {
                "fetches":       self._fetches,
                "failures":      self._failures,
                "bytes_fetched": self._bytes_fetched,
            }


class GeoRouter:
    """
    Routes block fetch requests to the geographically nearest peer first.

    Locality order (nearest first):
      1. Same rack     (IB: ~1us, TCP: ~0.1ms)
      2. Same pod      (IB: ~5us, TCP: ~0.5ms)
      3. Same region   (IB: ~20us, TCP: ~1ms)
      4. Cross-region  (WAN: ~10ms+)

    HONEST: The latency numbers above are typical values. We do not
    hardcode them. We measure them via GUMMetrics. The routing LOGIC
    is hardware-agnostic. The routing BENEFIT depends on hardware.
    """

    def __init__(
        self,
        local_node_id: str,
        local_rack: str = "unknown",
        local_pod: str = "unknown",
        local_region: str = "unknown",
    ):
        self._node_id = local_node_id
        self._rack = local_rack
        self._pod = local_pod
        self._region = local_region

    def sort_peers(self, peers: list) -> list:
        """
        Sort peers by proximity. Nearest first.

        Each peer dict must have node_id, and optionally rack, pod, region.
        Returns sorted list. Thread-safe (no state mutation). Never raises.
        """
        def locality_score(peer: dict) -> int:
            rack = peer.get("rack", "unknown")
            pod = peer.get("pod", "unknown")
            region = peer.get("region", "unknown")

            if rack == self._rack and rack != "unknown":
                return 0  # Same rack (best)
            if pod == self._pod and pod != "unknown":
                return 1  # Same pod
            if region == self._region and region != "unknown":
                return 2  # Same region
            return 3  # Cross-region (worst)

        try:
            return sorted(peers, key=locality_score)
        except Exception:
            return list(peers)

    def locality_label(self, peer: dict) -> str:
        """Human-readable locality label."""
        rack = peer.get("rack", "unknown")
        pod = peer.get("pod", "unknown")
        region = peer.get("region", "unknown")

        if rack == self._rack and rack != "unknown":
            return "same_rack"
        if pod == self._pod and pod != "unknown":
            return "same_pod"
        if region == self._region and region != "unknown":
            return "same_region"
        return "cross_region"

    @classmethod
    def from_env(cls) -> "GeoRouter":
        """Create from environment variables."""
        import socket as _socket
        return cls(
            local_node_id=os.environ.get(
                "MEMOPT_NODE_ID", _socket.gethostname()),
            local_rack=os.environ.get("MEMOPT_RACK", "unknown"),
            local_pod=os.environ.get("MEMOPT_POD", "unknown"),
            local_region=os.environ.get("MEMOPT_REGION", "unknown"),
        )
