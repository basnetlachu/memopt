"""
Transport layer — abstracts tensor send/recv across cluster nodes.

Shim layer: detects whether the C++ transport sidecar daemon is running
(checks for /dev/shm/memopt_transport_{node}_req). If running, creates
a SHMTransport client that communicates via shared memory ring buffer.
If not running, falls back to the original Python transport (TCPTransport
or UCXTransport).

All callers use make_transport() and never branch on the implementation.
The shim is transparent — identical AbstractTransport interface.
"""
from __future__ import annotations
import logging
import os
import socket

logger = logging.getLogger(__name__)

# Always import the full Python transport module for fallback and
# for re-exporting types that tests and __init__.py depend on.
from memopt.cluster._transport_py import (  # noqa: F401
    AbstractTransport,
    TCPTransport,
    UCXTransport,
    MemoryRegion,
    RemoteRegion,
)

# Re-export make_transport — detect daemon or use Python fallback.
_original_make_transport = None
try:
    from memopt.cluster._transport_py import make_transport as _original_make_transport
except ImportError:
    pass


def make_transport(
    listen_port: int = 18515,
    prefer_rdma: bool = True,
) -> AbstractTransport:
    """
    Return the best available transport for this environment.

    Selection order:
      1. C++ sidecar daemon (if running) — RDMA or TCP via ring buffer IPC.
      2. Python UCXTransport — if ucx-py installed.
      3. Python TCPTransport — always works.

    The sidecar daemon is detected by checking for shared memory files
    at /dev/shm/memopt_transport_{node_id}_req. This check is O(1)
    (stat() syscall, ~1µs) and never blocks.
    """
    # Check if C++ daemon is running
    node_id = os.environ.get("MEMOPT_NODE_ID", socket.gethostname())
    req_path = f"/dev/shm/memopt_transport_{node_id}_req"
    resp_path = f"/dev/shm/memopt_transport_{node_id}_resp"

    if os.path.exists(req_path) and os.path.exists(resp_path):
        try:
            from memopt.cluster._transport_shm import SHMTransport
            logger.info(
                f"memopt: C++ transport daemon detected "
                f"(node={node_id}, shm={req_path})"
            )
            return SHMTransport(node_id, req_path, resp_path)
        except Exception as e:
            logger.warning(
                f"memopt: SHM transport init failed ({e}) — "
                f"falling back to Python transport"
            )

    # Fall back to original Python make_transport
    if _original_make_transport is not None:
        return _original_make_transport(listen_port, prefer_rdma)

    # Last resort: TCPTransport
    return TCPTransport(listen_port=listen_port)


__all__ = [
    "AbstractTransport",
    "TCPTransport",
    "UCXTransport",
    "MemoryRegion",
    "RemoteRegion",
    "make_transport",
]
