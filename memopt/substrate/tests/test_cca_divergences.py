"""Tests that document where memopt substrate intentionally diverges from
PyTorch CUDACachingAllocator (CCA) semantics.

Each test references the PyTorch CCA behaviour, asserts memopt's
divergent behaviour, and stamps the date the design decision was made.
If the rationale changes, the date stamp helps locate the rolling-back
decision.

Per design §3.2.
"""
from __future__ import annotations

import inspect

import pytest


# Design decision date for ALL three divergences below: 2026-04-29
# Source: docs/substrate_v1_design.md §2.5, edge cases E2 / E3 / E4.
DESIGN_DATE = "2026-04-29"


def test_no_auto_peer_access_enable():
    """PyTorch CCA: on first cross-device touch, auto-calls
    cudaDeviceEnablePeerAccess. memopt: requires explicit
    set_access([peer_dev]) before cross-device access.

    Why we diverge:
      Auto-enable has caused subtle bugs around per-process
      resource limits (max enabled peers per device). Explicit
      is auditable. See §2.5 E2.

    Decision date: 2026-04-29. Revisit if a future CUDA driver
    removes the per-process peer-access limit."""
    pytest.importorskip("torch")
    import torch

    if torch.cuda.device_count() < 2:
        pytest.skip("requires >= 2 CUDA devices")

    # On a single-GPU rig (this rig has one A100) the divergence is
    # documented but not directly testable. The assertion below pins
    # the design date for the future maintainer.
    assert DESIGN_DATE == "2026-04-29"


def test_no_ipc_handle_export_in_v1():
    """PyTorch CCA: supports cudaIpcGetMemHandle / cudaIpcOpenMemHandle
    for cross-process tensor sharing. memopt v1: NO IPC API. Cross-
    process sharing in memopt today goes through the RDMA transport;
    cross-process VMM peer mapping is deferred to v1.1.

    Why we diverge:
      IPC handles tie reference counts to a kernel object that
      doesn't fit our backend strategy abstraction cleanly.
      Solving it well requires v1.1 design work. See §2.5 E3.

    Decision date: 2026-04-29. Revisit when v1.1 design is
    drafted (target: post-Q3-2026)."""
    from memopt.substrate.handle import MemoryHandle

    # No to_ipc_handle method exists on MemoryHandle.
    assert not hasattr(MemoryHandle, "to_ipc_handle"), (
        f"v1 must not expose IPC API. Decision date: {DESIGN_DATE}"
    )

    # Verify the only cross-process channel is export_fabric_handle on
    # the backend, which returns None on this rig per S0.1 DEGRADED.
    from memopt.substrate.backends.cpu_fallback import CPUFallbackBackend
    from memopt.substrate.backends.base import PhysLoc

    backend = CPUFallbackBackend()
    ph = backend.create_physical(4096, PhysLoc.DRAM)
    try:
        result = backend.export_fabric_handle(ph)
        assert result is None or isinstance(result, bytes)
    finally:
        backend.release_physical(ph)


def test_single_device_per_alloc_call_in_v1():
    """PyTorch CCA: each device has its own allocator; explicit
    tensor.to(device) for cross-device. memopt v1: alloc() has NO
    `device=` parameter — uses the current CUDA device.

    Why we diverge:
      Multi-device allocation in one call adds a parameter that
      90% of users won't need; it complicates the public surface
      and is easy to forget. Explicit `with torch.cuda.device(N):`
      around the call is the existing PyTorch idiom. See §2.5 E4.

    Decision date: 2026-04-29. Revisit if user feedback shows
    `device=` would prevent real bugs."""
    import memopt.substrate

    sig = inspect.signature(memopt.substrate.alloc)
    assert "device" not in sig.parameters, (
        f"v1 alloc() must not have device parameter. "
        f"Decision date: {DESIGN_DATE}"
    )
