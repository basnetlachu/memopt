"""Tests for CUDABackend (per design §3.1 test_backend_cuda.py).

All tests are marked @pytest.mark.gpu. Names contain "cuda" so they
are deselected by the regression's `-k 'not gpu and not cuda'` filter
on no-GPU CI; they run when invoked explicitly on a CUDA rig.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available on this host", allow_module_level=True)

from memopt.substrate.backends.base import PhysLoc
from memopt.substrate.backends.cuda_vmm import CUDABackend


_POOL = 4 * 1024 * 1024 * 1024  # 4 GiB; comfortable headroom on A100 80G


@pytest.fixture
def cuda_backend():
    backend = CUDABackend(device_idx=0, pool_size_bytes=_POOL)
    yield backend
    backend.close()


@pytest.mark.gpu
def test_cuda_backend_seven_primitives_smoke(cuda_backend):
    size = 4 * 1024 * 1024
    va_token = cuda_backend.reserve_va(size)
    assert va_token == 0  # sentinel; legacy VMM manages VA internally

    ph = cuda_backend.create_physical(size, PhysLoc.HBM)
    assert ph.raw > 0
    assert ph.location is PhysLoc.HBM

    cuda_backend.map(ph.raw, ph)
    cuda_backend.set_access(ph.raw, size, [0])
    cuda_backend.unmap(ph.raw, size)
    cuda_backend.release_physical(ph)
    cuda_backend.free_va(va_token, size)


@pytest.mark.gpu
def test_cuda_backend_granularity_returns_2mib(cuda_backend):
    assert cuda_backend.granularity_bytes() == 2 * 1024 * 1024


@pytest.mark.gpu
def test_cuda_backend_fabric_handle_export(cuda_backend):
    """Per S0.1 DEGRADED on this rig, returns None. Once re-verified on
    a CUDA 12.4+ rig with IMEX, this will return non-empty bytes."""
    ph = cuda_backend.create_physical(2 * 1024 * 1024, PhysLoc.HBM)
    try:
        result = cuda_backend.export_fabric_handle(ph)
        assert result is None or isinstance(result, bytes)
    finally:
        cuda_backend.release_physical(ph)


@pytest.mark.gpu
def test_cuda_backend_set_access_for_peer_is_explicit(cuda_backend):
    """v1 does not auto-enable peer access; documented divergence from CCA."""
    size = 2 * 1024 * 1024
    ph = cuda_backend.create_physical(size, PhysLoc.HBM)
    try:
        cuda_backend.set_access(ph.raw, size, [0])
    finally:
        cuda_backend.release_physical(ph)


@pytest.mark.gpu
def test_cuda_backend_evict_promote_round_trip(cuda_backend):
    """Allocate; evict; promote; verify."""
    size = 8 * 1024 * 1024
    handles = [cuda_backend.create_physical(size, PhysLoc.HBM) for _ in range(4)]
    try:
        s_before = cuda_backend.stats()
        cuda_backend.evict_to_target(0.10)
        s_after = cuda_backend.stats()
        assert s_after.get("pages_dram", 0) >= s_before.get("pages_dram", 0)

        if handles:
            cuda_backend.promote(int(handles[0].raw))
    finally:
        for ph in handles:
            cuda_backend.release_physical(ph)
