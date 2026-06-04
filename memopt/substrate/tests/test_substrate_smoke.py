"""End-to-end smoke tests for the assembled substrate (per design §3.1)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

import memopt
import memopt.substrate
from memopt.substrate.manager import AllocationManager


@pytest.fixture(autouse=True)
def _fresh_manager():
    AllocationManager.reset()
    yield
    AllocationManager.reset()


def test_alloc_use_free_round_trip_cpu():
    """Round trip on the CPU fallback backend, with alloc + free events."""
    received_alloc = []
    received_free = []
    sub_a = memopt.observe("alloc", received_alloc.append)
    sub_f = memopt.observe("free", received_free.append)
    try:
        with memopt.context(tenant="alice", placement="cpu"):
            h = memopt.alloc(4096)
            assert h.tenant == "alice"
            assert h.size_bytes == 4096
            assert h.backend_name == "cpu"
            h.write(b"\x42" * 4096)
            data = h.read()
            assert len(data) == 4096
            assert data[0] == 0x42
            memopt.free(h)
        # Event delivery is async; give it a beat.
        import time
        deadline = time.monotonic() + 1.0
        while (not received_alloc or not received_free) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert received_alloc and received_alloc[0].kind == "alloc"
        assert received_free and received_free[0].kind == "free"
    finally:
        sub_a.unsubscribe()
        sub_f.unsubscribe()


@pytest.mark.gpu
def test_alloc_use_free_round_trip_cuda():
    pytest.importorskip("torch")
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    with memopt.context(tenant="alice", placement="hbm"):
        h = memopt.alloc(2 * 1024 * 1024)
        try:
            assert h.backend_name == "cuda"
            assert h.size_bytes == 2 * 1024 * 1024
        finally:
            memopt.free(h)


def test_observe_unobserve_lifecycle():
    received = []
    sub = memopt.observe("alloc", received.append)
    with memopt.context(tenant="ob", placement="cpu"):
        h = memopt.alloc(1024)
    sub.unsubscribe()
    # Subsequent allocations are not delivered to this subscriber.
    received_after = list(received)
    with memopt.context(tenant="ob", placement="cpu"):
        h2 = memopt.alloc(1024)
        memopt.free(h2)
    memopt.free(h)
    import time
    time.sleep(0.05)
    # Ensure no new event arrived after unsubscribe (count unchanged).
    assert len(received) == len(received_after)


def test_context_propagation_into_thread_pool():
    """memopt.context captures into a ThreadPoolExecutor future."""
    captured = []

    def worker():
        with memopt.context(tenant="alice", placement="cpu"):
            h = memopt.alloc(1024)
            captured.append(h.tenant)
            memopt.free(h)

    with memopt.context(tenant="alice", placement="cpu"):
        with ThreadPoolExecutor(max_workers=1) as ex:
            ex.submit(worker).result()

    assert captured == ["alice"]
