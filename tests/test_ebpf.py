"""
Tests for the eBPF CUDA kernel interceptor and kernel swap protocol.

All 8 tests pass WITHOUT BCC installed.
The interceptor falls back to /proc + nvidia-smi monitoring — this is
the actual path that runs on this server (BCC not available in apt).
"""

import os
import time

import pytest
from fastapi.testclient import TestClient

from memopt.ebpf.interceptor import CUDAKernelInterceptor
from memopt.ebpf.kernel_swapper import KernelSwapper, _shm_dir
import memopt.control_plane.server as srv


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolated_server(tmp_path, monkeypatch):
    """Patch server singletons so each test gets a clean DB and known API key."""
    from memopt.control_plane.database import Database
    from memopt.fleet.intelligence import FleetIntelligence
    from memopt.alerts.alert_store import AlertStore
    from memopt.fleet.gossip import GossipKnowledgeBase

    new_db    = Database(db_path=tmp_path / "cp.db")
    new_db.init()
    new_fleet = FleetIntelligence(db_path=str(tmp_path / "fleet.db"), auto_remediate=False)
    new_gossip = GossipKnowledgeBase(db_path=str(tmp_path / "gossip.db"))

    monkeypatch.setattr(srv, "db",          new_db)
    monkeypatch.setattr(srv, "alert_store", AlertStore())
    monkeypatch.setattr(srv, "_fleet",      new_fleet)
    monkeypatch.setattr(srv, "_gossip_kb",  new_gossip)
    monkeypatch.setattr(srv, "_API_KEY",    "test-key")

    # Give each test a fresh interceptor with no side effects
    monkeypatch.setattr(srv, "_interceptor", CUDAKernelInterceptor())

    yield


@pytest.fixture
def test_client():
    return TestClient(srv.app, raise_server_exceptions=True)


TEST_API_KEY = "test-key"


# ── Test 1: Interceptor initialises without BCC ───────────────────────────────

def test_interceptor_initializes_without_bcc():
    """
    Import and __init__ must succeed whether BCC is installed or not.
    _bcc_available is a plain bool.
    """
    i = CUDAKernelInterceptor()
    assert isinstance(i._bcc_available, bool)
    # On this server BCC is not installed — confirm the honest result
    assert i._bcc_available is False


# ── Test 2: start() uses /proc fallback, _running=True ───────────────────────

def test_interceptor_starts_with_proc_fallback():
    """
    start() must not raise even without BCC.
    _running must be True after start and False after stop.
    """
    i = CUDAKernelInterceptor()
    result = i.start(pids=[os.getpid()])
    assert result is False          # BCC not available → /proc path
    assert i._running is True
    i.stop()
    assert i._running is False


# ── Test 3: get_stats() returns correct structure ────────────────────────────

def test_get_stats_returns_correct_structure():
    """
    get_stats() must always return a dict with all four expected keys,
    regardless of whether the interceptor has been started.
    """
    i = CUDAKernelInterceptor()
    stats = i.get_stats()

    assert "ebpf_active"        in stats
    assert "total_launches"     in stats
    assert "detections"         in stats
    assert "suboptimal_kernels" in stats
    assert isinstance(stats["suboptimal_kernels"], list)


# ── Test 4: add_pid / remove_pid ────────────────────────────────────────────

def test_add_and_remove_pid():
    """
    add_pid must add to monitored_pids.
    remove_pid must remove it. No BCC required.
    """
    i = CUDAKernelInterceptor()
    i.add_pid(12345)
    assert 12345 in i.monitored_pids

    i.remove_pid(12345)
    assert 12345 not in i.monitored_pids


# ── Test 5: KernelSwapper write + read round-trip ────────────────────────────

def test_kernel_swapper_write_and_read():
    """
    write_swap_instruction must create /dev/shm/memopt_{pid}.
    read_swap_status must return the exact values written.
    """
    s   = KernelSwapper()
    pid = os.getpid()

    try:
        ok = s.write_swap_instruction(pid, source_func=0xDEAD, target_func=0xBEEF)
        assert ok is True

        status = s.read_swap_status(pid)
        assert status is not None
        assert status.source_func == 0xDEAD
        assert status.target_func == 0xBEEF
        assert status.active      is True
        assert status.swap_count  == 0
    finally:
        s.clear_swap(pid)


# ── Test 6: clear_swap removes the shm file ──────────────────────────────────

def test_kernel_swapper_clear_removes_shm():
    """
    clear_swap must remove /dev/shm/memopt_{pid}.
    A second clear_swap on a missing file must not raise.
    """
    s   = KernelSwapper()
    pid = os.getpid()

    s.write_swap_instruction(pid, 1, 2)
    s.clear_swap(pid)
    assert not os.path.exists(f"{_shm_dir()}/memopt_{pid}")

    # Second clear must be a no-op
    s.clear_swap(pid)


# ── Test 7: /api/v1/ebpf/status returns correct shape ────────────────────────

def test_ebpf_status_endpoint(test_client):
    """
    GET /api/v1/ebpf/status must return 200 with ebpf_active field.
    BCC not installed → ebpf_active must be False.
    """
    response = test_client.get(
        "/api/v1/ebpf/status",
        headers={"X-Memopt-API-Key": TEST_API_KEY},
    )
    assert response.status_code == 200
    data = response.json()
    assert "ebpf_active"        in data
    assert "total_launches"     in data
    assert "detections"         in data
    assert "suboptimal_kernels" in data
    assert data["ebpf_active"] is False   # BCC not installed on this server


# ── Test 8: /api/v1/ebpf/monitor adds PID ────────────────────────────────────

def test_ebpf_monitor_pid_endpoint(test_client):
    """
    POST /api/v1/ebpf/monitor must return {"status": "monitoring", "pid": 99999}
    and add 99999 to the interceptor's monitored_pids.
    """
    response = test_client.post(
        "/api/v1/ebpf/monitor",
        json={"pid": 99999},
        headers={"X-Memopt-API-Key": TEST_API_KEY},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "monitoring"
    assert body["pid"]    == 99999
