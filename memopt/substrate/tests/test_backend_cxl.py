"""Tests for CXLBackend (per design §3.1; v1.0 ships env-var gated per
Step Zero S0.5 DEGRADED). Real CXL tests are marked @cxl and only run
when MEMOPT_CXL_PRESENT=1 is set on the host."""
from __future__ import annotations

import os

import pytest

from memopt.substrate.backends.cxl_numa import CXLBackend, _parse_node_list


def test_cxl_backend_import_smoke():
    """Construction must succeed even without CXL hardware."""
    backend = CXLBackend()
    assert backend.BACKEND_NAME == "cxl"
    assert backend.IS_REAL is True


def test_cxl_unavailable_when_env_var_missing(monkeypatch):
    monkeypatch.delenv("MEMOPT_CXL_NODES", raising=False)
    backend = CXLBackend()
    assert backend.is_available() is False


def test_cxl_unavailable_explicit_placement_raises(monkeypatch):
    """create_physical raises when the env var is unset (per §2.2 explicit
    placement rule + §3.1)."""
    monkeypatch.delenv("MEMOPT_CXL_NODES", raising=False)
    backend = CXLBackend()
    from memopt.substrate.backends.base import PhysLoc
    with pytest.raises(MemoryError):
        backend.create_physical(4096, PhysLoc.CXL)


def test_cxl_node_list_parser():
    assert _parse_node_list("") == ()
    assert _parse_node_list("0") == (0,)
    assert _parse_node_list("1,2") == (1, 2)
    assert _parse_node_list(" 1 , 2 ") == (1, 2)
    assert _parse_node_list("1,bad,3") == (1, 3)


@pytest.mark.cxl
def test_cxl_alloc_via_libnuma():
    """OPTIONAL: only runs on @cxl rig with MEMOPT_CXL_NODES set."""
    if not os.environ.get("MEMOPT_CXL_NODES"):
        pytest.skip("MEMOPT_CXL_NODES not set — CXL hardware not exposed")
    from memopt.substrate.backends.base import PhysLoc
    backend = CXLBackend()
    ph = backend.create_physical(4 * 1024 * 1024, PhysLoc.CXL)
    try:
        assert ph.raw > 0
    finally:
        backend.release_physical(ph)
