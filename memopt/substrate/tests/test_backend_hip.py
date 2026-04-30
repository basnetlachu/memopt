"""Tests for HIPBackend stub (per design §3.1; mirrors test_backend_l0_stub.py
because Step Zero S0.2 is DEGRADED on this rig — no AMD hardware)."""
from __future__ import annotations

import pytest

from memopt.substrate.backends.base import PhysHandle, PhysLoc
from memopt.substrate.backends.rocm_hip import HIPBackend


def test_hip_stub_is_available_is_false():
    """Per S0.2 DEGRADED, the stub returns False."""
    assert HIPBackend().is_available() is False


def test_hip_stub_calls_raise_clear_message():
    """Every method except is_available raises NotImplementedError citing
    S0.2 DEGRADED and the design+report doc paths."""
    backend = HIPBackend()
    ph = PhysHandle(backend_name="hip", location=PhysLoc.HBM, raw=0)
    targets = [
        ("granularity_bytes", []),
        ("reserve_va", [4096]),
        ("create_physical", [4096, PhysLoc.HBM]),
        ("map", [0, ph]),
        ("set_access", [0, 4096, [0]]),
        ("unmap", [0, 4096]),
        ("release_physical", [ph]),
        ("free_va", [0, 4096]),
        ("export_fabric_handle", [ph]),
    ]
    for name, args in targets:
        with pytest.raises(NotImplementedError) as exc_info:
            getattr(backend, name)(*args)
        msg = str(exc_info.value)
        assert "S0.2" in msg, f"{name}: missing S0.2 reference"
        assert "substrate_v1_design.md" in msg


def test_hip_stub_class_marks_is_real_false():
    """Class-level IS_REAL flag tells the manager this is an honest stub."""
    assert HIPBackend.IS_REAL is False
    assert HIPBackend.BACKEND_NAME == "hip"
