"""Tests for LevelZeroBackend stub (per design §3.1 test_backend_l0_stub.py)."""
from __future__ import annotations

from unittest import mock

import pytest

from memopt.substrate.backends.base import PhysHandle, PhysLoc
from memopt.substrate.backends import level_zero
from memopt.substrate.backends.level_zero import LevelZeroBackend


def test_l0_is_available_only_when_libze_loader_present():
    """is_available() returns True iff libze_loader.so is loadable AND
    zeInit(0) succeeds. Mock both branches."""
    backend = LevelZeroBackend()

    with mock.patch.object(level_zero, "_try_load_libze", return_value=None):
        assert backend.is_available() is False

    fake_lib = mock.MagicMock()
    with mock.patch.object(level_zero, "_try_load_libze", return_value=fake_lib), \
         mock.patch.object(level_zero, "_ze_init_succeeds", return_value=True):
        assert backend.is_available() is True

    with mock.patch.object(level_zero, "_try_load_libze", return_value=fake_lib), \
         mock.patch.object(level_zero, "_ze_init_succeeds", return_value=False):
        assert backend.is_available() is False


def test_l0_calls_raise_clear_message():
    """Every method except is_available raises NotImplementedError with a
    message that mentions v1.0 and the design doc path."""
    backend = LevelZeroBackend()
    ph = PhysHandle(backend_name="level_zero", location=PhysLoc.HBM, raw=0)

    targets = [
        ("granularity_bytes", []),
        ("reserve_va", [4096]),
        ("create_physical", [4096, PhysLoc.DRAM]),
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
        assert "v1.0" in msg, f"{name}: missing v1.0 in message: {msg}"
        assert "substrate_v1_design.md" in msg, f"{name}: missing design path"


def test_l0_does_not_register_with_manager():
    """Even when libze_loader is present and zeInit succeeds, the seven
    primitives raise NotImplementedError. A manager that defensively
    probes a primitive (e.g. granularity_bytes) before adopting the
    backend will treat LZ as unusable. We verify that contract here."""
    backend = LevelZeroBackend()
    fake_lib = mock.MagicMock()
    with mock.patch.object(level_zero, "_try_load_libze", return_value=fake_lib), \
         mock.patch.object(level_zero, "_ze_init_succeeds", return_value=True):
        assert backend.is_available() is True
        with pytest.raises(NotImplementedError):
            backend.granularity_bytes()
