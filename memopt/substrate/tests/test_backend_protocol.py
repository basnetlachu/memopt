"""Tests for BackendStrategy ABC (per design §3.1 test_backend_protocol.py)."""
from __future__ import annotations

import typing
from typing import ClassVar, List, Optional

import pytest

from memopt.substrate.backends.base import (
    BackendStrategy,
    PhysHandle,
    PhysLoc,
    SEVEN_PRIMITIVES,
)


class _MinimalConcreteBackend(BackendStrategy):
    """Minimal concrete subclass that only implements is_available + the
    seven primitives + granularity + export_fabric_handle, so we can
    introspect that no abstract method was missed."""

    BACKEND_NAME: ClassVar[str] = "_minimal"
    IS_REAL: ClassVar[bool] = False

    def __init__(self) -> None:
        self.calls: list[str] = []

    def is_available(self) -> bool:
        self.calls.append("is_available")
        return True

    def granularity_bytes(self) -> int:
        return 4096

    def reserve_va(self, size: int) -> int:
        return 0

    def create_physical(self, size: int, location: PhysLoc) -> PhysHandle:
        return PhysHandle(backend_name=self.BACKEND_NAME, location=location, raw=None)

    def map(self, va: int, ph: PhysHandle, offset: int = 0) -> None:
        return None

    def set_access(self, va: int, size: int, devices: List[int]) -> None:
        return None

    def unmap(self, va: int, size: int) -> None:
        return None

    def release_physical(self, ph: PhysHandle) -> None:
        return None

    def free_va(self, va: int, size: int) -> None:
        return None

    def export_fabric_handle(self, ph: PhysHandle) -> Optional[bytes]:
        return None


def test_all_backends_implement_seven_primitives():
    """The ABC must declare seven primitives + is_available + granularity_bytes
    + export_fabric_handle as abstractmethods, and a concrete subclass must
    override every one."""
    abstract_methods = BackendStrategy.__abstractmethods__
    expected = set(SEVEN_PRIMITIVES) | {
        "is_available",
        "granularity_bytes",
        "export_fabric_handle",
    }
    assert expected.issubset(abstract_methods), (
        f"missing abstractmethods: {expected - abstract_methods}"
    )

    backend = _MinimalConcreteBackend()
    for name in expected:
        assert callable(getattr(backend, name)), f"missing override: {name}"


def test_class_vars_declared_on_abc():
    """BACKEND_NAME and IS_REAL are ClassVar annotations on the ABC."""
    hints = typing.get_type_hints(BackendStrategy, include_extras=False)
    # Annotations on the class body should include both.
    annotations = BackendStrategy.__annotations__
    assert "BACKEND_NAME" in annotations
    assert "IS_REAL" in annotations


def test_cannot_instantiate_abc_directly():
    with pytest.raises(TypeError):
        BackendStrategy()  # type: ignore[abstract]


def test_is_available_is_pure(monkeypatch):
    """Calling is_available() many times must not initialise CUDA.

    We probe by checking torch.cuda.is_initialized() before and after if
    torch is importable; otherwise we just confirm the call is side-effect-
    free against the minimal backend.
    """
    backend = _MinimalConcreteBackend()

    try:
        import torch
        had_torch = True
        before = torch.cuda.is_initialized() if torch.cuda.is_available() else False
    except ImportError:
        had_torch = False
        before = False

    for _ in range(100):
        backend.is_available()

    if had_torch:
        after = torch.cuda.is_initialized() if torch.cuda.is_available() else False
        # The call must not have initialised CUDA when it wasn't before.
        if not before:
            assert not after, "is_available() initialised CUDA — must be pure"

    # The minimal backend recorded each call but had no other side effects.
    assert backend.calls == ["is_available"] * 100


def test_phys_handle_dataclass_shape():
    ph = PhysHandle(backend_name="x", location=PhysLoc.HBM, raw=42)
    assert ph.backend_name == "x"
    assert ph.location is PhysLoc.HBM
    assert ph.raw == 42


def test_phys_loc_enum_members():
    assert PhysLoc.HBM.value == "hbm"
    assert PhysLoc.DRAM.value == "dram"
    assert PhysLoc.CXL.value == "cxl"
    assert PhysLoc.NVME.value == "nvme"
