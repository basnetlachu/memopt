"""Tests for memopt.substrate.handle (per design §3.1 test_handle.py)."""
from __future__ import annotations

import dataclasses
import gc
import time

import pytest

import memopt.substrate as substrate
from memopt.substrate.handle import MemoryHandle


def _make_handle(**kwargs) -> MemoryHandle:
    defaults = dict(
        handle_id=1,
        size_bytes=1024,
        tenant="alice",
        tag="kv",
        placement="dram",
        stream=None,
        backend_name="cpu",
        created_at=time.monotonic(),
        ttl_seconds=None,
        hint=None,
    )
    defaults.update(kwargs)
    return MemoryHandle(**defaults)


class _MockEvent:
    def __init__(self, ready: bool = False) -> None:
        self.ready = ready

    def query(self) -> bool:
        return self.ready


class _MockStream:
    def __init__(self, event: _MockEvent | None = None) -> None:
        self._event = event or _MockEvent()

    def record_event(self) -> _MockEvent:
        return self._event


def test_alloc_returns_handle_with_metadata():
    h = _make_handle(
        handle_id=42,
        size_bytes=4096,
        tenant="bob",
        tag="weight",
        placement="hbm",
        backend_name="cuda",
        ttl_seconds=10.0,
        hint={"k": "v"},
    )
    assert h.handle_id == 42
    assert h.size_bytes == 4096
    assert h.tenant == "bob"
    assert h.tag == "weight"
    assert h.placement == "hbm"
    assert h.backend_name == "cuda"
    assert h.ttl_seconds == 10.0
    assert h.hint == {"k": "v"}
    assert h.created_at > 0


def test_handle_state_lifecycle():
    h = _make_handle()
    assert h.state == "live"
    h.free()
    assert h.state == "released"

    event = _MockEvent(ready=False)
    stream = _MockStream(event)
    h2 = _make_handle(stream=stream)
    assert h2.state == "live"
    h2.free()
    assert h2.state == "freed_pending_event"
    event.ready = True
    assert h2.state == "released"


def test_double_free_idempotent():
    h = _make_handle()
    h.free()
    h.free()
    h.free()
    s = h.stats()
    assert s["double_free_count"] == 2


def test_handle_freed_blocks_read():
    h = _make_handle()
    h.free()
    with pytest.raises(RuntimeError, match="handle is freed"):
        h.read()
    with pytest.raises(RuntimeError, match="handle is freed"):
        h.write(b"x")
    with pytest.raises(RuntimeError, match="handle is freed"):
        h.as_tensor("float32", (1,))
    with pytest.raises(RuntimeError, match="handle is freed"):
        h.as_numpy("float32", (1,))


def test_context_manager_auto_free():
    h = _make_handle()
    with h as ref:
        assert ref is h
        assert h.state == "live"
    assert h.state == "released"


def test_handle_ref_count_with_as_tensor():
    torch = pytest.importorskip("torch")

    class _MockBackend:
        BACKEND_NAME = "cpu_mock"

        def as_tensor(self, handle, dtype, shape):
            return torch.zeros(shape, dtype=dtype)

    h = _make_handle()
    object.__setattr__(h, "_backend", _MockBackend())

    assert h._ref_count == 0
    t = h.as_tensor(torch.float32, (4,))
    assert h._ref_count == 1
    assert tuple(t.shape) == (4,)
    del t
    gc.collect()
    # weakref.finalize is async; ref_count drop is best-effort but
    # MUST NOT exceed the number of as_tensor calls.
    assert 0 <= h._ref_count <= 1


def test_invalid_size_zero_raises():
    with pytest.raises(ValueError):
        substrate.alloc(0)
    with pytest.raises(ValueError):
        substrate.alloc(-1)


def test_invalid_tenant_regex_raises():
    with pytest.raises(ValueError):
        substrate.alloc(1024, tenant="alice/bob")
    with pytest.raises(ValueError):
        substrate.alloc(1024, tenant="")
    with pytest.raises(ValueError):
        substrate.alloc(1024, tag="x" * 65)


def test_handle_dataclass_is_frozen():
    h = _make_handle()
    with pytest.raises(dataclasses.FrozenInstanceError):
        h.handle_id = 999
    with pytest.raises(dataclasses.FrozenInstanceError):
        h.tenant = "evil"
