"""DECISION 2 verification (orchestrator v1 Commit 9; design §2.1
DECISION 2, §3.1.10). Layer 2 placement input is ADVISORY: alloc()
keeps its shape and never blocks on a Layer-2 callback. The substrate
remains the sole owner of hard-fail (C1)."""
from __future__ import annotations

import time

import pytest

import memopt
import memopt.orchestrator as orch
from memopt.substrate.manager import AllocationManager


@pytest.fixture(autouse=True)
def _fresh_world():
    try:
        orch.stop()
    except Exception:
        pass
    AllocationManager.reset()
    yield
    try:
        orch.stop()
    except Exception:
        pass
    AllocationManager.reset()


def test_alloc_succeeds_when_orchestrator_returns_none():
    # No policy registered → orchestrator never opines → alloc works.
    orch.start()
    h = memopt.alloc(4096, placement="cpu")
    assert h is not None
    memopt.free(h)


def test_alloc_succeeds_when_orchestrator_returns_hint():
    # A user policy is free to "hint" via its own state; the substrate
    # alloc() shape is unchanged. Verify alloc still returns a handle.
    orch.start()

    class HintPolicy:
        def evaluate(self, snapshot):
            return []  # advisory only — no Decisions emitted
    orch.register_policy(HintPolicy())
    h = memopt.alloc(4096, placement="cpu", hint={"opaque": True})
    assert h is not None
    assert h.hint == {"opaque": True}
    memopt.free(h)


def test_alloc_does_not_block_on_orchestrator_callback():
    # An adversarial policy that sleeps on every evaluate must not be
    # in the alloc() critical path. alloc() must stay sub-second even
    # when the policy callback sleeps for 5s on its own thread.
    orch.start()

    class SlowPolicy:
        def evaluate(self, snapshot):
            time.sleep(5.0)
            return []
    orch.register_policy(SlowPolicy())
    t0 = time.monotonic()
    h = memopt.alloc(4096, placement="cpu")
    elapsed = time.monotonic() - t0
    memopt.free(h)
    assert elapsed < 1.0


def test_alloc_failure_still_raises_memory_error():
    # C1: hard-fail belongs to the substrate. Asking for an unsupported
    # placement must raise MemoryError regardless of orchestrator state.
    orch.start()
    with pytest.raises(MemoryError):
        memopt.alloc(4096, placement="cxl")


def test_orchestrator_callback_exception_does_not_break_alloc():
    orch.start()

    class RaisingPolicy:
        def evaluate(self, snapshot):
            raise RuntimeError("boom")
    orch.register_policy(RaisingPolicy())
    # alloc() must still succeed; the policy fault is contained.
    h = memopt.alloc(4096, placement="cpu")
    assert h is not None
    memopt.free(h)
