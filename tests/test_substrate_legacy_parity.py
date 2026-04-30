"""Bridge test (CPU variant) — substrate / legacy byte-equivalence trust
anchor for Phase B (per design §2.9 + §3.3).

# Bridge test assertions follow OPTION Y from
# docs/substrate_v1_design.md §3.3 (decided 2026-04-29).
# Order of allocations is NOT asserted because no existing
# test in the regression net depends on it (audit recorded
# in §3.3 of the design doc). A future change that makes
# alloc order observable to consumers should re-run that
# audit before relaxing this test's strictness.

Workload (per §2.9):
  - 3 tenants: alice (67 allocs), bob (67), carol (66) = 200 total
  - Size mix per tenant: 50% 64 KiB, 30% 2 MiB, 15% 8 MiB, 5% 64 MiB
  - Pool: 256 MiB
  - Half the allocations are stream-bound (alternating between A and B)
  - Lifecycle: alloc 200 → access half → free 60 → realloc 60 → free all

Phase A (this commit): runs through memopt.substrate only and records the
workload's metrics. The legacy comparison fires in Phase B once
MEMOPT_VMM_USE_SUBSTRATE flips the VMM adapter to route through substrate.
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter

import pytest

import memopt
import memopt.substrate
from memopt.substrate.manager import AllocationManager


_SIZES = [
    (64 * 1024, 0.50),       # 64 KiB
    (2 * 1024 * 1024, 0.30), # 2 MiB
    (8 * 1024 * 1024, 0.15), # 8 MiB
    (64 * 1024 * 1024, 0.05),# 64 MiB
]


def _build_size_plan(n: int) -> list[int]:
    """Deterministic size plan from the §2.9 size mix."""
    plan: list[int] = []
    cum = 0.0
    counts = []
    for size, frac in _SIZES:
        c = int(round(n * frac))
        counts.append((size, c))
        cum += frac
    total = sum(c for _, c in counts)
    if total < n:
        counts[0] = (counts[0][0], counts[0][1] + (n - total))
    elif total > n:
        counts[0] = (counts[0][0], counts[0][1] - (total - n))
    for size, c in counts:
        plan.extend([size] * c)
    assert len(plan) == n, (len(plan), n)
    return plan


class _MockStream:
    """Stand-in stream object for CPU-variant bridge test."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __hash__(self) -> int:
        return hash(("mock_stream", self.name))

    def __eq__(self, other) -> bool:
        return isinstance(other, _MockStream) and other.name == self.name

    def record_event(self):
        # Returns a fake event whose query() returns True immediately, so
        # the manager's free path drains synchronously after the call.
        from memopt.substrate.manager import _ImmediateEvent
        ev = _ImmediateEvent()
        ev.record(stream=self)
        return ev


@pytest.fixture(autouse=True)
def _fresh_manager():
    AllocationManager.reset()
    yield
    AllocationManager.reset()


def test_substrate_bridge_cpu():
    """Run the §2.9 workload through substrate (CPU placement). Assert
    internal-consistency invariants per §3.3 OPTION Y; record metrics for
    Phase B comparison."""
    streamA = _MockStream("A")
    streamB = _MockStream("B")

    plan = _build_size_plan(200)
    tenants = ["alice"] * 67 + ["bob"] * 67 + ["carol"] * 66
    handles_per_tenant: dict[str, list] = {"alice": [], "bob": [], "carol": []}
    drops_observer = Counter()

    sub = memopt.observe("alloc", lambda e: drops_observer.update({"alloc": 1}))
    free_obs = memopt.observe("free", lambda e: drops_observer.update({"free": 1}))
    try:
        # 1. Alloc 200 — half stream-bound
        for i, (size, tenant) in enumerate(zip(plan, tenants)):
            stream = streamA if (i % 2 == 0) else streamB if (i % 2 == 1 and i < 100) else None
            with memopt.context(tenant=tenant, placement="cpu"):
                h = memopt.alloc(size, stream=stream)
            handles_per_tenant[tenant].append(h)

        snap_after_alloc = _snapshot()

        # 2. Access half — touch read/write
        flat = [(t, h) for t, lst in handles_per_tenant.items() for h in lst]
        for t, h in flat[: len(flat) // 2]:
            with memopt.context(tenant=t):
                h.write(b"\x01" * min(64, h.size_bytes))

        # 3. Free 60 (force the freelist to grow)
        freed = []
        for t, h in flat[:60]:
            with memopt.context(tenant=t):
                memopt.free(h)
            handles_per_tenant[t].remove(h)
            freed.append((t, h.size_bytes))

        snap_after_free60 = _snapshot()

        # 4. Realloc 60 — should hit freelist
        new_handles = []
        for t, sz in freed:
            with memopt.context(tenant=t, placement="cpu"):
                new_handles.append((t, memopt.alloc(sz)))
                handles_per_tenant[t].append(new_handles[-1][1])

        snap_after_realloc60 = _snapshot()

        # 5. Free everything
        for t, lst in handles_per_tenant.items():
            for h in lst:
                with memopt.context(tenant=t):
                    memopt.free(h)
            lst.clear()

        # Drain stream-locked pending so in_use_bytes reflects post-free reality.
        AllocationManager.get().drain_pending()

        snap_final = _snapshot()
    finally:
        sub.unsubscribe()
        free_obs.unsubscribe()

    # ===== Per-§3.3 OPTION Y assertions =====

    # (7) events_dropped == 0
    assert snap_final["events_dropped"] == 0, (
        f"events_dropped grew: {snap_final['events_dropped']}"
    )

    # G1 (per §3.3 #6): cross-tenant access must raise PermissionError.
    # (Verified by handle.read in another tenant's context, but at this
    # point all handles are freed; we exercise the check on a fresh handle.)
    with memopt.context(tenant="alice", placement="cpu"):
        h_alice = memopt.alloc(1024)
    try:
        with memopt.context(tenant="bob"):
            with pytest.raises(PermissionError):
                h_alice.read()
    finally:
        with memopt.context(tenant="alice"):
            memopt.free(h_alice)

    # Snapshots monotonic-ish: per tenant in_use_bytes drops to 0 at end.
    for t in ("alice", "bob", "carol"):
        ts = snap_final["tenants"].get(t, {})
        assert ts.get("in_use_bytes", 0) == 0, f"{t} leaked bytes: {ts}"

    # Persist metrics for Phase B comparison
    out_dir = "/tmp/memopt-bridge"
    os.makedirs(out_dir, exist_ok=True)
    sha = os.environ.get("GIT_COMMIT_SHA", "untracked")
    with open(os.path.join(out_dir, f"cpu_{sha}.json"), "w") as f:
        json.dump({
            "after_alloc": snap_after_alloc,
            "after_free60": snap_after_free60,
            "after_realloc60": snap_after_realloc60,
            "final": snap_final,
        }, f, indent=2, default=str)


def _snapshot() -> dict:
    """Aggregate stats with admin token bypass for tests."""
    os.environ["MEMOPT_ADMIN_TOKEN"] = "_test_token"
    try:
        return memopt.stats(tenant=None)
    finally:
        os.environ.pop("MEMOPT_ADMIN_TOKEN", None)
