"""Bridge test (GPU variant) — substrate / legacy byte-equivalence trust
anchor for Phase B (per design §2.9 + §3.3).

# Bridge test assertions follow OPTION Y from
# docs/substrate_v1_design.md §3.3 (decided 2026-04-29).
# Order of allocations is NOT asserted because no existing
# test in the regression net depends on it (audit recorded
# in §3.3 of the design doc). A future change that makes
# alloc order observable to consumers should re-run that
# audit before relaxing this test's strictness.

Same shape as test_substrate_legacy_parity.py but with placement="hbm",
exercising the CUDA backend on a CUDA-equipped rig.
"""
from __future__ import annotations

import json
import os

import pytest

import memopt
from memopt.substrate.manager import AllocationManager


@pytest.fixture(autouse=True)
def _fresh_manager():
    AllocationManager.reset()
    yield
    AllocationManager.reset()


@pytest.mark.gpu
def test_substrate_bridge_cuda():
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Smaller workload for GPU run to keep wall time reasonable.
    sizes = [(2 * 1024 * 1024, 30), (8 * 1024 * 1024, 10)]
    tenants = ["alice"] * 14 + ["bob"] * 14 + ["carol"] * 12  # 40 total
    plan = []
    for size, count in sizes:
        plan.extend([size] * count)
    plan = plan[: len(tenants)]

    handles = []
    for i, (size, tenant) in enumerate(zip(plan, tenants)):
        with memopt.context(tenant=tenant, placement="hbm"):
            h = memopt.alloc(size)
        handles.append((tenant, h))

    # Free all
    for tenant, h in handles:
        with memopt.context(tenant=tenant):
            memopt.free(h)

    os.environ["MEMOPT_ADMIN_TOKEN"] = "_test_token"
    try:
        snap = memopt.stats(tenant=None)
    finally:
        os.environ.pop("MEMOPT_ADMIN_TOKEN", None)

    assert snap["events_dropped"] == 0
    for t in ("alice", "bob", "carol"):
        ts = snap["tenants"].get(t, {})
        assert ts.get("in_use_bytes", 0) == 0, f"{t} leaked bytes: {ts}"

    out_dir = "/tmp/memopt-bridge"
    os.makedirs(out_dir, exist_ok=True)
    sha = os.environ.get("GIT_COMMIT_SHA", "untracked")
    with open(os.path.join(out_dir, f"gpu_{sha}.json"), "w") as f:
        json.dump(snap, f, indent=2, default=str)
