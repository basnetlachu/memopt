# memopt Orchestrator v1 — User Guide

The orchestrator is memopt's Layer 2 — a tenant-aware observation /
decision layer that sits on top of the substrate (Layer 1). In v1.0
(Phase A) it observes the substrate's event stream and exposes a
public Policy protocol for user extensions. It does NOT drive
eviction in Phase A; that is Phase B (`MEMOPT_USE_ORCHESTRATOR=1`,
v1.1).

| Function | Purpose |
| --- | --- |
| `memopt.orchestrator.start(*, config=None)` | Start the coordinator (idempotent) |
| `memopt.orchestrator.stop()` | Stop the coordinator (idempotent) |
| `memopt.orchestrator.stats()` | Snapshot the orchestrator's stats |
| `memopt.orchestrator.register_policy(policy)` | Register a custom Policy |
| `memopt.peek_handle(handle_id, *, tenant=None)` | Look up a live handle by id |

> Authoritative spec: `docs/orchestrator_v1_design.md` §2.
> Threat model: `docs/orchestrator_v1_threat_model.md`.

## Quickstart

```python
import memopt

# Allocate the substrate as usual.
with memopt.context(tenant="alice", placement="cpu"):
    h = memopt.alloc(1 << 20)        # 1 MiB

# Start the orchestrator. Observation only in v1.0.
memopt.orchestrator.start()

# ... do work; the orchestrator silently records access events ...

# Snapshot stats.
s = memopt.orchestrator.stats()
print(s["events_ingested"]["alloc"], "alloc events recorded")

# Free + stop.
memopt.free(h)
memopt.orchestrator.stop()
```

## Worked example: register a custom Policy

```python
import memopt
from memopt.orchestrator.policy import Decision

class MyEvictAllPolicy:
    """Demo policy: evict every handle it sees in the LRU candidate list."""

    def evaluate(self, snapshot):
        out = []
        for (tenant, placement), handle_ids in snapshot.lru_candidates.items():
            if placement != "hbm":
                continue
            for hid in handle_ids:
                out.append(Decision(
                    kind="evict",
                    handle_id=hid,
                    target_placement="dram",
                    reason="demo",
                    priority=1,
                ))
        return out

memopt.orchestrator.start()
memopt.orchestrator.register_policy(MyEvictAllPolicy())

# Drive some allocations:
with memopt.context(tenant="alice", placement="cpu"):
    handles = [memopt.alloc(1 << 18) for _ in range(8)]

# Observe the orchestrator's view:
print(memopt.orchestrator.stats())

for h in handles:
    memopt.free(h)
memopt.orchestrator.stop()
```

In Phase A the policy's decisions are evaluated but NOT applied
(DECISION 7 — observation-only). They become applied in Phase B
behind `MEMOPT_USE_ORCHESTRATOR=1`.

## Environment variables

| Var | Default | Purpose |
| --- | --- | --- |
| `MEMOPT_ORCH_CYCLE_MS` | 50 | Coordinator cycle period (ms). |
| `MEMOPT_ORCH_QUEUE_CAP` | 16384 | Event queue capacity. |
| `MEMOPT_ORCH_MAX_TRANSITIONS` | 100000 | Predictor row cap. |
| `MEMOPT_EVICT_HIGH` | 0.90 | Watermark high (shared with `TierManager`). |
| `MEMOPT_EVICT_LOW` | 0.75 | Watermark low (shared with `TierManager`). |
| `MEMOPT_ADMIN_TOKEN` | (unset) | Required to read `decisions_per_tenant` (G4). |
| `MEMOPT_AUTO_START_ORCHESTRATOR` | (unset) | **Phase-A test knob only.** Auto-starts the orchestrator at process import for regression runs. Removed in Phase B. |

`MEMOPT_USE_ORCHESTRATOR=1` is the public Phase B / Phase C flag —
**not yet honored in v1.0**. Setting it has no effect in Phase A.

## Phase A vs. Phase B boundary

| Behavior | Phase A (v1.0) | Phase B (v1.1) |
| --- | --- | --- |
| Coordinator starts on `start()` | Yes | Yes |
| Subscribes to substrate events | Yes | Yes |
| Decisions evaluated by policies | Yes | Yes |
| Decisions APPLIED (evict / promote / migrate) | **No** | Yes (behind `MEMOPT_USE_ORCHESTRATOR=1`) |
| `placement="auto"` resolved by Layer 2 | No | Yes (advisory; substrate retains hard-fail) |
| C++ `step_boundary` defers to Python callback | No | Yes |

The Phase A guarantee is byte-for-byte parity: starting and stopping
the orchestrator on a workload must produce identical per-tenant
in-use byte counts to a run that never started it
(`tests/test_orchestrator_legacy_parity.py`).

## Troubleshooting

### `queue_drops > 0`

Producer outpacing the coordinator. Check `stats()["coordinator"]["queue_drops"]`.
Either lower producer rate or raise `MEMOPT_ORCH_QUEUE_CAP`.

### Custom Policy stops being called

A buggy policy is auto-unregistered after 3 consecutive raises. Check
the WARNING log (`memopt.orchestrator.policy`) for the unregister
message.

### `peek_handle` returns `None`

Either (a) the handle was freed; (b) the id is unknown; or (c) the
caller's `memopt.context(tenant=...)` does not match the handle's
tenant — in case (c) `peek_handle` raises `PermissionError` rather
than returning `None`, so check for the exception too.

### Orchestrator does not appear to be running

`memopt.orchestrator.stats()` returns `{}` until `start()` is called.
After `start()`, `stats()["running"]` is `True` and the coordinator
thread `memopt-orchestrator-coordinator` is in `threading.enumerate()`.
