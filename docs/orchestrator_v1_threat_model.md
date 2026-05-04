# memopt Orchestrator v1 — Threat Model

| Field | Value |
| --- | --- |
| Status | v1.0 (Phase A) |
| Authoritative | `docs/orchestrator_v1_design.md` §2.5 |
| Builds on | `docs/substrate_v1_threat_model.md` G1–G4 (Layer 1) |

This document lists what the orchestrator (Layer 2) guarantees against
multi-tenant misuse and what it does not. It mirrors the substrate's
threat model shape and is derived from design §2.5.

The orchestrator inherits every Layer 1 guarantee. The list below
covers the additions Layer 2 introduces.

## Guarantees

### G1 — Per-tenant access histories isolated

Two tenants cannot read each other's access histories via the
orchestrator API. `AccessTracker.last_seen`, `lru_candidates`, and
`snapshot()` are scoped by `tenant`. Cross-tenant reads return empty
results; under a `memopt.context(tenant=...)`, `peek_handle` raises
`PermissionError("peek across tenants forbidden (G1)")`.

> Mitigation: `memopt/orchestrator/access.py:115-200` keeps state under
> per-tenant `threading.Lock`. `memopt/substrate/manager.py:393-396`
> enforces the cross-tenant check on `peek_handle`.

### G2 — Per-tenant policies enforced

`LRUWatermarkPolicy` computes per-tenant pressure independently and
cannot promote / evict across tenant boundaries. The
`PolicySnapshot.lru_candidates` dict is keyed by `(tenant, placement)`.
A buggy or adversarial policy that returns decisions targeting another
tenant's `handle_id` is filtered: each decision goes through
`peek_handle(handle_id)` which performs the G1 check before emit.

> Mitigation: `memopt/orchestrator/policy.py:130-170` (LRU is
> per-tenant); `memopt/orchestrator/coordinator.py:213-247`
> (peek_handle gate before emit).

### G3 — Predictor state partitioned by tenant

Every predictor key is the tuple `(tenant, tag)` (DECISION 6).
`Predictor.forget_tenant(tenant)` removes all transitions whose source
key has that tenant; used on tenant teardown. Tenants observing the
same tag values produce disjoint transitions.

> Mitigation: `memopt/orchestrator/predict.py:43-90` (key shape);
> `memopt/orchestrator/predict.py:144-160` (`forget_tenant`).

### G4 — Telemetry aggregates only without admin token

`memopt.orchestrator.stats()` returns per-kind aggregate counts. The
`decisions_per_tenant` block is exposed only when `MEMOPT_ADMIN_TOKEN`
is configured (constant-time `hmac.compare_digest`, parity with
substrate G2).

> Mitigation: `memopt/orchestrator/telemetry.py:88-110`.

## Non-guarantees (out of scope, made explicit)

### N1 — Cross-tenant pattern leakage via shared storage

If a future Layer 5 oracle aggregates transitions across tenants for
federated learning, that aggregation can leak access patterns across
trust boundaries. v1.0 does NOT do this; per-process predictor state
is the trust boundary.

> Why out of scope: federated orchestration is Layer 5 (separate layer
> per design §3.7). When that layer ships, it must re-run this threat
> model.
>
> Mitigation outside this layer: deploy in process-namespaces or
> per-tenant containers if cross-tenant inference is a concern.

### N2 — Side-channels through cache pressure

A tenant inferring another tenant's access pattern by observing its
own latency under shared HBM is a known timing side-channel. v1.0
does NOT attempt to constant-time eviction policy; Phase A is
observation-only and defers actual eviction to v1.1.

> Why out of scope: NVIDIA Confidential Compute (CC) mode and MIG
> partition resources at the hardware boundary; that is the
> appropriate layer for side-channel mitigation.
>
> Mitigation outside this layer: enable NVIDIA CC mode or MIG; treat
> tenants on the same GPU as a single trust domain otherwise.

### N3 — `register_policy` is unrestricted

Any caller can register a `Policy` in v1.0; production deployments
should gate this behind an admin role. v1.0 ships the protocol
without authentication.

> Why out of scope: Layer 4 (declarative policy DSL) introduces the
> policy registry as its own service with auth.
>
> Mitigation outside this layer: wrap `memopt.orchestrator.register_policy`
> in your service layer with your existing AuthN/AuthZ.

### N4 — Queue starvation under adversarial workloads

A producer that emits events at >> 16 384 events/second can saturate
the coordinator's bounded queue (`event_queue_capacity` default).
Under saturation, the oldest event is dropped and `queue_drops`
increments. The orchestrator silently degrades; it does NOT throttle
the producer.

> Why out of scope: rate-limiting per tenant belongs in the substrate
> alloc path, not the observation pipeline.
>
> Mitigation outside this layer: enforce per-tenant alloc rate limits
> upstream; monitor `stats()["coordinator"]["queue_drops"]` for early
> warning.

## Public API privilege expectations

| API | Privilege expected | Notes |
| --- | --- | --- |
| `memopt.orchestrator.start` | None (process-local) | Idempotent; same process trust boundary as substrate. |
| `memopt.orchestrator.stop` | None | Idempotent. |
| `memopt.orchestrator.stats` | None for aggregates; `MEMOPT_ADMIN_TOKEN` for `decisions_per_tenant` | G4. |
| `memopt.orchestrator.register_policy` | Unrestricted (N3) | Production should gate. |
| `memopt.peek_handle` | Tenant-scoped via `memopt.context` | G1 check at the substrate layer. |
