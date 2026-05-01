# Changelog

## 1.2.0 — Orchestrator v1 (Phase A)

Orchestrator v1 release — Phase A complete (per `docs/orchestrator_v1_design.md`).

  - **AccessTracker** (`memopt.orchestrator.access`) — tenant-aware
    LRU tracker driven by substrate event records. Per-tenant
    intrusive doubly-linked LRU per `(tenant, placement)` so candidate
    walks are O(count) not O(total).
  - **Predictor** (`memopt.orchestrator.predict`) — greenfield
    tenant-keyed first-order Markov chain with sequential / recency /
    fallback sources. Per DECISION 6, every key tuple is `(tenant, tag)`.
  - **PolicyEngine + LRUWatermarkPolicy** (`memopt.orchestrator.policy`)
    — fan-out evaluator with priority-based conflict resolution and
    auto-unregister of buggy policies after 3 consecutive raises.
  - **Coordinator** (`memopt.orchestrator.coordinator`) — single
    daemon thread `memopt-orchestrator-coordinator`; subscribes to all
    five `Event.kind` streams; drains queue + drives policy cycle. Per
    DECISION 7 v1.0 ships in observation-only mode (no decisions
    applied by default).
  - **TelemetryCollector** (`memopt.orchestrator.telemetry`) —
    counters live in their own namespace; `decisions_per_tenant` is
    gated by `MEMOPT_ADMIN_TOKEN` per G4.
  - **Public API** (`memopt.orchestrator.start / stop / stats /
    register_policy`) plus `memopt.peek_handle` re-exported from
    `memopt.substrate.AllocationManager`.
  - **Bridge test** (`tests/test_orchestrator_legacy_parity.py`) —
    Phase A assertions: per-tenant byte counts and event-drop counters
    are byte-for-byte identical between the orchestrator-stopped run
    and the orchestrator-on-but-observation-only run; `decisions == 0`.
  - **Threat model** (`docs/orchestrator_v1_threat_model.md`) and
    **user guide** (`docs/orchestrator_v1_user_guide.md`) shipped per
    §2.5 / §2.2.
  - Mac baseline: 752 → 876 PASSED (124 new orchestrator REQUIRED-CI
    tests); 22 → 34 SKIPPED (10 new @perf microbenches + 2 @gpu
    tests); 15 DESELECTED unchanged; 0 FAILED.
  - Substrate change scope (Commit 1 only): `peek_handle` and
    `_emit_orchestrator_event` added to `AllocationManager`. No
    existing substrate method body changed.

### What Phase A does NOT deliver

Verbatim from `docs/orchestrator_v1_design.md` §3.7:

  - **Phase B** (`MEMOPT_USE_ORCHESTRATOR=1` flag flip): advisory
    `placement="auto"` resolution + C++ step_boundary deferral.
    Separate engagement; gating in §2.7 Phase B.
  - **Phase C** (default-ON): three months of Phase B soak required.
    Separate engagement.
  - **Phase D** (remove legacy paths): deletes
    `TierManager._ensure_capacity`, `ElasticAllocator`,
    `MemoryGovernor`, the C++ direct-eviction path, and the
    VMM-internal Markov chain in `PrefetchEngine`. Separate
    engagement.
  - **Layer 3** (ML-driven prefetch oracle): separate package
    `memopt-prefetch-oracle`; v1.0 stays classical Markov.
  - **Layer 4** (declarative policy DSL): separate package
    `memopt-policy-dsl`; v1.0 ships only the `Policy` protocol.
  - **Layer 5** (federated orchestration across nodes): separate
    layer; the §2.3.2 predictor explicitly does not gossip across
    processes (DECISION 7 trade-off).
  - **vLLM integration.** `memopt-vllm` (separate package).
  - **Custom CUDA kernels for the predictor.** v1.0 is pure Python
    (DECISION 6 trade-off).
  - **Persistence of learned patterns across restarts.** Predictor
    starts cold every process.
  - **Cost / FinOps integration.** Lives in `memopt-trust`.
  - **Confidential computing / attestation.** Out of scope.

## 1.1.0 — Substrate v1 (Phase A)

  - Substrate v1 (Layer 1) — Memory management substrate added under
    `memopt.alloc / free / context / stats / observe`. See
    `docs/substrate_v1_design.md` for the design and
    `docs/substrate_v1_user_guide.md` for usage examples.
  - G12 fix: 16 GiB cap removed from `libmemopt_vmm.so`. The page table
    is now heap-allocated and sized at `memopt_allocator_create` from
    pool_size_bytes / 2 MiB plus 25% headroom. Verified end-to-end on
    A100-80GB by allocating 16385 × 2 MiB pages (above the legacy
    8192-page cap).
  - ROCm/HIP backend implemented as honest stub per Step Zero S0.2
    DEGRADED (no AMD hardware on the build rig). When a verified AMD
    rig becomes available, S0.2 must be re-run and the stub replaced
    with the real implementation per design §2.4 HIPBackend.
  - CUDA VMM backend wraps the existing `libmemopt_vmm.so` and exposes
    the seven primitives plus `granularity_bytes` and
    `export_fabric_handle`. Fabric handle support is DEGRADED on this
    rig per S0.1; returns None until re-verified.
  - Level Zero backend ships as honest stub per design §2.4
    LevelZeroBackend; real implementation deferred to v1.1.
  - CXL/NUMA backend gated on `MEMOPT_CXL_NODES` per S0.5 DEGRADED.
  - StreamRegistry implemented with PyTorch CCA semantics (DECISION 2).
    Three documented divergences (E2 no auto peer-access, E3 no IPC,
    E4 single-device-per-call) tested in `test_cca_divergences.py`
    with `DESIGN_DATE = "2026-04-29"` pinned.
  - EventRing uses `threading.Lock` per S0.6 fallback. Performance
    target relaxed to <1 µs (vs the <200 ns aspiration); ordering
    invariant `event_emit < alloc_warm` still enforced.
  - Bridge test added (`tests/test_substrate_legacy_parity.py` and
    `tests/test_substrate_legacy_parity_gpu.py`) — Phase B trust
    anchor. Asserts byte-equivalence per design §3.3 OPTION Y (no
    order assertion).
  - Threat model documented in `docs/substrate_v1_threat_model.md`
    with verified CVE references (S0.7 VERIFIED, 2026-04-29).
  - Existing public APIs unchanged. Pre-substrate baseline tests still
    passing.
