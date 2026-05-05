# Changelog

## 1.3.0a2 — Metadata + doc cleanup (ALPHA)

Patch on top of 1.3.0a1. No code change.

  - Maintainer email corrected to `lachu.basnet@sophisticatesai.com`
    (the 1.3.0a1 wheel carried the wrong email; PyPI does not allow
    re-uploading the same version, so this 1.3.0a2 republish carries
    the corrected metadata).
  - Removed `docs/oracle_v1_design.md` (Layer 3 Phase 1 internal
    discovery doc; Layer 3 itself is not in this release).
  - Softened the security/conduct boilerplate language.

Mac baseline unchanged: 1016 PASSED, 34 SKIPPED, 18 DESELECTED, 0 FAILED.

## 1.3.0a1 — Pillar revival + Apache-2.0 release prep (ALPHA)

**Alpha release.** Library is OSS-licensed and Mac baseline is green
(1016 PASSED, 0 FAILED), but GPU rig validation has not yet been
performed for this release. Do NOT run in production until item 1 in
README "Status — production readiness" is checked. Tag `v1.3.0a1`.



Reunifies the
`memopt-trust-archive` pillars (4, 6, 7) with the engine repo, ships
the public Apache-2.0 license, and wires the revived pillars to the
Layer 1 substrate + Layer 2 orchestrator through a thin adapter
package. Layer 1 substrate and Layer 2 orchestrator source unchanged.

  - **Pillar 4 — AI Compliance ledger** (`memopt/observability/`):
    `OptimizationLedger` + HMAC-signed entries + `CarbonCalculator`,
    plus `SavingsReport` / `ComplianceReport` exports
    (HTML/CSV/JSON/PDF). 4 REST endpoints in
    `memopt/api/server.py` rewired (no longer 501 stubs).
  - **Pillar 6 — Silicon certification**
    (`memopt/kernels/{certification,certify_daemon,drift_detector}.py`):
    correctness + throughput battery + drift detector + standalone
    `memopt certify` CLI subcommand + `memopt-certify` script entry
    point.
  - **Pillar 7 — GPU FinOps**
    (`memopt/finops/tracker.py`, `memopt/trust/receipt.py`):
    per-tenant utilization → dollar tracker + production trust
    receipt builder.
  - **Layer 1/2 wiring** (`memopt/integrations/`):
    `attach_ledger_to_substrate` (substrate events → ledger),
    `FinOpsPoller` (substrate stats → finops on a daemon thread),
    `assemble_production_receipt` (`ReceiptBuilder.build_for_request`
    wrapped to accept any subset of pillars). Pure consumers of the
    public Layer 1/2 API.
  - **Open-source readiness**: Apache-2.0 license + LICENSE +
    NOTICE + CONTRIBUTING.md + CODE_OF_CONDUCT.md (Contributor
    Covenant 2.1). `pyproject.toml` license field flipped from
    `"Proprietary"` → `"Apache-2.0"`; classifier added.
  - **Drift cleanup**: fixed `memopt/daemon/__init__.py` (was
    raising `ModuleNotFoundError` on import — unreached by tests
    but broken nonetheless), removed `memopt-wrap` ghost script
    entry from `pyproject.toml`, refreshed README test count.

Mac baseline: 876 → **1013 PASSED** (+137), 34 SKIPPED, 18 DESELECTED
(+3 cuda-named cert tests filtered by `-k 'not cuda'`), 0 FAILED.

### What v1.3.0 does NOT deliver

  - Phase B `MEMOPT_USE_ORCHESTRATOR=1` flag (Layer 2 stays
    observation-only per `docs/orchestrator_v1_design.md` DECISION 7).
  - GPU rig re-validation — Mac baseline only; the 2 `@gpu` tests
    skipped on Mac and the 18 cuda-named tests deselected on Mac
    must be re-run on a CUDA host before any release tag is pushed.
  - Layer 3 (Prefetch Oracle) — discovery doc only
    (planned future work).
  - Removal of dead `cmd_certificates` / `cmd_migrate` / phase3
    deferred-import surfaces. They sit inside try/except ImportError
    blocks and print "feature not available."
  - Cleanup of the `memopt-trust-archive` sibling repo. v1.3.0
    supersedes it; the archive should be marked read-only once the
    1.3.0 tag is pushed.

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
