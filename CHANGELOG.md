# Changelog

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
