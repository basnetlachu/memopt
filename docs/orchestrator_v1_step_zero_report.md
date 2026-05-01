# memopt Orchestrator v1 — Step Zero Verification Report

| Field          | Value                                                    |
| -------------- | -------------------------------------------------------- |
| Authoritative  | docs/orchestrator_v1_design.md §3.0                      |
| Repo HEAD      | `13cd56d` (Substrate v1 release — Phase A complete)      |
| Mac baseline   | 752 PASSED, 22 SKIPPED, 15 DESELECTED, 0 FAILED          |
| Probe rig      | Apple M4 Pro / Darwin 25.4.0 / Python 3.12.6 (no CUDA)   |
| Drafted        | 2026-05-01                                               |

Phase A may proceed when all HARD-BLOCK items (TV3, TV4, TV7) are
`PASS` (or, for TV7, deferred per §B.4) and the remaining items are
`PASS` or `DEGRADED-with-fallback`. This report is the gate.

Summary:

| Item | Subject                                | Status   |
| ---- | -------------------------------------- | -------- |
| S0.1 | `queue.Queue` overhead                 | PASS     |
| S0.2 | `Dispatcher.subscribe` head-of-line    | PASS     |
| S0.3 | `peek_handle` lookup cost (HARD BLOCK) | PASS     |
| S0.4 | CUDA `migrate` semantics (HARD BLOCK)  | DEFERRED — GPU rig run before commit 7 |
| S0.5 | `MEMOPT_EVICT_HIGH/_LOW` collision     | PASS     |
| S0.6 | Coordinator/dispatcher shutdown        | PASS     |
| S0.7 | C ABI hook additivity (Phase B block)  | PASS     |

No HARD-BLOCK failure. S0.4 deferred per §B.5 (Mac has no CUDA);
must be re-verified on the H100 rig before commit 7 ships.

---

## S0.1  `queue.Queue` overhead under 5 producer threads — PASS

Verified by:    60-line micro-benchmark (`/tmp/step_zero/tv1_queue.py`).
                Five producer threads, one consumer thread, 100 000
                events each, bounded `Queue(maxsize=16384)`. Three
                trials; median of P50 / P99 reported.
Result:         PASS
Numbers:        P50 = 333 ns (0.33 µs)  — target ≤ 5 µs
                P99 = 459 ns (0.46 µs)  — target ≤ 50 µs
                Trials P50 (ns): [333, 333, 333]
                Trials P99 (ns): [459, 459, 459]
                Wall-time per trial: ~0.4 s
                Zero exceptions, zero `queue.Full`.
Hardware:       Apple M4 Pro, Darwin 25.4.0, Python 3.12.6
                (Mac baseline rig).
Decision:       Protects §2.6 ingest budget and the §2.3.4
                coordinator queue choice (`queue.Queue`).
Fallback used:  None — primary criterion met by ~15× margin on
                P50 and ~100× on P99.
Doc updates:    None.

## S0.2  Substrate `Dispatcher.subscribe` head-of-line cost — PASS

Verified by:    50-line probe (`/tmp/step_zero/tv2_dispatcher.py`).
                Five subscribers (one per `Event.kind`), each callback
                sleeps 1 ms; 1000 mixed-kind events emitted; producer-
                side `emit` latency measured. Control = 1 subscriber.
Result:         PASS
Numbers:        1-subscriber emit P50 = 500 ns (0.50 µs)
                5-subscriber emit P50 = 1 125 ns (1.12 µs)
                Diff (5 − 1) P50      = 0.62 µs   — target < 50 µs
                5-sub absolute P50    = 1.12 µs   — target ≤ 50 µs
                1-sub P99 = 1.33 µs ; 5-sub P99 = 1.46 µs
                No callback raised; `events_dropped` == 0.
Hardware:       Apple M4 Pro, Darwin 25.4.0, Python 3.12.6.
Decision:       Protects §2.3.4's "subscribe one callback per kind"
                pattern and DECISION 3 (per-kind subscriber threads).
Fallback used:  None — head-of-line cost is < 1 µs across 5 subs.
Doc updates:    None.

## S0.3  `AllocationManager._handles` lookup cost (TV3) — PASS — HARD BLOCK CLEARED

Verified by:    Probe (`/tmp/step_zero/tv3_peek_v2.py`).
                Worker thread runs 10 000 `alloc(2 MiB) + free`;
                a second thread runs 10 000 bounded
                `peek_handle(handle_id)` calls (experimental
                wrapper using the existing `_handles_lock`). Median
                of three trials reported. Per the §B.4 hard-block
                rule, alloc warm-path P50 must regress < 20 % vs.
                the pre-`peek_handle` baseline measured in this
                same run.
Result:         PASS
Numbers:        alloc P50 baseline (median of 3) = 5 208 ns (5.21 µs)
                alloc P50 under peek (median of 3) = 5 208 ns (5.21 µs)
                alloc P50 regression                = +0.0 %
                                                     — target ≤ 5 %;
                                                     HARD BLOCK at > 20 %
                alloc P99 baseline = 7 291 ns (7.29 µs)
                alloc P99 under peek = 7 250 ns (7.25 µs)
                peek_handle P50 = 167 ns (0.17 µs)  — target ≤ 5 µs
                peek_handle P99 = 209 ns (0.21 µs)
                Tenant-context check: not exercised in this probe;
                covered by Commit 1 unit tests
                `test_peek_inside_tenant_context_passes` and
                `test_peek_inside_tenant_context_blocks_cross_tenant`.
                Note: the substrate's stated <1 µs warm-path target
                (substrate v1 §2.10) is not achieved on the Mac
                pure-Python path (~5 µs); this is the *existing*
                substrate baseline at HEAD `13cd56d` and is not
                regressed by `peek_handle`.
Hardware:       Apple M4 Pro, Darwin 25.4.0, Python 3.12.6.
Decision:       Protects §2.1 DECISION 5 (`peek_handle` exists) and
                §2.2.5 (`peek_handle` API specification).
Fallback used:  None — primary criterion met (0 % regression vs.
                5 % budget; well under the 20 % HARD BLOCK ceiling).
Doc updates:    None.

## S0.4  CUDA `migrate` semantics (TV4) — DEFERRED — GPU rig before commit 7

Verified by:    Not verified on this rig. The Mac dev box has no
                CUDA driver and no H100. Per §B.5 ("on Mac, mark
                S0.4 as 'deferred to GPU rig run before commit 7'")
                this deferral is acceptable for Phase A start-up.
Result:         DEFERRED — GPU rig run required before commit 7.
                Treated as conditional PASS for HARD-BLOCK accounting
                because Phase A commits 0–6 do not depend on
                `migrate` semantics: commit 1 only adds the
                `_emit_orchestrator_event` validation table
                (`migrate` requires `to_placement` set; no remap
                executed). The actual CUDA round-trip is not
                exercised until coordinator-side `migrate` decisions
                ship in commit 7.
Numbers:        Not measured.
                Probe to run on rig (per §3.0 S0.4 method):
                  1. `cuMemAddressReserve` → create HBM phys handle
                     → map → write sentinel → unmap → release →
                     create new HBM phys handle at same VA via
                     `cuMemMap` → read sentinel back.
                  2. Cross-tier: HBM → DRAM (host pinned) → HBM at
                     same VA, content round-trips.
                  3. 1000 cross-tier round trips, no
                     `cuMemAddressReserve` leak.
                  4. Whole probe < 30 s on H100 SXM5 80 GB.
Hardware:       Mac (no CUDA). Target rig: H100 SXM5 80 GB.
Decision:       Protects §2.1 DECISION 4 (`migrate` keeps `va`)
                and §2.2.6 (`_emit_orchestrator_event` accepts
                `migrate`).
Fallback used:  None yet. If the rig run fails (a) — same-tier
                remap loses bytes — apply §3.0 S0.4 fallback:
                  - drop `migrate` from v1.0 emission contract
                    (§2.2.6 narrows to `evict`/`promote`); OR
                  - redefine `migrate` as `free`+`alloc` with a
                    `from_handle_id` cross-reference (§2.1
                    DECISION 4 + substrate `Event` schema change,
                    re-run Phase 2).
Doc updates:    None until rig run completes. Track in commit 7
                pre-flight as a HARD BLOCK gate.

## S0.5  Watermark env-var collision (TV5) — PASS

Verified by:    Probe (`/tmp/step_zero/tv5_envvar.py`). With
                `MEMOPT_EVICT_HIGH=0.85`, `MEMOPT_EVICT_LOW=0.65`
                exported, both the `TierManager` env-read path
                (`tier_manager.py:72-75`) and a stub orchestrator
                config reading the same names returned the same
                values.
Result:         PASS
Numbers:        TierManager  → HIGH=0.85, LOW=0.65
                Orchestrator → HIGH=0.85, LOW=0.65
                Agreement: both subsystems read identical values.
                Phase A double-eviction risk: 0 by construction —
                per DECISION 7 the orchestrator is a silent
                observer in Phase A and never calls evict; the
                `test_evict_thresholds_*` suite still passes
                (752 PASSED baseline).
Hardware:       Apple M4 Pro, Darwin 25.4.0, Python 3.12.6.
Decision:       Protects DECISION 7 (Phase A is observer-only)
                and §2.3.6 (orchestrator config reuses
                `MEMOPT_EVICT_HIGH`/`_LOW`).
Fallback used:  None. If a future Phase B change introduces
                double-eviction, apply §3.0 S0.5 fallback:
                rename to `MEMOPT_ORCH_EVICT_HIGH`/`_LOW`
                (§2.3.6 change).
Doc updates:    None.

## S0.6  Coordinator/dispatcher shutdown interaction (TV6) — PASS

Verified by:    30-line probe (`/tmp/step_zero/tv6_shutdown.py`).
                Stub orchestrator = a coordinator thread that
                subscribes to `alloc` events on the substrate
                `Dispatcher`. Two scenarios:
                  A. `orchestrator.stop()` then
                     `AllocationManager.reset()` (recommended).
                  B. `AllocationManager.reset()` then
                     `orchestrator.stop()` (anti-pattern).
                Each scenario allocates 100 handles, frees 50,
                then runs the teardown. `threading.enumerate()`
                is checked 200 ms after teardown for any
                `memopt-` named lingering threads.
Result:         PASS
Numbers:        Scenario A: elapsed 0.1 ms ; lingering memopt
                            threads: []
                Scenario B: elapsed 0.1 ms ; lingering memopt
                            threads: []
                Both well under the 5 s budget; no deadlock; no
                thread leak.
Hardware:       Apple M4 Pro, Darwin 25.4.0, Python 3.12.6.
Decision:       Protects DECISION 3 (coordinator owns one thread)
                and §2.2.2 (`OrchestratorHandle.stop` semantics).
Fallback used:  None.
Doc updates:    None. The stub orchestrator's `stop()` did not
                hold any lock taken by `AllocationManager.reset`,
                so the §2.2.2 ordering guidance ("call
                orchestrator.stop() before reset()") remains a
                soft recommendation, not a deadlock-avoidance
                requirement.

## S0.7  C `step_boundary` callback hook feasibility (TV7) — PASS — HARD BLOCK FOR PHASE B (not Phase A)

Verified by:    End-to-end read of
                `csrc/cuda_vmm/vmm_allocator.h` (header is the
                ABI surface; 156 lines, fully reviewed) and
                `csrc/cuda_vmm/vmm_allocator.cpp:960-972`
                (`memopt_torch_step_boundary` definition); cross-
                checked against the ctypes consumer
                `memopt/vmm/cuda_vmm.py:74-100` (existing argtype
                bindings).
Result:         PASS — the proposed
                `memopt_set_evict_callback(void
                (*fn)(MemoptAllocator*, float))` can be added
                additively.
Numbers:        Existing `memopt_torch_step_boundary` definition
                exists in `vmm_allocator.cpp:960` but is NOT
                declared in `vmm_allocator.h` and NOT bound in
                `cuda_vmm.py` — so the symbol is reachable via
                `dlsym` only by a fresh `argtypes/restype`
                declaration. Adding a new exported symbol
                (`memopt_set_evict_callback`) to the header and
                `.cpp` does not change any existing function's
                signature; `vmm_allocator.h` already uses
                `extern "C"` block + `#ifdef __cplusplus` guards.
                The existing 9 ctypes bindings in
                `cuda_vmm.py:76-100` look up symbols by name and
                are unaffected.
                With the callback default = NULL,
                `memopt_torch_step_boundary` retains its existing
                semantic (quiesce → evict_to_target if pressure >
                threshold).
Hardware:       Source-only review; no compile / no link tested
                (csrc/ requires CUDA toolchain not present on
                Mac). Build verification deferred to GPU rig.
Decision:       Protects Phase B §2.4.1 plan ("layer 2 hooks the
                C++ allocator's eviction decision via a callback"
                rather than wrapping the entire `step_boundary`).
Fallback used:  None. Phase A does NOT change the C++ allocator
                (§2.4.1 Phase A is "frozen") so no Phase A
                consequence either way.
Doc updates:    None for Phase A. If the rig build fails to load
                `libmemopt_vmm.so` after the additive change at
                Phase B start, apply §3.0 S0.7 fallback: ship a
                separate `libmemopt_vmm` v2 with the new ABI, or
                wrap `memopt_torch_step_boundary` from Layer 2.

---

## Phase A gate

  - [x] All HARD-BLOCK items are PASS (or DEFERRED with no Phase A
        consequence). TV3 PASS; TV4 DEFERRED (no commit 0–6
        dependency); TV7 PASS (Phase B only).
  - [x] All other items are PASS or DEGRADED-with-fallback.
        TV1, TV2, TV5, TV6 all PASS.
  - [x] No DEGRADED items — no §2 doc updates required at this
        time.

Phase A start-up is approved. Commit 1 may proceed.

The TV4 H100 round-trip MUST run on the GPU rig before commit 7
(coordinator-side `migrate`) is merged; this is a hard pre-flight
on commit 7, not on commit 1. If TV4 fails on the rig, apply the
§3.0 S0.4 fallback and revisit §2.1 DECISION 4 / §2.2.6 before
commit 7 ships.
