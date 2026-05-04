# memopt Orchestrator v1 — Implementation Prompt

| Field           | Value                                                              |
| --------------- | ------------------------------------------------------------------ |
| Status          | Phase 4 prompt drafted. Ready for execution in fresh sessions.     |
| Authoritative   | docs/orchestrator_v1_design.md (Phases 1–3)                        |
| Repo            | /Users/lachumanbasnet/Personal/Sophisticates/MEMOPT/memopt         |
| Layer 1         | Substrate v1.1 at `13cd56d` (`Substrate v1 release — Phase A complete`) |
| Mac baseline    | 752 PASSED, 22 SKIPPED, 15 DESELECTED at HEAD `13cd56d` (§1.6.1)   |
| Drafted         | 2026-05-01                                                         |

This document is the operating manual for implementing Layer 2.
It is sequential: Step Zero, then 13 implementation commits in
5 milestones. Each commit prompt is self-contained — a future
contributor with this file and `docs/orchestrator_v1_design.md`
can rebuild Layer 2 from scratch without seeing the discovery /
design conversations.

The implementation runs as a series of fresh Claude Code
invocations. After every commit, the user reviews the change.
After every milestone, the user does a deeper review (reads each
new file end-to-end, verifies the milestone's promise).

Sections:

  A.  Baseline + setup
  B.  Step Zero verification (Commit 0)
  C.  Milestone 1 — Substrate-side prep (Commits 0–1)
  D.  Milestone 2 — Predictor + AccessTracker (Commits 2–3)
  E.  Milestone 3 — PolicyEngine + Config (Commits 4–5)
  F.  Milestone 4 — Coordinator + Telemetry + Public API (Commits 6–9)
  G.  Milestone 5 — Bridge + Docs + Final (Commits 10–13)
  H.  Final acceptance criteria

---

## Section A — Baseline + setup

### A.1 Baseline contract

```
BASELINE (HEAD 13cd56d on main, Mac/Darwin 25.4.0/Python 3.12, no CUDA):

  PASSED       752
  SKIPPED      22
  DESELECTED   15
  FAILED        0

ALLOWED deltas during Phase A:
  PASSED       monotonically increases by the count of new tests
               added in each commit (per §3.5 commit table).
  SKIPPED      stays at 22 unless a new test legitimately
               requires hardware not present on the Mac rig
               (e.g. new @gpu test). Each addition documented
               in the commit message.
  DESELECTED   stays at 15. The deselected set is gated by
               -k 'not gpu and not cuda'; new tests must
               either match that filter or be marked @gpu /
               @perf and skipped, not deselected.
  FAILED       stays at 0. Any new failure HALTS the commit;
               revert and fix.

DISALLOWED deltas during Phase A:
  - Any pre-existing PASSED test flips to FAILED or SKIPPED.
  - Any pre-existing SKIPPED test flips to FAILED.
  - Mac baseline numbers regress in any direction other than
    PASSED-up.

Substrate's existing GPU baseline (753 PASSED on the @gpu CI
rig per substrate v1 §H.1) must also be preserved. Commit 1
is the only commit that touches memopt/substrate/; its GPU
regression must be re-run on a rig before merge. All other
commits are GPU-neutral (Mac-only test changes).
```

### A.2 Regression command

```
PYTHONPATH=. python3 -m pytest tests/ memopt/ -q -p no:cacheprovider \
  -k 'not gpu and not cuda' \
  --ignore=memopt/vmm/tests/test_cuda_vmm_smoke.py \
  --ignore=memopt/vmm/tests/test_torch_allocator_sanity.py
```

This is the single source of truth for the §1.6.1 baseline. It
runs after every commit. Output is parsed for the four numbers
above.

For Scenario B (orchestrator started silently — §3.3.2):

```
MEMOPT_AUTO_START_ORCHESTRATOR=1 \
PYTHONPATH=. python3 -m pytest tests/ memopt/ -q -p no:cacheprovider \
  -k 'not gpu and not cuda' \
  --ignore=memopt/vmm/tests/test_cuda_vmm_smoke.py \
  --ignore=memopt/vmm/tests/test_torch_allocator_sanity.py
```

`MEMOPT_AUTO_START_ORCHESTRATOR` is a Phase-A-only test knob (NOT
a public env var). It is removed in Phase B in favour of
`MEMOPT_USE_ORCHESTRATOR` (§2.7 Phase B).

### A.3 Test categorization legend

```
Bucket            Marker        Where it runs
─────────────────  ───────────  ────────────────────────────────────
REQUIRED-CI       (none)        Mac / Linux without CUDA. Runs in
                                every commit's regression. Default.

REQUIRED-LOCAL    @gpu          Needs a CUDA rig. Runs locally and
                                in @gpu CI. Skipped on Mac.

OPTIONAL          @gpu_amd      AMD MI300X rig. Skipped elsewhere.
                  @cxl          CXL-capable host. Skipped elsewhere.
                  @perf         Microbenches. Skipped by default;
                                run with `-m perf`.
                  @imex         IMEX-enabled fabric host.
```

Per §3.1: most Layer-2 tests are REQUIRED-CI. The CUDA `migrate`
verification test (§3.1.4 `test_migrate_va_preserved`) is the
sole `@gpu` requirement; the perf microbench (§3.1.12) is
`@perf` (OPTIONAL).

### A.4 Dev environment assumptions

```
Python:           3.12.x (Mac baseline) or any 3.10+ on Linux
Pytest:           8.x with the existing memopt conftest rig
Torch:            optional; not required for any Layer-2 test
                  except @gpu tests on a rig
PYTHONPATH:       project root prepended (per A.2)
Git:              `main` is the integration branch; each commit
                  is a fresh PR or direct push gated by review.
Working tree:     clean before each commit prompt is launched.
                  `git status` shows no untracked or modified
                  files except the prompt's listed paths.
```

The Mac dev box has no CUDA; `@gpu` tests run on a rig
externally. All REQUIRED-CI tests must pass on the Mac.

### A.5 Authoritative design doc

`docs/orchestrator_v1_design.md` (Phases 1–3) is authoritative.
Where this prompt and the design doc disagree, the design doc
wins. The contributor:

  1. Halts.
  2. Updates the prompt to match the design.
  3. Notes the divergence in the commit message.

The prompt is a derivative work; the design doc is the
contract.

### A.6 Reverting protocol

If a commit fails its acceptance criteria (AC1–AC6, §3.5):

  - **AC1 / AC2 fail** (parse errors; new tests don't pass):
    fix in the same invocation. Do NOT amend a previous
    commit; create a new commit if the fix is non-trivial.
  - **AC3 fails** (regression in the 752 baseline): revert
    the commit (`git reset --hard HEAD~1`). Investigate
    whether the regression is real (Layer 2 has a bug — fix
    in the same invocation) or pre-existing flake (rerun;
    record in flake-tracking issue if persistent).
  - **AC4 fails** (touched a forbidden module): the commit
    is rejected. Restart the invocation with corrected
    scope.
  - **AC5 / AC6 fail** (commit message / revertibility):
    amend message; if revertibility is broken, split the
    commit.

Never `--no-verify`. Never `--force` push to `main`.

---

## Section B — Step Zero verification (Commit 0)

### B.1 Goal

Resolve every TODO-VERIFY (TV1–TV7) from §2.9 of the design
doc. Produce `docs/orchestrator_v1_step_zero_report.md`. Do not
write any production code.

### B.2 The 7 TV items (terse restatement)

```
TV1 / S0.1   queue.Queue overhead under 5 producer threads
             (target: P50 ≤ 5 µs, P99 ≤ 50 µs).
TV2 / S0.2   substrate Dispatcher.subscribe head-of-line cost
             with 5 subscriptions (target: ≤ 50 µs).
TV3 / S0.3   AllocationManager._handles lookup cost; alloc
             warm-path must not regress > 5 % under added
             peek_handle traffic. HARD BLOCK.
TV4 / S0.4   CUDA same-tier remap preserves bytes at the
             same VA (HBM→HBM round trip on H100). HARD BLOCK.
TV5 / S0.5   MEMOPT_EVICT_HIGH/_LOW shared with TierManager
             does not cause double-eviction in Phase A.
TV6 / S0.6   orchestrator stop + AllocationManager.reset in
             either order does not deadlock or leak threads.
TV7 / S0.7   C++ memopt_set_evict_callback can be added
             additively to the C ABI. HARD BLOCK FOR PHASE B
             (not Phase A).
```

### B.3 Report format (template)

The Step Zero report is a single markdown file with one entry
per S0.X item. Use this template verbatim:

```
## S0.X  <one-line summary> — <STATUS>

Verified by:    <method, with numbers>
Result:         <PASS | DEGRADED-with-fallback-{N} | FAIL>
Numbers:        <P50, P99, byte counts, etc.>
Hardware:       <hostname, OS, CPU, GPU if applicable>
Decision:       <which §2 decision this protects>
Fallback used:  <if DEGRADED, the documented fallback from §3.0>
Doc updates:    <if DEGRADED, the §2 lines that need editing>
```

STATUS legend (mirrors substrate v1 Step Zero conventions):

  - `PASS` — verification fully met.
  - `DEGRADED-with-fallback-N` — verification did not meet the
    primary criterion; the fallback labelled N in §3.0 is
    applied. Phase A may proceed; the design doc edit is
    documented.
  - `FAIL` — verification failed and no fallback applies. Phase
    A cannot begin until the design doc is revised and Phase 3
    re-run.

### B.4 HARD-BLOCK redesign protocol

Per §3.0 hard-block classification:

  - **TV3 (S0.3) — `peek_handle` lookup cost.**
    If alloc warm-path P50 regresses > 20 % vs. the
    pre-`peek_handle` baseline measured in commit 0 itself
    (against an in-memory experimental subclass), HALT. The
    fix is one of:
      a. Move `peek_handle` to a separate read-only side-
         table populated on alloc, lock-free reads. Re-run
         §2.2.5 specification.
      b. Drop DECISION 5 entirely. Layer 2 keeps its own
         immutable handle snapshots from `alloc` events.
         Re-run §2.1 DECISION 5 with rationale.
    Either choice triggers a Phase 2 doc update + Phase 3
    re-review before commit 1 begins.

  - **TV4 (S0.4) — CUDA same-tier remap.**
    If `cuMemUnmap` + `cuMemRelease` + `cuMemCreate` +
    `cuMemMap` to the same VA does not preserve bytes on
    H100, HALT. The fix is one of:
      a. Drop `migrate` from the v1.0 emission contract.
         §2.2.6 changes; `_emit_orchestrator_event`
         accepts only `evict` / `promote`.
      b. Redefine `migrate` as `free` + `alloc` with a
         `from_handle_id` cross-reference field. §2.1
         DECISION 4 changes; the `Event` schema gains a
         field (substrate change — re-run Phase 2).

  - **TV7 (S0.7) — C ABI hook.**
    HARD BLOCK FOR PHASE B, NOT PHASE A. If adding
    `memopt_set_evict_callback` is not ABI-additive, document
    the failure but DO NOT halt Commit 0. Phase A delivers
    no C++ changes; the ABI failure invalidates §2.4.1's
    Phase B plan and is re-resolved before Phase B begins.

  - **TV1, TV2, TV5, TV6 — graceful fallbacks.**
    Each has a documented degradation path in §3.0. Apply
    the fallback, mark `DEGRADED-with-fallback-N` in the
    report, document the §2 doc update needed, and proceed.

### B.5 Commit 0 prompt

```
=== Paste into a fresh Claude Code session ===

You are implementing memopt Orchestrator v1 Commit 0 — Step Zero
verification report.

Read first:
- docs/orchestrator_v1_design.md §3.0 (Step Zero S0.1–S0.7)
- docs/orchestrator_v1_design.md §2.1 (DECISIONS 1–8)
- docs/orchestrator_v1_implementation_prompt.md Sections A and B
  (this section)

Files to add:
- docs/orchestrator_v1_step_zero_report.md
    One entry per S0.X item, using the §B.3 template.

Files to edit:
- (none — Step Zero is verification-only)

Files NOT to edit (guardrail):
- Any file under memopt/, csrc/, tests/, or pyproject.toml.
- Any file under docs/ except the new step-zero report.

Verification work:
- Run a 60-line micro-benchmark for TV1 (queue.Queue under
  5 producer threads); record P50/P99 ns.
- Run a 50-line probe for TV2 (5 substrate subscriptions);
  record emit-side P50.
- For TV3, install an experimental peek_handle as a wrapper
  around AllocationManager._handles dict (do NOT commit this
  experiment); measure alloc-warm path P50 with and without
  10 000 concurrent peek_handle calls.
- For TV4, write the CUDA remap probe and run it on the
  available rig (or document SKIP-no-rig); on Mac, mark
  S0.4 as "deferred to GPU rig run before commit 7."
- For TV5, simulate the env-var collision by constructing a
  VMM and a mock orchestrator config in the same process
  with MEMOPT_EVICT_HIGH=0.85; verify both read 0.85.
- For TV6, write a 30-line test that starts a stub
  orchestrator (just the coordinator thread skeleton),
  calls AllocationManager.reset(), then orchestrator.stop()
  in both orders; confirm no thread leak via
  threading.enumerate() before/after.
- For TV7, read csrc/cuda_vmm/vmm_allocator.h end-to-end
  and document whether memopt_set_evict_callback can be
  added without breaking the existing ctypes consumers in
  memopt/vmm/cuda_vmm.py.

For each TV item:
- If PASS: record the numbers and proceed.
- If TV3 or TV4 fail: HALT and report. Do not write commit 1
  until the design doc is revised per §B.4.
- If TV7 fails: document but do not halt. Note Phase B impact.
- If TV1/TV2/TV5/TV6 fail their primary criterion: apply the
  fallback documented in §3.0, mark DEGRADED-with-fallback-N,
  list the §2 doc edits required.

Tests:
- (none — verification only)

Acceptance criteria (AC1-AC6 per §3.5):
- AC1: docs/orchestrator_v1_step_zero_report.md parses as
       valid markdown with one entry per S0.X.
- AC2: (no tests added).
- AC3: Mac baseline regression unchanged (752 PASSED, 22
       SKIPPED, 15 DESELECTED, 0 FAILED).
- AC4: Only docs/orchestrator_v1_step_zero_report.md is
       added; no other file is touched.
- AC5: Commit message: "Orchestrator v1 Commit 0: Step Zero
       verification report (per docs/orchestrator_v1_design.md
       §3.0)".
- AC6: Reverting the commit removes only the report.

After commit, print baseline contract status line:
  "Baseline contract status: PASSED 752, SKIPPED 22, DESELECTED 15, FAILED 0"

If TV3 or TV4 hard-block: print
  "HARD BLOCK on TV<N>: see docs/orchestrator_v1_step_zero_report.md."
and HALT.

=== End paste ===
```

---

## Section C — Milestone 1: Substrate-side prep (Commits 0–1)

### C.0 Milestone 1 promise

By the end of Milestone 1, Step Zero is complete and the
substrate has gained exactly two new public methods
(`peek_handle`, `_emit_orchestrator_event`). No Layer-2
component code exists yet. The substrate's 752-pass Mac baseline
holds; substrate's 753-pass GPU baseline must also hold (verify
on a rig before merge). Layer 2 has a place to plug into.

### C.1 Commit 0 — Step Zero (refer to Section B)

Section B contains the full prompt. Bucket: REQUIRED-CI (no
tests added; the regression run validates the unchanged
baseline).

### C.2 Commit 1 — substrate addition: peek_handle + emit hook

This is the ONLY commit that touches `memopt/substrate/`.
Per §2.2.5 and §2.2.6.

```
=== Paste into a fresh Claude Code session ===

You are implementing memopt Orchestrator v1 Commit 1 —
substrate addition: peek_handle + _emit_orchestrator_event.

Read first:
- docs/orchestrator_v1_design.md §1.5 (Layer 1/Layer 2 boundary)
- docs/orchestrator_v1_design.md §2.2.5 (peek_handle spec)
- docs/orchestrator_v1_design.md §2.2.6 (event-emission contract)
- docs/orchestrator_v1_design.md §3.0 S0.3 (TV3 lookup-cost
  budget; hard block on regression > 20 %)
- docs/orchestrator_v1_implementation_prompt.md Section A and C

Files to edit (additive only):
- memopt/substrate/manager.py
    Add a new method `peek_handle(self, handle_id: int, *,
    tenant: Optional[str] = None) -> Optional[MemoryHandle]`
    on AllocationManager. Behaviour:
      - Take the existing self._handles_lock.
      - Look up self._handles[handle_id].
      - If not present, return None.
      - Resolve effective tenant: explicit `tenant` arg,
        else current_tenant() context, else None.
      - If effective tenant is not None and the handle's
        tenant != effective tenant: raise PermissionError
        ("peek across tenants forbidden (G1)").
        Otherwise return the handle.
      - Frozen-dataclass: the returned MemoryHandle is a
        reference, not a copy; callers must treat it as
        immutable.
    Add a new method `_emit_orchestrator_event(self, event:
    Event) -> None`. Behaviour:
      - assert event.kind in ("evict","promote","migrate").
      - For "evict": assert from_placement is hotter than
        to_placement (use a static placement-rank table:
        hbm/hot=3, dram/warm=2, cxl=2, nvme/cold=1, cpu=1).
      - For "promote": assert from_placement is colder than
        to_placement.
      - For "migrate": assert to_placement is set.
      - Forward to self._dispatcher.emit(event).
    Do NOT modify any existing method body. Do NOT change
    any existing signature. Verify by `git diff` review:
    only net additions.
- memopt/substrate/__init__.py
    Re-export `peek_handle`:
      `def peek_handle(handle_id, *, tenant=None):
         return AllocationManager.get().peek_handle(
             handle_id, tenant=tenant)`
    Add to __all__.
- memopt/__init__.py
    Add `from .substrate import peek_handle` to the existing
    substrate re-export block. Add to __all__.

Files to add:
- memopt/substrate/tests/test_peek_handle.py
    10 tests:
      1. test_peek_returns_handle_for_known_id
      2. test_peek_returns_none_for_unknown_id
      3. test_peek_returns_none_after_free
      4. test_peek_with_explicit_tenant_arg_passes
      5. test_peek_with_explicit_tenant_arg_blocks_cross_tenant
      6. test_peek_inside_tenant_context_passes
      7. test_peek_inside_tenant_context_blocks_cross_tenant
      8. test_peek_returns_immutable_handle_reference
      9. test_peek_threadsafe_under_concurrent_alloc
     10. test_emit_orchestrator_event_round_trips_to_subscriber

Files NOT to edit (guardrail):
- Any file under memopt/vmm/, memopt/cluster/, memopt/serving/,
  memopt/api/, memopt/control_plane/, memopt/operator/,
  memopt/canary/, memopt/daemon/, memopt/profiler/,
  memopt/kernels/.
- Any file under memopt/substrate/ EXCEPT the three listed above.
- csrc/* (no C++ change in Phase A).

Guardrail audit:
- After the edit, run:
    git diff --stat memopt/substrate/manager.py
  Verify the patch shows only insertions (no removed lines,
  no modified existing methods). If any pre-existing method
  body changed, REVERT and restart the invocation.

GPU regression note:
- Print:
    "GPU regression note: substrate's 753-pass @gpu baseline
     must be re-run on a rig before this commit is merged
     (substrate v1 §H.1)."
- Do not block on it; the rig run is a human task.

Tests:
- REQUIRED-CI: all 10 tests in test_peek_handle.py.
- REQUIRED-LOCAL: (none — peek_handle is backend-agnostic).
- OPTIONAL: (none).

Acceptance criteria (AC1-AC6 per §3.5):
- AC1: Imports work; `python -W error -c "from memopt
       import peek_handle"` exits 0.
- AC2: All 10 new tests pass: `pytest
       memopt/substrate/tests/test_peek_handle.py -q`.
- AC3: Mac baseline regression: 752+10=762 PASSED, 22
       SKIPPED, 15 DESELECTED, 0 FAILED.
- AC4: `git diff --name-only HEAD~1..HEAD` shows exactly
       four paths:
         memopt/substrate/manager.py
         memopt/substrate/__init__.py
         memopt/__init__.py
         memopt/substrate/tests/test_peek_handle.py
- AC5: Commit message: "Orchestrator v1 Commit 1: substrate
       peek_handle + _emit_orchestrator_event (per
       docs/orchestrator_v1_design.md §2.2.5, §2.2.6)".
- AC6: Reverting removes the four files / restores the
       previous content; the substrate's 752 baseline
       returns; no test breaks.

After commit, print baseline contract status line:
  "Baseline contract status: PASSED 762, SKIPPED 22, DESELECTED 15, FAILED 0"

Halt and report if AC1-AC6 do not all pass.

=== End paste ===
```

### C.3 Milestone 1 review gate

Before launching Milestone 2:

  - [ ] Step Zero report exists with PASS / DEGRADED for all
        S0.1–S0.7.
  - [ ] No HARD-BLOCK item (TV3, TV4) is FAIL.
  - [ ] `peek_handle` works from Python: `import memopt;
        h = memopt.alloc(1024); assert
        memopt.peek_handle(h.handle_id) is h; memopt.free(h)`.
  - [ ] `_emit_orchestrator_event` is callable from a unit
        test and round-trips an event to a subscriber.
  - [ ] Mac baseline: 762 PASSED, 22 SKIPPED, 15 DESELECTED.
  - [ ] Substrate's @gpu baseline re-verified on a rig (753
        PASSED, no new failures).
  - [ ] `git log --diff-filter=M --name-only main..HEAD`
        shows only the four files listed in C.2.
  - [ ] §3.3 Scenario A regression passes.

Per §3.5 commit-ordering rationale: commit 1 ships before
every later commit because each Layer 2 component imports from
`memopt.substrate` and uses `peek_handle`; without commit 1
each later commit would require monkey-patching during tests.

---

## Section D — Milestone 2: Predictor + AccessTracker (Commits 2–3)

### D.0 Milestone 2 promise

By the end of Milestone 2, two pure-Python data structures exist
under `memopt/orchestrator/`: a tenant-aware access tracker
(§2.3.1) and a tenant-keyed Markov predictor (§2.3.2 / DECISION
6). Both are testable in isolation on the Mac baseline; neither
requires the substrate to be running.

### D.1 Commit 2 — AccessTracker + tenant LRU lists

Per §2.3.1.

```
=== Paste into a fresh Claude Code session ===

You are implementing memopt Orchestrator v1 Commit 2 —
AccessTracker + per-tenant LRU.

Read first:
- docs/orchestrator_v1_design.md §2.3.1 (AccessTracker spec)
- docs/orchestrator_v1_design.md §2.5 G1, G2 (tenant isolation)
- docs/orchestrator_v1_design.md §3.1.1 (test list)
- docs/orchestrator_v1_implementation_prompt.md Section A and D

Files to add:
- memopt/orchestrator/__init__.py
    Empty stub for now (filled in commit 8). One docstring
    line: "memopt orchestrator package — Layer 2."
- memopt/orchestrator/access.py
    Module containing:
      - AccessRecord dataclass (frozen): tag, placement,
        last_seen_ns, hit_count, size_bytes, lru_prev,
        lru_next (the last two are mutated; AccessRecord
        cannot be frozen — use __slots__).
      - AccessTracker class with public surface from
        §2.3.1: record(event), last_seen(tenant, tag,
        handle_id), lru_candidates(tenant, placement,
        count), forget_handle(handle_id), forget_tenant(
        tenant), snapshot().
      - Internal state: per-tenant dict of {handle_id:
        AccessRecord}, per-(tenant, placement) intrusive
        doubly-linked LRU list head/tail, per-tenant
        threading.Lock.
- memopt/orchestrator/tests/__init__.py (empty).
- memopt/orchestrator/tests/test_access_tracker.py
    13 tests per §3.1.1 list (each test ≤ 25 lines):
      test_record_alloc_inserts_record
      test_record_free_removes_record
      test_record_evict_updates_placement
      test_record_promote_updates_placement
      test_record_migrate_updates_placement
      test_lru_candidates_returns_oldest_first
      test_lru_candidates_respects_count
      test_lru_candidates_isolated_per_tenant
      test_forget_handle_removes_from_lru
      test_forget_tenant_clears_all_state
      test_concurrent_record_thread_safe
      test_unknown_handle_id_silently_dropped
      test_snapshot_returns_consistent_view

Files to edit:
- (none)

Files NOT to edit (guardrail):
- Any file under memopt/vmm/, memopt/cluster/, memopt/serving/,
  memopt/api/, memopt/control_plane/, memopt/operator/,
  memopt/canary/, memopt/daemon/, memopt/profiler/,
  memopt/kernels/.
- Any file under memopt/substrate/.
- memopt/__init__.py (no orchestrator re-export yet —
  package is internal until commit 8).

Tests:
- REQUIRED-CI: all 13 tests in test_access_tracker.py.
- REQUIRED-LOCAL: (none).
- OPTIONAL: (none).

Acceptance criteria (AC1-AC6 per §3.5):
- AC1: `python -W error -c "from memopt.orchestrator.access
       import AccessTracker"` exits 0.
- AC2: `pytest memopt/orchestrator/tests/test_access_tracker
       .py -q` passes 13/13.
- AC3: Mac baseline: 762+13=775 PASSED, 22 SKIPPED, 15
       DESELECTED, 0 FAILED.
- AC4: `git diff --name-only HEAD~1..HEAD` shows exactly
       four paths:
         memopt/orchestrator/__init__.py
         memopt/orchestrator/access.py
         memopt/orchestrator/tests/__init__.py
         memopt/orchestrator/tests/test_access_tracker.py
- AC5: Commit message: "Orchestrator v1 Commit 2:
       AccessTracker + per-tenant LRU (per
       docs/orchestrator_v1_design.md §2.3.1)".
- AC6: Reverting removes the orchestrator package; the 762
       baseline returns; no test breaks.

After commit, print baseline contract status line:
  "Baseline contract status: PASSED 775, SKIPPED 22, DESELECTED 15, FAILED 0"

Halt and report if AC1-AC6 do not all pass.

=== End paste ===
```

### D.2 Commit 3 — Predictor (Markov + sequential + recency)

Per §2.3.2 and DECISION 6.

```
=== Paste into a fresh Claude Code session ===

You are implementing memopt Orchestrator v1 Commit 3 —
Predictor (Markov + sequential + recency, tenant-keyed).

Read first:
- docs/orchestrator_v1_design.md §2.3.2 (Predictor spec)
- docs/orchestrator_v1_design.md §2.1 DECISION 6 (rationale
  for greenfield predictor)
- docs/orchestrator_v1_design.md §3.1.2 (test list)
- memopt/vmm/_oracle_py.py:40-251 (the existing oracle —
  for algorithm reference only; do NOT import it)
- docs/orchestrator_v1_implementation_prompt.md Section A and D

Files to add:
- memopt/orchestrator/predict.py
    Module containing:
      - Prediction dataclass (frozen): tenant, tag,
        confidence (float), source (one of "transition",
        "sequential", "recency", "fallback").
      - Predictor class with public surface from §2.3.2:
        observe(tenant, tag), predict(tenant, tag, top_k=10)
        -> List[Prediction], forget_tenant(tenant), stats().
      - Internal state:
        _transitions: Dict[(str,str), Counter[(str,str)]]
        _recency: OrderedDict[(str,str), int]
        _steps: Dict[str, int]   # per-tenant step counter
        _max_transitions, _min_confidence (from constructor)
        _lock: threading.RLock
      - DECISION 6 invariant: every key tuple is
        (tenant, tag); no int-only keys.
      - Bounded by _max_transitions (default 100_000); LRU
        eviction when full (mirrors _oracle_py.py:46).
- memopt/orchestrator/tests/test_predictor.py
    13 tests per §3.1.2 list (each ≤ 25 lines):
      test_observe_inserts_transition
      test_observe_increments_existing_transition
      test_predict_returns_top_k_sorted_by_confidence
      test_predict_includes_sequential_source
      test_predict_includes_recency_source
      test_predict_fallback_when_cold_start
      test_min_confidence_filters_predictions
      test_max_transitions_bounded
      test_predictor_keys_are_tenant_aware
      test_forget_tenant_removes_all_transitions
      test_predictor_thread_safe_under_rlock
      test_predictor_stats_shape
      test_predictor_does_not_share_state_with_memory_oracle

Files to edit:
- (none)

Files NOT to edit (guardrail):
- Any file under memopt/vmm/, memopt/cluster/, memopt/serving/,
  memopt/api/, memopt/control_plane/, memopt/operator/,
  memopt/canary/, memopt/daemon/, memopt/profiler/,
  memopt/kernels/, memopt/substrate/.
- The existing memopt/vmm/_oracle_py.py and oracle.py — read-
  only reference; the new code does NOT import them.

Tests:
- REQUIRED-CI: all 13 tests in test_predictor.py.
- REQUIRED-LOCAL: (none).
- OPTIONAL: (none).

Acceptance criteria (AC1-AC6 per §3.5):
- AC1: `python -W error -c "from memopt.orchestrator.predict
       import Predictor"` exits 0.
- AC2: `pytest memopt/orchestrator/tests/test_predictor.py
       -q` passes 13/13.
- AC3: Mac baseline: 775+13=788 PASSED, 22 SKIPPED, 15
       DESELECTED, 0 FAILED.
- AC4: `git diff --name-only HEAD~1..HEAD` shows exactly
       two paths:
         memopt/orchestrator/predict.py
         memopt/orchestrator/tests/test_predictor.py
- AC5: Commit message: "Orchestrator v1 Commit 3: Predictor
       (Markov + sequential + recency, tenant-keyed) (per
       docs/orchestrator_v1_design.md §2.3.2, DECISION 6)".
- AC6: Reverting removes the two files; the 775 baseline
       returns.

After commit, print baseline contract status line:
  "Baseline contract status: PASSED 788, SKIPPED 22, DESELECTED 15, FAILED 0"

Halt and report if AC1-AC6 do not all pass.

=== End paste ===
```

### D.3 Milestone 2 review gate

  - [ ] `memopt/orchestrator/access.py` and `predict.py` are
        importable.
  - [ ] AccessTracker stores no global state — instances are
        independent.
  - [ ] Predictor key shape is `(tenant, tag)` everywhere
        (DECISION 6 verification: grep `_transitions` for
        single-int keys; should return zero hits).
  - [ ] Mac baseline: 788 PASSED, 22 SKIPPED, 15 DESELECTED.
  - [ ] No file under memopt/substrate/ or memopt/vmm/ has
        been touched (per AC4 audit).

Per §3.5 ordering rationale: commits 2 and 3 precede commit 5
because `LRUWatermarkPolicy.evaluate` reads
`AccessTracker.lru_candidates` and `Predictor.predict` to rank
candidates. Either commit 2 or 3 could come first; we ship 2
first because PolicyEngine logically depends on access state
more directly than on predictions.

---

## Section E — Milestone 3: PolicyEngine + Config (Commits 4–5)

### E.0 Milestone 3 promise

By the end of Milestone 3, the orchestrator has a
configuration object with env-var overrides and a policy
engine with one built-in policy (`LRUWatermarkPolicy`). User
policies can register against the protocol but the engine is
not yet driven (the coordinator arrives in commit 7). The
shared env vars `MEMOPT_EVICT_HIGH/_LOW` (§2.3.6) are read
without disturbing `TierManager`'s reads.

### E.1 Commit 4 — OrchestratorConfig + env-var plumbing

Per §2.3.6.

```
=== Paste into a fresh Claude Code session ===

You are implementing memopt Orchestrator v1 Commit 4 —
OrchestratorConfig + env-var plumbing.

Read first:
- docs/orchestrator_v1_design.md §2.3.6 (config spec)
- docs/orchestrator_v1_design.md §3.0 S0.5 / TV5 (env-var
  collision contract)
- docs/orchestrator_v1_design.md §3.1.6 (test list)
- docs/orchestrator_v1_implementation_prompt.md Section A and E

Files to add:
- memopt/orchestrator/config.py
    Module containing:
      - OrchestratorConfig frozen dataclass with the 7
        fields from §2.3.6: cycle_period_ms (50),
        event_queue_capacity (16384), predictor_max_
        transitions (100_000), predictor_min_confidence
        (0.3), built_in_policy (True), lru_high_watermark
        (0.90), lru_low_watermark (0.75).
      - `from_env() -> OrchestratorConfig` classmethod that
        reads the documented env vars (MEMOPT_ORCH_CYCLE_MS,
        MEMOPT_ORCH_QUEUE_CAP, MEMOPT_ORCH_MAX_TRANSITIONS,
        MEMOPT_EVICT_HIGH, MEMOPT_EVICT_LOW). Invalid
        values fall back to defaults with a logged warning.
      - Validation: cycle_period_ms >= 1; capacity >= 1;
        0.5 <= low < high <= 1.0 (mirrors tier_manager.py:
        77-86 validation behaviour).
- memopt/orchestrator/tests/test_config.py
    8 tests per §3.1.6 list:
      test_default_config_has_documented_values
      test_env_overrides_cycle_period
      test_env_overrides_queue_capacity
      test_env_overrides_max_transitions
      test_env_overrides_evict_thresholds_shared_with_vmm
      test_env_invalid_falls_back_to_default
      test_config_is_frozen_dataclass
      test_built_in_policy_can_be_disabled

Files to edit:
- (none)

Files NOT to edit (guardrail):
- Any file under memopt/vmm/ (read-only reference for
  validation behaviour parity), memopt/cluster/,
  memopt/serving/, memopt/api/, memopt/control_plane/,
  memopt/operator/, memopt/canary/, memopt/daemon/,
  memopt/profiler/, memopt/kernels/, memopt/substrate/.

Tests:
- REQUIRED-CI: all 8 tests.
- REQUIRED-LOCAL: (none).
- OPTIONAL: (none).

Acceptance criteria (AC1-AC6):
- AC1: import works.
- AC2: 8/8 tests pass.
- AC3: Mac baseline: 788+8=796 PASSED, 22 SKIPPED, 15
       DESELECTED, 0 FAILED.
- AC4: only the two added files in the diff.
- AC5: Commit message: "Orchestrator v1 Commit 4:
       OrchestratorConfig + env-var plumbing (per
       docs/orchestrator_v1_design.md §2.3.6)".
- AC6: revert restores the 788 baseline.

After commit, print baseline contract status line:
  "Baseline contract status: PASSED 796, SKIPPED 22, DESELECTED 15, FAILED 0"

Halt and report if AC1-AC6 do not all pass.

=== End paste ===
```

### E.2 Commit 5 — PolicyEngine + LRUWatermarkPolicy

Per §2.3.3.

```
=== Paste into a fresh Claude Code session ===

You are implementing memopt Orchestrator v1 Commit 5 —
PolicyEngine + LRUWatermarkPolicy.

Read first:
- docs/orchestrator_v1_design.md §2.3.3 (PolicyEngine spec)
- docs/orchestrator_v1_design.md §3.1.3 (test list)
- docs/orchestrator_v1_implementation_prompt.md Section A and E

Files to add:
- memopt/orchestrator/policy.py
    Module containing:
      - PolicySnapshot frozen dataclass (per §2.3.3 fields).
      - Decision frozen dataclass (per §2.3.3 fields:
        kind, handle_id, target_placement, reason,
        priority).
      - Policy protocol (typing.Protocol) with the single
        method `evaluate(snapshot) -> List[Decision]`.
      - PolicyEngine class:
          register(policy) -> None
          unregister(policy) -> None
          evaluate(snapshot) -> List[Decision]
          conflict resolution: higher priority wins; ties
          break on registration order.
          buggy-policy auto-unregister after 3 raises.
      - LRUWatermarkPolicy(per_tenant_high, per_tenant_low):
          uses snapshot.per_tenant_pressure and
          snapshot.lru_candidates to emit Decision(kind=
          "evict", target_placement="dram", reason=
          "watermark_high") until pressure < low.
- memopt/orchestrator/tests/test_policy_engine.py
    12 tests per §3.1.3 list.

Files to edit:
- (none)

Files NOT to edit (guardrail):
- See E.1 list. Plus: do NOT import
  memopt/orchestrator/access.py or predict.py at module
  load time — to keep this commit's tests fast and
  independent. Tests may construct mock snapshots.

Tests:
- REQUIRED-CI: all 12 tests.
- REQUIRED-LOCAL: (none).
- OPTIONAL: (none).

Acceptance criteria (AC1-AC6):
- AC1: import works.
- AC2: 12/12 tests pass.
- AC3: Mac baseline: 796+12=808 PASSED, 22 SKIPPED, 15
       DESELECTED, 0 FAILED.
- AC4: only the two added files in the diff.
- AC5: Commit message: "Orchestrator v1 Commit 5:
       PolicyEngine + LRUWatermarkPolicy (per
       docs/orchestrator_v1_design.md §2.3.3)".
- AC6: revert restores the 796 baseline.

After commit, print baseline contract status line:
  "Baseline contract status: PASSED 808, SKIPPED 22, DESELECTED 15, FAILED 0"

Halt and report if AC1-AC6 do not all pass.

=== End paste ===
```

### E.3 Milestone 3 review gate

  - [ ] `OrchestratorConfig.from_env()` reads MEMOPT_EVICT_*
        without disturbing `TierManager`'s reads (verified by
        running both in the same process — quick REPL probe).
  - [ ] `PolicyEngine` accepts a user policy registered via
        the protocol.
  - [ ] `LRUWatermarkPolicy` emits decisions strictly per-
        tenant (G2 verification).
  - [ ] Mac baseline: 808 PASSED, 22 SKIPPED, 15 DESELECTED.
  - [ ] No file under memopt/vmm/ or memopt/substrate/ has
        been touched.

Per §3.5: commit 4 (Config) before commit 5 (PolicyEngine)
because PolicyEngine and the future Coordinator both consume
`OrchestratorConfig`. Commits 4–5 precede commit 7
(Coordinator) because the coordinator's `tick()` builds a
snapshot and invokes `PolicyEngine.evaluate`.

---

## Section F — Milestone 4: Coordinator + Telemetry + Public API (Commits 6–9)

### F.0 Milestone 4 promise

By the end of Milestone 4, the orchestrator is fully assembled:
the coordinator drives the cycle (DECISION 3), telemetry
collects counters (DECISION 8), and the public API is exposed
(§2.2). Decision verification + tenant isolation tests pin
DECISIONS 1, 2, 3, 4, 5, 6, 7, 8 and §2.5 G1–G4.

### F.1 Commit 6 — TelemetryCollector

Per §2.3.5 and DECISION 8.

```
=== Paste into a fresh Claude Code session ===

You are implementing memopt Orchestrator v1 Commit 6 —
TelemetryCollector.

Read first:
- docs/orchestrator_v1_design.md §2.3.5 (TelemetryCollector
  spec)
- docs/orchestrator_v1_design.md §2.1 DECISION 8 (separate
  stats namespace)
- docs/orchestrator_v1_design.md §3.1.5 (test list)
- docs/orchestrator_v1_implementation_prompt.md Section A and F

Files to add:
- memopt/orchestrator/telemetry.py
    Module containing:
      - TelemetryCollector class with __init__,
        increment(name, by=1), set(name, value),
        snapshot() -> dict, reset().
      - Counters required by §2.3.5: events_ingested[5
        kinds], decisions_emitted[3 kinds], policy_raises,
        queue_drops, coordinator_cycles.
      - Per-tenant decision counts gated by
        MEMOPT_ADMIN_TOKEN per §2.5 G4 (admin-token
        constant-time compare via hmac.compare_digest).
      - threading.Lock around aggregate read; per-counter
        increments use itertools.count or
        threading.atomic-equivalent (Python int + lock).
- memopt/orchestrator/tests/test_telemetry.py
    9 tests per §3.1.5 list.

Files to edit:
- (none)

Files NOT to edit (guardrail):
- See E.1 list.

Tests:
- REQUIRED-CI: all 9 tests.

Acceptance criteria (AC1-AC6):
- AC1: import works.
- AC2: 9/9 tests pass.
- AC3: Mac baseline: 808+9=817 PASSED, 22 SKIPPED, 15
       DESELECTED, 0 FAILED.
- AC4: only the two added files in the diff.
- AC5: Commit message: "Orchestrator v1 Commit 6:
       TelemetryCollector (per
       docs/orchestrator_v1_design.md §2.3.5, DECISION 8)".
- AC6: revert restores the 808 baseline.

After commit, print baseline contract status line:
  "Baseline contract status: PASSED 817, SKIPPED 22, DESELECTED 15, FAILED 0"

Halt and report if AC1-AC6 do not all pass.

=== End paste ===
```

### F.2 Commit 7 — Coordinator + event-emission contract

Per §2.3.4 and §2.2.6.

```
=== Paste into a fresh Claude Code session ===

You are implementing memopt Orchestrator v1 Commit 7 —
OrchestratorCoordinator + event-emission contract.

Read first:
- docs/orchestrator_v1_design.md §2.3.4 (Coordinator spec)
- docs/orchestrator_v1_design.md §2.2.6 (Layer 2 emission)
- docs/orchestrator_v1_design.md §2.1 DECISIONS 3, 4, 7
- docs/orchestrator_v1_design.md §3.1.4 (coordinator tests),
  §3.1.8 (event emission tests)
- docs/orchestrator_v1_implementation_prompt.md Section A and F

Files to add:
- memopt/orchestrator/coordinator.py
    Module containing:
      - OrchestratorCoordinator class:
          __init__(manager, config, tracker, predictor,
                   policy_engine, telemetry)
          start() -> None
          stop(timeout=2.0) -> None
          is_alive() -> bool
          tick() -> None  (test-visible single-cycle drive)
      - Internal: _event_queue (queue.Queue,
        config.event_queue_capacity), _subs (5
        SubscriptionHandles), _thread, _stopping
        (threading.Event), _cycle_period_ms.
      - Subscribe loop: 5 calls to
        AllocationManager.observe(kind, _enqueue)
        producing one subscription per Event.kind.
      - Coordinator loop:
          drain queue (apply each event to
            tracker.record + predictor.observe)
          on cycle boundary:
            build PolicySnapshot
            policy_engine.evaluate(snapshot)
            for each decision in priority order:
              construct Event(kind=...,
                              from_placement=...,
                              to_placement=...,
                              reason=...)
              call manager._emit_orchestrator_event(event)
              telemetry.increment("decisions_emitted." + kind)
      - DECISION 7 gate: in v1.0, only emit decisions when
        config has a Phase-A test mode flag set
        (`_decision_mode_for_phase_a_only=True`); default
        False. The default behaviour is observation-only.
- memopt/orchestrator/tests/test_coordinator.py
    15 tests per §3.1.4 list. Includes test_migrate_va_
    preserved which has TWO modes: REQUIRED-CI verifies
    the Python-side _va field is unchanged using a CPU
    backend; REQUIRED-LOCAL @gpu verifies via real CUDA
    remap (skip on Mac).
- memopt/orchestrator/tests/test_event_emission.py
    11 tests per §3.1.8 list.

Files to edit:
- (none)

Files NOT to edit (guardrail):
- See E.1 list.
- memopt/substrate/* — uses peek_handle and
  _emit_orchestrator_event from commit 1; does not modify
  any substrate file.

Tests:
- REQUIRED-CI: all 26 tests except test_migrate_va_preserved
  in @gpu mode.
- REQUIRED-LOCAL: test_migrate_va_preserved (@gpu).
- OPTIONAL: (none).

Acceptance criteria (AC1-AC6):
- AC1: import works.
- AC2: 25/25 REQUIRED-CI tests pass on Mac. (The 1 @gpu
       test is skipped on Mac and run on a rig.)
- AC3: Mac baseline: 817+25=842 PASSED, 22+1=23 SKIPPED,
       15 DESELECTED, 0 FAILED.
       The new SKIPPED is the @gpu test.
- AC4: only the two added files in the diff.
- AC5: Commit message: "Orchestrator v1 Commit 7:
       Coordinator + event-emission contract (per
       docs/orchestrator_v1_design.md §2.3.4, §2.2.6,
       DECISIONS 3, 4, 7)".
- AC6: revert restores the 817 baseline.

After commit, print baseline contract status line:
  "Baseline contract status: PASSED 842, SKIPPED 23, DESELECTED 15, FAILED 0"

Halt and report if AC1-AC6 do not all pass.

=== End paste ===
```

### F.3 Commit 8 — Public API surface assembly

Per §2.2 (excluding §2.2.5 / §2.2.6 which shipped in commit 1).

```
=== Paste into a fresh Claude Code session ===

You are implementing memopt Orchestrator v1 Commit 8 —
public API surface (start, stop, stats, register_policy).

Read first:
- docs/orchestrator_v1_design.md §2.2 (public API)
- docs/orchestrator_v1_design.md §3.1.7 (public API tests)
- docs/orchestrator_v1_implementation_prompt.md Section A and F

Files to edit:
- memopt/orchestrator/__init__.py
    Replace the empty stub with:
      - Module-level docstring.
      - Public symbols: start, stop, stats,
        register_policy, OrchestratorHandle,
        OrchestratorConfig.
      - Internal singleton holding the OrchestratorCoordinator.
      - start(*, config=None) -> OrchestratorHandle:
          idempotent; constructs the singleton once;
          calls coordinator.start(); returns handle.
      - stop() -> None: idempotent; coordinator.stop();
        drops singleton.
      - stats() -> dict: empty {} when not running; else
        delegates to TelemetryCollector.snapshot()
        merged with predictor.stats() and coordinator
        snapshot per §2.2.3.
      - register_policy(policy): forwards to PolicyEngine.
      - OrchestratorHandle: small class with stop() and
        is_alive() per §2.2.1.
- memopt/__init__.py
    Add `from . import orchestrator as orchestrator`
    so callers can write `memopt.orchestrator.start()`.
    Add to __all__.

Files to add:
- memopt/orchestrator/tests/test_public_api.py
    15 tests per §3.1.7 list. Includes a static
    import-shape snapshot test (test_substrate_api_unchanged)
    that asserts memopt.alloc, free, context, stats,
    observe, MemoryHandle, peek_handle still exist with
    unchanged signatures (C1).

Files NOT to edit (guardrail):
- See E.1 list.

Tests:
- REQUIRED-CI: all 15 tests in test_public_api.py.
- REQUIRED-LOCAL: test_peek_handle_safe_in_cuda_callback
  (@gpu) — runs on a rig only.

Acceptance criteria (AC1-AC6):
- AC1: `python -W error -c "import memopt;
       memopt.orchestrator.start(); memopt.orchestrator.stop()"`
       exits 0 and leaves no zombie threads.
- AC2: 14/14 REQUIRED-CI tests pass on Mac.
- AC3: Mac baseline: 842+14=856 PASSED, 23+1=24 SKIPPED,
       15 DESELECTED, 0 FAILED.
       The new SKIPPED is the @gpu peek-in-callback test.
- AC4: `git diff --name-only HEAD~1..HEAD` shows exactly
       three paths:
         memopt/orchestrator/__init__.py
         memopt/__init__.py
         memopt/orchestrator/tests/test_public_api.py
- AC5: Commit message: "Orchestrator v1 Commit 8: public
       API surface (start/stop/stats/register_policy) (per
       docs/orchestrator_v1_design.md §2.2)".
- AC6: revert restores the 842 baseline + the empty
       __init__.py stub.

After commit, print baseline contract status line:
  "Baseline contract status: PASSED 856, SKIPPED 24, DESELECTED 15, FAILED 0"

Halt and report if AC1-AC6 do not all pass.

=== End paste ===
```

### F.4 Commit 9 — Decision-verification + isolation tests

Per §3.1.9, §3.1.10, §3.1.11.

```
=== Paste into a fresh Claude Code session ===

You are implementing memopt Orchestrator v1 Commit 9 —
decision-verification + tenant-isolation tests.

Read first:
- docs/orchestrator_v1_design.md §2.1 DECISIONS 1 and 2
- docs/orchestrator_v1_design.md §2.5 G1–G4 + N3, N4
- docs/orchestrator_v1_design.md §3.1.9, §3.1.10, §3.1.11
- docs/orchestrator_v1_implementation_prompt.md Section A and F

Files to add:
- memopt/orchestrator/tests/test_decision_d1_greenfield.py
    5 tests per §3.1.9: static import-scan tests using
    the `ast` module to assert the orchestrator package
    does NOT import tier_manager, elastic_allocator,
    memory_governor, prefetch_engine, or call
    memopt_evict_to_target via ctypes.
- memopt/orchestrator/tests/test_decision_d2_advisory.py
    5 tests per §3.1.10: alloc-time advisory callback
    behaviour. Uses a stub Layer-2 policy that returns
    None / a hint / raises; verifies alloc() is unaffected
    in shape and never blocks.
- memopt/orchestrator/tests/test_tenant_isolation.py
    9 tests per §3.1.11: G1–G4 + N3 (warning) + N4 (queue
    drop under starvation).

Files to edit:
- (none)

Files NOT to edit (guardrail):
- See E.1 list.

Tests:
- REQUIRED-CI: all 19 tests.
- REQUIRED-LOCAL: (none).

Acceptance criteria (AC1-AC6):
- AC1: imports work.
- AC2: 19/19 tests pass.
- AC3: Mac baseline: 856+19=875 PASSED, 24 SKIPPED, 15
       DESELECTED, 0 FAILED.
- AC4: only the three added test files in the diff.
- AC5: Commit message: "Orchestrator v1 Commit 9:
       decision + tenant-isolation tests (per
       docs/orchestrator_v1_design.md §3.1.9–11)".
- AC6: revert restores the 856 baseline.

After commit, print baseline contract status line:
  "Baseline contract status: PASSED 875, SKIPPED 24, DESELECTED 15, FAILED 0"

Halt and report if AC1-AC6 do not all pass.

=== End paste ===
```

### F.5 Milestone 4 review gate

  - [ ] `memopt.orchestrator.start()` followed by
        `memopt.orchestrator.stop()` is round-trip clean
        (no zombie threads).
  - [ ] §3.3 Scenario A passes: 875 PASSED on Mac.
  - [ ] §3.3 Scenario B passes:
        `MEMOPT_AUTO_START_ORCHESTRATOR=1 pytest ...`
        also shows 875 PASSED with `decisions == 0`
        (DECISION 7).
  - [ ] DECISIONS 1, 2, 3, 4, 5, 6, 7, 8 each have ≥1
        passing test (verified via §3.1.14 cross-check).
  - [ ] Mac baseline: 875 PASSED, 24 SKIPPED, 15
        DESELECTED, 0 FAILED.

Per §3.5 ordering rationale: 6 (Telemetry) before 7
(Coordinator) because the coordinator increments telemetry
counters every cycle. 7 before 8 because `start()` constructs
and starts the coordinator. 9 after 8 because the decision-
verification tests use the public API to drive scenarios.

---

## Section G — Milestone 5: Bridge + Docs + Final (Commits 10–13)

### G.0 Milestone 5 promise

By the end of Milestone 5, the bridge test (§3.2) verifies
legacy parity, the perf microbench (§3.4) writes its first
JSON record, the threat model + user guide are written, the
version is bumped, and the CHANGELOG entry is in. Phase A is
complete and the §3.6 28-checkbox gate passes.

### G.1 Commit 10 — Bridge test (legacy parity, Phase A assertions)

Per §3.2.

```
=== Paste into a fresh Claude Code session ===

You are implementing memopt Orchestrator v1 Commit 10 —
Bridge test (legacy parity, Phase A assertions).

Read first:
- docs/orchestrator_v1_design.md §3.2 (bridge test spec)
- docs/orchestrator_v1_design.md §3.2.6 (OPTION Y order
  decision)
- docs/orchestrator_v1_implementation_prompt.md Section A and G

Files to add:
- tests/test_orchestrator_legacy_parity.py
    The bridge test described in §3.2:
      - 3 tenants (alice, bob, carol).
      - 200 deterministic allocations from
        random.Random(0xMEMOPT2) (or equivalent stable
        seed; choose a literal int and document it in the
        test).
      - 100 stream-bound (using the existing MockStream
        pattern from substrate test_events.py).
      - 5 lifecycle phases per §3.2.1.
      - Two runs:
          run A: orchestrator.start() then immediately
                 orchestrator.stop() — the orchestrator
                 was instantiated but is silent.
          run B: orchestrator.start() and left running
                 in observation mode (no MEMOPT_USE_
                 ORCHESTRATOR flag in v1.0).
      - Phase A assertions per §3.2.3 (exact equality of
        per-tenant byte counts and event-drop counters
        between runs A and B; decisions == 0 in run B).
      - The future-proofing comment block from §3.2.6
        verbatim.

Files to edit:
- (none)

Files NOT to edit (guardrail):
- See E.1 list.

Tests:
- REQUIRED-CI: the new bridge test (single test class
  with the 5 phase assertions).
- REQUIRED-LOCAL: (none).

Acceptance criteria (AC1-AC6):
- AC1: import works.
- AC2: bridge test passes on Mac.
- AC3: Mac baseline: 875+1=876 PASSED (the bridge test
       counts as one parameterized test class), 24
       SKIPPED, 15 DESELECTED, 0 FAILED.
- AC4: only tests/test_orchestrator_legacy_parity.py in
       the diff.
- AC5: Commit message: "Orchestrator v1 Commit 10: bridge
       test (legacy parity, Phase A assertions) (per
       docs/orchestrator_v1_design.md §3.2)".
- AC6: revert restores the 875 baseline.

After commit, print baseline contract status line:
  "Baseline contract status: PASSED 876, SKIPPED 24, DESELECTED 15, FAILED 0"

Halt and report if AC1-AC6 do not all pass.

=== End paste ===
```

### G.2 Commit 11 — Performance microbench (ordering only)

Per §3.4.

```
=== Paste into a fresh Claude Code session ===

You are implementing memopt Orchestrator v1 Commit 11 —
performance microbench (ordering-only assertions).

Read first:
- docs/orchestrator_v1_design.md §2.6 (perf targets)
- docs/orchestrator_v1_design.md §3.4 (microbench spec)
- docs/orchestrator_v1_implementation_prompt.md Section A and G

Files to add:
- memopt/orchestrator/tests/test_perf_microbench.py
    10 tests per §3.1.12, all marked @pytest.mark.perf.
    File structure per §3.4.1: 6 _bench_* helpers + 4
    ordering-assertion tests + 1 JSON-write test (10 in
    total per §3.1.12 list — 6 print-only timing tests +
    3 ordering tests + 1 JSON-write test).
    JSON written to /tmp/memopt-orchestrator-bench/<sha>.json.

Files to edit:
- (none)

Files NOT to edit (guardrail):
- See E.1 list.

Tests:
- REQUIRED-CI: (none — all 10 are @perf, OPTIONAL).
- OPTIONAL: all 10 @perf tests, run via `pytest -m perf`.

Acceptance criteria (AC1-AC6):
- AC1: import works.
- AC2: `pytest -m perf memopt/orchestrator/tests/
       test_perf_microbench.py` passes 10/10 on the Mac.
- AC3: Mac baseline (without `-m perf`): 876 PASSED, 24+10
       =34 SKIPPED, 15 DESELECTED, 0 FAILED.
       (The 10 perf tests SKIP by default per the @perf
        marker.)
- AC4: only the test file in the diff.
- AC5: Commit message: "Orchestrator v1 Commit 11: perf
       microbench (ordering only) (per
       docs/orchestrator_v1_design.md §3.4)".
- AC6: revert restores the 876 baseline.

After commit, print baseline contract status line:
  "Baseline contract status: PASSED 876, SKIPPED 34, DESELECTED 15, FAILED 0"

Halt and report if AC1-AC6 do not all pass.

=== End paste ===
```

### G.3 Commit 12 — Documentation: threat model, user guide, README

```
=== Paste into a fresh Claude Code session ===

You are implementing memopt Orchestrator v1 Commit 12 —
threat model + user guide + README pointer.

Read first:
- docs/orchestrator_v1_design.md §2.5 (G1–G4, N1–N4)
- docs/orchestrator_v1_design.md §2.2 (public API)
- docs/substrate_v1_threat_model.md (existing — for shape
  reference only)
- docs/substrate_v1_user_guide.md (existing — for shape
  reference only)
- docs/orchestrator_v1_implementation_prompt.md Section A and G

Files to add:
- docs/orchestrator_v1_threat_model.md
    Following the substrate threat model shape:
      - G1–G4 each with a "Mitigation" subsection citing
        the file:line(s) that enforce the guarantee.
      - N1–N4 each with a "Why out of scope" subsection
        and a "Mitigation outside this layer" pointer
        (e.g. NVIDIA CC mode, MIG, process namespaces).
      - One section per public API method noting any
        privilege expectation (e.g. register_policy is
        unrestricted in v1.0; production should gate it
        per N3).
- docs/orchestrator_v1_user_guide.md
    Following the substrate user guide shape:
      - Quickstart: import, start(), free, stop().
      - Worked example: register a custom Policy, observe
        decisions in stats().
      - Env-var reference (MEMOPT_ORCH_*, MEMOPT_EVICT_*,
        MEMOPT_AUTO_START_ORCHESTRATOR test knob).
      - "Phase A vs. Phase B" boundary.
      - Troubleshooting (queue drops, policy auto-
        unregister, peek_handle returning None).

Files to edit:
- README.md
    Add a section "Orchestrator (Layer 2)" with one
    paragraph + a pointer to docs/orchestrator_v1_user_guide.md
    and docs/orchestrator_v1_design.md.

Files NOT to edit (guardrail):
- Any code file. Docs only.

Tests:
- (none — docs only)

Acceptance criteria (AC1-AC6):
- AC1: docs/*.md parse as valid markdown.
- AC2: (no tests).
- AC3: Mac baseline unchanged: 876 PASSED, 34 SKIPPED,
       15 DESELECTED, 0 FAILED.
- AC4: `git diff --name-only HEAD~1..HEAD` shows exactly
       three paths:
         docs/orchestrator_v1_threat_model.md
         docs/orchestrator_v1_user_guide.md
         README.md
- AC5: Commit message: "Orchestrator v1 Commit 12: threat
       model + user guide + README pointer (per
       docs/orchestrator_v1_design.md §2.5, §2.2)".
- AC6: revert restores the prior README + drops the new
       docs.

After commit, print baseline contract status line:
  "Baseline contract status: PASSED 876, SKIPPED 34, DESELECTED 15, FAILED 0"

Halt and report if AC1-AC6 do not all pass.

=== End paste ===
```

### G.4 Commit 13 — Final regression + version bump

```
=== Paste into a fresh Claude Code session ===

You are implementing memopt Orchestrator v1 Commit 13 —
final regression run + version bump + CHANGELOG.

Read first:
- docs/orchestrator_v1_design.md §3.6 (Phase A acceptance
  gate, 28 checkboxes)
- docs/orchestrator_v1_design.md §3.7 (what Phase A does
  NOT deliver)
- CHANGELOG.md (existing — for shape reference)
- pyproject.toml (existing version)
- docs/orchestrator_v1_implementation_prompt.md Sections A,
  G, H

Files to edit:
- pyproject.toml
    Bump version: existing "1.1.0" → "1.2.0" (mirrors
    substrate's "1.0.0" → "1.1.0" Phase A bump).
- memopt/__init__.py
    Bump __version__ to "1.2.0".
- CHANGELOG.md
    Add a top entry for v1.2.0:
      - Title: "Orchestrator v1 release — Phase A complete
        (per docs/orchestrator_v1_design.md)"
      - Sub-bullets: AccessTracker, Predictor, PolicyEngine,
        Coordinator, TelemetryCollector, public API, bridge
        test, threat model, user guide.
      - "What Phase A does NOT deliver" sub-section
        verbatim from §3.7 (one paragraph per item).

Files NOT to edit (guardrail):
- All code files. The version bump is the only allowed
  code-adjacent change.

Tests:
- (none — release pass)

Final regression sweep — must run BEFORE the commit:
  1. `<§A.2 command>` — assert PASSED 876, SKIPPED 34,
     DESELECTED 15, FAILED 0.
  2. `MEMOPT_AUTO_START_ORCHESTRATOR=1 <§A.2 command>` —
     assert same numbers.
  3. `pytest -m perf memopt/orchestrator/tests/
     test_perf_microbench.py` — assert 10/10 PASSED.
  4. `python -c "import memopt; memopt.orchestrator.start();
     memopt.orchestrator.stop(); import threading;
     assert all(t.name != 'memopt-orchestrator-coordinator'
                for t in threading.enumerate())"` exits 0.
  5. `git log --diff-filter=M --name-only main..HEAD` —
     manually verify only the four allowed substrate paths
     (commit 1) are listed under memopt/substrate/.

Acceptance criteria (AC1-AC6):
- AC1: pyproject.toml + memopt/__init__.py both report
       1.2.0.
- AC2: (no tests).
- AC3: Mac baseline: 876 PASSED, 34 SKIPPED, 15 DESELECTED,
       0 FAILED.
- AC4: `git diff --name-only HEAD~1..HEAD` shows exactly
       three paths:
         pyproject.toml
         memopt/__init__.py
         CHANGELOG.md
- AC5: Commit message: "Orchestrator v1 release — Phase A
       complete (per docs/orchestrator_v1_design.md)".
- AC6: revert restores 1.1.0 and the prior CHANGELOG.

After commit, print baseline contract status line:
  "Baseline contract status: PASSED 876, SKIPPED 34, DESELECTED 15, FAILED 0"

Then verify the §H.1 28-checkbox gate by going through it
top to bottom; print the result of each checkbox. If any
checkbox is unchecked, HALT and report.

Tag the commit:
  `git tag orchestrator-v1.0.0`
  (Do not push the tag; the user reviews first.)

Halt and report if AC1-AC6 do not all pass.

=== End paste ===
```

### G.5 Milestone 5 review gate

  - [ ] All 28 checkboxes in §H.1 are checked.
  - [ ] §3.3 Scenarios A and B both show 876 PASSED.
  - [ ] Bridge test (§3.2.3) Phase A assertions all pass
        EXACT equality.
  - [ ] Threat model documents G1–G4, N1–N4 with mitigations.
  - [ ] User guide includes a worked example.
  - [ ] CHANGELOG entry includes "What Phase A does NOT
        deliver" pointing at §3.7.
  - [ ] `git tag orchestrator-v1.0.0` created (not pushed).

---

## Section H — Final acceptance criteria

### H.1 Phase A completion gate (28 checkboxes — verbatim from §3.6)

Phase A is complete when ALL of the following hold:

```
[ ] Step Zero report exists at
    docs/orchestrator_v1_step_zero_report.md and shows PASS
    or DEGRADED-with-fallback for every S0.1–S0.7 item.
[ ] All HARD-BLOCK items (S0.3 / TV3, S0.4 / TV4) are PASS.
[ ] memopt/orchestrator/ contains:
        __init__.py
        access.py
        predict.py
        policy.py
        coordinator.py
        telemetry.py
        config.py
        tests/__init__.py
        tests/test_access_tracker.py
        tests/test_predictor.py
        tests/test_policy_engine.py
        tests/test_coordinator.py
        tests/test_telemetry.py
        tests/test_config.py
        tests/test_public_api.py
        tests/test_event_emission.py
        tests/test_decision_d1_greenfield.py
        tests/test_decision_d2_advisory.py
        tests/test_tenant_isolation.py
        tests/test_perf_microbench.py
[ ] memopt/substrate/manager.py has the new peek_handle
    method and the internal _emit_orchestrator_event method
    (the only commit-1 substrate edits).
[ ] memopt/substrate/__init__.py re-exports peek_handle.
[ ] memopt/__init__.py re-exports peek_handle and exposes
    the orchestrator subpackage.
[ ] memopt/substrate/tests/test_peek_handle.py exists.
[ ] tests/test_orchestrator_legacy_parity.py exists and
    passes the §3.2 Phase A assertions.
[ ] docs/orchestrator_v1_design.md exists (Phases 1–3).
[ ] docs/orchestrator_v1_step_zero_report.md exists.
[ ] docs/orchestrator_v1_threat_model.md exists.
[ ] docs/orchestrator_v1_user_guide.md exists.
[ ] docs/orchestrator_v1_implementation_prompt.md exists
    (this file).
[ ] CHANGELOG.md has the orchestrator v1 entry.
[ ] pyproject.toml version is bumped (1.1.0 → 1.2.0).
[ ] No existing module under memopt/vmm/, memopt/cluster/,
    memopt/serving/, memopt/api/, memopt/control_plane/,
    memopt/operator/, memopt/canary/, memopt/daemon/,
    memopt/profiler/, memopt/kernels/ was edited.
[ ] No existing file under memopt/substrate/ was edited
    EXCEPT memopt/substrate/manager.py and
    memopt/substrate/__init__.py in commit 1.
[ ] Mac baseline (§1.6.1): 752 originally passing tests STILL
    pass; orchestrator tests add to the PASSED count;
    SKIPPED stays at 22 (+ orchestrator @gpu / @perf adds);
    DESELECTED stays at 15; FAILED == 0.
[ ] The substrate's existing GPU-rig baseline (753 PASSED on
    the substrate's @gpu CI) is preserved (commit 1 changes
    do not regress any substrate test).
[ ] Bridge test (§3.2.3) Phase A assertions all pass:
    exact equality on per-tenant byte counts and event-
    drop counters; decisions == 0.
[ ] Decision-to-test cross-check (§3.1.14): every D1–D8
    has at least one PASSING test verifying the chosen
    behaviour.
[ ] Gap-to-test cross-check (§3.1.13): every O1–O11 has at
    least one PASSING test.
[ ] Public-API cross-check (§2.2): every public symbol
    (start, stop, stats, register_policy, peek_handle,
    OrchestratorHandle, OrchestratorConfig) has at least
    one PASSING test in test_public_api.py or
    test_config.py.
[ ] Performance microbench (§3.4) writes a JSON record
    at /tmp/memopt-orchestrator-bench/<sha>.json on at
    least one rig; the ordering assertions pass.
[ ] `python -c "import memopt; memopt.orchestrator.start();
    memopt.orchestrator.stop()"` runs without error and
    leaves no zombie threads.
[ ] AC4 audit: `git diff --name-only main..HEAD` shows
    only paths under memopt/orchestrator/,
    memopt/substrate/__init__.py, memopt/substrate/manager.py,
    memopt/substrate/tests/test_peek_handle.py,
    memopt/__init__.py, tests/test_orchestrator_legacy_parity.py,
    docs/orchestrator_v1_*, CHANGELOG.md, pyproject.toml.
[ ] Phase A version tag created on `main` (e.g.
    `orchestrator-v1.0.0`).
[ ] CHANGELOG entry includes a "What Phase A does NOT
    deliver" section pointing at §3.7.
[ ] User guide has a worked example: start() → register a
    policy → observe a decision in stats() → stop().
[ ] Threat model enumerates §2.5 G1–G4 and N1–N4 with
    mitigations.
[ ] No `TODO` or `FIXME` introduced in commits 1–13 that
    is not paired with an issue link.
```

### H.2 What Phase A does NOT deliver

Verbatim from §3.7:

  - **Phase B** (`MEMOPT_USE_ORCHESTRATOR=1` flag flip):
    advisory `placement="auto"` resolution + C++
    step_boundary deferral. Separate engagement; gating
    in §2.7 Phase B.
  - **Phase C** (default-ON): three months of Phase B
    soak required. Separate engagement.
  - **Phase D** (remove legacy paths): deletes
    `TierManager._ensure_capacity`, `ElasticAllocator`,
    `MemoryGovernor`, the C++ direct-eviction path, and
    the VMM-internal Markov chain in `PrefetchEngine`.
    Separate engagement.
  - **Layer 3** (ML-driven prefetch oracle): separate
    package `memopt-prefetch-oracle`; v1.0 stays
    classical Markov.
  - **Layer 4** (declarative policy DSL): separate package
    `memopt-policy-dsl`; v1.0 ships only the `Policy`
    protocol.
  - **Layer 5** (federated orchestration across nodes):
    separate layer; the §2.3.2 predictor explicitly does
    not gossip across processes (DECISION 7 trade-off).
  - **vLLM integration.** `memopt-vllm` (separate package).
  - **Custom CUDA kernels for the predictor.** v1.0 is
    pure Python (DECISION 6 trade-off).
  - **Persistence of learned patterns across restarts.**
    Predictor starts cold every process.
  - **Cost / FinOps integration.** Lives in `memopt-trust`.
  - **Confidential computing / attestation.** Out of scope.

### H.3 Per-commit test bucket reference

```
Commit  Buckets (REQUIRED-CI / REQUIRED-LOCAL / OPTIONAL)
0       —                                / —                                      / —
1       test_peek_handle.py (10)         / —                                      / —
2       test_access_tracker.py (13)      / —                                      / —
3       test_predictor.py (13)           / —                                      / —
4       test_config.py (8)               / —                                      / —
5       test_policy_engine.py (12)       / —                                      / —
6       test_telemetry.py (9)            / —                                      / —
7       test_coordinator.py (14),        / test_migrate_va_preserved (1, @gpu)    / —
        test_event_emission.py (11)
8       test_public_api.py (14)          / test_peek_handle_safe_in_cuda_callback / —
                                            (1, @gpu)
9       test_decision_d1_greenfield.py (5),  —                                    / —
        test_decision_d2_advisory.py (5),
        test_tenant_isolation.py (9)
10      test_orchestrator_legacy_parity.py   —                                    / —
        (1)
11      —                                / —                                      / test_perf_microbench.py
                                                                                     (10, @perf)
12      —                                / —                                      / —
13      —                                / —                                      / —

Cumulative Mac baseline (REQUIRED-CI only, @gpu and @perf SKIPPED):
0:  752  /  1:  762  /  2:  775  /  3:  788  /  4:  796  /  5:  808
6:  817  /  7:  842  /  8:  856  /  9:  875  /  10: 876  /  11: 876
12: 876  /  13: 876
```

### H.4 Long-term contract (20-year horizon)

The following items are the long-term invariants of the
Layer-1 + Layer-2 stack. They are not subject to revision in
the orchestrator's lifetime (until v2.0 / v3.0 explicitly
revises them with a new design doc):

  - **The seven primitives.** `reserve_va`, `create_physical`,
    `map`, `set_access`, `unmap`, `release_physical`,
    `free_va` (substrate `BackendStrategy` ABC). Layer 2 NEVER
    calls them directly; it goes through `memopt.alloc()`
    (boundary §1.5.3 surface S2).
  - **`MemoryHandle` field stability.** `(handle_id,
    size_bytes, tenant, tag, placement, stream, backend_name,
    created_at, ttl_seconds, hint)` are the public-facing
    fields per `memopt/substrate/handle.py:64-83`. Layer 2's
    `peek_handle` returns these unchanged. Adding a public
    field to `MemoryHandle` requires a substrate Phase 2
    redesign; removing one is forbidden.
  - **The five event kinds.** `("alloc", "free", "evict",
    "promote", "migrate")` (substrate `events.py:33`). Adding
    a sixth kind requires substrate change + Layer 2 design
    doc revision. Layer 1 produces alloc/free; Layer 2
    produces evict/promote/migrate. The producer split is
    pinned by §2.2.6.
  - **G1–G4 isolation guarantees** (Layer 1 §2.6 + Layer 2
    §2.5). Per-tenant arenas, per-tenant access histories,
    per-tenant predictor keys, admin-token gating on cross-
    tenant aggregates. These are user-visible promises.
  - **`MEMOPT_USE_ORCHESTRATOR` flag stability.** The flag
    name is the public Phase B / C interface. Renaming it
    is a breaking change; it must persist through v2.0 even
    after Phase C makes it the default (then it is the
    opt-out flag, not the opt-in flag).
  - **Bridge test path stability.**
    `tests/test_orchestrator_legacy_parity.py` is the
    long-term parity contract. Its assertions in §3.2.3
    (Phase A) and §3.2.4 (Phase B) must continue to be
    asserted as long as the legacy paths in §1.2 exist.
    The test file path itself is part of the contract;
    moving it requires a CHANGELOG entry.
  - **`DECISION 4 migrate semantics`.** A `migrate` event
    preserves `va` and `handle_id`; `free`+`alloc` does not.
    This distinction is pinned for as long as backends with
    in-place remap support exist. A backend that cannot
    preserve `va` declines the migrate path.
  - **OPTION Y order non-assertion** (§3.2.6). The bridge
    test does NOT assert the order of allocations / evictions
    / decisions. Future tests must inherit this discipline;
    relaxing it requires re-running the regression-net audit
    documented in substrate v1 §3.3.
  - **Phase B `MEMOPT_AUTO_START_ORCHESTRATOR` is NOT
    public.** It is a Phase-A-only test knob (§A.2). It is
    removed when Phase B introduces the public
    `MEMOPT_USE_ORCHESTRATOR=1`. Anyone reading this in 20
    years and finding `MEMOPT_AUTO_START_ORCHESTRATOR` in
    code should delete it.

---

*End of Phase 4 implementation prompt.*
