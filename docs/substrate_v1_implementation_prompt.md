# memopt Substrate v1 — Implementation Prompt

| Field           | Value                                                              |
| --------------- | ------------------------------------------------------------------ |
| Status          | Phase 4 prompt drafted. Ready for execution in fresh sessions.     |
| Authoritative   | docs/substrate_v1_design.md (Phases 1–3)                           |
| Repo            | /Users/lachumanbasnet/Personal/Sophisticates/MEMOPT/memopt         |
| Baseline commit | 6f0678d on `main`                                                  |
| Drafted         | 2026-04-29                                                         |

This document is the operating manual for implementing Layer 1.
It is sequential: Step Zero, then 15 commits in 5 milestones.
Each commit prompt is self-contained — a future contributor with
this file and `docs/substrate_v1_design.md` can rebuild Layer 1
from scratch without seeing the discovery / design conversations.

The implementation runs as a series of fresh Claude Code
invocations. After every commit, the user reviews the change.
After every milestone, the user does a deeper review (reads each
new file end-to-end, verifies the milestone's promise).

Sections:

  A.  Baseline + setup
  B.  Step Zero verification (Commit 0)
  C.  Milestone 1 — Verification + G12 fix
  D.  Milestone 2 — Handle + base + CPU
  E.  Milestone 3 — Real backends
  F.  Milestone 4 — Arena + Stream + Events
  G.  Milestone 5 — Manager + Bridge + Docs + Final
  H.  Final acceptance criteria

---

## Section A — Baseline + setup

### A.1 Baseline contract

```
BASELINE (commit 6f0678d on main):
    PASSED:     682
    SKIPPED:     19
    DESELECTED:  11
    FAILED:       0

ALLOWED DELTAS during Phase A:
    PASSED      may grow (new substrate tests)
    SKIPPED     may grow (new hardware-gated tests that
                legitimately skip on macOS / no-GPU CI)
    DESELECTED  unchanged
    FAILED      must always be 0

DISALLOWED DELTAS (any of these halts the sequence):
    Any test that previously PASSED is now SKIPPED.
    Any test that previously PASSED is now FAILED.
    Any test in the original 682 changes assertion text,
        expected value, or behaviour.
```

After every commit, Claude Code MUST print:

```
Baseline contract status: PASSED 682+M, SKIPPED 19+K, DESELECTED 11, FAILED 0
```

with `M` = new substrate tests added in this commit (cumulative
since baseline), `K` = new hardware-skipped tests (cumulative).
The 682 must always be a subset of the new PASSED count. If the
status would print anything other than `FAILED 0` or with the
original 682 not all passing, halt and report.

### A.2 Regression command

```
PYTHONPATH=. python3 -m pytest tests/ memopt/ -q -p no:cacheprovider \
    -k "not gpu and not cuda" \
    --ignore=memopt/vmm/tests/test_cuda_vmm_smoke.py \
    --ignore=memopt/vmm/tests/test_torch_allocator_sanity.py \
    2>&1 | tail -3
```

Expected last line on a passing baseline:

```
682 passed, 19 skipped, 11 deselected, 5 warnings in <N>s
```

Any deviation other than the allowed deltas above halts.

### A.3 Test categorization legend (Refinement 2)

Every test in §3.1 of the design doc belongs to exactly ONE
bucket below. Each commit prompt lists which tests in each
bucket apply.

```
REQUIRED-CI
    Must pass in CI on every commit. Runs on macOS + Linux
    without a GPU. No @gpu, @gpu_amd, @cxl, @perf markers.
    Failure here halts the commit.

REQUIRED-LOCAL
    Must pass on the developer's GPU rig. Includes @gpu tests
    (NVIDIA hardware required). Failure here halts the commit
    when run by a developer who has the rig; CI does not gate
    on it.

OPTIONAL
    Skipped without specific hardware. @gpu_amd, @cxl, @perf
    tests fall here. CI does not gate; developer rigs gate
    only when the hardware is present. Failure on absent
    hardware = skip, not fail.
```

The complete bucket assignments are in Section H.3 (one table
per commit). Each commit prompt also restates its bucket subset.

### A.4 Dev environment assumptions

  - Repo: `/Users/lachumanbasnet/Personal/Sophisticates/MEMOPT/memopt`
  - Python: 3.10–3.13 (pyproject.toml `requires-python = ">=3.8"`,
    but baseline tests use 3.12)
  - Working tree must be clean before each commit (git status
    has no uncommitted changes other than the commit's own
    edits).
  - `pip install -e .` installs the local package; no rebuild
    needed for pure-Python edits.
  - C/C++ rebuild for Commit 1 (G12) requires:
    `cd csrc/cuda_vmm && mkdir -p build && cd build && cmake .. && make`
    on a CUDA-equipped host. Mac dev does not rebuild — ship
    Commit 1 from a Linux+CUDA rig, run regression on Mac after.

### A.5 Authoritative design doc

Every commit prompt cites sections of
`docs/substrate_v1_design.md`. That doc is authoritative for
behaviour, signatures, and design rationale. If a commit prompt
and the design doc disagree, the design doc wins; halt the
commit and update the prompt.

### A.6 Reverting

Each commit is independently revertible. A failed commit:

  1. `git reset --hard HEAD~1` rolls back the failed commit.
  2. The next commit can resume from the previous green state.
  3. NEVER `git push --force` to fix a bad commit; revert and
     re-commit.

---

## Section B — Step Zero verification (Commit 0)

### B.1 Goal

Produce `docs/substrate_v1_step_zero_report.md` with the result
of every TODO-VERIFY item from §3.0 of the design doc. Phase A
implementation does NOT begin until this report shows every
item passed (or fell back to a documented degradation).

### B.2 The 8 items

For each, run the verification method in design doc §3.0
(S0.1 through S0.8) and record:

  - `STATUS`: VERIFIED | DEGRADED | FAILED
  - `EVIDENCE`: command output, URL+commit, or hardware probe
    result that justifies the status
  - `IMPACT`: which design section is affected if degraded
  - `NEXT STEP`: nothing | apply documented fallback (cite
    design doc §3.0 fallback paragraph) | halt for redesign

### B.3 Report format

```markdown
# Substrate v1 — Step Zero Verification Report

| Field        | Value                                  |
| ------------ | -------------------------------------- |
| Run by       | <name>                                 |
| Run date     | <YYYY-MM-DD>                           |
| Test rigs    | <hostnames or descriptions>            |
| CUDA version | <output of nvcc --version, or N/A>     |
| ROCm version | <output of /opt/rocm/bin/rocminfo, or> |
| Python       | <python3 --version>                    |

## S0.1 — CUDA fabric handle
STATUS:   <VERIFIED | DEGRADED | FAILED>
EVIDENCE: <inline output or paste of probe result>
IMPACT:   <none | design doc §X.Y must update | halt>
NEXT:     <action>

## S0.2 — HIP VMM capability
... (same template) ...

(repeat for S0.3 through S0.8)

## Summary
<one-line statement: PASS | DEGRADED with N items | HALT>
```

### B.4 S0.3 hard-block redesign protocol (Refinement 4)

S0.3 (PyTorch CCA semantics) is the only item that hard-blocks
Phase A on mismatch. The StreamRegistry is the foundational
guarantee of the substrate; even a "small" redesign without
user review can introduce silent semantic drift that fails the
Milestone 5 bridge test. ALWAYS halt and report on any S0.3
mismatch — the cost of halting (5 minutes of user review) is
always less than the cost of a StreamRegistry semantics bug.

If S0.3 verification finds memopt's StreamRegistry semantic in
design doc §2.5 does NOT match the current PyTorch CCA semantic
exactly:

```
STEP A — Read CCA actual.
    Read c10/cuda/CUDACachingAllocator.cpp at the current
    PyTorch main. Document what CCA actually does for the
    affected case.

STEP B — HALT. Do NOT proceed to Phase A.

    Print a report to the user with:
        - What CCA actually does (with file:line citations).
        - What memopt design doc §2.5 says.
        - The specific delta.
        - Proposed redesign sketch (no code).
        - Estimated effort.
        - Recommended path forward.

    Wait for user decision before any further action.
```

### B.5 Commit 0 prompt

> ## Prompt for Commit 0 (paste into a fresh Claude Code session)
>
> You are implementing memopt Substrate v1 Commit 0 (Step Zero
> verification).
>
> Read first:
> - `docs/substrate_v1_design.md` §3.0 (entire section)
> - `docs/substrate_v1_implementation_prompt.md` Sections A and B
>
> Tasks (in order):
>
> 1. Open or create `docs/substrate_v1_step_zero_report.md`.
> 2. For each item S0.1 through S0.8 in design §3.0, run the
>    "Method" specified there. Record `STATUS`, `EVIDENCE`,
>    `IMPACT`, `NEXT STEP` per the template in B.3.
> 3. If any item lands in `FAILED` and is the S0.3 hard block,
>    follow the redesign protocol in B.4. Otherwise apply the
>    documented fallback from §3.0 of the design doc and record
>    DEGRADED with the fallback path.
> 4. Run the regression command from A.2. Confirm the baseline
>    contract: 682 passed, 19 skipped, 11 deselected, 0 failed.
> 5. Print the baseline contract status line per A.1:
>    `Baseline contract status: PASSED 682+0, SKIPPED 19+0,
>    DESELECTED 11, FAILED 0`
> 6. `git add docs/substrate_v1_step_zero_report.md && git
>    commit -m "Substrate v1 Commit 0: Step Zero verification
>    (per design §3.0)"`.
>
> Acceptance:
> - The report contains an entry for every S0.1–S0.8 item.
> - Every entry has STATUS, EVIDENCE, IMPACT, NEXT STEP.
> - The summary line is one of: PASS / DEGRADED with N items /
>   HALT.
> - The baseline contract status prints exactly the line above.
>
> Test buckets for Commit 0:
> - REQUIRED-CI: regression command in A.2 must show baseline.
> - REQUIRED-LOCAL: none.
> - OPTIONAL: none.
>
> Do NOT proceed to Commit 1 if Step Zero summary is HALT.

---

## Section C — Milestone 1: Verification + G12 fix

### C.0 Milestone 1 promise

> After Milestone 1, Step Zero is verified (or has documented
> fallbacks for every degraded item) AND the C VMM allocator
> supports pools larger than 16 GiB on a real big-HBM rig. The
> existing libmemopt_vmm.so ABI is preserved; legacy ctypes
> consumers compile unchanged.

### C.1 Commit 0 — Step Zero (refer to Section B)

See Section B above.

### C.2 Commit 1 — csrc G12 fix (dynamic page cap)

> ## Prompt for Commit 1 (fresh Claude Code session)
>
> You are implementing memopt Substrate v1 Commit 1 — the C++
> G12 fix (dynamic page cap). The 16 GiB cap from
> `MEMOPT_MAX_PAGES = 8192` must be removed by making the page
> array dynamic, sized at allocator construction.
>
> Read first:
> - `docs/substrate_v1_design.md` §2.12 (exact line changes)
> - `docs/substrate_v1_design.md` §1.2 (current C++ surface)
>
> Files to edit:
> - `csrc/cuda_vmm/vmm_allocator.h`     — remove the
>   `MEMOPT_MAX_PAGES` define per design §2.12
> - `csrc/cuda_vmm/vmm_allocator.cpp`   — replace inline pages
>   array with heap-allocated `MemoptPage*`, add `n_pages_cap`
>   field, allocate in `memopt_allocator_create`, free in
>   `memopt_allocator_destroy`. Update lines 245, 320, 584.
>
> Files to add:
> - `csrc/cuda_vmm/test_vmm_allocator.cpp` — extend with
>   `test_dynamic_capacity_above_legacy_cap` per design §2.12.
>   Skip when `cuda::mem_get_info().free < 34 GiB`.
>
> Files NOT to edit (guardrail):
> - Any other file under `memopt/`, `csrc/core/`,
>   `csrc/cuda_backend/`, `csrc/paged/`, `csrc/hooks/`,
>   `csrc/transport/`, `csrc/rocm/`, `csrc/simd/`, or `tests/`.
>
> Tests:
> - REQUIRED-LOCAL only:
>   `csrc/cuda_vmm/test_vmm_allocator.cpp::
>    test_dynamic_capacity_above_legacy_cap` on a >= 80 GiB
>    HBM rig. Allocate 16385 × 2 MiB pages; free; destroy.
> - REQUIRED-CI: regression command in A.2 must still print
>   682 passed (no Python tests changed).
> - OPTIONAL: none.
>
> ABI compatibility check (every developer rig with the .so
> built):
> - `python3 -c "from memopt.vmm.cuda_vmm import
>   CUDAVMMAllocator; print('ok')"` must succeed without
>   editing `memopt/vmm/cuda_vmm.py`. The ctypes signatures
>   must remain identical.
>
> Acceptance criteria (AC1–AC6 per design §3.5):
> - AC1: cmake + make under `csrc/cuda_vmm/build/` succeed
>   with no new warnings.
> - AC2: the new test passes on a big-HBM rig OR is correctly
>   skipped on smaller rigs with a clear reason string.
> - AC3: regression command in A.2 prints the baseline.
> - AC4: only the three files listed above were touched.
> - AC5: commit message format:
>   "Substrate v1 Commit 1: G12 fix — dynamic page cap
>   (per design §2.12)".
> - AC6: revertible: `git revert HEAD` restores the static
>   `MEMOPT_MAX_PAGES`-based allocator and the test passes
>   again on the previous (capped) build.
>
> After commit:
> - Print baseline contract status:
>   `Baseline contract status: PASSED 682+0, SKIPPED 19+1,
>   DESELECTED 11, FAILED 0`
>   (the +1 SKIPPED is the new C++ test on rigs without 32 GiB
>    free HBM; on a big-HBM rig it counts in PASSED).
>
> Halt and report if AC1–AC6 do not all pass.

### C.3 Milestone 1 review gate (Refinement 1)

User reviews:

  - Read `docs/substrate_v1_step_zero_report.md` end-to-end.
    Confirm every S0.x item has STATUS, EVIDENCE, IMPACT, NEXT
    STEP.
  - Read `csrc/cuda_vmm/vmm_allocator.h` and
    `csrc/cuda_vmm/vmm_allocator.cpp` end-to-end. Verify only
    the lines listed in design §2.12 changed.
  - Read `csrc/cuda_vmm/test_vmm_allocator.cpp` end-to-end.
    Verify the skip condition is correct and the test
    exercises 16385 pages.
  - Verify AC1–AC6 on Commits 0 and 1.
  - Verify Milestone 1 promise: Step Zero verified AND
    G12 fix passes on at least one big-HBM rig.

Sign-off: user approves, then Milestone 2 begins. If anything
is wrong, revert the offending commit and re-prompt.

---

## Section D — Milestone 2: Handle + base + CPU

### D.0 Milestone 2 promise

> After Milestone 2, the substrate scaffold compiles and the
> CPU fallback backend works end-to-end. `memopt.alloc(...,
> placement="dram")` returns a usable `MemoryHandle` on a
> macOS / no-GPU CI host. No real GPU backends yet; no
> tenant arena yet; no stream registry yet.

### D.1 Commit 2 — substrate shell + handle dataclass

> ## Prompt for Commit 2 (fresh session)
>
> You are implementing memopt Substrate v1 Commit 2: the
> top-level substrate package shell, the `MemoryHandle`
> dataclass, and `test_handle.py`.
>
> Read first:
> - `docs/substrate_v1_design.md` §2.1 (Public API surface)
> - `docs/substrate_v1_design.md` §3.1 (test_handle.py spec)
>
> Files to add:
> - `memopt/substrate/__init__.py` — re-exports `alloc`, `free`,
>   `context`, `stats`, `observe`, `MemoryHandle`. In Commit 2
>   these are stubs that raise `NotImplementedError("substrate
>   not assembled until Commit 12")`. The point of this commit
>   is the dataclass + the package shell.
> - `memopt/substrate/handle.py` — defines `MemoryHandle`
>   dataclass per design §2.1. Fields: handle_id, size_bytes,
>   tenant, tag, placement, stream, backend_name, created_at,
>   ttl_seconds, hint, plus internal fields prefixed _va,
>   _physical, _backend, _state. Methods: read, write,
>   as_tensor, as_numpy, stats, free, __enter__, __exit__.
>   read/write/as_tensor/as_numpy raise
>   `RuntimeError("handle is freed")` when state == "released",
>   `NotImplementedError("backend assembly pending Commit 12")`
>   when state == "live" (since no backend is connected yet).
>   The dataclass IS frozen for the public-facing fields; the
>   internal fields use `field(repr=False)`.
> - `memopt/substrate/tests/__init__.py` — empty.
> - `memopt/substrate/tests/test_handle.py` — every test
>   listed in design §3.1. The tests that depend on real
>   backend behaviour (`test_alloc_returns_handle_with_metadata`
>   etc.) construct a `MemoryHandle` directly with a mock
>   backend object, so they don't need a real backend.
>
> Files NOT to edit:
> - Anything outside `memopt/substrate/`.
>
> Tests:
> - REQUIRED-CI: every test in `memopt/substrate/tests/
>   test_handle.py`. They run on macOS + Linux without GPU.
>   `test_handle_ref_count_with_as_tensor` uses CPU torch
>   only; if torch is absent, skip with a reason string.
> - REQUIRED-LOCAL: none.
> - OPTIONAL: none.
>
> Acceptance:
> - AC1: `python3 -c "from memopt.substrate import
>   MemoryHandle"` succeeds.
> - AC2: pytest on `memopt/substrate/tests/test_handle.py`
>   passes 100 % (count: at least 8 tests per design §3.1).
> - AC3: regression command (A.2) still prints baseline 682.
> - AC4: only `memopt/substrate/` files were touched.
> - AC5: commit message:
>   "Substrate v1 Commit 2: handle dataclass + substrate
>   shell (per design §2.1, §3.1)".
> - AC6: revertible.
>
> After commit, print baseline:
> `Baseline contract status: PASSED 682+8, SKIPPED 19+1,
> DESELECTED 11, FAILED 0`
> (number of new tests is the actual count from the run.)
>
> Halt if any AC fails.

### D.2 Commit 3 — backend protocol (ABC)

> ## Prompt for Commit 3 (fresh session)
>
> You are implementing memopt Substrate v1 Commit 3: the
> abstract base class for backends.
>
> Read first:
> - `docs/substrate_v1_design.md` §2.3 (BackendStrategy ABC)
> - `docs/substrate_v1_design.md` §3.1 (test_backend_protocol.py)
>
> Files to add:
> - `memopt/substrate/backends/__init__.py` — empty (real
>   exports added when backends land in Commits 4–8).
> - `memopt/substrate/backends/base.py` — defines `PhysLoc`
>   enum, `PhysHandle` dataclass, and `BackendStrategy` ABC
>   exactly per design §2.3. Every method is `@abstractmethod`.
>   The class has class-level `BACKEND_NAME: ClassVar[str]`
>   and `IS_REAL: ClassVar[bool]` annotations.
> - `memopt/substrate/tests/test_backend_protocol.py` per
>   design §3.1.
>
> Tests:
> - REQUIRED-CI: every test in `test_backend_protocol.py`.
>   The "all backends implement seven primitives" test runs
>   with an empty backend registry in Commit 3 (no concrete
>   backends yet); it asserts the ABC has the seven
>   abstractmethods and that `IS_REAL` and `BACKEND_NAME` are
>   ClassVar. The "is_available is pure" test mocks a
>   subclass and confirms calling its `is_available()` does
>   not import torch / hip / etc.
> - REQUIRED-LOCAL: none.
> - OPTIONAL: none.
>
> Acceptance:
> - AC1–AC6 as Commit 2.
> - Commit message: "Substrate v1 Commit 3: backend ABC
>   (per design §2.3)".
> - Baseline status: `682+M, 19+1, 11, 0` with M = total
>   substrate tests so far (from Commits 2 and 3 combined).

### D.3 Commit 4 — CPU fallback backend

> ## Prompt for Commit 4 (fresh session)
>
> You are implementing memopt Substrate v1 Commit 4: the
> CPU fallback backend.
>
> Read first:
> - `docs/substrate_v1_design.md` §2.4 CPUFallbackBackend
> - `docs/substrate_v1_design.md` §3.1 test_backend_cpu.py
>
> Files to add:
> - `memopt/substrate/backends/cpu_fallback.py` — implements
>   `BackendStrategy` per design §2.4. Reserve_va via mmap
>   PROT_NONE. Create_physical via anonymous mmap. Map and
>   unmap follow the simplification documented in §2.4 (map
>   is no-op when va == ph.raw; document this divergence in a
>   docstring at the top of the file). Granularity returns
>   sysconf(_SC_PAGESIZE) by default, attempts MAP_HUGETLB on
>   `MEMOPT_FORCE_HUGE=1`. Set_access is a no-op.
>   Export_fabric_handle returns None.
> - `memopt/substrate/backends/__init__.py` — re-export
>   `CPUFallbackBackend`.
> - `memopt/substrate/tests/test_backend_cpu.py` — every
>   test from design §3.1.
>
> Tests:
> - REQUIRED-CI: all 5 tests in `test_backend_cpu.py`.
> - REQUIRED-LOCAL: none additional.
> - OPTIONAL: none.
>
> Acceptance:
> - AC1: `from memopt.substrate.backends import
>   CPUFallbackBackend; CPUFallbackBackend().is_available()`
>   returns True.
> - AC2: pytest passes for CPU backend tests.
> - AC3: baseline regression still 682.
> - AC4: only `memopt/substrate/` touched.
> - AC5: commit message format consistent.
> - AC6: revertible.
>
> Print baseline status with cumulative substrate test count.

### D.4 Milestone 2 review gate

User reads end-to-end:

  - `memopt/substrate/__init__.py`
  - `memopt/substrate/handle.py`
  - `memopt/substrate/backends/__init__.py`
  - `memopt/substrate/backends/base.py`
  - `memopt/substrate/backends/cpu_fallback.py`
  - All four new test files.

Verifies:

  - AC1–AC6 on Commits 2, 3, 4 individually.
  - Milestone 2 promise: substrate compiles, CPU fallback
    works on macOS, baseline still 682 passing.
  - The `memopt/__init__.py` of the existing package was NOT
    edited (we add re-exports later, in Commit 12 where the
    manager assembles).

Sign-off → Milestone 3.

---

## Section E — Milestone 3: Real backends

### E.0 Milestone 3 promise

> After Milestone 3, all four real-or-stub backends are
> implemented. CUDA passes on the @gpu rig. HIP passes
> on the @gpu_amd rig OR is honestly stubbed per S0.2's
> fallback. Level Zero stub raises NotImplementedError
> from non-is_available methods. CXL detects a CXL node
> on the @cxl rig OR ships behind the MEMOPT_CXL_NODES
> env var per S0.5's fallback.

### E.1 Commit 5 — CUDA VMM backend

> ## Prompt for Commit 5 (fresh session)
>
> You are implementing memopt Substrate v1 Commit 5: the
> CUDA VMM backend that wraps libmemopt_vmm.so.
>
> Read first:
> - `docs/substrate_v1_design.md` §2.4 CUDABackend
> - `docs/substrate_v1_design.md` §3.1 test_backend_cuda.py
> - `docs/substrate_v1_design.md` §1.2 (existing C ABI of
>   libmemopt_vmm.so — the seven primitives are already there
>   in spirit; this commit exposes them as the BackendStrategy)
> - The Step Zero S0.1 entry — fabric handle support
>   classification.
>
> Files to add:
> - `memopt/substrate/backends/cuda_vmm.py` — concrete
>   `BackendStrategy` subclass `CUDABackend`. Wraps the
>   existing `memopt.vmm.cuda_vmm.CUDAVMMAllocator` and the
>   ctypes-loaded libmemopt_vmm.so for the seven primitives.
>   Adds the eighth (`export_fabric_handle`) using
>   `cuMemExportToShareableHandle` via ctypes — gated on
>   CUDA 12.4 detection at runtime. is_available() returns
>   True iff `torch.cuda.is_available()` AND the .so loads.
> - `memopt/substrate/tests/test_backend_cuda.py` — all
>   tests from design §3.1, marked `@pytest.mark.gpu`.
>
> Files NOT to edit:
> - `memopt/vmm/cuda_vmm.py` (still authoritative for the
>   legacy path)
> - `csrc/cuda_vmm/*` (not touched in this commit)
>
> Tests:
> - REQUIRED-CI: regression command (A.2) still 682 passing,
>   substrate test count stays at the cumulative level (the
>   new tests skip with "no CUDA" reason on macOS / no-GPU
>   CI). Confirm SKIPPED count grew by the @gpu test count.
> - REQUIRED-LOCAL: every test in `test_backend_cuda.py` on
>   the @gpu rig.
> - OPTIONAL: none.
>
> Acceptance:
> - AC1: `python3 -c "from memopt.substrate.backends.cuda_vmm
>   import CUDABackend"` succeeds on macOS (importable even
>   without CUDA — is_available() returns False).
> - AC2: on the @gpu rig, all tests pass.
> - AC3: baseline regression still shows 682 passed.
> - AC4: only `memopt/substrate/backends/cuda_vmm.py` and
>   `memopt/substrate/tests/test_backend_cuda.py` added; no
>   existing files edited.
> - AC5: commit message: "Substrate v1 Commit 5: CUDA VMM
>   backend (per design §2.4 CUDABackend)".
> - AC6: revertible.

### E.2 Commit 6 — ROCm/HIP backend (real or stub per S0.2)

> ## Prompt for Commit 6 (fresh session)
>
> You are implementing memopt Substrate v1 Commit 6: the
> ROCm/HIP backend. Whether this is the real implementation
> or a stub is determined by S0.2's outcome in the Step Zero
> report.
>
> Read first:
> - `docs/substrate_v1_design.md` §2.4 HIPBackend
> - `docs/substrate_v1_step_zero_report.md` §S0.2
> - The `_rocm_backend_py.py` reference at
>   `memopt/vmm/backends/_rocm_backend_py.py` (existing
>   contributor reference)
>
> If S0.2 STATUS is VERIFIED:
>   Implement `memopt/substrate/backends/rocm_hip.py` with
>   real HIP VMM calls per design §2.4. Tests in
>   `test_backend_hip.py` mark `@pytest.mark.gpu_amd`.
>
> If S0.2 STATUS is DEGRADED (capability bit returns 0 OR
> symbols missing):
>   Implement `memopt/substrate/backends/rocm_hip.py` as a
>   stub mirroring `LevelZeroBackend` (Commit 7): is_available
>   returns False; every other method raises
>   NotImplementedError with a clear message. Tests in
>   `test_backend_hip.py` test only the stub behaviour and
>   are NOT marked @gpu_amd (they run on CI).
>
> Files NOT to edit:
> - `memopt/vmm/backends/rocm_backend.py` (legacy stub
>   stays — different code path).
>
> Tests:
> - REQUIRED-CI: stub-mode tests when DEGRADED. On VERIFIED,
>   only the import-only test runs on CI.
> - REQUIRED-LOCAL: none on NVIDIA-only rigs.
> - OPTIONAL: real HIP tests on @gpu_amd rig (only when
>   VERIFIED).
>
> AC1–AC6 as previous commits.

### E.3 Commit 7 — Level Zero stub

> ## Prompt for Commit 7 (fresh session)
>
> You are implementing memopt Substrate v1 Commit 7: the
> Level Zero honest stub (v1.0 — real implementation
> deferred).
>
> Read first:
> - `docs/substrate_v1_design.md` §2.4 LevelZeroBackend
>
> Files to add:
> - `memopt/substrate/backends/level_zero.py` — class
>   `LevelZeroBackend(BackendStrategy)`. `is_available()`
>   returns True iff `libze_loader.so` is loadable AND
>   `zeInit(0)` succeeds. Every other abstract method raises
>   `NotImplementedError("Intel Level Zero backend not
>   implemented in v1.0. See docs/substrate_v1_design.md
>   §2.4.")`.
> - `memopt/substrate/tests/test_backend_l0_stub.py` — all
>   tests from design §3.1.
>
> Tests:
> - REQUIRED-CI: every test in `test_backend_l0_stub.py`.
>   They mock the ze loader so they run anywhere.
> - REQUIRED-LOCAL: none additional.
> - OPTIONAL: none.
>
> AC1–AC6.

### E.4 Commit 8 — CXL/NUMA backend

> ## Prompt for Commit 8 (fresh session)
>
> You are implementing memopt Substrate v1 Commit 8: the
> CXL/NUMA backend.
>
> Read first:
> - `docs/substrate_v1_design.md` §2.4 CXLBackend
> - `docs/substrate_v1_step_zero_report.md` §S0.5 (heuristic
>   verification)
>
> Files to add:
> - `memopt/substrate/backends/cxl_numa.py` —
>   `CXLBackend(BackendStrategy)`. is_available() checks
>   for libnuma + the CXL detection heuristic per S0.5
>   outcome:
>     * If S0.5 VERIFIED with /sys/devices heuristic: use it.
>     * If S0.5 VERIFIED with /sys/bus/cxl/ preferred: use it.
>     * If S0.5 DEGRADED: require `MEMOPT_CXL_NODES` env var
>       (comma-separated NUMA node ids). Without the env var,
>       is_available returns False.
>   Allocate via numa_alloc_onnode (libnuma). For GPU access,
>   when CUDA is also active, register via cuMemHostRegister.
> - `memopt/substrate/tests/test_backend_cxl.py` — tests
>   from design §3.1, marked `@pytest.mark.cxl`.
>
> Tests:
> - REQUIRED-CI: import-only smoke (CXLBackend can be
>   instantiated even without CXL hardware).
> - REQUIRED-LOCAL: none on a non-CXL dev rig.
> - OPTIONAL: real CXL tests on @cxl rig.
>
> AC1–AC6.

### E.5 Milestone 3 review gate

User reads end-to-end:

  - All four new backend files (`cuda_vmm.py`, `rocm_hip.py`,
    `level_zero.py`, `cxl_numa.py`).
  - All four new test files.
  - Re-reads `docs/substrate_v1_step_zero_report.md` to
    confirm Commits 6 and 8 followed the recorded S0.2 / S0.5
    outcomes.

Verifies:

  - AC1–AC6 on each of Commits 5, 6, 7, 8.
  - Milestone 3 promise: every backend is constructible,
    is_available() is honest, real backends work on their
    hardware (when present), stubs raise clearly.
  - Baseline still 682 passing on macOS / no-GPU CI.

Sign-off → Milestone 4.

---

## Section F — Milestone 4: Arena + Stream + Events

### F.0 Milestone 4 promise

> After Milestone 4, the substrate's core data structures
> are in place: per-tenant arenas with size-class freelists,
> a PyTorch CCA-equivalent stream registry, and an event
> ring with a dispatcher. The three documented divergences
> from PyTorch CCA (no auto peer-access, no IPC, single-
> device per call) are tested. The manager (Commit 12) can
> then assemble these.

### F.1 Commit 9 — TenantArena + tenant.py

> ## Prompt for Commit 9 (fresh session)
>
> You are implementing memopt Substrate v1 Commit 9: the
> tenant arena and tenant-isolation guarantees.
>
> Read first:
> - `docs/substrate_v1_design.md` §2.5 TenantArena
> - `docs/substrate_v1_design.md` §2.6 (G1–G4, N1–N4)
> - `docs/substrate_v1_design.md` §2.7 (size class scheme,
>   fragmentation policy)
>
> Files to add:
> - `memopt/substrate/arena.py` — `TenantArena` class with
>   size-class freelists per design §2.5. Per-tenant
>   `threading.Lock`. Background fragmentation reclamation
>   thread per design §2.7.
> - `memopt/substrate/tenant.py` — `_sanitize_id` (using the
>   regex from existing `_cuda_backend_py.py:20`), tenant
>   context (contextvars-based), G1 enforcement helper.
>   NVMe path generator per design §2.6 G3 (sha256-prefix
>   hash, mode 0700, realpath escape check).
> - `memopt/substrate/tests/test_arena.py` — every test
>   from design §3.1.
> - `memopt/substrate/tests/test_tenant_isolation.py` —
>   every test from design §3.1.
>
> Tests:
> - REQUIRED-CI: every test in `test_arena.py` (CPU only)
>   and every test in `test_tenant_isolation.py` EXCEPT
>   `test_n2_direct_cudaMalloc_bypasses_memopt`.
> - REQUIRED-LOCAL: `test_n2_direct_cudaMalloc_bypasses_memopt`
>   on the @gpu rig.
> - OPTIONAL: none.
>
> AC1–AC6.

### F.2 Commit 10 — StreamRegistry + CCA divergence tests

> ## Prompt for Commit 10 (fresh session)
>
> You are implementing memopt Substrate v1 Commit 10: the
> stream registry that matches PyTorch CCA semantics
> exactly per DECISION 2 of the design.
>
> Read first:
> - `docs/substrate_v1_design.md` §2.5 StreamRegistry
> - `docs/substrate_v1_design.md` §2.5 edge cases E1–E9
> - `docs/substrate_v1_step_zero_report.md` §S0.3 (CCA
>   verification outcome — if redesign happened, it's
>   recorded there)
> - `docs/substrate_v1_design.md` §3.2 (CCA divergence
>   tests)
>
> Files to add:
> - `memopt/substrate/stream_registry.py` — `StreamRegistry`
>   class with stream_uses, pending events FIFO, event_pool,
>   per-stream lock. Algorithm per design §2.5 (alloc /
>   record_stream / free / process-pending), matching
>   PyTorch CCA exactly. CUDA Graphs E1 handling: detect
>   capture via cudaStreamIsCapturing; route to per-capture
>   private mempool.
> - `memopt/substrate/tests/test_stream_registry.py` —
>   every test from design §3.1, marked `@pytest.mark.gpu`.
> - `memopt/substrate/tests/test_cca_divergences.py` —
>   exactly the three tests in design §3.2 with the
>   `DESIGN_DATE = "2026-04-29"` constant.
>
> Tests:
> - REQUIRED-CI: import-only smoke for the stream registry
>   (no @gpu tests run; they skip with reason).
> - REQUIRED-LOCAL: every @gpu test in
>   `test_stream_registry.py` and `test_cca_divergences.py`.
> - OPTIONAL: none.
>
> AC1–AC6.

### F.3 Commit 11 — EventRing + observer dispatch

> ## Prompt for Commit 11 (fresh session)
>
> You are implementing memopt Substrate v1 Commit 11: the
> event ring and observer dispatcher.
>
> Read first:
> - `docs/substrate_v1_design.md` §2.5 EventRing
> - `docs/substrate_v1_design.md` §2.8 (Event dataclass,
>   D1–D6 delivery contract)
> - `docs/substrate_v1_step_zero_report.md` §S0.6 (atomic
>   primitive selection)
>
> Files to add:
> - `memopt/substrate/events.py` — `Event` frozen
>   dataclass. `EventRing` with the primitive selected by
>   S0.6 (default: `threading.Lock` if S0.6 fell back).
>   Subscriber dispatcher thread per `memopt.observe`.
>   `SubscriptionHandle` with `unsubscribe()` and
>   `__exit__`.
> - `memopt/substrate/tests/test_events.py` — every test
>   from design §3.1.
>
> Tests:
> - REQUIRED-CI: every test in `test_events.py` EXCEPT
>   `test_event_emit_under_200ns_target` (it's @perf).
> - REQUIRED-LOCAL: same as REQUIRED-CI.
> - OPTIONAL: `test_event_emit_under_200ns_target` (@perf).
>
> AC1–AC6.

### F.4 Milestone 4 review gate

User reads end-to-end:

  - `memopt/substrate/arena.py`
  - `memopt/substrate/tenant.py`
  - `memopt/substrate/stream_registry.py`
  - `memopt/substrate/events.py`
  - All four new test files (arena, tenant_isolation,
    stream_registry, events).
  - `test_cca_divergences.py` end-to-end — the design rationale
    in each docstring must match design §2.5 E2/E3/E4.

Verifies:

  - AC1–AC6 on Commits 9, 10, 11.
  - Milestone 4 promise: tenant arena + stream registry +
    events all pass on macOS / no-GPU CI; stream tests pass
    on the @gpu rig; all three CCA divergences tested per
    Refinement 2.

Sign-off → Milestone 5.

---

## Section G — Milestone 5: Manager + Bridge + Docs + Final

### G.0 Milestone 5 promise

> After Milestone 5, the AllocationManager is assembled,
> the bridge test passes byte-equivalent metrics between
> legacy VMM and substrate-backed paths, the user-facing
> documentation exists, and the version is bumped. The
> substrate is ready for Phase B (the flag-on flip), which
> is a separate engagement.

### G.1 Commit 12 — AllocationManager + smoke + perf bench

> ## Prompt for Commit 12 (fresh session)
>
> You are implementing memopt Substrate v1 Commit 12: the
> AllocationManager that assembles every prior commit into
> the live `memopt.alloc / free / context / stats / observe`
> public API.
>
> Read first:
> - Entire `docs/substrate_v1_design.md` §2 (the design
>   surface). The manager is where everything plugs in.
>
> Files to add/edit:
> - `memopt/substrate/manager.py` — `AllocationManager`
>   process singleton. Wires HandleRegistry + TenantArena
>   per tenant + StreamRegistry + EventRing + Backends
>   list. Implements the alloc / free / context / stats /
>   observe algorithms from §2.5.
> - `memopt/substrate/__init__.py` — replace the stubs
>   from Commit 2 with real `alloc`, `free`, `context`,
>   `stats`, `observe`, `MemoryHandle` exports.
> - `memopt/__init__.py` — add re-exports `from memopt
>   import alloc, free, context, stats, observe,
>   MemoryHandle`. THIS IS THE ONE EDIT TO AN EXISTING
>   FILE in the substrate sequence; the edit is purely
>   additive (new exports added; nothing existing removed
>   or renamed).
> - `memopt/substrate/tests/test_substrate_smoke.py` —
>   every test from design §3.1.
> - `memopt/substrate/tests/test_perf_microbench.py` —
>   every bench from design §3.1; asserts ordering only.
>
> Files NOT to edit:
> - Anything in `memopt/vmm/`, `memopt/cluster/`,
>   `memopt/serving/`, `memopt/api/`, `memopt/control_plane/`,
>   `memopt/operator/`, `memopt/canary/`, `memopt/daemon/`.
>
> Tests:
> - REQUIRED-CI: `test_substrate_smoke.py::
>   test_alloc_use_free_round_trip_cpu`,
>   `test_observe_unobserve_lifecycle`,
>   `test_context_propagation_into_thread_pool`.
> - REQUIRED-LOCAL: `test_alloc_use_free_round_trip_cuda`
>   (@gpu).
> - OPTIONAL: `test_alloc_use_free_round_trip_hip`
>   (@gpu_amd), every `test_perf_microbench.py` (@gpu @perf).
>
> AC1–AC6. Confirm `memopt.alloc(1<<20)` succeeds on macOS
> (CPU fallback) and returns a usable handle that can be
> read/written.

### G.2 Commit 13 — Bridge test (legacy parity)

> ## Prompt for Commit 13 (fresh session)
>
> You are implementing memopt Substrate v1 Commit 13: the
> bridge test that compares legacy VMM and substrate-backed
> outputs for byte-equivalence.
>
> Read first:
> - `docs/substrate_v1_design.md` §2.9 bridge test
>   specification
> - `docs/substrate_v1_design.md` §3.3 (OPTION Y — order
>   not asserted)
>
> Files to add:
> - `tests/test_substrate_legacy_parity.py` — the bridge
>   test per design §2.9 + §3.3. Includes the future-proofing
>   comment from §3.3 verbatim.
>   Workload: 200 allocations across 3 tenants, the size mix
>   from §2.9, 256 MiB pool, half stream-bound. Asserts the
>   seven assertions from §3.3. CPU variant runs on CI.
> - `tests/test_substrate_legacy_parity_gpu.py` — @gpu
>   variant of the same workload using CUDA backend.
>
> Files NOT to edit:
> - Any existing `tests/test_*.py` (the bridge test is new
>   and standalone).
> - `memopt/` — the bridge test calls public APIs only.
>
> Tests:
> - REQUIRED-CI: `test_substrate_legacy_parity.py` (CPU
>   variant). Pass criteria: every one of the 7 assertions
>   from §3.3 holds.
> - REQUIRED-LOCAL: `test_substrate_legacy_parity_gpu.py`
>   (@gpu).
> - OPTIONAL: none.
>
> AC1–AC6. The CPU variant is the trust anchor for Phase B.

### G.3 Commit 14 — Documentation

> ## Prompt for Commit 14 (fresh session)
>
> You are implementing memopt Substrate v1 Commit 14:
> the user-facing documentation.
>
> Read first:
> - `docs/substrate_v1_design.md` (full design)
>
> Files to add:
> - `docs/substrate_v1_threat_model.md` — single-page
>   threat model derived from design §2.6. Lists G1–G4
>   guarantees with one example each. Lists N1–N4 non-
>   guarantees with one example each. Cites the CVE
>   identifiers as recorded in the Step Zero report (use
>   the verified set; if all were unverified, cite the
>   NVIDIA Security Bulletin index URL only).
> - `docs/substrate_v1_user_guide.md` — usage examples for
>   `memopt.alloc / free / context / stats / observe`. One
>   example per public API. Honest performance section
>   citing §2.10 targets and labelling them as TARGETS.
>
> Files to edit:
> - `README.md` — add a "Memory substrate" section pointing
>   at the user guide. ONE paragraph addition. NO
>   performance numbers in the README; only "see user
>   guide" pointers.
>
> Files NOT to edit:
> - Any source file under `memopt/` or `csrc/`.
> - Any test file.
>
> Tests:
> - REQUIRED-CI: regression command (A.2). The 682 must
>   stay green; no new tests in this commit.
> - REQUIRED-LOCAL: same.
> - OPTIONAL: none.
>
> AC1–AC6. Commit message: "Substrate v1 Commit 14:
> threat model + user guide + README pointer (per design
> §2.6, §2.10)".

### G.4 Commit 15 — Final regression run + version bump

> ## Prompt for Commit 15 (fresh session)
>
> You are implementing memopt Substrate v1 Commit 15: the
> final regression sweep and the version bump.
>
> Read first:
> - `pyproject.toml` (version line)
> - `CHANGELOG.md` if it exists; otherwise create one.
> - All previous Commit 0–14 reports / commit messages
>   to summarize.
>
> Tasks:
>
> 1. Run regression (A.2). Confirm baseline contract:
>    `Baseline contract status: PASSED 682+M, SKIPPED 19+K,
>    DESELECTED 11, FAILED 0`
>    where M is the cumulative substrate test count and K
>    is the cumulative hardware-skipped count.
> 2. Bump `pyproject.toml` version per the project's
>    convention (likely `1.0.0` -> `1.1.0` or whatever the
>    project uses; preserve format).
> 3. Add `CHANGELOG.md` entry (or create the file) listing:
>    - "Substrate v1 (Layer 1) — Memory management substrate
>       added under `memopt.alloc / free / context / stats /
>       observe`. See docs/substrate_v1_design.md."
>    - "G12 fix: 16 GiB cap removed from libmemopt_vmm.so."
>    - "ROCm/HIP backend implemented (real or stub per
>       Step Zero S0.2)."
>    - "Bridge test added (tests/test_substrate_legacy_parity
>       .py) — Phase B trust anchor."
>    - "Existing public APIs unchanged. 682 baseline tests
>       still passing."
>
> Files NOT to edit:
> - Anything else.
>
> Tests:
> - REQUIRED-CI: full regression command. Must show baseline.
> - REQUIRED-LOCAL: dev rig runs `pytest -m "gpu"` against
>   `memopt/substrate/tests/`. All @gpu tests pass.
> - OPTIONAL: @gpu_amd, @cxl, @perf — record results in the
>   commit message.
>
> AC1–AC6. The version bump is the last edit. Commit
> message: "Substrate v1 release — Phase A complete
> (per docs/substrate_v1_design.md)".

### G.5 Milestone 5 review gate

User reads end-to-end:

  - `memopt/substrate/manager.py`
  - `memopt/substrate/__init__.py` (now has real exports)
  - The single line edit in `memopt/__init__.py` (re-exports)
  - `tests/test_substrate_legacy_parity.py` and the @gpu
    variant
  - `docs/substrate_v1_threat_model.md`
  - `docs/substrate_v1_user_guide.md`
  - The README pointer
  - `CHANGELOG.md` entry

Verifies:

  - AC1–AC6 on Commits 12, 13, 14, 15.
  - Milestone 5 promise: bridge test passes byte-equivalent;
    docs are honest (no fake numbers); version bumped.
  - Baseline: 682 still passing PLUS substrate tests passing.
  - The 5 milestones together meet Section H below.

Sign-off → Phase A complete.

---

## Section H — Final acceptance criteria

### H.1 Phase A completion gate

Phase A is complete when ALL of the following hold:

```
[ ] Step Zero report exists at
    docs/substrate_v1_step_zero_report.md and shows PASS or
    DEGRADED-with-fallbacks for every S0.1–S0.8 item.
[ ] G12 fix is in csrc; the dynamic-capacity test passes on
    a >= 80 GiB HBM rig.
[ ] memopt/substrate/ contains:
        __init__.py
        handle.py
        manager.py
        arena.py
        tenant.py
        stream_registry.py
        events.py
        backends/__init__.py
        backends/base.py
        backends/cpu_fallback.py
        backends/cuda_vmm.py
        backends/rocm_hip.py        (real or stub per S0.2)
        backends/level_zero.py
        backends/cxl_numa.py
        tests/__init__.py
        tests/test_arena.py
        tests/test_backend_cpu.py
        tests/test_backend_cuda.py
        tests/test_backend_cxl.py
        tests/test_backend_hip.py
        tests/test_backend_l0_stub.py
        tests/test_backend_protocol.py
        tests/test_cca_divergences.py
        tests/test_events.py
        tests/test_handle.py
        tests/test_perf_microbench.py
        tests/test_stream_registry.py
        tests/test_substrate_smoke.py
        tests/test_tenant_isolation.py
[ ] tests/test_substrate_legacy_parity.py and
    test_substrate_legacy_parity_gpu.py exist.
[ ] docs/substrate_v1_design.md exists (Phases 1–3).
[ ] docs/substrate_v1_implementation_prompt.md exists
    (this file).
[ ] docs/substrate_v1_step_zero_report.md exists.
[ ] docs/substrate_v1_threat_model.md exists.
[ ] docs/substrate_v1_user_guide.md exists.
[ ] CHANGELOG.md has the substrate v1 entry.
[ ] pyproject.toml version is bumped.
[ ] memopt/__init__.py has the new re-exports (alloc,
    free, context, stats, observe, MemoryHandle).
[ ] No existing module under memopt/vmm/, memopt/cluster/,
    memopt/serving/, memopt/api/, memopt/control_plane/,
    memopt/operator/, memopt/canary/, memopt/daemon/ was
    edited (except the additive re-exports in
    memopt/__init__.py — verified by `git log --diff-filter=M
    --name-only` over the substrate commits showing only
    memopt/__init__.py outside the substrate path).
[ ] Baseline contract: 682 tests still pass; substrate tests
    add to PASSED; only hardware-gated tests appear in
    SKIPPED; FAILED == 0; DESELECTED unchanged at 11.
[ ] Bridge test passes the byte-equivalent assertions from
    design §3.3.
[ ] Three CCA divergence tests pass on the @gpu rig with
    the documented behaviour and the DESIGN_DATE constant
    pinned.
```

### H.2 What Phase A does NOT deliver

  - Phase B: VMM adapter that routes to substrate under
    `MEMOPT_VMM_USE_SUBSTRATE=1`. Separate engagement.
  - Phase C: flipping the default to substrate-backed.
    Separate engagement.
  - Phase D: removing the legacy code path. Separate
    engagement.
  - Layer 2: tier orchestrator that consumes the event
    surface. Separate design doc.
  - Layer 3+: prefetch oracle, policy DSL, etc.

### H.3 Per-commit test bucket reference

```
Commit  Buckets (REQUIRED-CI / REQUIRED-LOCAL / OPTIONAL)
0       —  /  —  /  —              (verification only)
1       —  /  G12 dynamic capacity test on big-HBM rig  /  —
2       test_handle.py (8 tests)  /  —  /  —
3       test_backend_protocol.py  /  —  /  —
4       test_backend_cpu.py (5 tests)  /  —  /  —
5       —  /  test_backend_cuda.py @gpu (5 tests)  /  —
6       (stub-mode tests when DEGRADED) or import smoke
        when VERIFIED  /  —  /  test_backend_hip.py @gpu_amd
7       test_backend_l0_stub.py (3 tests)  /  —  /  —
8       import smoke  /  —  /  test_backend_cxl.py @cxl
9       test_arena.py + test_tenant_isolation.py except N2 /
        test_n2_direct_cudaMalloc (@gpu)  /  —
10      —  /  test_stream_registry.py + test_cca_divergences
        .py (@gpu)  /  —
11      test_events.py (all but @perf)  /  same  /
        test_event_emit_under_200ns_target (@perf)
12      test_substrate_smoke.py CPU variant  /
        cuda variant (@gpu)  /  hip variant (@gpu_amd) and
        test_perf_microbench.py (@gpu @perf)
13      test_substrate_legacy_parity.py (CPU)  /
        test_substrate_legacy_parity_gpu.py (@gpu)  /  —
14      —  /  —  /  —              (docs only)
15      full regression  /  full regression on @gpu rig  /
        @gpu_amd, @cxl, @perf where hardware allows
```

### H.4 Long-term contract

The substrate ships with a 20-year horizon. The following
contracts must hold from v1.0 onward:

  - The seven primitives in `BackendStrategy` are stable.
    Removing any is a breaking change.
  - `MemoryHandle` field names and types are stable.
    Adding new optional fields with defaults is allowed;
    removing or renaming is breaking.
  - The five event kinds (`alloc`, `free`, `evict`,
    `promote`, `migrate`) are stable. Adding kinds is
    additive; subscribers filter by kind.
  - Tenant isolation guarantees G1–G4 are stable. Weakening
    any is breaking.
  - The `MEMOPT_VMM_USE_SUBSTRATE` flag is stable through
    Phases B–D.
  - Bridge test path `tests/test_substrate_legacy_parity
    .py` stays as the trust anchor for any future migration.

Future maintainers reading this doc 20 years from now should
be able to:

  1. Read `docs/substrate_v1_design.md` for the why.
  2. Read this file for the how.
  3. Read `docs/substrate_v1_step_zero_report.md` for the
     state of dependencies at v1.0 ship.
  4. Trust that the 682 baseline tests still capture the
     pre-substrate behaviour, because nothing in this
     sequence edited them.

---

End of implementation prompt. Phase 4 deliverable complete.
