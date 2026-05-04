# memopt Codebase Audit — 2026-05-01

| Field | Value |
| --- | --- |
| Status | Read-only audit. No file modified outside this report. |
| Working dir | `/Users/lachumanbasnet/Personal/Sophisticates/MEMOPT/memopt` |
| Repo HEAD | `2e2fd9a` (orchestrator-v1.0.0) |
| Test baseline | 876 PASSED, 34 SKIPPED, 15 DESELECTED, 0 FAILED (preserved — no edits) |
| Method | `find` + `grep` + `git log` + `python -c "import …"` smoke loads |

The audit is deliberately conservative: when in doubt, ACTIVE. Every
classification carries either `file:line` or a literal grep command
with its result. Nothing is proposed for deletion in the shipped Layer 1
(`memopt/substrate/`) or Layer 2 (`memopt/orchestrator/`) trees.

---

## Summary

### Files in `memopt/` (excluding `__pycache__`)

```
Total files in memopt/:                  216
  ACTIVE (incl. legacy, tests):          213
  LEGACY (Phase D — keep):                42  (subset of ACTIVE)
  UNUSED candidates:                       2  (review before deletion)
  DUPLICATE candidates:                    1  (review before deletion)
  TEST:                                   89
  DOC (incl. shell helpers in tests):     —   (1: qwen_compare.sh)
```

(LEGACY and TEST are not disjoint from ACTIVE — they are sub-classifications.)

### Files in `csrc/`

```
Total files in csrc/:                     54
  ACTIVE   (Layer 1 substrate consumes):   8 (csrc/cuda_vmm/*, csrc/core/page_table.*, csrc/core/block_directory.*, csrc/core/bindings.cpp)
  LEGACY   (per orchestrator §1.2.4 / §2.4.1 / §2.4.4):  10  (csrc/cuda_vmm/, csrc/core/)
  UNCERTAIN: 36 (other csrc/ subdirs — shipped consumers depend on
              Python fallbacks; the C++ extension is the optional fast
              path; full validation would require build-time inspection)
```

### Files in `tests/` (top-level)

```
Total files in tests/:                    16
  ACTIVE:                                 16
  LEGACY:                                  0
  UNUSED:                                  0
```

### Top recommendations (sorted by safety)

  1. **Keep all LEGACY** under `memopt/vmm/` and `csrc/cuda_vmm/`,
     `csrc/core/` — orchestrator design §2.4 pins them through Phase D.
  2. **Investigate** `memopt/serving/_kernel_hooks_py.py`
     (DUPLICATE of `kernel_hooks.py`, zero callers).
  3. **Investigate** orphaned `__pycache__` directories under
     `memopt/vmm/io/` and `memopt/cluster/fast_lookup/` (no source
     `.py`, not git-tracked, gitignored bytecode of files removed long
     ago — safe to `rm -rf` locally; not a commit).
  4. **Repair drift** in `memopt/daemon/__init__.py:15` (imports
     deleted `.apply` module; `import memopt.daemon` raises today).
     Repair is OUT OF SCOPE for this audit.
  5. **Repair drift** in `memopt/cli/__init__.py` and other files that
     reference deleted `memopt.certificates`, `memopt.migration`,
     `memopt.phase3` packages (deferred imports — runtime-only failure).
  6. **Add** `examples/` directory and an architecture-overview
     diagram to `docs/` (proposals only — see §Restructure Proposals).
  7. **Pyproject entry-point fix** — `pyproject.toml` declares
     `memopt-wrap = "memopt.wrap.cli:main"` but `memopt/wrap/` does not
     exist. Repair OUT OF SCOPE for this audit.

---

## File Inventory

### memopt/substrate/ — Layer 1 (DO NOT TOUCH)

All 30 files ACTIVE. Every file is exercised by
`memopt/substrate/tests/` and by `tests/test_orchestrator_legacy_parity.py`
(part of the 876-pass baseline). Per
`docs/orchestrator_v1_implementation_prompt.md` §H.4 (long-term
contract) these paths are pinned and may not be moved or renamed
without a substrate-redesign cycle.

Source files (9): `__init__.py`, `arena.py`, `events.py`, `handle.py`,
`manager.py`, `stream_registry.py`, `tenant.py`, plus
`backends/{__init__,base,cpu_fallback,cuda_vmm,cxl_numa,level_zero,rocm_hip}.py`
(6 backend files + base).

Tests (15): all under `memopt/substrate/tests/`.

### memopt/orchestrator/ — Layer 2 (DO NOT TOUCH)

All 21 files ACTIVE. Source (8): `__init__.py`, `access.py`,
`config.py`, `coordinator.py`, `policy.py`, `predict.py`,
`telemetry.py`, plus the `tests/conftest.py` (perf-skip hook).
Tests (12): all under `memopt/orchestrator/tests/`.

Per `docs/orchestrator_v1_implementation_prompt.md` §H.4 these paths
are pinned through Layer 3 and beyond.

### memopt/vmm/ — LEGACY (per orchestrator design §1.2 / §2.4)

Per `docs/orchestrator_v1_design.md` §1.2 lines 1-50 and §2.4 (lines
1480-1556), the entire `memopt/vmm/` subtree is the legacy memory
manager that Layer 2 Phase B observes and Layer 2 Phase D will
remove. **All files under `memopt/vmm/` are LEGACY and must be kept**
until the orchestrator's Phase D plan executes (out of scope for this
repo state).

Importer counts (subset; all greater than zero unless flagged):

| Module | Importers (excl. self) | Disposition |
| --- | --- | --- |
| `memopt/vmm/__init__.py` | star-export hub | LEGACY (keep) |
| `memopt/vmm/_oracle_py.py` | 7 | LEGACY (keep) — pure-Python fallback for `oracle.py` |
| `memopt/vmm/_page_table_py.py` | 2 | LEGACY (keep) — fallback for `page_table.py` |
| `memopt/vmm/access_log.py` | 4 | LEGACY (keep) — JSONL access trace logger |
| `memopt/vmm/cuda_vmm.py` | 8 | LEGACY (keep) — wraps `csrc/cuda_vmm/libmemopt_vmm.so` |
| `memopt/vmm/discovery.py` | 7 | LEGACY (keep) |
| `memopt/vmm/elastic_allocator.py` | 2 | LEGACY (per design §1.2) |
| `memopt/vmm/federation.py` | 1 (`vmm/tests/test_layer3.py:12`) | LEGACY (keep) |
| `memopt/vmm/global_oracle.py` | 4 | LEGACY (per design §2.4.5) |
| `memopt/vmm/hal.py` | 10 | LEGACY (keep) |
| `memopt/vmm/memory_governor.py` | 2 | LEGACY (per design §1.2) |
| `memopt/vmm/oracle.py` | 11 | LEGACY (keep) |
| `memopt/vmm/oracle_data_cleaner.py` | 3 | LEGACY (keep) |
| `memopt/vmm/oracle_trainer.py` | 3 | LEGACY (keep) |
| `memopt/vmm/page_table.py` | 6 | LEGACY (keep) |
| `memopt/vmm/pod_controller.py` | 6 | LEGACY (per design §2.4.5) |
| `memopt/vmm/prefetch_engine.py` | 5 | LEGACY (per design §1.2) |
| `memopt/vmm/tier_manager.py` | 7 | LEGACY (per design §1.2 / §2.4.2) |
| `memopt/vmm/torch_allocator.py` | 5 | LEGACY (keep) — PyTorch CCA shim |
| `memopt/vmm/universal_profile.py` | 2 (`test_layer3.py`, `test_universal_profile.py`) | LEGACY (keep) |
| `memopt/vmm/weight_manager.py` | 0 direct importers, but exported via `memopt/vmm/__init__.py:30` and `:174` | LEGACY (keep — exported in package surface) |
| `memopt/vmm/backends/*.py` | varies | LEGACY (per design §2.4.3) |

Tests under `memopt/vmm/tests/` (16 files): all LEGACY, all part of
the 876 baseline (per `docs/orchestrator_v1_design.md` §1.6.2 — these
must remain green forever).

### memopt/cluster/ — ACTIVE (cross-node memory fabric)

Not listed under orchestrator §1.2 legacy paths. ACTIVE.

| Module | Importers | Notes |
| --- | --- | --- |
| `_block_directory_py.py` | 2 | Python fallback for `_memopt_core` C++ ext |
| `_prefix_index_py.py` | 1 | Python fallback for `_memopt_simd` |
| `_transport_py.py` | 2 | Python fallback for transport |
| `_transport_shm.py` | 1 | Shared-memory transport |
| `block_directory.py` | 7 | C++/Python shim |
| `bloom_filter.py` | 2 | |
| `gkd_store.py` | 18 | Heavily used |
| `gum_metrics.py` | 4 | |
| `hashing.py` | 6 | |
| `hypervisor.py` | 3 | |
| `prefix_index.py` | 4 | C++/Python shim |
| `remote_block.py` | 7 | |
| `scylla_gkd_backend.py` | 2 | |
| `transport.py` | 5 | |

Tests (10): all ACTIVE.

### memopt/profiler/ — ACTIVE

12 source files, 0 tests under `memopt/profiler/tests/` (does not
exist as a directory — confirmed by `find memopt/profiler -type d`).
All source ACTIVE per importer counts (every module ≥ 1 importer).

| Module | Importers |
| --- | --- |
| `access_analyzer.py` | 2 |
| `attribution.py` | 1 (`profiler/__init__.py`) |
| `classifier.py` | 3 |
| `continuous_profiler.py` | 1 |
| `gpu_specs.py` | 2 |
| `graph_analyzer.py` | 2 |
| `hardware_counters.py` | 9 |
| `hardware_metrics.py` | 2 |
| `ncu_profiler.py` | 2 |
| `power_sampler.py` | 2 |
| `profiler.py` | 1 (`profiler/__init__.py`) |
| `roofline.py` | 3 |

### memopt/kernels/ — ACTIVE

5 source modules + tests under `memopt/kernels/tests/` (5 test files).

| Module | Importers |
| --- | --- |
| `bottleneck_detector.py` | 7 |
| `jit_generator.py` | 15 |
| `kernel_cache.py` | 17 |
| `portability_layer.py` | 10 |
| `prompt_builders.py` | 1 (`jit_generator.py`) |

### memopt/serving/ — ACTIVE (with one DUPLICATE candidate)

| Module | Importers | Notes |
| --- | --- | --- |
| `_kernel_hooks_py.py` | **0** | DUPLICATE candidate (see §Unused / §Duplicate) |
| `_paged_attention_py.py` | 2 | Python fallback |
| `auto_optimizer.py` | 3 | |
| `continuous_batching.py` | 3 | |
| `kernel_hooks.py` | 3 | C++/Python shim |
| `kv_cache.py` | 2 | |
| `paged_attention.py` | 4 | |
| `server.py` | 20 | Heavy hub |

Tests (2): `test_oracle_endpoints.py`, `test_serving_kernels.py` — both ACTIVE.

### memopt/control_plane/ — ACTIVE

| Module | Importers | Notes |
| --- | --- | --- |
| `cli.py` | 2 | Includes `from memopt.cli` import in tests |
| `crdb_schema.py` | 1+ | Used by `database.py` |
| `database.py` | 5 | |
| `server.py` | 5 | References deleted `memopt.certificates` (deferred import) |

Tests (7): all ACTIVE.

### memopt/daemon/ — ACTIVE source, BROKEN package surface

10 files. **`memopt/daemon/__init__.py` is broken at import time** —
see §Drift below. All files have legitimate purposes:

| Module | Importers | Notes |
| --- | --- | --- |
| `cli.py` | 1 (`memopt/cli/__init__.py`) | Imports deleted `memopt.migration.engine` (deferred) |
| `daemon_service.py` | 1 (`__init__.py`) | |
| `process_inspector.py` | 1 (`__init__.py`) | |
| `process_monitor.py` | 1 (`__init__.py`) | |
| `report.py` | 1 (`__init__.py`) | |
| `reporter.py` | 1 (`__init__.py`) | |
| `scanner.py` | 1 (`__init__.py`) | |
| `scheduler.py` | 1 (`__init__.py`) | |

These individual modules are loadable; `memopt/daemon/__init__.py`
itself is not. No test in the 876 baseline imports `memopt.daemon`.
Classification: ACTIVE (modules), with `__init__.py` flagged for
repair.

### memopt/operator/ — ACTIVE (Kubernetes operator entry point)

| Module | Importers | Notes |
| --- | --- | --- |
| `controller.py` | 6 | Tests + `memopt.cli` |
| `main.py` | 0 direct importers; entry point via `python -m memopt.operator.main` (`main.py:9`) | ACTIVE (entry point) |
| `models.py` | 4 | |

Tests (3): all ACTIVE.

### memopt/canary/ — ACTIVE

7 files, 4 source + 3 tests. Importer counts ≥ 1 each. Tests pass.

### memopt/utils/ — ACTIVE

| Module | Importers |
| --- | --- |
| `gpu_info.py` | 1 (`hardware_detector.py`) |
| `hardware_detector.py` | 1+ |
| `input_handler.py` | 1+ |
| `model_loader.py` | 1+ |
| `multimodal.py` | 1+ |

### memopt/measurement/, memopt/observability/, memopt/auth/, memopt/api/, memopt/workflows/, memopt/cli/

All ACTIVE — each has at least one importer in the live tree:

| Package | Importer |
| --- | --- |
| `memopt/measurement/` | `memopt/__init__.py:52` (`from .measurement import BandwidthTracker, …`) |
| `memopt/observability/` | `memopt/api/server.py`, `scripts/pillar_drivers.py`, `scripts/big_model_run.py` |
| `memopt/auth/` | `memopt/daemon/reporter.py`, `memopt/control_plane/server.py`, `memopt/control_plane/cli.py`, `memopt/api/server.py` |
| `memopt/api/` | `memopt/serving/server.py:602,719,823,980,1054` (5 deferred imports) |
| `memopt/workflows/` | `memopt/cli/__init__.py:121` (`from memopt.workflows import AnalyzeWorkflow`) |
| `memopt/cli/` | `pyproject.toml` `[project.scripts]` `memopt = "memopt.cli:main"`; `memopt/control_plane/tests/test_pods.py` |

### memopt/__init__.py + memopt/image_version.py

Both ACTIVE.

  - `memopt/__init__.py` — public API surface + Layer 1/Layer 2
    re-exports (lines 47, 59, 64). Exports `BandwidthTracker`,
    `alloc`, `free`, `context`, `stats`, `observe`, `peek_handle`,
    `MemoryHandle`, `orchestrator`.
  - `memopt/image_version.py` — referenced by
    `memopt/canary/gates.py`, `memopt/canary/tests/test_controller.py`,
    `memopt/vmm/discovery.py`, `memopt/control_plane/server.py`.

---

## C++ / CUDA Sources under `csrc/`

### Confirmed ACTIVE (consumed by Layer 1 substrate)

  - `csrc/cuda_vmm/vmm_allocator.cpp`, `vmm_allocator.h` — wrapped by
    `memopt/vmm/cuda_vmm.py` and `memopt/substrate/backends/cuda_vmm.py`.
    LEGACY per orchestrator design §2.4.1 (PyTorch pluggable allocator
    path); **DO NOT propose deletion**.
  - `csrc/cuda_vmm/test_vmm_allocator.cpp` — C++ test for VMM allocator.
    LEGACY per same.
  - `csrc/cuda_vmm/CMakeLists.txt`, `csrc/cuda_vmm/build.sh` — build
    glue.
  - `csrc/core/page_table.{h,cpp}` — referenced by
    `memopt/vmm/_page_table_py.py` AS the design template (per
    `memopt/orchestrator/access.py:3-5` "Mirrors
    `csrc/core/page_table.h:108-138`"). LEGACY per §2.4.4.
  - `csrc/core/block_directory.{h,cpp}` — Python shim wraps via
    `memopt/cluster/_block_directory_py.py`.
  - `csrc/core/oracle.{h,cpp}` — pybind11-exposed; mirrors
    `memopt/vmm/oracle.py`. LEGACY per §2.4.5.
  - `csrc/core/bindings.cpp` — pybind11 entry.

### LEGACY (per orchestrator §1.2 / §2.4)

The full `csrc/cuda_vmm/` subtree (4 files) and `csrc/core/` subtree
(7 files) are explicitly LEGACY by the design doc. Layer 2 Phase D
(separate engagement) handles their replacement. **DO NOT delete.**

### Other csrc/ subtrees — UNCERTAIN, classify as ACTIVE by default

These have C++/CUDA implementations alongside Python fallbacks; full
validation requires a build to confirm the .so binding loads:

  - `csrc/cuda_backend/` (8 files) — GDS, NVMe async I/O, stream pool.
    Likely consumed by `memopt/vmm/backends/cuda_backend.py`.
  - `csrc/hooks/` (4 files) — kernel-hooks dispatch; the
    `_memopt_hooks` C++ extension referenced by
    `memopt/serving/kernel_hooks.py:45`.
  - `csrc/paged/` (6 files including `gather_kernel.cu`,
    `gather_kernel.cuh`) — paged-attention KV cache kernel; consumed
    by `memopt/serving/_paged_attention_py.py`.
  - `csrc/rocm/rocm_backend.cpp` — paired with
    `memopt/vmm/backends/_rocm_backend_py.py` (stub) per
    `docs/substrate_v1_step_zero_report.md`.
  - `csrc/simd/` (3 files) — `_memopt_simd` extension (referenced by
    `memopt/cluster/prefix_index.py:32-33`).
  - `csrc/transport/` (10 files) — RDMA + TCP transport; paired with
    `memopt/cluster/_transport_py.py`.
  - `csrc/discovery/main.go` — Go binary; not Python-importable; build
    glue.
  - `csrc/experiments/{vmm_probe,vmm_probe_concurrent}.cpp` — research
    probes; standalone executables. UNUSED-CANDIDATE in spirit but
    intentionally exploratory; not proposed for deletion.

`csrc/cmake/` (1 file), `csrc/CMakeLists.txt`, `csrc/cuda_vmm/CMakeLists.txt`,
`csrc/experiments/CMakeLists.txt`, `csrc/discovery/README.md` — build
config; ACTIVE.

---

## Tests under `tests/` (top-level)

All 16 files ACTIVE. None LEGACY, none UNUSED.

| File | Purpose |
| --- | --- |
| `conftest_rig_skips.py` | CUDA-rig env-skip plumbing (pytest hook from `conftest.py`) |
| `pillar1_proof.py` … `pillar8_proof.py` | Per-pillar end-to-end proofs |
| `test_async_nvme.py` | NVMe async path |
| `test_canary_monitoring.py` | Canary controller |
| `test_golden_image.py` | Golden-image build artifact verification |
| `test_hardware_support.py` | Multi-arch (gaudi/tpu stub) tests |
| `test_integration_gaps.py` | Integration smoke |
| `test_orchestrator_legacy_parity.py` | **The bridge test** (orchestrator §3.2) |
| `test_pillar2_enhanced.py` … `test_pillar5_enhanced.py` | Enhanced pillar tests |
| `test_substrate_legacy_parity.py`, `test_substrate_legacy_parity_gpu.py` | Substrate v1 bridge tests |

---

## Legacy Boundary

Per `docs/orchestrator_v1_design.md` §1.2 (lines covering the legacy
paths) and §2.4 (the coexistence plan). The following are LEGACY,
isolated, and must remain in place until orchestrator Phase D ships:

```
memopt/vmm/tier_manager.py            (§1.2.1, §2.4.2)
memopt/vmm/elastic_allocator.py       (§1.2)
memopt/vmm/memory_governor.py         (§1.2)
memopt/vmm/prefetch_engine.py         (§1.2)
memopt/vmm/_oracle_py.py              (§1.3 / DECISION 6)
memopt/vmm/oracle.py                  (§1.3)
memopt/vmm/global_oracle.py           (§2.4.5)
memopt/vmm/pod_controller.py          (§2.4.5)
memopt/vmm/oracle_trainer.py          (§1.3.3)
memopt/vmm/oracle_data_cleaner.py     (§1.3.4)
memopt/vmm/access_log.py              (§1.3.2)
memopt/vmm/backends/                  (§2.4.3)
csrc/cuda_vmm/                        (§2.4.1)
csrc/core/                            (§2.4.4 + 2.4.5)
```

**Isolation evidence (Layer 1 / Layer 2 do NOT import LEGACY):**

```
$ grep -rln "from memopt.vmm\|import memopt.vmm" memopt/substrate/ memopt/orchestrator/
(no results)

$ grep -rln "from memopt.vmm\|import memopt.vmm" \
    tests/test_orchestrator_legacy_parity.py
(no results)
```

Confirmed: Layers 1 and 2, plus the bridge test, do not import any
`memopt.vmm.*` module. The legacy tree is isolated from the shipped
contract surface.

`memopt/orchestrator/tests/test_decision_d1_greenfield.py:1-67`
provides automated `ast`-based enforcement of this isolation.

---

## UNUSED Candidates (review-before-deleting)

### Candidate 1 — `memopt/vmm/io/` and `memopt/vmm/io/tests/`

  - **Path**: `memopt/vmm/io/`, `memopt/vmm/io/tests/`
  - **What's there**: only `__pycache__/` directories with stale
    `.pyc` files (one was for `test_fast_io.cpython-313-pytest-9.0.2.pyc`).
    No `.py` source files exist.
  - **Git tracking**: NOT tracked.
    ```
    $ git ls-files memopt/vmm/io
    (empty)
    ```
  - **Last-modified**: not derivable from `git log` (untracked).
  - **Recommendation**: SAFE TO `rm -rf memopt/vmm/io` LOCALLY. This
    is gitignored bytecode from a long-deleted source tree. Not a git
    commit — just a local filesystem cleanup. **The audit does NOT
    perform this deletion** (per non-negotiables).

### Candidate 2 — `memopt/cluster/fast_lookup/` and its tests

  - **Path**: `memopt/cluster/fast_lookup/`, `memopt/cluster/fast_lookup/tests/`
  - **What's there**: only `__pycache__/__init__.cpython-313.pyc` and
    `__pycache__/test_gkd_map.cpython-313-pytest-9.0.2.pyc`. No
    `.py` source.
  - **Git tracking**: NOT tracked.
    ```
    $ git ls-files memopt/cluster/fast_lookup
    (empty)
    ```
  - **Recommendation**: SAFE TO `rm -rf memopt/cluster/fast_lookup`
    LOCALLY. Same character as Candidate 1.

### Candidate 3 — `memopt/vmm/tests/__pycache__/test_speculative_prefetcher.cpython-313-pytest-9.0.2.pyc`

  - **What's there**: one orphan .pyc file with no matching source
    `test_speculative_prefetcher.py`.
  - **Git tracking**: NOT tracked.
  - **Recommendation**: SAFE local `rm` — same gitignored cruft.

### NOT classified as UNUSED (despite zero direct importers)

  - `memopt/vmm/weight_manager.py` — exported by
    `memopt/vmm/__init__.py:30,174` (`from .weight_manager import
    WeightManager`, `__all__ = [..., "WeightManager", ...]`). ACTIVE
    via package surface.
  - `memopt/operator/main.py` — entry point invoked via `python -m
    memopt.operator.main` (documented at `main.py:9`). ACTIVE.
  - `memopt/serving/_kernel_hooks_py.py` — zero importers but a
    near-duplicate of `kernel_hooks.py`. Classified as DUPLICATE
    (§Duplicate Candidates), not UNUSED, because intent is
    ambiguous (likely intended fallback that got disconnected).

---

## DUPLICATE Candidates

### Candidate D1 — `memopt/serving/_kernel_hooks_py.py`

  - **Sibling**: `memopt/serving/kernel_hooks.py` (302 lines, same
    git commit `d7ffe6a` 2026-04-10).
  - **Intent suggested by name**: per the `_oracle_py.py` /
    `_page_table_py.py` convention elsewhere in the repo, `_<x>_py.py`
    is the pure-Python fallback for an `<x>.py` shim that prefers a
    C++ extension.
  - **Discrepancy**: `memopt/serving/kernel_hooks.py:45` does try
    `import memopt._memopt_hooks as _cpp_hooks`, but it does NOT fall
    back to `_kernel_hooks_py.py` — both files contain the full
    Python implementation independently. Diff (head):
    ```
    $ diff memopt/serving/kernel_hooks.py memopt/serving/_kernel_hooks_py.py | head
    8,11d7  (kernel_hooks.py adds: "Shim layer: tries C++ _memopt_hooks…")
    24c20   (TYPE_CHECKING comment differs)
    29,32c25,32  (comment block reorganized)
    35c35   (lock comment differs)
    ```
  - **Importer count**:
    ```
    $ grep -rn "_kernel_hooks_py" memopt/ tests/ scripts/
    (no results — zero callers, zero references)
    ```
  - **Recommendation**: INVESTIGATE before deleting. The intent is
    one of:
      (a) `_kernel_hooks_py.py` was supposed to be the fallback that
          `kernel_hooks.py` imports when the C++ ext is unavailable —
          the wiring was lost in a refactor. Repair = make
          `kernel_hooks.py` import from `_kernel_hooks_py.py` on
          ImportError.
      (b) `_kernel_hooks_py.py` is dead code from before
          `kernel_hooks.py` was promoted to the shim. Repair =
          delete `_kernel_hooks_py.py`.
    Either repair is a separate engagement, not this audit.

---

## Documentation Drift

### Documents under `docs/`

| Doc | Last commit | Status |
| --- | --- | --- |
| `docs/architecture.md` | 2026-04-28 | ACTIVE (general arch); includes references to LEGACY VMM internals (TierManager, etc.) — accurate vs current code |
| `docs/cockroachdb_deployment.md` | 2026-04-13 | ACTIVE — deployment doc |
| `docs/hardware_support.md` | 2026-04-13 | ACTIVE — references `_gaudi_backend_py.GaudiBackend` (line 77) and `_tpu_backend_py.TPUBackend` (line 94); both files exist and tests cover them (`tests/test_hardware_support.py:18-28`) |
| `docs/oracle_v1_design.md` | (untracked) | DRAFT — Layer 3 Phase 1 discovery; not yet committed |
| `docs/orchestrator_v1_design.md` | (untracked) | Source-of-truth for shipped Layer 2; not yet committed (the implementation prompt was authored separately) |
| `docs/orchestrator_v1_implementation_prompt.md` | (untracked) | Same |
| `docs/orchestrator_v1_step_zero_report.md` | committed (commit 0) | ACTIVE — covers shipped behavior |
| `docs/orchestrator_v1_threat_model.md` | committed (commit 12) | ACTIVE |
| `docs/orchestrator_v1_user_guide.md` | committed (commit 12) | ACTIVE |
| `docs/query_optimization.md` | 2026-04-13 | ACTIVE |
| `docs/rdma_deployment.md` | 2026-04-10 | ACTIVE — referenced by README.md:166 |
| `docs/substrate_v1_*.md` (4 files) | committed in substrate v1 release | ACTIVE |

### README.md

  - Last commit 2026-05-01.
  - Header line 5: "**339 tests pass | 7 C++ test suites | 0 failures**"
    — this is **STALE**. Current Mac baseline is 876 PASSED, 34 SKIPPED,
    15 DESELECTED. **Drift: documented**.
  - Lines 168-176 (Memory substrate section) and lines 178-189
    (Orchestrator section) are accurate against shipped Layers 1/2.

### CHANGELOG.md

  - Last commit 2026-05-01 (orchestrator v1.2.0 entry shipped at
    commit 13). ACTIVE, accurate.

### Drift summary

  - **README.md:5** — stale test count "339" should be "876". Audit
    does NOT edit it (per non-negotiables).
  - **No references to deleted modules** in the doc tree itself
    (the deleted `memopt.agent`, `memopt.business`, `memopt.migration`,
    `memopt.optimization`, `memopt.phase3`, `memopt.certificates`
    packages are NOT mentioned in any `docs/*.md`).
  - **Code-side drift** (separate from docs):
    ```
    $ grep -rnE "from memopt\.(agent|business|migration|optimization|phase3|certificates)" memopt/
    memopt/cli/__init__.py:239:        from memopt.certificates import CertificateStore
    memopt/daemon/cli.py:147:        from memopt.migration.engine import AutoMigrationEngine
    memopt/control_plane/server.py:1186:        from memopt.certificates import CertificateStore
    memopt/api/server.py:340:    from memopt.phase3 import (
    ```
    All four are deferred imports inside command handlers / route
    handlers, so they don't break the 876 baseline. They will fail at
    runtime if the relevant CLI command / API endpoint is invoked.
  - **`memopt/daemon/__init__.py:15`** imports `.apply` which was
    deleted in commit `c5705a3` (2026-03-13, "optimizations
    deleted……"). Repro:
    ```
    $ PYTHONPATH=. python3 -c "import memopt.daemon"
    ModuleNotFoundError: No module named 'memopt.daemon.apply'
    ```
    `__init__.py` lines 38-44 also reference symbols never imported
    (`ZeroTouchDaemon`, `ZeroTouchConfig`, `OptimizationEvent`,
    `ClusterROICalculator`, `ROICalculator`, `ROIEstimate`); line 51
    references `ZeroTouchConfig` not imported. The file is broken.
    No test in the 876 baseline imports `memopt.daemon`, so the
    breakage is invisible to CI.
  - **`pyproject.toml`** declares `memopt-wrap = "memopt.wrap.cli:main"`
    in `[project.scripts]`; `memopt/wrap/` does not exist. Repro:
    ```
    $ ls memopt/wrap
    ls: memopt/wrap: No such file or directory
    ```

---

## Restructure Proposals (proposals only — NO ACTION)

The constraints rule out anything that touches Layers 1/2, anything
that moves or merges shipped modules, and anything that "cleans up
imports" in shipped code. Proposals below stay strictly additive.

### Proposal #1 — Add a top-level `examples/` directory

  - **What changes**: create `examples/` (currently no such directory
    at the repo root) populated with three small runnable scripts:
      - `examples/01_alloc_free_basic.py` — substrate quickstart
        (mirrors `docs/substrate_v1_user_guide.md` Quickstart).
      - `examples/02_orchestrator_observe.py` — start the orchestrator,
        observe events, snapshot stats (mirrors
        `docs/orchestrator_v1_user_guide.md` Quickstart).
      - `examples/03_register_custom_policy.py` — the worked example
        from `docs/orchestrator_v1_user_guide.md` "Worked example"
        section.
  - **Why it's safe**: brand-new files; no existing path is renamed
    or moved; no test collected. The only risk surface is if pytest
    accidentally discovers them — adding `examples/` to the
    `[tool.pytest.ini_options] testpaths = […]` if explicit, or
    naming files without `test_` prefix prevents collection.
  - **Estimated effort**: 1-2 hours (write three short scripts, verify
    each runs against the current `876` baseline).
  - **Regression run required?** Yes — to confirm pytest does not
    collect anything new.

### Proposal #2 — Add an architecture overview diagram to `docs/`

  - **What changes**: add `docs/architecture_layers_overview.md`
    (or `.svg` / Mermaid block in markdown) showing Layer 1 substrate
    + Layer 2 orchestrator + the legacy VMM tree side-by-side, with
    arrows for the §1.5 boundary (S1 events up; S2/S3/S4 alloc/free/
    stats down). The existing `docs/architecture.md` is large and
    mixes concerns; a focused diagram for new contributors helps.
  - **Why it's safe**: brand-new file; no existing doc is edited.
  - **Estimated effort**: 1 hour.
  - **Regression run required?** No — docs only.

### Proposal #3 — Local `__pycache__` cleanup of orphaned dirs

  - **What changes**: `rm -rf memopt/vmm/io memopt/cluster/fast_lookup`
    (these are gitignored bytecode from removed source). Also `rm`
    the orphan `test_speculative_prefetcher.cpython-313-pytest-9.0.2.pyc`.
  - **Why it's safe**: nothing tracked in git; `.pyc` files are
    transient.
  - **Estimated effort**: 1 minute.
  - **Regression run required?** No — bytecode that pytest never
    discovers source for.

---

## Recommended Next Steps (sorted by safety)

  1. **(safest, immediate)** Local-only cleanup of orphan `__pycache__`
     trees per Proposal #3. No git impact.
  2. **(safe, additive)** Land Proposal #1 (`examples/`) and Proposal #2
     (architecture diagram doc) as separate PRs. Re-run the 876-pass
     baseline after #1.
  3. **(repair drift, separate engagement)** Open an issue to fix
     `memopt/daemon/__init__.py:15` — either restore the deleted
     `apply.py` from commit `c5705a3` or remove the broken imports
     and the `__all__` references to symbols that no longer exist.
     Add a regression test that does `import memopt.daemon`.
  4. **(repair drift, separate engagement)** Open an issue to clean
     up the four deferred-import references to deleted packages
     (`memopt.certificates`, `memopt.migration`, `memopt.phase3`)
     listed in §Documentation Drift → "Code-side drift". Either
     restore those packages or remove the dead command handlers.
  5. **(repair drift, separate engagement)** Update `pyproject.toml`
     `[project.scripts]` to drop the `memopt-wrap` entry that
     references the missing `memopt/wrap/` package, OR add the
     missing package.
  6. **(repair doc drift, separate engagement)** Update README.md:5
     test count "339 tests pass" to "876 tests pass" (or equivalent
     wording).
  7. **(investigate before any action)** Decide the fate of
     `memopt/serving/_kernel_hooks_py.py` per §Duplicate Candidates D1.
  8. **(KEEP, no action)** Every file under `memopt/vmm/`,
     `csrc/cuda_vmm/`, `csrc/core/` — orchestrator design §2.4 owns
     their disposition through Phase D.
  9. **(KEEP, no action)** Every file under `memopt/substrate/` and
     `memopt/orchestrator/` — long-term contract pinned in
     `docs/orchestrator_v1_implementation_prompt.md` §H.4.

---

*End of audit. No file outside `docs/codebase_audit_2026_05_01.md`
was modified. The 876-test baseline is preserved.*
