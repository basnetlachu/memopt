# memopt Pillar Revival + Open-Source Plan

| Field | Value |
| --- | --- |
| Status | Triage doc — milestone 1. Read-only inventory. No edits to source. |
| Repo HEAD | `2e2fd9a` (orchestrator-v1.0.0) |
| Test baseline | 876 PASSED, 34 SKIPPED, 15 DESELECTED, 0 FAILED |
| Archive sibling | `/Users/lachumanbasnet/Personal/Sophisticates/MEMOPT/memopt-trust-archive` (separate git repo, master branch) |
| Drafted | 2026-05-04 |

This document plans the revival of the trust archive into the engine
repo + the open-source pass. **Nothing is moved or deleted yet.** Each
milestone below ships as its own additive PR with the 876 baseline
preserved at every step.

---

## What's in the archive (verified by `find` + `head` 2026-05-04)

22 Python files split across 3 pillars:

| Pillar | Files | LOC | Description |
| --- | --- | --- | --- |
| **4 — Compliance ledger** | `memopt/observability/{ledger.py,certificate.py,report_exporter.py,arbitrage.py}`, `memopt/observability/tests/{test_ledger_verify.py,test_ledger_hardening.py,test_observability.py}` | ~2 700 | Optimization-event ledger + HMAC-signed certificates + savings reports + GPU-arbitrage engine |
| **6 — Silicon certification** | `memopt/kernels/{certification.py,certify_daemon.py,drift_detector.py}`, `memopt/kernels/tests/{test_certification.py,test_certify_daemon-related,test_drift_detector.py,test_drift_hardening.py,test_feedback_loop_drift.py,test_control_plane_callback.py}`, `memopt/cli/certify.py` | ~2 200 | Correctness + throughput battery + drift detection + standalone `memopt-certify` CLI |
| **7 — GPU FinOps** | `memopt/finops/tracker.py`, `memopt/trust/receipt.py` | ~800 | GPU-utilization-to-dollar tracker + production trust receipt |
| **+ tests** | `tests/{pillar4_proof.py,pillar6_proof.py,pillar7_proof.py,test_pillar4_enhanced.py,test_pillar6_enhanced.py,test_production_receipt.py}` | ~1 600 | End-to-end pillar proofs |

Total: 13 296 LOC.

The archive's `memopt/cli/__init__.py` is **larger** than main's
(836 vs 781 lines) — the difference is one new `cmd_certify` function
+ subparser block (verified by `diff`). All other CLI symbols are
identical.

The archive's `memopt/observability/__init__.py` is a 4-import re-export.
Main has a 1-import version (only `MetricsCollector`). The archive
version subsumes main's (it imports the same `collector` symbols
plus the new ones).

---

## Dependency map (archive → main)

Every `from memopt.*` import in the archive points at code that lives
in main today. No phantom dependencies.

```
ARCHIVE FILE                                   IMPORTS FROM MAIN
─────────────────────────────────────────────  ───────────────────────────────────────
memopt/kernels/certification.py                memopt.profiler.hardware_counters.GPU_SPECS
memopt/kernels/certify_daemon.py               memopt.kernels.{jit_generator, kernel_cache, portability_layer}
                                                memopt.kernels.{drift_detector, certification}  (sibling-archive)
memopt/observability/ledger.py                 memopt.observability.certificate          (sibling-archive)
memopt/observability/{arbitrage,certificate,
   report_exporter}.py                         (no main imports)
memopt/finops/tracker.py                       (no main imports)
memopt/trust/receipt.py                        (no main imports)
memopt/cli/__init__.py (added cmd)             memopt.kernels.certification
memopt/cli/certify.py                          memopt.kernels.certification
```

All listed main-side targets exist:
  - `memopt/profiler/hardware_counters.py` (line 1+; module loads)
  - `memopt/kernels/{jit_generator.py, kernel_cache.py, portability_layer.py}` (all present)

No revival blocker on the dependency graph.

---

## File-by-file revival plan

For each archive file: target path in main, action (COPY / MERGE / SKIP),
and integration notes.

### Pillar 4 — Compliance ledger

| Archive file | Target in main | Action | Notes |
| --- | --- | --- | --- |
| `memopt/observability/__init__.py` | `memopt/observability/__init__.py` | **MERGE** | Append the 3 new exports (ledger, certificate, arbitrage) to the existing 1-export file |
| `memopt/observability/ledger.py` | `memopt/observability/ledger.py` | COPY | New file |
| `memopt/observability/certificate.py` | `memopt/observability/certificate.py` | COPY | New file |
| `memopt/observability/report_exporter.py` | `memopt/observability/report_exporter.py` | COPY | New file |
| `memopt/observability/arbitrage.py` | `memopt/observability/arbitrage.py` | COPY | New file |
| `memopt/observability/tests/__init__.py` | new (0 bytes) | COPY | Create tests dir |
| `memopt/observability/tests/test_ledger_verify.py` | same | COPY | New file |
| `memopt/observability/tests/test_ledger_hardening.py` | same | COPY | New file |
| `memopt/observability/tests/test_observability.py` | same | COPY | New file |
| `tests/pillar4_proof.py` | `tests/pillar4_proof.py` | COPY | E2E proof |
| `tests/test_pillar4_enhanced.py` | `tests/test_pillar4_enhanced.py` | COPY | Enhanced suite |

**Layer 1/2 wiring (Phase 4):**
  - `OptimizationLedger.record_event(...)` becomes a Layer-2 subscriber
    via `memopt.observe("alloc"|"free"|"evict"|...)`.
  - When an optimization happens, the ledger entry's `handle_id` /
    `tenant` / `tag` come from the substrate `Event` fields directly.
  - `sign_entry` keeps using HMAC over the full entry; key rotation
    via `MEMOPT_SIGNING_KEY` env var (already present).

### Pillar 6 — Silicon certification

| Archive file | Target in main | Action | Notes |
| --- | --- | --- | --- |
| `memopt/kernels/certification.py` | same | COPY | 805 LOC — main certification logic |
| `memopt/kernels/certify_daemon.py` | same | COPY | Imports `jit_generator`, `kernel_cache`, `portability_layer` (all in main) |
| `memopt/kernels/drift_detector.py` | same | COPY | New |
| `memopt/kernels/tests/test_certification.py` | same | COPY | |
| `memopt/kernels/tests/test_drift_detector.py` | same | COPY | |
| `memopt/kernels/tests/test_drift_hardening.py` | same | COPY | |
| `memopt/kernels/tests/test_feedback_loop_drift.py` | same | COPY | |
| `memopt/kernels/tests/test_control_plane_callback.py` | same | COPY | |
| `memopt/cli/__init__.py` | `memopt/cli/__init__.py` | **MERGE** | Add `cmd_certify` (lines 326-363 in archive) and the `certify` subparser block (lines 742-759 in archive) |
| `memopt/cli/certify.py` | same | COPY | Standalone `memopt-certify` entry point logic |
| `tests/pillar6_proof.py` | same | COPY | |
| `tests/test_pillar6_enhanced.py` | same | COPY | |

**Layer 1/2 wiring (Phase 4):**
  - `run_certification(...)` does not need substrate hooks; runs
    standalone GPU benchmarks. Optional integration: register a
    Layer-2 Policy that triggers re-certification when drift
    detector fires.
  - `certify_daemon` already imports kernel cache + JIT generator
    cleanly from main. No substrate change.

### Pillar 7 — GPU FinOps

| Archive file | Target in main | Action | Notes |
| --- | --- | --- | --- |
| `memopt/finops/__init__.py` | `memopt/finops/__init__.py` | COPY (empty) | New package |
| `memopt/finops/tracker.py` | same | COPY | 391 LOC — GPU $/hr cost tracker |
| `memopt/trust/__init__.py` | `memopt/trust/__init__.py` | COPY (empty) | New package |
| `memopt/trust/receipt.py` | same | COPY | 409 LOC — production trust receipt |
| `tests/pillar7_proof.py` | same | COPY | |
| `tests/test_production_receipt.py` | same | COPY | |

**Layer 1/2 wiring (Phase 4):**
  - `GPUFinOpsTracker` periodically polls `memopt.stats(tenant=t)`
    for in-use bytes; converts utilization → cost. Pure consumer of
    the substrate stats API; no substrate change.
  - `production_receipt(...)` aggregates `memopt.orchestrator.stats()`
    + finops + ledger into one signed receipt. Pure consumer.

### Things deleted in `c5705a3` but NOT in the archive (excluded from revival)

These were removed in the same commit but did not migrate anywhere.
Per the commit message ("optimizations deleted……") they look like
deliberate dead-code removal. **Excluded from the revival plan**
unless you say otherwise:

```
memopt/agent/optimization_agent.py             1341 LOC
memopt/agent/training_agent.py                  976 LOC
memopt/business/__init__.py                       1 LOC
memopt/migration/{__init__,engine}.py           521 LOC
memopt/optimization/memory_coalescer.py         615 LOC
memopt/optimization/__init__.py                   9 LOC
memopt/phase3/{__init__,auto_optimizer,
   flash_attention}.py                          981 LOC
memopt/daemon/{apply,roi_calculator,zero_touch}.py
                                                1690 LOC
memopt/certificates/{__init__,certificate}.py   280 LOC
                                            ─────────
                                            ~6 414 LOC
```

**Drift consequence (separate from revival):**
  - `memopt/daemon/__init__.py:15` imports the deleted
    `daemon/apply.py` → `import memopt.daemon` raises today.
  - `memopt/cli/__init__.py:239`, `memopt/control_plane/server.py:1186`
    reference deleted `memopt.certificates`.
  - `memopt/api/server.py:340` references deleted `memopt.phase3`.
  - `memopt/daemon/cli.py:147` references deleted `memopt.migration.engine`.

These will be cleaned up in milestone 5 (OSS hardening).

---

## Milestones

Each milestone is one PR. Each PR keeps the 876 baseline green and
adds tests at every step. No file in `memopt/substrate/` or
`memopt/orchestrator/` is touched.

### Milestone 1 — This document (DONE)

  - Output: `docs/pillar_revival_plan.md` (read-only, no source edits).
  - Verify with you that the archive scope matches your intent.

### Milestone 2 — Open-source license + housekeeping

  - **Add** `LICENSE` (Apache-2.0 — the standard for AI infra OSS;
    PyTorch, vLLM, Triton, llama.cpp all use it). Default unless you
    redirect to MIT / BSD-3 / AGPL.
  - **Add** `NOTICE` (Apache-2.0 boilerplate + third-party attributions).
  - **Add** `CONTRIBUTING.md` (how to file issues, PR rules, the
    876-baseline contract, the Layer 1/2 long-term-pin §H.4 rule).
  - **Add** `CODE_OF_CONDUCT.md` (Contributor Covenant).
  - **Edit** `pyproject.toml:10` `license = {text = "Proprietary"}`
    → `license = {text = "Apache-2.0"}` (single-line; reversible).
  - **Edit** `pyproject.toml:8` description: keep, harmless.
  - **Edit** `pyproject.toml` `classifiers`: add
    `"License :: OSI Approved :: Apache Software License"`.
  - Diff size: ~5 new files + 2 small `pyproject.toml` edits.
  - Regression: 876 must hold (no code change).

### Milestone 3 — Pillar 7 revival (smallest, no merges)

  - `memopt/finops/{__init__,tracker}.py` (COPY)
  - `memopt/trust/{__init__,receipt}.py` (COPY)
  - `tests/pillar7_proof.py` (COPY)
  - `tests/test_production_receipt.py` (COPY)
  - 6 new files; 0 main-repo files edited.
  - Regression target: 876 + N where N = #passing pillar 7 tests on Mac.

### Milestone 4 — Pillar 4 revival

  - 4 new source files under `memopt/observability/`
  - **MERGE** `memopt/observability/__init__.py` (3 new exports)
  - 3 new test files under `memopt/observability/tests/`
  - 2 new tests at `tests/`
  - Regression target: 876 + (M3 delta) + N4

### Milestone 5 — Pillar 6 revival + CLI merge

  - 3 new source files under `memopt/kernels/`
  - 5 new test files under `memopt/kernels/tests/`
  - **MERGE** `memopt/cli/__init__.py` (new `cmd_certify` + subparser)
  - `memopt/cli/certify.py` (COPY)
  - 2 new tests at `tests/`

### Milestone 6 — Layer 1/2 wiring

  - One commit per integration point:
    - Ledger subscribes to substrate events via `memopt.observe(...)`.
    - Optional `RecertifyOnDriftPolicy` registered via
      `memopt.orchestrator.register_policy(...)` (only on Phase B).
    - FinOps tracker pulls per-tenant in-use bytes via
      `memopt.stats(tenant=t)` on a poll loop.
    - Production receipt merges `memopt.orchestrator.stats()` +
      ledger + finops outputs.
  - Each commit is a new file + one wire-up; no shipped Layer 1/2
    file edited.

### Milestone 7 — OSS hardening + drift cleanup

  - Fix `memopt/daemon/__init__.py:15` (broken import — either
    re-add a no-op `apply.py` or remove the dead imports).
  - Remove deferred imports of deleted modules
    (`memopt.certificates`, `memopt.migration`, `memopt.phase3`)
    from `cli/__init__.py:239`, `daemon/cli.py:147`,
    `control_plane/server.py:1186`, `api/server.py:340`.
  - Remove `memopt-wrap` ghost entry from
    `pyproject.toml [project.scripts]`.
  - Update `README.md:5` test count `"339"` → current.
  - Add an OSS dependency-license audit (e.g. `pip-licenses`).
  - Scrub any internal hostnames / credentials from `tests/`,
    `scripts/`, `deploy/`.

### Milestone 8 — Public release prep

  - `CHANGELOG.md` v1.3.0 entry: pillars 4/6/7 revived; OSS license.
  - GitHub Actions CI matrix (Linux / Mac, Python 3.10–3.12).
  - Issue + PR templates.
  - `docs/architecture.md` refresh — current 8-pillar map + Layer
    1/2 boundary.
  - Tag `v1.3.0` and the OSS public-release tag.

---

## Risk register

| Risk | Severity | Mitigation |
| --- | --- | --- |
| Archive code uses old `memopt.*` paths that have since moved | LOW | Verified above — only 6 unique import lines, all targets exist in main |
| `memopt/cli/__init__.py` merge conflicts with future CLI work | LOW | Strict additive — only inserts a new `cmd_certify` and one subparser; existing handlers untouched |
| Pillar 4 ledger writes signed entries with secrets in repo | MEDIUM | Use `MEMOPT_SIGNING_KEY` env var (already the convention). OSS hardening pass scrubs any literal keys |
| Test count regression: pillar tests rely on hardware | MEDIUM | Use `pytest.importorskip("torch") + skipif` on Mac (substrate convention) |
| Open-source license choice irreversible | HIGH | Apache-2.0 default; one-line `pyproject.toml` change is easy to flip BEFORE first public release; impossible to take back AFTER |
| Reviving deleted dead code by accident | MEDIUM | Plan explicitly lists ~6 414 LOC of NON-archive deletions excluded from revival |
| Substrate Layer 1 / Orchestrator Layer 2 contracts broken | LOW | None of the milestone PRs touch `memopt/substrate/` or `memopt/orchestrator/` source. AC4 audit at every commit |

---

## What I need from you to proceed

  1. **License choice** — Apache-2.0 (default) or something else?
  2. **Trust archive disposition** — archive becomes obsolete when we
     finish revival. Delete the archive repo, mark it read-only, or
     leave it alongside?
  3. **Excluded ~6 414 LOC** — leave deleted (per `c5705a3` intent)
     or revive any of `agent/` / `phase3/` / `migration/` /
     `optimization/` / `daemon/{apply,zero_touch,roi_calculator}.py`?
  4. **Order** — proceed milestone-by-milestone with you reviewing
     each, or batch milestones 2-5 (license + 3 pillar revivals)
     into one larger PR? Recommended: milestone-by-milestone for
     non-destructive review.

Once you answer these I'll start milestone 2 (license + housekeeping)
immediately.
