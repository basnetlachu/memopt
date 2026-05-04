# Production Readiness Checklist

> **Read this before deploying memopt to real GPU traffic.**
>
> The release maintainer (the author of `v1.3.0a1`) tested on Mac
> only. The 18 cuda-named tests + 2 `@gpu` tests are
> SKIPPED / DESELECTED on the release host. Anything you find on
> your hardware is your finding to work through.

This document is the single source of truth for "is my deployment
production-ready?" Run through every section below in order. Capture
results in a `production_readiness_<date>.md` file in your fork.

---

## Section A — GPU regression sweep

**Required before any production traffic. No exceptions.**

```bash
# 1. Clone + install on the target GPU host (NOT Mac).
git clone <your-fork-url>
cd memopt
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,daemon,api]"

# 2. Run the FULL regression — no -k filter this time.
PYTHONPATH=. python3 -m pytest tests/ memopt/ -q -p no:cacheprovider \
  --ignore=memopt/vmm/tests/test_torch_allocator_sanity.py

# 3. Capture the final summary line. Expected:
#    ~1036 passed (1016 + ~20 GPU-specific), 0 failed.
```

**Pass criterion:** 0 FAILED. Anything else needs to be triaged before
proceeding to Section B.

**If FAILED > 0:**

  - For each failing test, read the error. CUDA-specific bugs
    (driver version mismatches, NVML version skew, CUDA-12 vs CUDA-13
    syntax differences) are typical.
  - Fix the bug or open an issue at the upstream repo.
  - Do not push a release tag until 0 FAILED is restored.

---

## Section B — Public-API smoke on GPU

```bash
MEMOPT_SIGNING_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))") \
python3 -c "
import memopt
import memopt.orchestrator
import memopt.integrations
import memopt.observability
import memopt.finops.tracker
import memopt.trust.receipt
import memopt.kernels.certification

# Layer 1 + Layer 2 round-trip on real GPU.
with memopt.context(tenant='prod-test', placement='hbm'):
    h = memopt.alloc(4 << 20)        # 4 MiB on real HBM
    memopt.free(h)

# Orchestrator round-trip.
memopt.orchestrator.start()
print(memopt.orchestrator.stats())
memopt.orchestrator.stop()

# Pillar 6 silicon cert on real device.
from memopt.kernels.certification import run_certification
cert = run_certification(node_id='prod-host-1')
print(f'cert.all_passed = {cert.all_passed}')
print(f'cert.device_name = {cert.device_name}')
print(f'cert.compute_cap = {cert.compute_cap}')

print('PRODUCTION SMOKE OK')
"
```

**Pass criterion:** prints `PRODUCTION SMOKE OK` and `cert.all_passed = True`.

---

## Section C — Soak test (≥ 24 h)

Run a representative inference workload with all pillars wired:

```python
import memopt
import memopt.orchestrator as orch
from memopt.integrations import (
    attach_ledger_to_substrate, FinOpsPoller,
)
from memopt.observability.ledger import OptimizationLedger
from memopt.finops.tracker import GPUFinOpsTracker

orch.start()
ledger = OptimizationLedger()
sub_handles = attach_ledger_to_substrate(ledger)
finops = GPUFinOpsTracker()
poller = FinOpsPoller(finops, tenants=["prod"], period_s=10.0)
poller.start()

# ... your workload (memopt-serve, vLLM, etc.) runs for ≥ 24 h ...
```

Every hour, log:

  - `memopt.orchestrator.stats()["coordinator"]["queue_drops"]`
  - `memopt.orchestrator.stats()["coordinator"]["cycles"]`
  - `memopt.stats(tenant=t)["events_dropped"]` for each tenant
  - Process RSS (e.g. `psutil.Process().memory_info().rss`)
  - GPU memory (`nvidia-smi --query-gpu=memory.used --format=csv`)

**Pass criteria:**

  - `queue_drops` stays at 0 (or grows < 0.1% of `events_ingested.alloc`)
  - `events_dropped` stays at 0
  - Process RSS does not grow unbounded (allow ~10% drift over 24 h)
  - GPU memory does not leak

If any of these regress, capture the timeline and open an issue
**before** moving to production.

---

## Section D — Security checklist

  - [ ] `MEMOPT_SIGNING_KEY` is a high-entropy secret (≥ 32 random bytes,
        NOT the default `""` and NOT a placeholder like
        `"production-test-key"`). Store in your secret manager (Vault,
        AWS Secrets Manager, GCP Secret Manager), not in shell history
        or `.env` files.
  - [ ] `MEMOPT_API_KEY` rotated per deployment.
  - [ ] `MEMOPT_ADMIN_TOKEN` only set on hosts that need
        `memopt.stats(tenant=None)` aggregate access (G2 isolation).
  - [ ] `MEMOPT_NVME_DIR` points at a directory with mode `0700` owned
        by the service user (G3 tenant-namespacing).
  - [ ] Network ports for `memopt-serve` and the control plane are
        firewalled.
  - [ ] If exposing the FastAPI surface to the internet, put it
        behind authenticated TLS (the API key alone is not enough).

---

## Section E — Layer-2 mode selection

**v1.3.0a1 ships Layer 2 in observation-only mode** per
`docs/orchestrator_v1_design.md` DECISION 7. Layer 2 records access
events and offers stats, but does NOT drive evictions or rewrite
placements. The legacy `TierManager` (Layer 1 / Pillar 1) is the
active eviction policy.

**To enable Layer 2 to drive placement decisions:**

  - **Phase B (`MEMOPT_USE_ORCHESTRATOR=1`)** — not yet shipped in
    this release. Track in `docs/orchestrator_v1_design.md` §2.7
    "Phase B." Targeted for v1.4.0.

If your deployment requires Layer 2 to actually optimize placement,
**this release is not the right release.** Pin to a future v1.4.0+
and re-run this checklist when it ships.

---

## Section F — Operational hygiene

  - [ ] Logs ingested into your observability stack (loki / cloudwatch /
        stackdriver). Grep targets: `memopt.orchestrator.coordinator`,
        `memopt.observability.ledger`, `memopt.kernels.certification`.
  - [ ] Prometheus scrape configured against the
        `prometheus_client` ASGI app at `/metrics`. The metric name
        prefix is `memopt_*`.
  - [ ] Pillar 6 (silicon certification) re-runs daily via
        `memopt-certify --node-id <host>`. Compare to the prior day's
        cert hash; alert on drift.
  - [ ] Pillar 4 ledger DB (`~/.memopt/ledger.db` or
        `MEMOPT_LEDGER_DB_PATH`) backed up nightly.
  - [ ] Pillar 7 finops report exported weekly as audit trail.

---

## Section G — Sign-off

When all sections above are checked:

  - [ ] Section A — GPU regression: 0 FAILED on target hardware.
  - [ ] Section B — Public-API smoke prints `PRODUCTION SMOKE OK`.
  - [ ] Section C — 24h soak: counters stable.
  - [ ] Section D — All secrets rotated, no defaults in use.
  - [ ] Section E — Layer-2 mode matches your deployment intent.
  - [ ] Section F — Observability + cert + backups wired.

Then — and only then — bump the version from `1.3.0a1` to a beta
(`1.3.0b1`) or stable (`1.3.0`) tag in your fork and announce.
