"""
Pillar 2 proof — Agentic KV Memory.

Tests:
  1. Basic dedup hit rate >= 60% on shared prefix (uses existing
     token-based GKDStore.lookup/register; LCP supplies prefix hits)
  2. Workflow-scoped lookup works (new register_workflow_block /
     lookup_workflow)
  3. TTL expiry works (per-entry expires_at)
  4. Cross-tenant isolation (uses MEMOPT_GKD_TENANT_ISOLATION=true so
     identical tokens hash differently per tenant)

Spec deviations vs original prompt — see PILLAR2 SUMMARY in agent
output. The repo's GKDStore.lookup/register take (token_ids, seq_len),
not (content_hash). compute_hash() in memopt.cluster.hashing also takes
token_ids, not bytes. result.is_hit doesn't exist — lookup returns
Optional[GKDHit] (None == miss).
"""
import os
# Tenant isolation must be set BEFORE importing/instantiating GKDStore.
os.environ["MEMOPT_GKD_TENANT_ISOLATION"] = "true"

import sys, time, json, hashlib
sys.path.insert(0, ".")

from memopt.cluster.gkd_store import GKDStore

print("=" * 50)
print("PILLAR 2 — AGENTIC KV MEMORY PROOF")
print("=" * 50)

store = GKDStore()
results = {}


def text_to_tokens(s: str):
    """Map UTF-8 bytes → token IDs for the token-based register/lookup."""
    return list(s.encode("utf-8"))


# ────────────────────────────────────────────────────────────────────
# Test 1: hit rate on shared prefix workload (token-based API + LCP)
# ────────────────────────────────────────────────────────────────────
print("\n[Test 1] Hit rate on shared prefix workload")
SHARED = "System: You are a helpful AI. " * 100
hits = 0
total = 100

for i in range(total):
    if i % 10 < 7:
        content = SHARED + f"User: question {i % 20}"
    else:
        content = f"Unique request {i} " * 20
    tids = text_to_tokens(content)

    result = store.lookup(tids, len(tids), tenant_id="t1")
    if result is not None:
        hits += 1
    else:
        store.register(
            tids, len(tids),
            block_ref=f"block_{i}",
            node_id="node1",
            size_bytes=len(content),
            tenant_id="t1",
        )

hit_rate = hits / total * 100
passed1 = hit_rate >= 60.0
results["hit_rate"] = {
    "hit_rate_pct": round(hit_rate, 1),
    "passed": passed1,
}
print(f"  Hit rate: {hit_rate:.1f}%")
print(f"  Test 1: {'PASS' if passed1 else 'FAIL'}")


# ────────────────────────────────────────────────────────────────────
# Test 2: workflow-scoped lookup (new API)
# ────────────────────────────────────────────────────────────────────
print("\n[Test 2] Workflow-scoped lookup")
content = "Agent context step 1 " * 50
h = hashlib.sha256(content.encode()).hexdigest()

store.register_workflow_block(
    content_hash=h,
    block_ref="wf_block_1",
    node_id="node1",
    size_bytes=len(content),
    tenant_id="t1",
    workflow_id="wf_abc",
    ttl_seconds=3600,
)

r1 = store.lookup_workflow(h, tenant_id="t1", workflow_id="wf_abc")
r2 = store.lookup_workflow(h, tenant_id="t1", workflow_id="wf_xyz")
passed2 = (r1 is not None) and (r2 is None)
results["workflow_scoping"] = {"passed": passed2}
print(f"  Correct workflow: hit={r1 is not None}")
print(f"  Wrong workflow:   hit={r2 is not None}")
print(f"  Test 2: {'PASS' if passed2 else 'FAIL'}")


# ────────────────────────────────────────────────────────────────────
# Test 3: TTL expiry
# ────────────────────────────────────────────────────────────────────
print("\n[Test 3] TTL expiry")
content = "Short lived block " * 20
h = hashlib.sha256(content.encode()).hexdigest()

store.register_workflow_block(
    content_hash=h,
    block_ref="ttl_block",
    node_id="node1",
    size_bytes=len(content),
    tenant_id="t1",
    workflow_id="wf_ttl",
    ttl_seconds=1,
)

r1 = store.lookup_workflow(h, tenant_id="t1", workflow_id="wf_ttl")
time.sleep(2)
r2 = store.lookup_workflow(h, tenant_id="t1", workflow_id="wf_ttl")
passed3 = (r1 is not None) and (r2 is None)
results["ttl_expiry"] = {"passed": passed3}
print(f"  Before expiry: hit={r1 is not None}")
print(f"  After expiry:  hit={r2 is not None}")
print(f"  Test 3: {'PASS' if passed3 else 'FAIL'}")


# ────────────────────────────────────────────────────────────────────
# Test 4: cross-tenant isolation (token-based API + isolation flag)
# ────────────────────────────────────────────────────────────────────
print("\n[Test 4] Cross-tenant isolation")
content = "Shared content both tenants " * 30
tids = text_to_tokens(content)

store.register(
    tids, len(tids),
    block_ref="t2_block",
    node_id="node1",
    size_bytes=len(content),
    tenant_id="t2",
)

r_t2 = store.lookup(tids, len(tids), tenant_id="t2")
r_t3 = store.lookup(tids, len(tids), tenant_id="t3")
passed4 = (r_t2 is not None) and (r_t3 is None)
results["tenant_isolation"] = {"passed": passed4}
print(f"  Correct tenant: hit={r_t2 is not None}")
print(f"  Wrong tenant:   hit={r_t3 is not None}")
print(f"  Test 4: {'PASS' if passed4 else 'FAIL'}")


# ────────────────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────────────────
all_passed = all(v.get("passed", False) for v in results.values())
passed_count = sum(1 for v in results.values() if v.get("passed", False))

print(f"\n{'=' * 50}")
print("PILLAR 2 RESULT")
print(f"{'=' * 50}")
print(f"Tests passed: {passed_count}/{len(results)}")
print(f"PASS: {all_passed}")

with open("/tmp/pillar2_result.json", "w") as f:
    json.dump(
        {
            "passed": all_passed,
            "tests_passed": passed_count,
            "total_tests": len(results),
            "results": results,
        },
        f, indent=2,
    )
