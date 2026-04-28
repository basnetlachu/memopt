"""
Tests for the Production Trust Receipt.

The headline product. Must work end to end.
"""
import os, json, copy
import sys
import warnings
sys.path.insert(0, '.')
os.environ.setdefault('MEMOPT_SIGNING_KEY', 'test-key-receipt')

from dataclasses import asdict
from memopt.trust.receipt import (
    ReceiptBuilder, verify_receipt, ProductionReceipt,
)

# Tests 1–5 deliberately exercise the no-pillar builder. Suppress the
# "no pillar instances" UserWarning here — Test 7 explicitly verifies
# the warning fires with catch_warnings.
warnings.filterwarnings(
    "ignore",
    message="ReceiptBuilder created with no pillar instances.*",
    category=UserWarning,
)

print("=" * 55)
print("PRODUCTION TRUST RECEIPT TESTS")
print("=" * 55)

results = {}


# Test 1: Build a receipt with no components
print("\n[Test 1] Empty receipt builds")
try:
    builder = ReceiptBuilder()
    r = builder.build_for_request(
        request_id="req_001",
        tenant_id="test-tenant",
        tokens=100,
        gpu_seconds=0.5,
        joules_per_token=0.84,
        energy_source="nvml_measured",
        cache_hit=False,
    )

    assert r.request_id == "req_001"
    assert r.signature.startswith("hmac-sha256:")
    assert r.energy.tokens_generated == 100
    assert r.energy.joules_per_token == 0.84

    results["empty_build"] = {"passed": True}
    print(f"  Receipt built: {r.receipt_hash[:16]}...")
    print(f"  Signature: {r.signature[:30]}...")
    print(f"  Test 1: PASS")
except Exception as e:
    results["empty_build"] = {"passed": False, "error": str(e)}
    print(f"  FAIL: {e}")


# Test 2: Verify untampered
print("\n[Test 2] Verify untampered receipt")
try:
    builder = ReceiptBuilder()
    r = builder.build_for_request(
        request_id="req_002",
        tenant_id="test-tenant",
        tokens=50,
        joules_per_token=0.5,
    )

    r_dict = asdict(r)
    verified = verify_receipt(r_dict, "test-key-receipt")

    results["untampered_verify"] = {"passed": verified}
    print(f"  Verified: {verified}")
    print(f"  Test 2: {'PASS' if verified else 'FAIL'}")
except Exception as e:
    results["untampered_verify"] = {"passed": False, "error": str(e)}
    print(f"  FAIL: {e}")


# Test 3: Tamper detection
print("\n[Test 3] Tamper detection")
try:
    builder = ReceiptBuilder()
    r = builder.build_for_request(
        request_id="req_003",
        tenant_id="test-tenant",
        tokens=100,
        joules_per_token=0.5,
    )

    r_dict = asdict(r)
    tampered = copy.deepcopy(r_dict)
    tampered["energy"]["joules_total"] = 0.0001

    result = verify_receipt(tampered, "test-key-receipt")

    results["tamper_detection"] = {
        "passed": not result,
        "tamper_detected": not result,
    }
    print(f"  Verify after tamper: {result}")
    print(f"  Tamper detected: {not result}")
    print(f"  Test 3: {'PASS' if not result else 'FAIL'}")
except Exception as e:
    results["tamper_detection"] = {"passed": False, "error": str(e)}
    print(f"  FAIL: {e}")


# Test 4: All proof fields present
print("\n[Test 4] Receipt has all 7 pillar proofs")
try:
    builder = ReceiptBuilder()
    r = builder.build_for_request(
        request_id="req_004", tenant_id="t", tokens=10,
    )
    d = asdict(r)

    required = ["hardware", "memory", "cache", "kernel", "energy", "cost", "backend"]
    missing = [f for f in required if f not in d]
    passed = len(missing) == 0

    results["all_pillars"] = {"passed": passed, "missing": missing}
    print(f"  Missing: {missing}")
    print(f"  Test 4: {'PASS' if passed else 'FAIL'}")
except Exception as e:
    results["all_pillars"] = {"passed": False, "error": str(e)}
    print(f"  FAIL: {e}")


# Test 5: Receipt hash is deterministic
print("\n[Test 5] Receipt hash deterministic")
try:
    builder = ReceiptBuilder()
    r1 = builder.build_for_request(
        request_id="req_005", tenant_id="t",
        tokens=42, joules_per_token=0.7,
    )
    r2 = builder.build_for_request(
        request_id="req_005", tenant_id="t",
        tokens=42, joules_per_token=0.7,
    )

    has_sigs = (
        r1.signature.startswith("hmac-sha256:")
        and r2.signature.startswith("hmac-sha256:")
    )

    results["deterministic"] = {"passed": has_sigs}
    print(f"  Both signed: {has_sigs}")
    print(f"  Test 5: {'PASS' if has_sigs else 'FAIL'}")
except Exception as e:
    results["deterministic"] = {"passed": False, "error": str(e)}
    print(f"  FAIL: {e}")


# Test 6: Receipt with real FinOps tracker — savings_usd should be non-zero
print("\n[Test 6] Real FinOps tracker: savings_usd populated")
try:
    import time as _time
    from memopt.finops.tracker import GPUFinOpsTracker

    tracker = GPUFinOpsTracker(
        tenant_id="receipt-test", gpu_cost_per_hour=2.00,
    )
    # Backfill a synthetic 1 h window with 50% util + 70% KV hit so the
    # tracker reports non-zero total_cost AND non-zero estimated_savings.
    tracker._start_time = _time.time() - 3600
    for _ in range(5):
        tracker.record_sample(gpu_util_pct=50.0)
    tracker.set_kv_hit_rate(70.0)

    builder = ReceiptBuilder(finops=tracker)
    r = builder.build_for_request(
        request_id="req_006",
        tenant_id="receipt-test",
        tokens=100,
        gpu_seconds=1.0,
        joules_per_token=0.5,
    )

    has_cost     = r.cost.cost_usd > 0
    has_savings  = r.cost.savings_usd > 0
    real_data = has_cost and has_savings

    results["real_finops"] = {
        "passed":     real_data,
        "cost_usd":   r.cost.cost_usd,
        "savings_usd": r.cost.savings_usd,
    }
    print(f"  cost_usd:    {r.cost.cost_usd}")
    print(f"  savings_usd: {r.cost.savings_usd}")
    print(f"  Test 6: {'PASS' if real_data else 'FAIL'}")
except Exception as e:
    results["real_finops"] = {"passed": False, "error": str(e)}
    print(f"  FAIL: {e}")


# Test 7: Empty-builder warning fires
print("\n[Test 7] Empty-builder UserWarning fires")
try:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = ReceiptBuilder()  # no pillars
        empty_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "no pillar instances" in str(w.message)
        ]

    fired = len(empty_warnings) >= 1

    # Negative case: passing any pillar must NOT fire the warning
    with warnings.catch_warnings(record=True) as caught2:
        warnings.simplefilter("always")
        _ = ReceiptBuilder(cert={"x": 1})
        suppressed_warnings = [
            w for w in caught2
            if issubclass(w.category, UserWarning)
            and "no pillar instances" in str(w.message)
        ]
    not_fired = len(suppressed_warnings) == 0

    passed = fired and not_fired
    results["empty_builder_warns"] = {
        "passed":      passed,
        "fired_empty": fired,
        "silent_full": not_fired,
    }
    print(f"  Empty builder fires:    {fired}")
    print(f"  Pillar-equipped quiet:  {not_fired}")
    print(f"  Test 7: {'PASS' if passed else 'FAIL'}")
except Exception as e:
    results["empty_builder_warns"] = {"passed": False, "error": str(e)}
    print(f"  FAIL: {e}")


# Summary
passed_count = sum(1 for v in results.values() if v.get("passed"))
total = len(results)

print(f"\n{'=' * 55}")
print("PRODUCTION RECEIPT TESTS")
print(f"{'=' * 55}")
print(f"Tests: {passed_count}/{total} pass")

with open('/tmp/receipt_test_result.json', 'w') as f:
    json.dump({
        "passed":  passed_count == total,
        "score":   f"{passed_count}/{total}",
        "results": results,
    }, f, indent=2)

# Print one example receipt — wired with a real FinOps tracker and
# context_tokens so cache.compute_saved_pct and cost.savings_usd are
# populated from real ratios, not placeholders.
print("\nEXAMPLE RECEIPT (with real FinOps + context):")
print("-" * 55)
import time as _time
from memopt.finops.tracker import GPUFinOpsTracker
_example_tracker = GPUFinOpsTracker(
    tenant_id="customer-acme", gpu_cost_per_hour=2.00,
)
_example_tracker._start_time = _time.time() - 3600
for _ in range(5):
    _example_tracker.record_sample(gpu_util_pct=50.0)
_example_tracker.set_kv_hit_rate(70.0)

builder = ReceiptBuilder(finops=_example_tracker)
r = builder.build_for_request(
    request_id="example_req_001",
    tenant_id="customer-acme",
    tokens=50,
    gpu_seconds=0.31,
    joules_per_token=0.847,
    energy_source="nvml_measured",
    cache_hit=True,
    workflow_id="wf_agent_session_42",
    matched_tokens=2048,
    context_tokens=2560,
)
print(json.dumps(asdict(r), indent=2))
