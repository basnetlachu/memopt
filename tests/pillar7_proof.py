"""
Pillar 7 proof — GPU FinOps Intelligence.
All tests run locally without GPU.
"""
import sys, os, json, time
sys.path.insert(0, '.')
os.environ.setdefault('MEMOPT_SIGNING_KEY', 'test-key-p7')

print("=" * 50)
print("PILLAR 7 — GPU FINOPS INTELLIGENCE")
print("=" * 50)

results = {}


# Test 1: Module imports
print("\n[Test 1] Module imports")
try:
    from memopt.finops.tracker import (
        GPUFinOpsTracker, FinOpsReport,
        estimate_annual_waste, get_gpu_cost_per_hour,
    )
    results["imports"] = {"passed": True}
    print("  All imports: PASS")
except Exception as e:
    results["imports"] = {"passed": False, "error": str(e)}
    print(f"  FAIL: {e}")


# Test 2: Cost calculation
print("\n[Test 2] Waste cost calculation")
try:
    from memopt.finops.tracker import GPUFinOpsTracker

    tracker = GPUFinOpsTracker(tenant_id="test", gpu_cost_per_hour=2.00)
    tracker._start_time = time.time() - 3600

    for _ in range(10):
        tracker.record_sample(
            gpu_util_pct=5.0, memory_util_pct=20.0, power_watts=100.0,
        )

    report = tracker.get_report(sign=False)

    passed2 = (
        report.waste_pct > 90
        and report.wasted_cost_usd > 0
        and report.total_cost_usd > 0
    )
    results["cost_calculation"] = {
        "passed":     passed2,
        "waste_pct":  report.waste_pct,
        "wasted_usd": report.wasted_cost_usd,
        "total_usd":  report.total_cost_usd,
    }
    print(f"  Utilization:  5%")
    print(f"  Waste:        {report.waste_pct}%")
    print(f"  Wasted USD:   ${report.wasted_cost_usd}")
    print(f"  Total USD:    ${report.total_cost_usd}")
    print(f"  Test 2: {'PASS' if passed2 else 'FAIL'}")
except Exception as e:
    results["cost_calculation"] = {"passed": False, "error": str(e)}
    print(f"  FAIL: {e}")


# Test 3: KV savings estimate
print("\n[Test 3] KV cache savings estimate")
try:
    from memopt.finops.tracker import GPUFinOpsTracker

    tracker = GPUFinOpsTracker(tenant_id="test", gpu_cost_per_hour=2.00)
    tracker._start_time = time.time() - 3600
    tracker.set_kv_hit_rate(69.8)

    for _ in range(5):
        tracker.record_sample(gpu_util_pct=50.0)

    report = tracker.get_report(sign=False)

    passed3 = report.estimated_savings_usd > 0
    results["kv_savings"] = {
        "passed":      passed3,
        "hit_rate":    report.kv_cache_hit_rate_pct,
        "savings_usd": report.estimated_savings_usd,
    }
    print(f"  KV hit rate:  69.8%")
    print(f"  Savings est.: ${report.estimated_savings_usd}")
    print(f"  Test 3: {'PASS' if passed3 else 'FAIL'}")
except Exception as e:
    results["kv_savings"] = {"passed": False, "error": str(e)}
    print(f"  FAIL: {e}")


# Test 4: Signature round-trip
print("\n[Test 4] HMAC signature round-trip")
try:
    from memopt.finops.tracker import GPUFinOpsTracker

    tracker = GPUFinOpsTracker(tenant_id="test")
    tracker._start_time = time.time() - 60
    tracker.record_sample(gpu_util_pct=10.0)

    report = tracker.get_report(sign=True)

    has_prefix   = report.signature.startswith("hmac-sha256:")
    verified     = tracker.verify_signature(report)
    report.waste_pct = 99.9                       # tamper
    tamper_fails = not tracker.verify_signature(report)

    passed4 = has_prefix and verified and tamper_fails

    results["signature"] = {
        "passed":          passed4,
        "has_prefix":      has_prefix,
        "verified":        verified,
        "tamper_detected": tamper_fails,
    }
    print(f"  Has prefix:       {has_prefix}")
    print(f"  Verified:         {verified}")
    print(f"  Tamper detected:  {tamper_fails}")
    print(f"  Test 4: {'PASS' if passed4 else 'FAIL'}")
except Exception as e:
    results["signature"] = {"passed": False, "error": str(e)}
    print(f"  FAIL: {e}")


# Test 5: Annual waste estimate
print("\n[Test 5] Annual waste estimate")
try:
    from memopt.finops.tracker import estimate_annual_waste

    est = estimate_annual_waste(
        avg_util_pct=5.0, gpu_count=1000, gpu_cost_per_hour=2.00,
    )

    passed5 = (
        est["annual_wasted_usd"] > 10_000_000
        and est["waste_pct"] == 95.0
    )
    results["annual_estimate"] = {
        "passed":        passed5,
        "annual_wasted": est["annual_wasted_usd"],
        "waste_pct":     est["waste_pct"],
    }
    print(f"  1000 GPUs × $2/hr × 8760hrs:")
    print(f"  Annual waste: ${est['annual_wasted_usd']:,.0f}")
    print(f"  Waste pct:    {est['waste_pct']}%")
    print(f"  Test 5: {'PASS' if passed5 else 'FAIL'}")
except Exception as e:
    results["annual_estimate"] = {"passed": False, "error": str(e)}
    print(f"  FAIL: {e}")


# Test 6: JSON export
print("\n[Test 6] JSON export")
try:
    from memopt.finops.tracker import GPUFinOpsTracker

    tracker = GPUFinOpsTracker(tenant_id="test")
    tracker._start_time = time.time() - 60
    tracker.record_sample(gpu_util_pct=15.0)

    j = tracker.to_json()
    data = json.loads(j)

    required = ["tenant_id", "waste_pct", "wasted_cost_usd", "generated_at"]
    missing = [f for f in required if f not in data]
    passed6 = len(missing) == 0

    results["json_export"] = {"passed": passed6, "missing": missing}
    print(f"  Missing fields: {missing}")
    print(f"  Test 6: {'PASS' if passed6 else 'FAIL'}")
except Exception as e:
    results["json_export"] = {"passed": False, "error": str(e)}
    print(f"  FAIL: {e}")


# Summary
passed_count = sum(1 for v in results.values() if v.get("passed"))
total = len(results)

print(f"\n{'=' * 50}")
print("PILLAR 7 RESULT")
print(f"{'=' * 50}")
print(f"Tests: {passed_count}/{total} pass")
print(f"PASS: {passed_count == total}")

with open('/tmp/pillar7_result.json', 'w') as f:
    json.dump({
        "passed":  passed_count == total,
        "score":   f"{passed_count}/{total}",
        "results": results,
    }, f, indent=2)
