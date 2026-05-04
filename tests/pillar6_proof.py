"""
Pillar 6 proof — Silicon Certification.

Tests what can be verified without a GPU.
GPU tests are deferred to server.
"""
import sys, os, json
sys.path.insert(0, '.')
os.environ.setdefault('MEMOPT_SIGNING_KEY', 'test-key-p6')

print("=" * 50)
print("PILLAR 6 — SILICON CERTIFICATION")
print("=" * 50)

results = {}


# Test 1: CLI module imports cleanly
print("\n[Test 1] CLI imports")
try:
    from memopt.cli.certify import (
        format_text_output,
        compare_with_baseline,
        save_baseline,
        load_baseline,
        get_cert_history_path,
    )
    results["cli_imports"] = {"passed": True}
    print("  All CLI functions importable: PASS")
except Exception as e:
    results["cli_imports"] = {"passed": False, "error": str(e)}
    print(f"  FAIL: {e}")


# Test 2: format_text_output
print("\n[Test 2] Text formatter")
try:
    from memopt.cli.certify import format_text_output

    fake_cert = {
        "gpu":                   "NVIDIA H100 80GB HBM3",
        "timestamp":             "2026-04-27T10:00:00Z",
        "ops_passed":            10,
        "ops_total":             10,
        "bandwidth_pct_of_peak": 88.78,
        "power_watts_nvml":      115.5,
        "certificate_status":    "CERTIFIED",
        "signature_status":      "hmac-sha256:abc123",
    }

    output = format_text_output(fake_cert)
    required = ["CERTIFIED", "88.78", "H100", "10/10"]
    missing = [r for r in required if r not in output]
    passed2 = len(missing) == 0

    results["text_formatter"] = {"passed": passed2, "missing": missing}
    print(f"  Missing: {missing}")
    print(f"  Test 2: {'PASS' if passed2 else 'FAIL'}")
except Exception as e:
    results["text_formatter"] = {"passed": False, "error": str(e)}
    print(f"  FAIL: {e}")


# Test 3: baseline save/load round trip
print("\n[Test 3] Baseline save/load")
try:
    from memopt.cli.certify import save_baseline, load_baseline
    import tempfile, unittest.mock as mock

    fake_cert = {
        "certificate_status":    "CERTIFIED",
        "bandwidth_pct_of_peak": 88.78,
        "timestamp":             "2026-04-27T10:00:00Z",
    }

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "baseline.json")
        with mock.patch(
            "memopt.cli.certify.get_cert_history_path",
            return_value=path,
        ):
            save_baseline(fake_cert)
            loaded = load_baseline()

    passed3 = (
        loaded.get("certificate_status") == "CERTIFIED"
        and loaded.get("bandwidth_pct_of_peak") == 88.78
    )
    results["baseline_roundtrip"] = {"passed": passed3}
    print(f"  Round trip: {passed3}")
    print(f"  Test 3: {'PASS' if passed3 else 'FAIL'}")
except Exception as e:
    results["baseline_roundtrip"] = {"passed": False, "error": str(e)}
    print(f"  FAIL: {e}")


# Test 4: drift comparison
print("\n[Test 4] Drift detection")
try:
    from memopt.cli.certify import compare_with_baseline

    baseline = {"bandwidth_pct_of_peak": 90.0}

    current_ok    = {"bandwidth_pct_of_peak": 89.0}
    current_drift = {"bandwidth_pct_of_peak": 84.0}
    r1 = compare_with_baseline(current_ok,    baseline)
    r2 = compare_with_baseline(current_drift, baseline)

    passed4 = r1.get("degraded") is False and r2.get("degraded") is True
    results["drift_detection"] = {
        "passed":            passed4,
        "no_drift_degraded": r1.get("degraded"),
        "drift_degraded":    r2.get("degraded"),
    }
    print(f"  No drift flagged:   {not r1.get('degraded')}")
    print(f"  Real drift flagged: {r2.get('degraded')}")
    print(f"  Test 4: {'PASS' if passed4 else 'FAIL'}")
except Exception as e:
    results["drift_detection"] = {"passed": False, "error": str(e)}
    print(f"  FAIL: {e}")


# Test 5: SiliconCertification exists
print("\n[Test 5] SiliconCertification class")
try:
    from memopt.kernels.certification import (
        SiliconCertification, CertificationConfig,
    )
    config = CertificationConfig()
    cert = SiliconCertification(config=config)

    has_run = hasattr(cert, "run")
    results["cert_class"] = {"passed": has_run, "has_run_method": has_run}
    print(f"  Has run() method: {has_run}")
    print(f"  Test 5: {'PASS' if has_run else 'FAIL'}")
except Exception as e:
    results["cert_class"] = {"passed": False, "error": str(e)}
    print(f"  FAIL: {e}")


# Test 6: GPU certification (deferred)
print("\n[Test 6] Full certification (GPU required)")
print("  SKIP — requires CUDA hardware")
print("  Will run on GPU server")
results["gpu_cert"] = {"passed": None, "note": "GPU required"}


# Summary
local = {k: v for k, v in results.items() if v.get("passed") is not None}
passed_count = sum(1 for v in local.values() if v.get("passed"))
total = len(local)

print(f"\n{'=' * 50}")
print("PILLAR 6 RESULT (local)")
print(f"{'=' * 50}")
print(f"Local: {passed_count}/{total} pass")
print("GPU: deferred to server")
print(f"PASS: {passed_count == total}")

with open('/tmp/pillar6_result.json', 'w') as f:
    json.dump({
        "passed":  passed_count == total,
        "score":   f"{passed_count}/{total}",
        "results": results,
    }, f, indent=2)
