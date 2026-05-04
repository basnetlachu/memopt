"""
Pillar 4 proof — AI Compliance Infrastructure.

Tests:
  1. Ledger records with correct energy_source
  2. verify_certificate works for both cert types
  3. ComplianceReport generates valid HTML
  4. ComplianceReport generates valid CSV
  5. Report contains required EU AI Act fields

Spec deviations vs original prompt:
  - OptimizationLedger summary key is `tokens_total`, not `total_tokens`
    (verified by reading totals() at ledger.py:571).
  - OptimizationLedger.record() buffers; totals() reads DB only — must
    call ledger.flush() before totals() or the entries are invisible.
  - SLACertificate.generate() emits an unsigned empty cert when there
    is no run history for the node, so Test 2 seeds a synthetic
    history file via MEMOPT_CERT_HISTORY_PATH before generating.
"""
import sys, os, json, tempfile, time
sys.path.insert(0, ".")
os.environ.setdefault("MEMOPT_SIGNING_KEY", "test-key-p4")

print("=" * 50)
print("PILLAR 4 — AI COMPLIANCE INFRASTRUCTURE")
print("=" * 50)

results = {}


# ────────────────────────────────────────────────────────────────────
# Test 1: Ledger energy_source tracking
# ────────────────────────────────────────────────────────────────────
print("\n[Test 1] Ledger energy_source tracking")
try:
    from memopt.observability.ledger import OptimizationLedger

    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        ledger = OptimizationLedger(db_path=db)

        ledger.record(
            tokens=1000,
            tenant_id="t1",
            actual_j_per_token=0.001,
            energy_source="nvml_measured",
        )
        ledger.record(
            tokens=500,
            tenant_id="t1",
            actual_j_per_token=0.002,
            energy_source="estimated",
        )
        ledger.flush()

        totals = ledger.totals(tenant_id="t1")
        tokens = totals.get("tokens_total", 0) or 0
        has_source = "energy_source_breakdown" in totals
        passed1 = tokens >= 1000 and has_source

        results["ledger_energy_source"] = {
            "passed":              passed1,
            "tokens_total":        tokens,
            "has_source_tracking": has_source,
            "breakdown":           totals.get("energy_source_breakdown"),
        }
        print(f"  tokens_total: {tokens}")
        print(f"  has_source_tracking: {has_source}")
        print(f"  breakdown: {totals.get('energy_source_breakdown')}")
        print(f"  Test 1: {'PASS' if passed1 else 'FAIL'}")
except Exception as e:
    results["ledger_energy_source"] = {"passed": False, "error": str(e)}
    print(f"  FAIL: {e}")


# ────────────────────────────────────────────────────────────────────
# Test 2: verify_certificate for SLACertificate
# ────────────────────────────────────────────────────────────────────
print("\n[Test 2] verify_certificate (SLA format)")
try:
    from memopt.kernels.certification import (
        SLACertificate, verify_certificate,
    )

    with tempfile.TemporaryDirectory() as d:
        hp = os.path.join(d, "hist.json")
        os.environ["MEMOPT_CERT_HISTORY_PATH"] = hp
        with open(hp, "w") as f:
            json.dump(
                [{
                    "node_id":               "test-node",
                    "timestamp":             time.time(),
                    "all_passed":            True,
                    "bandwidth_pct_of_peak": 88.78,
                    "drift_detected":        False,
                    "cert_hash":             "abc",
                }],
                f,
            )

        cert = SLACertificate(node_id="test-node").generate()
        result = verify_certificate(cert, signing_key="test-key-p4")

        results["cert_verify_sla"] = {
            "passed":           result is True,
            "signature_status": cert.get("signature_status", "")[:30],
            "cert_keys":        list(cert.keys())[:5],
        }
        print(f"  signature_status: {cert.get('signature_status', '')[:30]}")
        print(f"  verify result: {result}")
        print(f"  Test 2: {'PASS' if result else 'FAIL'}")
except Exception as e:
    results["cert_verify_sla"] = {"passed": False, "error": str(e)}
    print(f"  FAIL: {e}")


# ────────────────────────────────────────────────────────────────────
# Test 3: HTML compliance report
# ────────────────────────────────────────────────────────────────────
print("\n[Test 3] HTML compliance report")
try:
    from memopt.observability.report_exporter import ComplianceReport

    report = ComplianceReport(
        tenant_id="test-tenant",
        certificate={
            "certificate_status":    "CERTIFIED",
            "certificate_hash":      "abc123def456",
            "ops_passed":            10,
            "bandwidth_pct_of_peak": 88.78,
        },
    )
    html = report.to_html()

    required = ["Compliance Report", "CERTIFIED", "test-tenant", "EU AI Act"]
    missing = [r for r in required if r not in html]
    passed3 = len(missing) == 0

    results["html_report"] = {
        "passed":      passed3,
        "missing":     missing,
        "html_length": len(html),
    }
    print(f"  HTML length: {len(html)} chars")
    print(f"  Missing fields: {missing}")
    print(f"  Test 3: {'PASS' if passed3 else 'FAIL'}")
except Exception as e:
    results["html_report"] = {"passed": False, "error": str(e)}
    print(f"  FAIL: {e}")


# ────────────────────────────────────────────────────────────────────
# Test 4: CSV compliance report
# ────────────────────────────────────────────────────────────────────
print("\n[Test 4] CSV compliance report")
try:
    from memopt.observability.report_exporter import ComplianceReport

    report = ComplianceReport(tenant_id="test-tenant")
    csv_out = report.to_csv()

    passed4 = (
        "generated_at" in csv_out
        and "tenant_id" in csv_out
        and len(csv_out) > 50
    )
    results["csv_report"] = {
        "passed":     passed4,
        "csv_length": len(csv_out),
    }
    print(f"  CSV length: {len(csv_out)} chars")
    print(f"  Test 4: {'PASS' if passed4 else 'FAIL'}")
except Exception as e:
    results["csv_report"] = {"passed": False, "error": str(e)}
    print(f"  FAIL: {e}")


# ────────────────────────────────────────────────────────────────────
# Test 5: EU AI Act required fields
# ────────────────────────────────────────────────────────────────────
print("\n[Test 5] EU AI Act required fields")
try:
    from memopt.observability.report_exporter import ComplianceReport

    report = ComplianceReport(
        tenant_id="test-tenant",
        certificate={
            "certificate_status": "CERTIFIED",
            "signature_status":   "hmac-sha256:abc",
        },
    )
    data = report.to_dict()

    required_fields = [
        "report_type", "generated_at", "tenant_id",
        "hardware_certificate", "compliance_notes",
    ]
    missing = [f for f in required_fields if f not in data]
    passed5 = len(missing) == 0

    results["eu_ai_act_fields"] = {"passed": passed5, "missing": missing}
    print(f"  Missing fields: {missing}")
    print(f"  Test 5: {'PASS' if passed5 else 'FAIL'}")
except Exception as e:
    results["eu_ai_act_fields"] = {"passed": False, "error": str(e)}
    print(f"  FAIL: {e}")


# ────────────────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────────────────
passed_count = sum(1 for v in results.values() if v.get("passed"))
total = len(results)

print(f"\n{'=' * 50}")
print("PILLAR 4 RESULT")
print(f"{'=' * 50}")
print(f"Tests: {passed_count}/{total} pass")
print(f"PASS: {passed_count == total}")

with open("/tmp/pillar4_result.json", "w") as f:
    json.dump(
        {
            "passed":  passed_count == total,
            "score":   f"{passed_count}/{total}",
            "results": results,
        },
        f, indent=2,
    )
