"""
Tests for Pillar 4 enhancements:
  - energy_source tracking in ledger
  - CSV export
  - HTML/PDF report generation
  - CarbonCalculator with regional grid intensity
  - Export endpoints
  - Honest measurement labelling

All tests run without GPU, NVML, or pynvml.
"""
import os
import tempfile
import time
from unittest import mock

import pytest


def _make_ledger():
    """Create a ledger with a temp file that persists across connections."""
    from memopt.observability.ledger import OptimizationLedger
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return OptimizationLedger(tmp.name), tmp.name


# ── Ledger energy_source ────────────────────────────────────────────


def test_ledger_energy_source_auto_detected():
    """energy_source auto-detects from actual_j_per_token presence."""
    from memopt.observability.ledger import OptimizationLedger

    ledger, path = _make_ledger()
    try:
        entry = ledger.record(tokens=100, actual_j_per_token=0.001)
        assert entry.energy_source == "nvml_measured"

        entry2 = ledger.record(tokens=100, gkd_hit_rate_pct=90.0)
        assert entry2.energy_source == "estimated"

        entry3 = ledger.record(tokens=100)
        assert entry3.energy_source == "unmeasured"
    finally:
        os.unlink(path)


def test_ledger_energy_source_stored():
    from memopt.observability.ledger import OptimizationLedger

    ledger, path = _make_ledger()
    try:
        ledger.record(tokens=100, tenant_id="test",
                      actual_j_per_token=0.001,
                      energy_source="nvml_measured")
        ledger.record(tokens=100, tenant_id="test",
                      actual_j_per_token=None,
                      energy_source="unmeasured")
        ledger.flush()

        totals = ledger.totals(tenant_id="test")
        breakdown = totals.get("energy_source_breakdown", {})

        assert breakdown.get("nvml_measured", 0) >= 1
        assert breakdown.get("unmeasured", 0) >= 1
    finally:
        os.unlink(path)


def test_ledger_j_per_token_none_without_nvml():
    """Without NVML: j_per_token is None, energy_source is unmeasured."""
    from memopt.observability.ledger import OptimizationLedger

    ledger, path = _make_ledger()
    try:
        entry = ledger.record(tokens=100, actual_j_per_token=None,
                              energy_source="unmeasured")
        assert entry.actual_j_per_token is None

        ledger.flush()
        totals = ledger.totals()
        breakdown = totals.get("energy_source_breakdown", {})
        assert breakdown.get("unmeasured", 0) >= 1
    finally:
        os.unlink(path)


# ── CSV export ──────────────────────────────────────────────────────


def test_ledger_csv_export():
    from memopt.observability.ledger import OptimizationLedger

    ledger, path = _make_ledger()
    try:
        ledger.record(tokens=512, tenant_id="csv_test",
                      actual_j_per_token=0.002,
                      energy_source="nvml_measured")
        ledger.flush()

        csv_data = ledger.export_csv(tenant_id="csv_test")
        assert len(csv_data) > 0
        lines = csv_data.strip().split("\n")
        assert len(lines) >= 2  # header + 1 row

        header = lines[0]
        assert "tokens_generated" in header
        assert "energy_source" in header
        assert "entry_hash" in header
    finally:
        os.unlink(path)


def test_ledger_csv_empty_on_no_data():
    from memopt.observability.ledger import OptimizationLedger

    ledger, path = _make_ledger()
    try:
        csv_data = ledger.export_csv(tenant_id="nobody")
        lines = [l for l in csv_data.strip().split("\n") if l]
        assert len(lines) == 1  # header only
    finally:
        os.unlink(path)


def test_ledger_csv_time_filter():
    from memopt.observability.ledger import OptimizationLedger

    ledger, path = _make_ledger()
    try:
        ledger.record(tokens=100, tenant_id="time_test")
        ledger.flush()

        csv_data = ledger.export_csv(
            tenant_id="time_test",
            start_time=time.time() + 3600)

        lines = [l for l in csv_data.strip().split("\n") if l]
        assert len(lines) == 1  # header only
    finally:
        os.unlink(path)


# ── Savings report ──────────────────────────────────────────────────


def test_savings_report_html_generated():
    from memopt.observability.report_exporter import SavingsReport

    report = SavingsReport(
        tenant_id="acme", period_label="Last 24h",
        totals={
            "tokens_total": 100000, "energy_saved_kwh": 0.05,
            "co2_saved_kg": 0.012, "cost_saved_usd": None,
        },
        source_breakdown={
            "nvml_measured": 80, "estimated": 15, "unmeasured": 5,
        },
        config={"grid_intensity": 0.233, "cost_per_1k": 0.0})

    html = report.to_html()
    assert len(html) > 100
    assert "acme" in html
    assert "100,000" in html
    assert "measurement" in html.lower()
    assert "not constitute" in html


def test_savings_report_no_fake_dollars():
    """Report shows N/A when no pricing configured."""
    from memopt.observability.report_exporter import SavingsReport

    report = SavingsReport(
        tenant_id="test", period_label="Test",
        totals={"cost_saved_usd": 0},
        source_breakdown={},
        config={"cost_per_1k": 0.0})

    html = report.to_html()
    assert "N/A" in html


def test_savings_report_csv_summary():
    from memopt.observability.report_exporter import SavingsReport

    report = SavingsReport(
        tenant_id="acme", period_label="Last 30d",
        totals={"tokens_total": 1000000, "energy_saved_kwh": 0.5},
        source_breakdown={"nvml_measured": 900, "estimated": 100},
        config={})

    csv_data = report.to_csv_summary()
    lines = csv_data.strip().split("\n")
    assert len(lines) == 2  # header + 1 row
    assert "acme" in csv_data
    assert "1000000" in csv_data


def test_savings_report_pdf_none_without_reportlab():
    from memopt.observability.report_exporter import SavingsReport

    report = SavingsReport(
        tenant_id="test", period_label="test",
        totals={}, source_breakdown={}, config={})

    result = report.to_pdf()
    assert result is None or isinstance(result, bytes)


# ── Carbon calculator ───────────────────────────────────────────────


def test_carbon_calculator_regions():
    from memopt.observability.ledger import CarbonCalculator

    old_region = os.environ.get("MEMOPT_GRID_REGION")
    old_intensity = os.environ.get("MEMOPT_GRID_INTENSITY_KG_KWH")
    try:
        os.environ.pop("MEMOPT_GRID_INTENSITY_KG_KWH", None)

        os.environ["MEMOPT_GRID_REGION"] = "us"
        calc_us = CarbonCalculator()
        assert calc_us._intensity == 0.380

        os.environ["MEMOPT_GRID_REGION"] = "france"
        calc_fr = CarbonCalculator()
        assert calc_fr._intensity == 0.052

        assert calc_fr._intensity < calc_us._intensity
    finally:
        if old_region is not None:
            os.environ["MEMOPT_GRID_REGION"] = old_region
        else:
            os.environ.pop("MEMOPT_GRID_REGION", None)
        if old_intensity is not None:
            os.environ["MEMOPT_GRID_INTENSITY_KG_KWH"] = old_intensity


def test_carbon_calculator_kwh_to_co2():
    from memopt.observability.ledger import CarbonCalculator

    old = os.environ.get("MEMOPT_GRID_INTENSITY_KG_KWH")
    try:
        os.environ["MEMOPT_GRID_INTENSITY_KG_KWH"] = "0.233"
        calc = CarbonCalculator()
        co2 = calc.kwh_to_co2_kg(1.0)
        assert abs(co2 - 0.233) < 1e-6
        assert calc.kwh_to_co2_kg(-1.0) == 0.0
    finally:
        if old is not None:
            os.environ["MEMOPT_GRID_INTENSITY_KG_KWH"] = old
        else:
            os.environ.pop("MEMOPT_GRID_INTENSITY_KG_KWH", None)


def test_carbon_calculator_annual_projection():
    from memopt.observability.ledger import CarbonCalculator

    calc = CarbonCalculator()
    proj = calc.annual_projection(daily_kwh_saved=1.0)

    assert "annual_kwh_saved" in proj
    assert "annual_co2_kg" in proj
    assert "disclaimer" in proj
    assert "not a certified" in proj["disclaimer"].lower()
    assert abs(proj["annual_kwh_saved"] - 365.0) < 0.01


def test_carbon_disclaimer_present():
    from memopt.observability.ledger import CarbonCalculator

    calc = CarbonCalculator()
    proj = calc.annual_projection(1.0)
    disclaimer = proj.get("disclaimer", "")
    assert "not" in disclaimer.lower()
    assert "certified" in disclaimer.lower()


def test_carbon_calculator_stats():
    from memopt.observability.ledger import CarbonCalculator

    calc = CarbonCalculator()
    stats = calc.stats()
    assert "region" in stats
    assert "intensity" in stats
    assert "available_regions" in stats
    assert "eu" in stats["available_regions"]


# ── Export endpoints ────────────────────────────────────────────────


def _api_test_client():
    """Create a TestClient with valid auth for the api server."""
    from fastapi.testclient import TestClient
    import memopt.api.server as api_srv

    # Set env var so authenticate() accepts "test_key"
    old = os.environ.get("MEMOPT_API_KEY")
    os.environ["MEMOPT_API_KEY"] = "test_key"
    # Re-read the key so verify_key matches
    api_srv._API_KEY = "test_key"

    client = TestClient(api_srv.app)
    headers = {"x-memopt-api-key": "test_key"}
    return client, headers, old


def _restore_api_key(old):
    if old is not None:
        os.environ["MEMOPT_API_KEY"] = old
    else:
        os.environ.pop("MEMOPT_API_KEY", None)


def test_export_csv_endpoint():
    client, headers, old = _api_test_client()
    try:
        r = client.get("/ledger/export/csv", headers=headers)
        assert r.status_code == 200
    finally:
        _restore_api_key(old)


def test_export_html_endpoint():
    client, headers, old = _api_test_client()
    try:
        r = client.get("/ledger/export/report.html", headers=headers)
        assert r.status_code == 200
        assert "memopt" in r.text.lower()
    finally:
        _restore_api_key(old)


def test_export_carbon_endpoint():
    client, headers, old = _api_test_client()
    try:
        r = client.get("/ledger/export/carbon", headers=headers)
        assert r.status_code == 200
        data = r.json()
        assert "disclaimer" in data
        assert "certified" in data["disclaimer"].lower()
    finally:
        _restore_api_key(old)


def test_export_carbon_has_projection():
    client, headers, old = _api_test_client()
    try:
        r = client.get("/ledger/export/carbon", headers=headers)
        data = r.json()
        assert "annual_projection" in data
        proj = data["annual_projection"]
        assert "annual_kwh_saved" in proj
        assert "annual_co2_kg" in proj
    finally:
        _restore_api_key(old)


# ── Generate report from ledger ─────────────────────────────────────


def test_generate_report_from_ledger():
    from memopt.observability.ledger import OptimizationLedger
    from memopt.observability.report_exporter import generate_report

    ledger, path = _make_ledger()
    try:
        for i in range(10):
            ledger.record(
                tokens=1000, tenant_id="report_test",
                actual_j_per_token=0.0005 if i % 2 == 0 else None,
                gkd_hit_rate_pct=85.0)
        ledger.flush()

        report = generate_report(
            ledger=ledger, tenant_id="report_test", period_hours=24)
        assert report.tenant_id == "report_test"
        html = report.to_html()
        assert len(html) > 100
        assert "report_test" in html
    finally:
        os.unlink(path)


# ── PowerSampler availability ───────────────────────────────────────


def test_power_sampler_does_not_crash_without_nvml():
    """PowerSampler starts without crash even without pynvml."""
    from memopt.profiler.power_sampler import PowerSampler

    sampler = PowerSampler(interval_ms=100)
    assert sampler._idle_watts is None or isinstance(sampler._idle_watts, float)
    assert sampler.current_avg_watts() == 0.0
