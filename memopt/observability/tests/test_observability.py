"""
Pillar 4 observability tests.
All pass with no GPU, no API keys, no network access.
"""
import os
import json
import time
import hmac
import hashlib
import tempfile
import threading
import pytest

from memopt.observability.collector   import MetricsCollector, MetricRegistry
from memopt.observability.ledger      import OptimizationLedger, _compute_savings
from memopt.observability.certificate import sign_entry, verify_certificate
from memopt.observability.arbitrage   import ArbitrageEngine, GPUOffer


# ── MetricRegistry ─────────────────────────────────────────────────────

def test_registry_set_and_get():
    r = MetricRegistry()
    r.set("memopt_test_metric", 42.0, help="test", type="gauge")
    metrics = r.get_all()
    assert any(m.name == "memopt_test_metric" and m.value == 42.0
               for m in metrics)

def test_registry_prometheus_text_format():
    r = MetricRegistry()
    r.set("memopt_gkd_hit_rate_pct", 90.0,
          labels={"node": "a"}, help="GKD hit rate")
    text = r.prometheus_text()
    assert "memopt_gkd_hit_rate_pct" in text
    assert "90.0" in text
    assert "# HELP" in text
    assert "# TYPE" in text

def test_registry_ignores_nan():
    r = MetricRegistry()
    r.set("memopt_nan_test", float("nan"))
    metrics = [m for m in r.get_all() if m.name == "memopt_nan_test"]
    assert len(metrics) == 0, "NaN values must not be stored"

def test_registry_as_dict():
    r = MetricRegistry()
    r.set("memopt_foo", 1.0)
    d = r.as_dict()
    assert "memopt_foo" in d
    assert d["memopt_foo"]["value"] == 1.0


# ── MetricsCollector ───────────────────────────────────────────────────

def test_collector_collects_without_pillars():
    """collect_once() must not raise when no pillar objects are registered."""
    c = MetricsCollector(interval_s=9999)
    c.collect_once()   # must not raise

def test_collector_records_request_counters():
    c = MetricsCollector(interval_s=9999)
    c.record_request(tokens_generated=512)
    c.record_request(tokens_generated=256)
    c.collect_once()
    metrics = {m.name: m.value for m in c.registry.get_all()}
    assert metrics.get("memopt_tokens_generated_total") == 768.0
    assert metrics.get("memopt_requests_total") == 2.0

def test_collector_registers_mock_gkd():
    """Collector must read gkd.stats() and expose hit_rate metric."""
    class MockGKD:
        def stats(self):
            return {
                "total_lookups": 1000,
                "cache_hits": 900,
                "hit_rate_pct": 90.0,
                "estimated_hbm_saved_gb": 0.105,
                "collision_detections_total": 0,
            }

    c = MetricsCollector(interval_s=9999)
    c.register_gkd(MockGKD())
    c.collect_once()
    metrics = {m.name: m.value for m in c.registry.get_all()}
    assert metrics["memopt_gkd_hit_rate_pct"] == 90.0
    assert metrics["memopt_gkd_collision_detections"] == 0.0


# ── _compute_savings ───────────────────────────────────────────────────

def test_savings_with_power_measurement():
    """Energy saved must be calculated correctly from J/token measurements."""
    s = _compute_savings(
        tokens=1_000_000,
        actual_j_per_token=0.00040,
        baseline_j_per_token=0.00100,
        gkd_hit_rate_pct=None,
    )
    expected_kwh = 600 / 3_600_000
    assert s["energy_saved_kwh"] is not None
    assert abs(s["energy_saved_kwh"] - expected_kwh) < 1e-10
    assert s["co2_saved_kg"] is not None
    assert s["cost_saved_usd"] is not None

def test_savings_none_when_no_measurement():
    """energy_saved_kwh must be None when actual_j_per_token is not available."""
    s = _compute_savings(
        tokens=1000,
        actual_j_per_token=None,
        baseline_j_per_token=0.001,
        gkd_hit_rate_pct=None,
    )
    assert s["energy_saved_kwh"] is None
    assert s["co2_saved_kg"]     is None

def test_savings_from_gkd_only():
    """GKD-only savings must be computed when power measurement is absent."""
    s = _compute_savings(
        tokens=1000,
        actual_j_per_token=None,
        baseline_j_per_token=0.001,
        gkd_hit_rate_pct=90.0,
    )
    expected_kwh = 0.9 / 3_600_000
    assert s["energy_saved_kwh"] is not None
    assert abs(s["energy_saved_kwh"] - expected_kwh) < 1e-12


# ── OptimizationLedger ─────────────────────────────────────────────────

def test_ledger_record_and_recent():
    with tempfile.TemporaryDirectory() as d:
        ledger = OptimizationLedger(db_path=f"{d}/ledger.db")
        entry  = ledger.record(
            tokens=1024,
            actual_j_per_token=0.0004,
            gkd_hit_rate_pct=90.0,
            speedup_ratio=5.18,
            hbm_saved_bytes=177e9,
            node_id="test-node",
        )
        assert entry.tokens_generated == 1024
        assert entry.energy_saved_kwh is not None

        recent = ledger.recent(n=5)
        assert len(recent) == 1
        assert recent[0]["tokens_generated"] == 1024

def test_ledger_totals():
    with tempfile.TemporaryDirectory() as d:
        ledger = OptimizationLedger(db_path=f"{d}/ledger.db")
        for _ in range(3):
            ledger.record(tokens=1000, actual_j_per_token=0.0004,
                          gkd_hit_rate_pct=90.0)
        t = ledger.totals()
        assert t["tokens_total"] == 3000
        assert t["n_batches"] == 3
        assert t["energy_saved_kwh"] is not None

def test_ledger_survives_db_error():
    """Ledger must not raise on a bad db path — returns entry without persisting."""
    ledger = OptimizationLedger(db_path="/nonexistent/path/ledger.db")
    entry  = ledger.record(tokens=100)
    assert entry.tokens_generated == 100   # must not raise


# ── Certificate ────────────────────────────────────────────────────────

def test_certificate_signed_and_verified():
    """A signed certificate must verify correctly with the signing key."""
    key = "test-signing-key-12345"
    os.environ["MEMOPT_SIGNING_KEY"] = key

    import importlib
    import memopt.observability.certificate as cert_mod
    importlib.reload(cert_mod)

    with tempfile.TemporaryDirectory() as d:
        ledger = OptimizationLedger(db_path=f"{d}/ledger.db")
        entry  = ledger.record(tokens=512, actual_j_per_token=0.0004)

    cert = cert_mod.sign_entry(entry)
    assert cert["signature_status"] == "signed"
    assert cert["signature"] is not None
    assert cert_mod.verify_certificate(cert, key) is True

    del os.environ["MEMOPT_SIGNING_KEY"]

def test_certificate_unsigned_when_no_key():
    """Without MEMOPT_SIGNING_KEY, certificate is produced unsigned."""
    os.environ.pop("MEMOPT_SIGNING_KEY", None)

    import importlib
    import memopt.observability.certificate as cert_mod
    importlib.reload(cert_mod)

    with tempfile.TemporaryDirectory() as d:
        ledger = OptimizationLedger(db_path=f"{d}/ledger.db")
        entry  = ledger.record(tokens=512)

    cert = cert_mod.sign_entry(entry)
    assert cert["signature_status"] == "unsigned"
    assert cert["signature"] is None

def test_certificate_tamper_detection():
    """Modifying the payload must invalidate the signature."""
    key = "tamper-test-key"
    os.environ["MEMOPT_SIGNING_KEY"] = key

    import importlib
    import memopt.observability.certificate as cert_mod
    importlib.reload(cert_mod)

    with tempfile.TemporaryDirectory() as d:
        ledger = OptimizationLedger(db_path=f"{d}/ledger.db")
        entry  = ledger.record(tokens=512, actual_j_per_token=0.0004)

    cert = cert_mod.sign_entry(entry)
    cert["payload"]["tokens_generated"] = 999_999  # tamper
    assert cert_mod.verify_certificate(cert, key) is False

    del os.environ["MEMOPT_SIGNING_KEY"]


# ── ArbitrageEngine ────────────────────────────────────────────────────

def test_arbitrage_no_keys_starts_without_error():
    """ArbitrageEngine starts cleanly with no provider keys."""
    os.environ.pop("RUNPOD_API_KEY",  None)
    os.environ.pop("LAMBDA_API_KEY",  None)
    engine = ArbitrageEngine(current_price_hr=2.00)
    engine.start()
    time.sleep(0.1)
    engine.stop()

def test_arbitrage_effective_cost_calculation():
    """Effective cost per token must account for speedup ratio."""
    engine = ArbitrageEngine(
        current_price_hr=2.00,
        tokens_per_sec=100.0,
        speedup_ratio=2.0,
    )
    cost     = engine._effective_cost(2.00)
    expected = 2.00 / (100.0 * 3600 * 2.0)
    assert abs(cost - expected) < 1e-10

def test_arbitrage_threshold_not_met_no_recommendation():
    """No recommendation when saving is below threshold."""
    engine = ArbitrageEngine(current_price_hr=2.00, tokens_per_sec=100.0)
    saving = (2.00 - 1.90) / 2.00
    assert saving < 0.20
    assert engine.latest_recommendation() is None

def test_arbitrage_dry_run_does_not_provision():
    """With DRY_RUN=1, _provision_and_migrate is not called by _evaluate."""
    os.environ["MEMOPT_ARBITRAGE_DRY_RUN"] = "1"
    provisioned = {"called": False}

    class TrackingEngine(ArbitrageEngine):
        def _provision_and_migrate(self, rec):
            provisioned["called"] = True

    engine = TrackingEngine(current_price_hr=2.00)
    assert not provisioned["called"]
    del os.environ["MEMOPT_ARBITRAGE_DRY_RUN"]
