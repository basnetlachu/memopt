"""
Tests for the Executive ROI Dashboard endpoints.

All tests use FastAPI TestClient (no GPU required).
The control-plane is started in-process with a temporary SQLite database.
"""

import pytest
from fastapi.testclient import TestClient

import memopt.control_plane.server as srv


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolated_server(tmp_path, monkeypatch):
    """
    Run each test with a fresh in-memory control-plane:
    - New Database (temp SQLite file)
    - New FleetIntelligence (separate temp SQLite)
    - Known API key "test-key-exec"
    """
    from memopt.control_plane.database import Database
    from memopt.fleet.intelligence import FleetIntelligence
    from memopt.alerts.alert_store import AlertStore

    new_db    = Database(db_path=tmp_path / "cp.db")
    new_db.init()
    new_fleet = FleetIntelligence(
        db_path=str(tmp_path / "fleet.db"),
        auto_remediate=False,
    )
    new_alerts = AlertStore()

    monkeypatch.setattr(srv, "db",          new_db)
    monkeypatch.setattr(srv, "alert_store", new_alerts)
    monkeypatch.setattr(srv, "_fleet",      new_fleet)
    monkeypatch.setattr(srv, "_API_KEY",    "test-key-exec")
    yield


@pytest.fixture
def client():
    return TestClient(srv.app, raise_server_exceptions=True)


@pytest.fixture
def auth(client):
    """Returns headers dict with a valid API key."""
    return {"X-Memopt-API-Key": "test-key-exec"}


# ── Test 1: HTML page is served ────────────────────────────────────────────

def test_executive_html_served(client):
    """GET /executive returns 200 with HTML dashboard content."""
    r = client.get("/executive")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    body = r.text
    assert "Executive ROI Dashboard" in body
    assert "memopt" in body.lower()
    # Must contain ESG section
    assert "esg" in body.lower() or "ESG" in body


# ── Test 2: Unauthenticated summary returns 401 ────────────────────────────

def test_executive_summary_requires_auth(client):
    """GET /api/v1/executive/summary without API key → 401."""
    r = client.get("/api/v1/executive/summary")
    assert r.status_code == 401


# ── Test 3: Summary response shape is correct ──────────────────────────────

def test_executive_summary_structure(client, auth):
    """
    GET /api/v1/executive/summary with valid key → 200 with all required keys.
    Works with an empty database (zero savings are valid).
    """
    r = client.get("/api/v1/executive/summary?hours=24", headers=auth)
    assert r.status_code == 200, r.text

    data = r.json()
    required_keys = {
        "period_hours", "total_nodes", "total_gpus", "optimized_gpus",
        "throughput_multiplier", "gpu_hours_saved", "dollar_savings",
        "dollar_savings_annual", "gpu_cost_per_hour",
        "electricity_savings_usd", "power_reduction_pct", "top_savings_nodes",
    }
    missing = required_keys - data.keys()
    assert not missing, f"Missing keys in summary response: {missing}"

    # Types check
    assert isinstance(data["period_hours"],           float)
    assert isinstance(data["dollar_savings_annual"],  (int, float))
    assert isinstance(data["throughput_multiplier"],  (int, float))
    assert isinstance(data["top_savings_nodes"],      list)


# ── Test 4: Summary period_hours parameter is respected ────────────────────

def test_executive_summary_period_hours(client, auth):
    """period_hours query param is echoed back correctly."""
    for hours in (1.0, 168.0, 720.0, 8760.0):
        r = client.get(f"/api/v1/executive/summary?hours={hours}", headers=auth)
        assert r.status_code == 200, f"hours={hours}: {r.text}"
        assert r.json()["period_hours"] == pytest.approx(hours)


# ── Test 5: Carbon endpoint structure and equivalencies ────────────────────

def test_executive_carbon_structure(client, auth):
    """
    GET /api/v1/executive/carbon → 200 with carbon keys and equivalencies dict.
    Equivalencies must be non-negative numbers.
    """
    r = client.get("/api/v1/executive/carbon?hours=24", headers=auth)
    assert r.status_code == 200, r.text

    data = r.json()
    required_keys = {
        "period_hours", "power_reduction_pct", "kwh_saved",
        "kg_co2_saved", "tonnes_co2_saved", "carbon_source",
        "equivalencies", "annual_projection",
    }
    missing = required_keys - data.keys()
    assert not missing, f"Missing keys in carbon response: {missing}"

    eq = data["equivalencies"]
    assert set(eq.keys()) >= {"cars_removed", "trees_planted", "flights_avoided"}
    for k, v in eq.items():
        assert isinstance(v, (int, float)), f"equivalencies.{k} must be numeric"
        assert v >= 0, f"equivalencies.{k} must be non-negative"

    proj = data["annual_projection"]
    assert set(proj.keys()) >= {"kwh_saved", "kg_co2_saved", "tonnes_co2_saved"}

    # carbon_source is one of the expected values
    assert data["carbon_source"] in ("us_average", "watttime")
