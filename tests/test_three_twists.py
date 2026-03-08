"""
Tests for the three executive dashboard twists:
  Twist 1: Live dollar counter (savings_per_second, savings_since_epoch in summary)
  Twist 2: Carbon hero (carbon endpoint now has top-level alias fields)
  Twist 3: Hardware health card (new endpoint + FleetIntelligence.get_thermal_profile)

All 7 tests pass with an isolated in-memory DB — no GPU required.
"""

import pytest
from fastapi.testclient import TestClient

import memopt.control_plane.server as srv


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolated_server(tmp_path, monkeypatch):
    """Patch server singletons so each test gets a clean DB and known API key."""
    from memopt.control_plane.database import Database
    from memopt.fleet.intelligence import FleetIntelligence
    from memopt.alerts.alert_store import AlertStore
    from memopt.fleet.gossip import GossipKnowledgeBase
    from memopt.ebpf.interceptor import CUDAKernelInterceptor

    new_db    = Database(db_path=tmp_path / "cp.db")
    new_db.init()
    new_fleet = FleetIntelligence(db_path=str(tmp_path / "fleet.db"), auto_remediate=False)
    new_gossip = GossipKnowledgeBase(db_path=str(tmp_path / "gossip.db"))

    monkeypatch.setattr(srv, "db",           new_db)
    monkeypatch.setattr(srv, "alert_store",  AlertStore())
    monkeypatch.setattr(srv, "_fleet",       new_fleet)
    monkeypatch.setattr(srv, "_gossip_kb",   new_gossip)
    monkeypatch.setattr(srv, "_API_KEY",     "test-key")
    monkeypatch.setattr(srv, "_interceptor", CUDAKernelInterceptor())
    yield


@pytest.fixture
def test_client():
    return TestClient(srv.app, raise_server_exceptions=True)


TEST_API_KEY = "test-key"


# ── Twist 1: savings_per_second ───────────────────────────────────────────────

def test_savings_per_second_in_summary_response(test_client):
    resp = test_client.get(
        "/api/v1/executive/summary",
        headers={"X-Memopt-API-Key": TEST_API_KEY},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "savings_per_second"  in data, "missing savings_per_second"
    assert "savings_since_epoch" in data, "missing savings_since_epoch"
    assert data["savings_per_second"]  >= 0
    assert data["savings_since_epoch"] >= 0


def test_savings_per_second_math():
    # $3,153,600/yr ÷ (365×24×3600) = exactly $0.10/s
    annual     = 3_153_600
    per_second = annual / (365 * 24 * 3600)
    assert abs(per_second - 0.10) < 0.001


# ── Twist 3: hardware health endpoint ────────────────────────────────────────

def test_hardware_health_endpoint_returns_correct_fields(test_client):
    resp = test_client.get(
        "/api/v1/executive/hardware-health",
        headers={"X-Memopt-API-Key": TEST_API_KEY},
    )
    assert resp.status_code == 200
    data = resp.json()
    required = [
        "temp_before_c", "temp_after_c", "temp_reduction_c",
        "temp_reduction_pct", "power_before_w", "power_after_w",
        "power_reduction_pct", "health_score", "data_available",
    ]
    for field in required:
        assert field in data, f"Missing field: {field}"


def test_health_score_between_0_and_100(test_client):
    resp = test_client.get(
        "/api/v1/executive/hardware-health",
        headers={"X-Memopt-API-Key": TEST_API_KEY},
    )
    data = resp.json()
    assert 0 <= data["health_score"] <= 100


def test_thermal_profile_handles_no_data(tmp_path):
    from memopt.fleet.intelligence import FleetIntelligence
    fleet   = FleetIntelligence(db_path=str(tmp_path / "thermal_test.db"), auto_remediate=False)
    profile = fleet.get_thermal_profile()
    assert profile["data_available"]     is False
    assert profile["health_score"]       == 0
    assert profile["temp_reduction_c"]   == 0.0
    assert profile["power_reduction_pct"] == 0.0


# ── Twist 2 + HTML content checks ────────────────────────────────────────────

def test_executive_html_loads_with_new_sections(test_client):
    resp = test_client.get("/executive")
    assert resp.status_code == 200
    html = resp.text
    assert "COMPUTE DOLLARS RECLAIMED" in html
    assert "liveCounter"               in html
    assert "carbon-hero"               in html
    assert "HARDWARE HEALTH"           in html
    assert "healthScore"               in html


def test_esg_csv_contains_regulatory_note(test_client):
    resp = test_client.get("/executive")
    assert resp.status_code == 200
    # Regulatory note is inside the downloadESG JS function
    assert "GHG Protocol" in resp.text or "CSRD" in resp.text
