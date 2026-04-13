"""
HTTP tests for the canary rollout endpoints.

Each test enters the TestClient context manager so FastAPI's startup
event fires, creates a fresh CanaryController (the startup always
replaces `_canary`), and sets the module-level API key.
"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client_and_key():
    """
    TestClient fixture that yields (client, api_key).

    `with TestClient(app)` triggers startup → sets `_API_KEY` and
    creates a fresh `_canary`. On context exit, shutdown stops the
    canary and clears the module var so the next test gets a clean
    in-memory state. The DB persists across tests (shared module-
    level instance), which is fine because rollout_ids are unique.
    """
    import memopt.control_plane.server as cp_mod
    with TestClient(cp_mod.app) as client:
        yield client, cp_mod._API_KEY


def _register_version(client, api_key, version):
    """Helper: register an image version so rollouts can target it."""
    return client.post(
        "/api/v1/images",
        json={
            "version":      version,
            "git_commit":   "test",
            "build_date":   "2026-04-13T00:00:00Z",
            "cuda_version": "12.4",
            "sm_targets":   "86",
            "image_url":    f"registry/memopt:{version}",
        },
        headers={"X-Memopt-API-Key": api_key})


# ══════════════════════════════════════════════════════════════════════
#  POST /api/v1/rollouts
# ══════════════════════════════════════════════════════════════════════


def test_start_rollout_success(client_and_key):
    client, api_key = client_and_key
    _register_version(client, api_key, "v2.0.0-api")

    response = client.post(
        "/api/v1/rollouts",
        json={
            "target_version":  "v2.0.0-api",
            "current_version": "v1.0.0",
        },
        headers={"X-Memopt-API-Key": api_key})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "started"
    assert "rollout_id" in data
    assert len(data["rollout_id"]) > 0
    assert data["target_version"] == "v2.0.0-api"


def test_start_rollout_missing_version_404(client_and_key):
    client, api_key = client_and_key
    # No image registered for v9.9.9 → 404
    response = client.post(
        "/api/v1/rollouts",
        json={
            "target_version":  "v9.9.9-missing",
            "current_version": "v1.0.0",
        },
        headers={"X-Memopt-API-Key": api_key})
    assert response.status_code == 404


def test_start_rollout_missing_fields_400(client_and_key):
    client, api_key = client_and_key
    response = client.post(
        "/api/v1/rollouts",
        json={"target_version": "v1.0.0"},  # no current_version
        headers={"X-Memopt-API-Key": api_key})
    assert response.status_code == 400


def test_start_rollout_requires_auth(client_and_key):
    client, _ = client_and_key
    response = client.post(
        "/api/v1/rollouts",
        json={
            "target_version":  "v1.0.0",
            "current_version": "v0.9.0",
        })
    assert response.status_code == 401


def test_start_rollout_concurrent_rejected_409(client_and_key):
    client, api_key = client_and_key
    _register_version(client, api_key, "v3.0.0-conc")
    _register_version(client, api_key, "v4.0.0-conc")

    r1 = client.post(
        "/api/v1/rollouts",
        json={
            "target_version":  "v3.0.0-conc",
            "current_version": "v1.0.0",
        },
        headers={"X-Memopt-API-Key": api_key})
    assert r1.status_code == 200

    r2 = client.post(
        "/api/v1/rollouts",
        json={
            "target_version":  "v4.0.0-conc",
            "current_version": "v1.0.0",
        },
        headers={"X-Memopt-API-Key": api_key})
    assert r2.status_code == 409
    assert "already active" in r2.json()["detail"]


# ══════════════════════════════════════════════════════════════════════
#  GET /api/v1/rollouts/active
# ══════════════════════════════════════════════════════════════════════


def test_get_active_rollout_none(client_and_key):
    client, api_key = client_and_key
    response = client.get(
        "/api/v1/rollouts/active",
        headers={"X-Memopt-API-Key": api_key})
    assert response.status_code == 200
    data = response.json()
    assert "active" in data
    # Fresh context means no active rollout
    assert data["active"] is False


def test_get_active_rollout_after_start(client_and_key):
    client, api_key = client_and_key
    _register_version(client, api_key, "v5.0.0-act")
    client.post(
        "/api/v1/rollouts",
        json={
            "target_version":  "v5.0.0-act",
            "current_version": "v1.0.0",
        },
        headers={"X-Memopt-API-Key": api_key})

    response = client.get(
        "/api/v1/rollouts/active",
        headers={"X-Memopt-API-Key": api_key})
    assert response.status_code == 200
    data = response.json()
    assert data["active"] is True
    assert data["target_version"] == "v5.0.0-act"
    assert "rollout_id" in data
    assert "current_stage" in data


# ══════════════════════════════════════════════════════════════════════
#  Pause / Resume / Abort
# ══════════════════════════════════════════════════════════════════════


def test_pause_resume_abort_flow(client_and_key):
    client, api_key = client_and_key
    _register_version(client, api_key, "v6.0.0-flow")
    client.post(
        "/api/v1/rollouts",
        json={
            "target_version":  "v6.0.0-flow",
            "current_version": "v1.0.0",
        },
        headers={"X-Memopt-API-Key": api_key})

    # Pause
    r = client.post(
        "/api/v1/rollouts/pause",
        json={"reason": "manual test"},
        headers={"X-Memopt-API-Key": api_key})
    assert r.status_code == 200
    assert r.json()["status"] == "paused"
    assert r.json()["reason"] == "manual test"

    # Verify state reflects pause
    r = client.get(
        "/api/v1/rollouts/active",
        headers={"X-Memopt-API-Key": api_key})
    assert r.json()["current_stage"] == "paused"

    # Resume
    r = client.post(
        "/api/v1/rollouts/resume",
        headers={"X-Memopt-API-Key": api_key})
    assert r.status_code == 200
    assert r.json()["status"] == "resumed"

    # Abort
    r = client.post(
        "/api/v1/rollouts/abort",
        json={"reason": "test done"},
        headers={"X-Memopt-API-Key": api_key})
    assert r.status_code == 200
    assert r.json()["status"] == "aborted"

    # After abort the rollout is in FAILED terminal state.
    r = client.get(
        "/api/v1/rollouts/active",
        headers={"X-Memopt-API-Key": api_key})
    assert r.json()["current_stage"] == "failed"


def test_pause_with_default_reason(client_and_key):
    """Empty pause body → default reason."""
    client, api_key = client_and_key
    _register_version(client, api_key, "v6.1.0-pause")
    client.post(
        "/api/v1/rollouts",
        json={
            "target_version":  "v6.1.0-pause",
            "current_version": "v1.0.0",
        },
        headers={"X-Memopt-API-Key": api_key})

    r = client.post(
        "/api/v1/rollouts/pause",
        json={},
        headers={"X-Memopt-API-Key": api_key})
    assert r.status_code == 200
    assert r.json()["reason"] == "manual pause"


# ══════════════════════════════════════════════════════════════════════
#  GET /api/v1/rollouts/{id}/events
# ══════════════════════════════════════════════════════════════════════


def test_rollout_events_endpoint(client_and_key):
    client, api_key = client_and_key
    _register_version(client, api_key, "v7.0.0-evt")

    r = client.post(
        "/api/v1/rollouts",
        json={
            "target_version":  "v7.0.0-evt",
            "current_version": "v1.0.0",
        },
        headers={"X-Memopt-API-Key": api_key})
    rollout_id = r.json()["rollout_id"]

    # Pause to generate a second event
    client.post(
        "/api/v1/rollouts/pause",
        json={"reason": "generate event"},
        headers={"X-Memopt-API-Key": api_key})

    r = client.get(
        f"/api/v1/rollouts/{rollout_id}/events",
        headers={"X-Memopt-API-Key": api_key})
    assert r.status_code == 200
    data = r.json()
    assert data["rollout_id"] == rollout_id
    assert data["total"] >= 2
    events = data["events"]
    event_names = [e["event"] for e in events]
    assert "started" in event_names
    assert "paused" in event_names

    # Events carry rollout_id and recorded_at
    for e in events:
        assert e["rollout_id"] == rollout_id
        assert "recorded_at" in e


def test_rollout_events_unknown_id_empty(client_and_key):
    client, api_key = client_and_key
    r = client.get(
        "/api/v1/rollouts/does-not-exist/events",
        headers={"X-Memopt-API-Key": api_key})
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 0
    assert data["events"] == []
