"""
Tests for oracle export endpoints (Phase 2b extension).

Covers:
  - /oracle/stats structured transition export
  - /oracle/health pod controller health check
  - /oracle/merge pod→node transition injection
  - _get_top_transitions confidence calculation
"""
import pytest

from memopt.vmm.oracle import MemoryOracle
from memopt.serving.server import _get_top_transitions


# ══════════════════════════════════════════════════════════════════════════
#  _get_top_transitions unit tests
# ══════════════════════════════════════════════════════════════════════════


def test_get_top_transitions_with_confidence():
    oracle = MemoryOracle()

    # Build transitions: seq1 visits 1→2→2→3
    oracle.observe("seq1", 1)
    oracle.observe("seq1", 2)
    oracle.observe("seq1", 2)
    oracle.observe("seq1", 3)

    transitions = _get_top_transitions(oracle, top_k=10)

    assert len(transitions) > 0
    for t in transitions:
        assert "from_block" in t
        assert "to_block" in t
        assert "count" in t
        assert "confidence" in t
        assert 0.0 <= t["confidence"] <= 1.0
        assert "total_from" in t
        assert t["total_from"] >= t["count"]


def test_get_top_transitions_sorted_by_count():
    oracle = MemoryOracle()

    # Create transitions with varying counts
    for _ in range(10):
        oracle.observe("s1", 0)
        oracle.observe("s1", 1)
    for _ in range(5):
        oracle.observe("s2", 0)
        oracle.observe("s2", 2)

    transitions = _get_top_transitions(oracle, top_k=50)
    counts = [t["count"] for t in transitions]
    assert counts == sorted(counts, reverse=True)


def test_get_top_transitions_top_k_limits():
    oracle = MemoryOracle()

    # Create many distinct transitions
    for i in range(20):
        oracle.observe(f"s{i}", i)
        oracle.observe(f"s{i}", i + 100)

    transitions = _get_top_transitions(oracle, top_k=5)
    assert len(transitions) <= 5


def test_get_top_transitions_empty_oracle():
    oracle = MemoryOracle()
    transitions = _get_top_transitions(oracle, top_k=10)
    assert transitions == []


def test_get_top_transitions_confidence_sums():
    """For a single from_block, confidences of all targets must sum to 1.0."""
    oracle = MemoryOracle()

    # from_block=0 → to_block in {1, 2, 3}
    for _ in range(6):
        oracle.observe("s1", 0)
        oracle.observe("s1", 1)
    for _ in range(3):
        oracle.observe("s2", 0)
        oracle.observe("s2", 2)
    for _ in range(1):
        oracle.observe("s3", 0)
        oracle.observe("s3", 3)

    transitions = _get_top_transitions(oracle, top_k=100)
    from_0 = [t for t in transitions if t["from_block"] == 0]

    total_conf = sum(t["confidence"] for t in from_0)
    assert abs(total_conf - 1.0) < 0.01


# ══════════════════════════════════════════════════════════════════════════
#  Endpoint integration tests (via TestClient)
# ══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from memopt.serving.server import app
    return TestClient(app)


def test_oracle_stats_endpoint_structure(client):
    response = client.get("/oracle/stats")
    assert response.status_code == 200
    data = response.json()

    required = [
        "node_id", "transitions", "total_transitions",
        "horizon", "exported_at",
    ]
    for key in required:
        assert key in data, f"Missing key: {key}"

    assert isinstance(data["transitions"], list)
    assert isinstance(data["total_transitions"], int)
    assert isinstance(data["exported_at"], float)


def test_oracle_stats_transitions_have_confidence(client):
    """Even when empty, the shape contract holds."""
    response = client.get("/oracle/stats")
    data = response.json()

    # With no engine, transitions list is empty — that's fine
    # But if there were transitions, each must have the new fields
    for t in data["transitions"]:
        assert "from_block" in t
        assert "to_block" in t
        assert "count" in t
        assert "confidence" in t
        assert "total_from" in t


def test_oracle_health_endpoint(client):
    response = client.get("/oracle/health")
    assert response.status_code == 200
    data = response.json()

    required = [
        "node_id", "oracle_active", "transition_count",
        "horizon", "prediction_accuracy", "healthy",
    ]
    for key in required:
        assert key in data, f"Missing key: {key}"


def test_oracle_health_inactive_when_no_engine(client):
    """With no engine running, oracle should report inactive."""
    response = client.get("/oracle/health")
    data = response.json()
    assert data["oracle_active"] is False
    assert data["healthy"] is False


def test_oracle_merge_endpoint(client):
    payload = {
        "source": "pod",
        "pod_id": "pod-001",
        "transitions": [
            {"from_block": 1, "to_block": 2,
             "count": 10, "confidence": 0.8},
            {"from_block": 2, "to_block": 3,
             "count": 5, "confidence": 0.3},
        ],
    }

    response = client.post("/oracle/merge", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "merged" in data
    assert "skipped" in data
    # With no engine, nothing merges — but the endpoint doesn't crash
    # The low-confidence transition (0.3 < 0.5) would be skipped
    # if oracle were active


def test_oracle_merge_no_engine_returns_reason(client):
    payload = {
        "source": "pod",
        "pod_id": "pod-001",
        "transitions": [],
    }
    response = client.post("/oracle/merge", json=payload)
    data = response.json()
    assert data.get("reason") == "oracle_not_active"
