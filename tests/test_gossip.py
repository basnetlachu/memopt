"""
Tests for the Cross-Node Gossip Protocol.

All tests are pure-Python — no GPU, no torch, no network required.
The GossipKnowledgeBase uses a temp SQLite file per test.
The API tests use FastAPI TestClient with an isolated server instance.
"""

import time

import pytest
from fastapi.testclient import TestClient

from memopt.fleet.gossip import GossipKnowledgeBase, OptimizationRecipe
import memopt.control_plane.server as srv


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def kb(tmp_path):
    """Fresh GossipKnowledgeBase backed by a temp SQLite file."""
    return GossipKnowledgeBase(db_path=str(tmp_path / "gossip.db"))


@pytest.fixture
def sample_recipe(kb):
    """A valid OptimizationRecipe for mistral/a100/float16."""
    return OptimizationRecipe(
        recipe_hash=OptimizationRecipe.compute_hash("mistral", "a100", "float16"),
        model_family="mistral",
        gpu_family="a100",
        dtype="float16",
        backend="vllm",
        batch_size=32,
        max_model_len=4096,
        flash_attention="flash_attention_2",
        gpu_memory_util=0.85,
        tensor_parallel=1,
        extra_flags={},
        measured_speedup=62.3,
        measured_tps=1517.0,
        verified_by_node="node-a100-01",
        verified_at=time.time(),
        verification_count=1,
    )


@pytest.fixture(autouse=True)
def _isolated_server(tmp_path, monkeypatch):
    """Patch server singletons so each test gets a clean DB and known API key."""
    from memopt.control_plane.database import Database
    from memopt.fleet.intelligence import FleetIntelligence
    from memopt.alerts.alert_store import AlertStore

    new_db    = Database(db_path=tmp_path / "cp.db")
    new_db.init()
    new_fleet = FleetIntelligence(db_path=str(tmp_path / "fleet.db"), auto_remediate=False)
    new_gossip = GossipKnowledgeBase(db_path=str(tmp_path / "gossip.db"))

    monkeypatch.setattr(srv, "db",          new_db)
    monkeypatch.setattr(srv, "alert_store", AlertStore())
    monkeypatch.setattr(srv, "_fleet",      new_fleet)
    monkeypatch.setattr(srv, "_gossip_kb",  new_gossip)
    monkeypatch.setattr(srv, "_API_KEY",    "test-key")
    yield


@pytest.fixture
def client():
    return TestClient(srv.app, raise_server_exceptions=True)


@pytest.fixture
def auth():
    return {"X-Memopt-API-Key": "test-key"}


# ── Test 1: Hash is deterministic ─────────────────────────────────────────

def test_recipe_hash_is_deterministic():
    """Same inputs must always produce the same 16-char hex hash."""
    h1 = OptimizationRecipe.compute_hash("mistral", "a100", "float16")
    h2 = OptimizationRecipe.compute_hash("mistral", "a100", "float16")
    assert h1 == h2
    assert len(h1) == 16
    assert all(c in "0123456789abcdef" for c in h1)


# ── Test 2: Different inputs → different hashes ────────────────────────────

def test_different_inputs_produce_different_hashes():
    """Every distinct (model, gpu, dtype) combination must hash differently."""
    pairs = [
        ("mistral",  "a100", "float16"),
        ("llama",    "a100", "float16"),
        ("mistral",  "h100", "float16"),
        ("mistral",  "a100", "bfloat16"),
    ]
    hashes = [OptimizationRecipe.compute_hash(*p) for p in pairs]
    assert len(hashes) == len(set(hashes)), "Hash collision detected"


# ── Test 3: Store and lookup round-trip ───────────────────────────────────

def test_knowledge_base_store_and_lookup(kb, sample_recipe):
    """Stored recipe must be retrievable with all fields intact."""
    kb.store(sample_recipe)

    found = kb.lookup("mistral", "a100", "float16")

    assert found is not None
    assert found.recipe_hash      == sample_recipe.recipe_hash
    assert found.measured_speedup == pytest.approx(62.3)
    assert found.measured_tps     == pytest.approx(1517.0)
    assert found.backend          == "vllm"
    assert found.batch_size       == 32
    assert found.gpu_memory_util  == pytest.approx(0.85)
    assert found.extra_flags      == {}


# ── Test 4: Gossip API POST and GET ───────────────────────────────────────

def test_gossip_api_post_and_get(client, auth, sample_recipe):
    """POST /api/v1/gossip/recipes stores a recipe; GET retrieves it correctly."""
    from dataclasses import asdict
    recipe_data = asdict(sample_recipe)

    post = client.post("/api/v1/gossip/recipes", json=recipe_data, headers=auth)
    assert post.status_code == 201, post.text
    assert post.json()["status"] == "stored"
    assert post.json()["recipe_hash"] == sample_recipe.recipe_hash

    get = client.get(
        f"/api/v1/gossip/recipes/{sample_recipe.recipe_hash}", headers=auth
    )
    assert get.status_code == 200, get.text
    data = get.json()
    assert data["measured_speedup"] == pytest.approx(62.3)
    assert data["backend"] == "vllm"
    assert data["verified_by_node"] == "node-a100-01"


# ── Test 5: verification_count increments on duplicate ────────────────────

def test_verification_count_increments_on_duplicate(client, auth, sample_recipe):
    """
    POSTing the same recipe_hash a second time (from a different node)
    must increment verification_count, not insert a duplicate row.
    """
    from dataclasses import asdict
    recipe_data = asdict(sample_recipe)

    post1 = client.post("/api/v1/gossip/recipes", json=recipe_data, headers=auth)
    assert post1.status_code == 201

    # Second node confirms same recipe
    recipe_data2 = {**recipe_data, "verified_by_node": "node-a100-02"}
    post2 = client.post("/api/v1/gossip/recipes", json=recipe_data2, headers=auth)
    assert post2.status_code == 201
    assert post2.json()["verification_count"] == 2

    # List endpoint must still show only one recipe
    lst = client.get("/api/v1/gossip/recipes", headers=auth)
    assert lst.status_code == 200
    assert lst.json()["count"] == 1


# ── Test 6: Recipe expires after 7 days ───────────────────────────────────

def test_recipe_expires_after_7_days(kb, sample_recipe):
    """
    GossipKnowledgeBase.lookup() must return None for entries whose
    local_cache_time is older than 7 days.
    """
    import sqlite3 as _sq

    # Store with a cache time 8 days in the past
    kb.store(sample_recipe)
    eight_days_ago = time.time() - 8 * 86400
    with _sq.connect(kb.db_path) as conn:
        conn.execute(
            "UPDATE recipes SET local_cache_time = ? WHERE recipe_hash = ?",
            (eight_days_ago, sample_recipe.recipe_hash),
        )
        conn.commit()

    # lookup() must treat it as expired
    result = kb.lookup("mistral", "a100", "float16")
    assert result is None, "Expected expired recipe to return None"
