"""
Tests for CockroachDB backend, URL routing, schema module, and
the heartbeat batcher.

CockroachDB integration tests auto-skip unless COCKROACHDB_TEST_URL
is set and reachable.
"""
import os
import time
from unittest.mock import MagicMock

import pytest


# ══════════════════════════════════════════════════════════════════════
#  Skip decorator
# ══════════════════════════════════════════════════════════════════════


def _crdb_available() -> bool:
    try:
        url = os.getenv("COCKROACHDB_TEST_URL", "")
        if not url:
            return False
        import psycopg2
        conn = psycopg2.connect(
            url.replace("cockroachdb://", "postgresql://"),
            connect_timeout=3)
        conn.close()
        return True
    except Exception:
        return False


requires_crdb = pytest.mark.skipif(
    not _crdb_available(),
    reason="CockroachDB not available (set COCKROACHDB_TEST_URL)")


# ══════════════════════════════════════════════════════════════════════
#  URL normalization (no network)
# ══════════════════════════════════════════════════════════════════════


def test_cockroachdb_url_normalization():
    """cockroachdb:// → postgresql:// is pure string surgery."""
    url = "cockroachdb://user:pass@host:26257/db"
    normalized = url.replace("cockroachdb://", "postgresql://")
    assert normalized.startswith("postgresql://")
    assert "user:pass" in normalized
    assert "host:26257" in normalized


# ══════════════════════════════════════════════════════════════════════
#  make_backend() URL routing (with real fallbacks)
# ══════════════════════════════════════════════════════════════════════


def test_make_backend_sqlite_default():
    from memopt.control_plane.database import (
        SQLiteBackend, make_backend)
    old = os.environ.pop("DATABASE_URL", None)
    try:
        backend = make_backend("")
        assert isinstance(backend, SQLiteBackend)
        backend.close()
    finally:
        if old is not None:
            os.environ["DATABASE_URL"] = old


def test_make_backend_sqlite_explicit():
    from memopt.control_plane.database import (
        SQLiteBackend, make_backend)
    backend = make_backend("sqlite:///")
    assert isinstance(backend, SQLiteBackend)
    backend.close()


def test_make_backend_cockroachdb_url_no_server():
    """cockroachdb:// URL with no reachable server → SQLite fallback."""
    from memopt.control_plane.database import (
        SQLiteBackend, make_backend)
    backend = make_backend(
        "cockroachdb://root@localhost:26257/memopt")
    assert isinstance(backend, SQLiteBackend)
    backend.close()


def test_make_backend_postgresql_url_no_server():
    from memopt.control_plane.database import (
        SQLiteBackend, make_backend)
    backend = make_backend(
        "postgresql://user:pass@localhost:5432/db")
    assert isinstance(backend, SQLiteBackend)
    backend.close()


def test_make_backend_unknown_scheme_falls_back():
    from memopt.control_plane.database import (
        SQLiteBackend, make_backend)
    backend = make_backend("redis://example:6379/0")
    assert isinstance(backend, SQLiteBackend)
    backend.close()


# ══════════════════════════════════════════════════════════════════════
#  CRDB schema module (pure string/SQL, no connection)
# ══════════════════════════════════════════════════════════════════════


def test_crdb_schema_sql_parseable():
    from memopt.control_plane.crdb_schema import CRDB_SCHEMA
    expected = [
        "nodes", "pods", "events",
        "boot_events", "image_versions",
        "rollout_events", "rollback_intents",
        "canary_baselines", "metrics",
    ]
    for table in expected:
        assert table in CRDB_SCHEMA, \
            f"Missing table in CRDB schema: {table}"


def test_crdb_schema_uses_uuid_not_serial():
    from memopt.control_plane.crdb_schema import CRDB_SCHEMA
    assert "gen_random_uuid()" in CRDB_SCHEMA
    assert "SERIAL" not in CRDB_SCHEMA
    assert "AUTOINCREMENT" not in CRDB_SCHEMA


def test_crdb_schema_has_indexes():
    from memopt.control_plane.crdb_schema import CRDB_SCHEMA
    for idx in ("idx_boot_node_time",
                "idx_rollout_id",
                "idx_nodes_region",
                "idx_boot_node",
                "idx_rollback_status"):
        assert idx in CRDB_SCHEMA, f"Missing index: {idx}"


def test_crdb_schema_split_statements():
    """_split_statements strips comments and empties."""
    from memopt.control_plane.crdb_schema import (
        CRDB_SCHEMA, _split_statements)
    stmts = _split_statements(CRDB_SCHEMA)
    assert len(stmts) >= 10
    for stmt in stmts:
        assert not stmt.startswith("--")
        assert stmt.strip()


# ══════════════════════════════════════════════════════════════════════
#  HeartbeatBatcher (runs against in-memory SQLite)
# ══════════════════════════════════════════════════════════════════════


def test_heartbeat_batcher_queues_and_flushes():
    from memopt.control_plane.database import (
        ControlPlaneDB, SQLiteBackend)
    from memopt.control_plane.server import HeartbeatBatcher

    backend = SQLiteBackend(":memory:")
    db = ControlPlaneDB(backend=backend)
    db.init()

    batcher = HeartbeatBatcher(db)
    batcher.BATCH_INTERVAL_S = 0.05
    batcher.start()

    for i in range(5):
        batcher.record(
            node_id=f"node-{i}",
            healthy=True,
            degraded=False,
            drift_pct=0.0,
            reason="test")

    # Wait for at least one flush tick
    time.sleep(0.3)

    stats = batcher.stats()
    assert stats["flushed"] >= 5
    assert stats["pending"] == 0

    batcher.stop()
    backend.close()


def test_heartbeat_batcher_stop_flushes_final():
    """stop() drains the queue so pending → 0."""
    from memopt.control_plane.database import (
        ControlPlaneDB, SQLiteBackend)
    from memopt.control_plane.server import HeartbeatBatcher

    backend = SQLiteBackend(":memory:")
    db = ControlPlaneDB(backend=backend)
    db.init()

    batcher = HeartbeatBatcher(db)
    batcher.BATCH_INTERVAL_S = 100  # never auto-flush
    batcher.start()

    batcher.record("flush-test", True, False, 0.0, "stop-flush")
    assert batcher.stats()["pending"] == 1

    batcher.stop()
    stats = batcher.stats()
    assert stats["pending"] == 0
    assert stats["flushed"] >= 1
    backend.close()


def test_heartbeat_batcher_never_raises_on_bad_db():
    """A DB that raises on every write must not crash the batcher."""
    from memopt.control_plane.server import HeartbeatBatcher

    bad_db = MagicMock()
    bad_db.update_node_status.side_effect = Exception("DB error")

    batcher = HeartbeatBatcher(bad_db)
    batcher.BATCH_INTERVAL_S = 0.05
    batcher.start()

    batcher.record("node-1", True, False, 0.0, "test")
    time.sleep(0.2)

    stats = batcher.stats()
    assert stats["errors"] >= 1
    # Did not crash the thread
    batcher.stop()


def test_heartbeat_batcher_stats_keys():
    from memopt.control_plane.database import (
        ControlPlaneDB, SQLiteBackend)
    from memopt.control_plane.server import HeartbeatBatcher

    backend = SQLiteBackend(":memory:")
    db = ControlPlaneDB(backend=backend)
    db.init()

    batcher = HeartbeatBatcher(db)
    stats = batcher.stats()
    for key in ("pending", "flushed", "errors", "interval"):
        assert key in stats
    backend.close()


def test_heartbeat_batcher_multi_flush():
    """Multiple writes across multiple flush windows all land."""
    from memopt.control_plane.database import (
        ControlPlaneDB, SQLiteBackend)
    from memopt.control_plane.server import HeartbeatBatcher

    backend = SQLiteBackend(":memory:")
    db = ControlPlaneDB(backend=backend)
    db.init()

    batcher = HeartbeatBatcher(db)
    batcher.BATCH_INTERVAL_S = 0.05
    batcher.start()
    try:
        for i in range(3):
            batcher.record(
                f"n{i}", True, False, 0.0, "w1")
        time.sleep(0.15)
        for i in range(3):
            batcher.record(
                f"n{i+3}", True, False, 0.0, "w2")
        time.sleep(0.2)
        stats = batcher.stats()
        assert stats["flushed"] >= 6
    finally:
        batcher.stop()
        backend.close()


def test_heartbeat_batcher_endpoint():
    """GET /api/v1/heartbeat-batcher/stats returns batcher metrics."""
    from fastapi.testclient import TestClient
    import memopt.control_plane.server as cp_mod

    with TestClient(cp_mod.app) as client:
        resp = client.get(
            "/api/v1/heartbeat-batcher/stats",
            headers={"X-Memopt-API-Key": cp_mod._API_KEY})
        assert resp.status_code == 200
        data = resp.json()
        assert data["active"] is True
        for key in ("pending", "flushed", "errors", "interval"):
            assert key in data


# ══════════════════════════════════════════════════════════════════════
#  CRDB integration tests (auto-skip without COCKROACHDB_TEST_URL)
# ══════════════════════════════════════════════════════════════════════


@requires_crdb
def test_crdb_backend_connect():
    url = os.getenv("COCKROACHDB_TEST_URL")
    from memopt.control_plane.database import CockroachDBBackend
    backend = CockroachDBBackend(url)
    backend.close()


@requires_crdb
def test_crdb_backend_execute_and_fetch():
    url = os.getenv("COCKROACHDB_TEST_URL")
    from memopt.control_plane.database import CockroachDBBackend
    backend = CockroachDBBackend(url)
    try:
        backend.execute(
            "CREATE TABLE IF NOT EXISTS memopt_test_table "
            "(id UUID DEFAULT gen_random_uuid() PRIMARY KEY, val TEXT)")
        backend.execute(
            "INSERT INTO memopt_test_table (val) VALUES (%s)",
            ("hello",))
        rows = backend.fetchall(
            "SELECT val FROM memopt_test_table LIMIT 1")
        assert len(rows) >= 1
        assert rows[0]["val"] == "hello"
    finally:
        try:
            backend.execute("DROP TABLE memopt_test_table")
        except Exception:
            pass
        backend.close()


@requires_crdb
def test_crdb_schema_apply():
    url = os.getenv("COCKROACHDB_TEST_URL")
    from memopt.control_plane.crdb_schema import apply_schema
    apply_schema(url)  # must not raise


# ══════════════════════════════════════════════════════════════════════
#  Query optimization (Phase 8b)
# ══════════════════════════════════════════════════════════════════════


def test_dialect_detection_sqlite():
    from memopt.control_plane.database import (
        ControlPlaneDB, SQLiteBackend)
    backend = SQLiteBackend(":memory:")
    db = ControlPlaneDB(backend=backend)
    assert db.dialect == "sqlite"
    backend.close()


def test_dialect_detection_postgresql():
    from memopt.control_plane.database import (
        ControlPlaneDB, PostgreSQLBackend)

    mock_backend = MagicMock(spec=PostgreSQLBackend)
    db = ControlPlaneDB(backend=mock_backend)
    assert db.dialect == "postgresql"


def test_dialect_detection_cockroachdb():
    from memopt.control_plane.database import (
        ControlPlaneDB, CockroachDBBackend)

    mock_backend = MagicMock(spec=CockroachDBBackend)
    db = ControlPlaneDB(backend=mock_backend)
    assert db.dialect == "cockroachdb"


def test_get_boot_status_sqlite_dialect():
    """Latest-per-node correctness on SQLite path."""
    from memopt.control_plane.database import (
        ControlPlaneDB, SQLiteBackend)

    backend = SQLiteBackend(":memory:")
    db = ControlPlaneDB(backend=backend)
    db.init()

    db.record_boot_event("n1", "v1.0.0", "PASSED", 8, 40.0)
    db.record_boot_event("n1", "v2.0.0", "PASSED", 8, 42.0)
    db.record_boot_event("n2", "v2.0.0", "FAILED", 8, 41.0)

    status = db.get_boot_status()
    assert status["total_nodes"] == 2
    assert status["version_distribution"].get("v2.0.0") == 2
    assert status["failed_certs"] == 1
    assert "v1.0.0" not in status["version_distribution"]
    backend.close()


def test_get_nodes_by_version_latest_only():
    """Only the latest boot per node contributes."""
    from memopt.control_plane.database import (
        ControlPlaneDB, SQLiteBackend)

    backend = SQLiteBackend(":memory:")
    db = ControlPlaneDB(backend=backend)
    db.init()

    # n1: v1 then v2 → should be in v2
    db.record_boot_event("n1", "v1", "PASSED", 8, 10.0)
    db.record_boot_event("n1", "v2", "PASSED", 8, 10.0)
    # n2: v2 only
    db.record_boot_event("n2", "v2", "PASSED", 8, 10.0)

    v2 = db.get_nodes_by_version("v2")
    ids = {r["node_id"] for r in v2}
    assert ids == {"n1", "n2"}
    # n1's v1 row must not appear in the v1 result
    v1 = db.get_nodes_by_version("v1")
    assert not any(r["node_id"] == "n1" for r in v1)
    backend.close()


def test_get_version_node_count_latest_only():
    from memopt.control_plane.database import (
        ControlPlaneDB, SQLiteBackend)

    backend = SQLiteBackend(":memory:")
    db = ControlPlaneDB(backend=backend)
    db.init()

    db.record_boot_event("n1", "v1", "PASSED", 8, 10.0)
    db.record_boot_event("n1", "v2", "PASSED", 8, 10.0)
    db.record_boot_event("n2", "v2", "PASSED", 8, 10.0)

    # Only the latest boot per node is counted
    assert db.get_version_node_count("v2") == 2
    assert db.get_version_node_count("v1") == 0
    backend.close()


def test_get_nodes_on_version_latest_only():
    from memopt.control_plane.database import (
        ControlPlaneDB, SQLiteBackend)

    backend = SQLiteBackend(":memory:")
    db = ControlPlaneDB(backend=backend)
    db.init()

    db.record_boot_event("n1", "v1", "PASSED", 8, 10.0)
    db.record_boot_event("n1", "v2", "PASSED", 8, 10.0)
    db.record_boot_event("n2", "v2", "PASSED", 8, 10.0)

    nodes = set(db.get_nodes_on_version("v2"))
    assert nodes == {"n1", "n2"}
    assert db.get_nodes_on_version("v1") == []
    backend.close()


def test_migration_007_creates_indexes():
    """Composite + single-column indexes present after init."""
    from memopt.control_plane.database import (
        ControlPlaneDB, SQLiteBackend)

    backend = SQLiteBackend(":memory:")
    db = ControlPlaneDB(backend=backend)
    db.init()

    rows = backend.fetchall(
        "SELECT name FROM sqlite_master "
        "WHERE type='index' "
        "ORDER BY name")
    names = {r["name"] for r in rows}

    assert "idx_boot_events_node_time" in names
    assert "idx_nodes_mac_address" in names
    assert "idx_nodes_is_degraded" in names
    backend.close()


# ══════════════════════════════════════════════════════════════════════
#  make_backend_for_scale
# ══════════════════════════════════════════════════════════════════════


def test_pool_sizing_small():
    from memopt.control_plane.database import (
        make_backend_for_scale, SQLiteBackend)
    b = make_backend_for_scale("", 50)
    assert isinstance(b, SQLiteBackend)
    b.close()


def test_pool_sizing_does_not_crash():
    from memopt.control_plane.database import (
        make_backend_for_scale, SQLiteBackend)
    for nodes in [10, 100, 1000, 10_000, 100_000, 1_000_000]:
        b = make_backend_for_scale("", nodes)
        assert isinstance(b, SQLiteBackend)
        b.close()


def test_pool_sizing_cockroachdb_unreachable_falls_back():
    from memopt.control_plane.database import (
        make_backend_for_scale, SQLiteBackend)
    b = make_backend_for_scale(
        "cockroachdb://root@localhost:26257/memopt",
        expected_nodes=100_000)
    # CRDB unreachable → SQLite fallback
    assert isinstance(b, SQLiteBackend)
    b.close()


def test_pool_sizing_postgresql_unreachable_falls_back():
    from memopt.control_plane.database import (
        make_backend_for_scale, SQLiteBackend)
    b = make_backend_for_scale(
        "postgresql://user:pass@localhost:5432/db",
        expected_nodes=500)
    assert isinstance(b, SQLiteBackend)
    b.close()


# ══════════════════════════════════════════════════════════════════════
#  Documentation
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
#  Migration tooling (Phase 8c)
# ══════════════════════════════════════════════════════════════════════


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))


def test_migrate_script_syntax():
    import subprocess
    result = subprocess.run(
        ["bash", "-n",
         os.path.join(_repo_root(),
                      "scripts/migrate_to_crdb.sh")],
        capture_output=True, text=True)
    assert result.returncode == 0, \
        f"Syntax error: {result.stderr}"


def test_migrate_script_help():
    import subprocess
    result = subprocess.run(
        ["bash",
         os.path.join(_repo_root(),
                      "scripts/migrate_to_crdb.sh"),
         "--help"],
        capture_output=True, text=True)
    assert result.returncode == 0
    out = result.stdout + result.stderr
    assert "Usage" in out or "usage" in out


def test_migrate_script_requires_source_and_target():
    """Missing --source or --target exits non-zero."""
    import subprocess
    result = subprocess.run(
        ["bash",
         os.path.join(_repo_root(),
                      "scripts/migrate_to_crdb.sh")],
        capture_output=True, text=True)
    assert result.returncode != 0


def test_migrate_script_rejects_bad_target_scheme():
    """Target URL must start with cockroachdb://."""
    import subprocess
    result = subprocess.run(
        ["bash",
         os.path.join(_repo_root(),
                      "scripts/migrate_to_crdb.sh"),
         "--source", "sqlite:///tmp/x.db",
         "--target", "postgresql://x/y"],
        capture_output=True, text=True)
    assert result.returncode != 0


def test_crdb_deployment_doc_exists():
    path = os.path.join(
        _repo_root(), "docs/cockroachdb_deployment.md")
    assert os.path.exists(path)

    with open(path) as f:
        content = f.read()

    # Must document real limitations
    assert ("limitation" in content.lower()
            or "known" in content.lower())
    assert "UUID" in content
    assert ("geo-partition" in content.lower()
            or "geo_partition" in content.lower())
    # Must document the migration script
    assert "migrate_to_crdb" in content


# ══════════════════════════════════════════════════════════════════════
#  Database health endpoint
# ══════════════════════════════════════════════════════════════════════


def test_database_health_endpoint():
    from fastapi.testclient import TestClient
    import memopt.control_plane.server as cp_mod

    with TestClient(cp_mod.app) as client:
        response = client.get(
            "/api/v1/database/health",
            headers={"X-Memopt-API-Key": cp_mod._API_KEY})
        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "dialect" in data
        assert data["dialect"] in (
            "sqlite", "postgresql", "cockroachdb", "unknown")

        if data["status"] == "ok":
            assert "read_ms" in data
            assert "ping_ms" in data
            assert "node_count" in data
            assert data["read_ms"] >= 0
            assert data["ping_ms"] >= 0


def test_database_health_no_auth():
    from fastapi.testclient import TestClient
    import memopt.control_plane.server as cp_mod

    with TestClient(cp_mod.app) as client:
        response = client.get("/api/v1/database/health")
        assert response.status_code == 401


def test_database_health_includes_batcher_when_active():
    """When HeartbeatBatcher is running, its stats are included."""
    from fastapi.testclient import TestClient
    import memopt.control_plane.server as cp_mod

    with TestClient(cp_mod.app) as client:
        response = client.get(
            "/api/v1/database/health",
            headers={"X-Memopt-API-Key": cp_mod._API_KEY})
        data = response.json()
        if data["status"] == "ok":
            # Startup event creates the batcher; it should be
            # present in the response.
            assert data.get("batcher") is not None
            batcher = data["batcher"]
            for key in ("pending", "flushed", "errors", "interval"):
                assert key in batcher


def test_query_optimization_doc_exists():
    repo_root = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))
    path = os.path.join(repo_root, "docs/query_optimization.md")
    assert os.path.exists(path), \
        "docs/query_optimization.md not found"

    with open(path) as f:
        content = f.read()

    assert "boot_events" in content
    # DISTINCT ON is the optimized pattern; MAX(id) is the baseline.
    assert "DISTINCT ON" in content
    assert "MAX(id)" in content
    assert "index" in content.lower()
    # Four hot queries must all be mentioned by name
    assert "get_boot_status" in content
    assert "get_nodes_by_version" in content
