"""
Tests for database abstraction — SQLite backend, migrations, pod ops.
No PostgreSQL required (uses SQLite :memory:).
"""
import time


from memopt.control_plane.database import (
    Database, SQLiteBackend, make_backend,
    _pg_to_sqlite,
)


def test_sqlite_backend_execute_and_fetch():
    """SQLiteBackend CRUD works."""
    backend = SQLiteBackend(":memory:")
    backend.executescript(
        "CREATE TABLE test (id INTEGER, val TEXT)")
    backend.execute(
        "INSERT INTO test VALUES (?, ?)", (1, "hello"))
    rows = backend.fetchall("SELECT * FROM test")
    assert len(rows) == 1
    assert rows[0]["val"] == "hello"
    backend.close()


def test_sqlite_backend_fetchone():
    """fetchone returns dict or None."""
    backend = SQLiteBackend(":memory:")
    backend.executescript(
        "CREATE TABLE test (id INTEGER, val TEXT)")
    backend.execute(
        "INSERT INTO test VALUES (?, ?)", (1, "world"))
    row = backend.fetchone(
        "SELECT * FROM test WHERE id = ?", (1,))
    assert row is not None
    assert row["val"] == "world"
    none_row = backend.fetchone(
        "SELECT * FROM test WHERE id = ?", (999,))
    assert none_row is None
    backend.close()


def test_pg_to_sqlite_conversion():
    """$1,$2 converted to ? for SQLite."""
    assert _pg_to_sqlite("SELECT $1") == "SELECT ?"
    assert _pg_to_sqlite(
        "WHERE a=$1 AND b=$2") == "WHERE a=? AND b=?"
    assert _pg_to_sqlite("SELECT *") == "SELECT *"


def test_make_backend_sqlite_default():
    """Empty URL → SQLite."""
    backend = make_backend("")
    assert isinstance(backend, SQLiteBackend)
    backend.close()


def test_make_backend_postgresql_no_psycopg2():
    """PostgreSQL URL without psycopg2 → SQLite fallback."""
    import sys

    # Temporarily make psycopg2 unimportable
    orig = sys.modules.get("psycopg2")
    sys.modules["psycopg2"] = None
    try:
        backend = make_backend(
            "postgresql://user:pass@localhost/db")
        assert isinstance(backend, SQLiteBackend)
    finally:
        if orig is not None:
            sys.modules["psycopg2"] = orig
        else:
            sys.modules.pop("psycopg2", None)
    backend.close()


def test_migration_runs_idempotent():
    """Migrations can run multiple times without error."""
    backend = SQLiteBackend(":memory:")
    db = Database(backend=backend)
    db.init()
    db.init()  # second run — must not fail

    # Verify tables exist
    rows = backend.fetchall(
        "SELECT name FROM sqlite_master "
        "WHERE type='table'")
    names = [r["name"] for r in rows]
    assert "nodes" in names
    assert "events" in names
    assert "pods" in names
    assert "migrations" in names
    backend.close()


def test_database_node_operations():
    """All node methods work on SQLite."""
    backend = SQLiteBackend(":memory:")
    db = Database(backend=backend)
    db.init()

    db.update_node_status(
        "node-001", healthy=True,
        degraded=False, drift_pct=0.0)

    node = db.get_node("node-001")
    assert node is not None
    assert node["node_name"] == "node-001"

    all_nodes = db.get_all_nodes()
    assert len(all_nodes) >= 1

    degraded = db.get_degraded_nodes()
    assert isinstance(degraded, list)
    backend.close()


def test_database_pod_operations():
    """Pod upsert, get, list all work."""
    backend = SQLiteBackend(":memory:")
    db = Database(backend=backend)
    db.init()

    db.upsert_pod("pod-001", {
        "node_count": 100,
        "healthy_nodes": 98,
        "pod_oracle_size": 50000,
        "avg_hbm_free_gb": 45.2,
        "gkd_hit_rate_pct": 87.3,
        "reported_at": time.time(),
    })

    pod = db.get_pod("pod-001")
    assert pod is not None
    assert pod["node_count"] == 100

    pods = db.list_pods()
    assert len(pods) >= 1

    # Upsert again — should update not duplicate
    db.upsert_pod("pod-001", {
        "node_count": 200,
        "healthy_nodes": 195,
    })
    pod = db.get_pod("pod-001")
    assert pod["node_count"] == 200
    backend.close()


def test_database_cluster_summary():
    """get_cluster_summary works on empty DB."""
    backend = SQLiteBackend(":memory:")
    db = Database(backend=backend)
    db.init()

    summary = db.get_cluster_summary()
    assert summary["total_nodes"] == 0
    assert summary["online_nodes"] == 0
    backend.close()
