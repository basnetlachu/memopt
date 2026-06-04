"""
Tests for control plane degradation tracking.
All pass without GPU, without Redis.
"""


def test_database_degradation_columns(tmp_path):
    """New degradation columns exist after init."""
    from memopt.control_plane.database import Database
    db = Database(db_path=tmp_path / "test.db")
    db.init()

    # Insert a node, verify degradation columns have defaults
    import sqlite3
    conn = sqlite3.connect(str(tmp_path / "test.db"))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        INSERT INTO nodes (node_name, last_seen, status, current_workloads, updated_at)
        VALUES ('test-node', 1000.0, 'online', '[]', 1000.0)
    """)
    conn.commit()

    row = conn.execute("SELECT * FROM nodes WHERE node_name='test-node'").fetchone()
    assert row["is_degraded"] == 0
    assert row["drift_pct"] == 0
    assert row["degraded_since"] is None
    conn.close()


def test_update_node_status_creates_node(tmp_path):
    """update_node_status creates a node that doesn't exist yet."""
    from memopt.control_plane.database import Database
    db = Database(db_path=tmp_path / "test.db")
    db.init()

    result = db.update_node_status(
        node_name="new-node",
        healthy=False,
        degraded=True,
        drift_pct=8.5,
        reason="test",
    )
    assert result is True

    node = db.get_node("new-node")
    assert node is not None
    assert node["is_degraded"] == 1
    assert node["drift_pct"] == 8.5
    assert node["status"] == "degraded"


def test_update_node_status_updates_existing(tmp_path):
    """update_node_status updates an existing node."""
    from memopt.control_plane.database import Database, NodeRecord
    db = Database(db_path=tmp_path / "test.db")
    db.init()

    # Create initial node
    db.upsert_node(NodeRecord(
        node_name="gpu-01",
        last_seen=1000.0,
        gpu_count=4,
        total_vram_gb=320.0,
        active_processes=2,
        optimizations_applied=5,
        dollar_saved_today=10.0,
        dollar_saved_total=100.0,
        status="online",
        current_workloads="[]",
    ))

    # Mark degraded
    db.update_node_status(
        node_name="gpu-01",
        healthy=False,
        degraded=True,
        drift_pct=12.3,
    )

    node = db.get_node("gpu-01")
    assert node["is_degraded"] == 1
    assert node["drift_pct"] == 12.3
    assert node["status"] == "degraded"
    # Original fields preserved
    assert node["gpu_count"] == 4
    assert node["total_vram_gb"] == 320.0


def test_update_node_clears_degraded(tmp_path):
    """Setting healthy=True clears degradation."""
    from memopt.control_plane.database import Database
    db = Database(db_path=tmp_path / "test.db")
    db.init()

    db.update_node_status("node-a", healthy=False, degraded=True, drift_pct=5.0)
    db.update_node_status("node-a", healthy=True, degraded=False)

    node = db.get_node("node-a")
    assert node["is_degraded"] == 0
    assert node["status"] == "online"


def test_get_degraded_nodes(tmp_path):
    """get_degraded_nodes returns only degraded nodes."""
    from memopt.control_plane.database import Database
    db = Database(db_path=tmp_path / "test.db")
    db.init()

    db.update_node_status("node-a", healthy=True, degraded=False)
    db.update_node_status("node-b", healthy=False, degraded=True, drift_pct=7.0)
    db.update_node_status("node-c", healthy=False, degraded=True, drift_pct=15.0)

    degraded = db.get_degraded_nodes()
    assert len(degraded) == 2
    names = {n["node_name"] for n in degraded}
    assert names == {"node-b", "node-c"}


def test_get_degraded_nodes_empty(tmp_path):
    """No degraded nodes returns empty list."""
    from memopt.control_plane.database import Database
    db = Database(db_path=tmp_path / "test.db")
    db.init()

    db.update_node_status("node-a", healthy=True, degraded=False)
    assert db.get_degraded_nodes() == []
