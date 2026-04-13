"""
Tests for PXE boot endpoints + boot_events DB methods.

HTTP tests use `with TestClient(app) as client:` so FastAPI's startup
event fires and sets the module-level API key. DB tests run against
an in-memory SQLite backend for isolation.
"""
import os

import pytest


# ══════════════════════════════════════════════════════════════════════
#  HTTP endpoints
# ══════════════════════════════════════════════════════════════════════


def test_boot_config_endpoint_returns_defaults():
    from fastapi.testclient import TestClient
    from memopt.control_plane.server import app

    with TestClient(app) as client:
        response = client.get(
            "/boot/config/aa:bb:cc:dd:ee:ff")
        assert response.status_code == 200
        data = response.json()

        required = ["node_id", "mac_address",
                    "image_url", "image_version",
                    "kernel_args", "generated_at",
                    "rack", "pod", "region"]
        for key in required:
            assert key in data, f"Missing: {key}"

        assert isinstance(data["kernel_args"], list)
        assert data["mac_address"] == "aa:bb:cc:dd:ee:ff"
        # kernel_args lines are non-empty strings
        for arg in data["kernel_args"]:
            assert isinstance(arg, str)
            assert arg


def test_boot_config_normalizes_mac():
    from fastapi.testclient import TestClient
    from memopt.control_plane.server import app

    with TestClient(app) as client:
        response = client.get(
            "/boot/config/AA-BB-CC-DD-EE-FF")
        assert response.status_code == 200
        data = response.json()
        assert data["mac_address"] == "aa:bb:cc:dd:ee:ff"


def test_boot_config_no_auth_required():
    """Boot endpoints intentionally have no auth."""
    from fastapi.testclient import TestClient
    from memopt.control_plane.server import app

    with TestClient(app) as client:
        # No X-Memopt-API-Key header
        response = client.get(
            "/boot/config/aa:bb:cc:dd:ee:ff")
        assert response.status_code == 200


def test_boot_callback_records_event():
    from fastapi.testclient import TestClient
    from memopt.control_plane.server import app

    with TestClient(app) as client:
        response = client.post(
            "/boot/callback",
            json={
                "node_id": "test-node-boot-01",
                "mac_address": "aa:bb:cc:dd:ee:01",
                "image_version": "v1.0.0",
                "boot_time_seconds": 45.2,
                "gpu_count": 8,
                "cert_status": "PASSED",
            })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["node_id"] == "test-node-boot-01"


def test_boot_callback_missing_node_id():
    from fastapi.testclient import TestClient
    from memopt.control_plane.server import app

    with TestClient(app) as client:
        response = client.post(
            "/boot/callback",
            json={"image_version": "v1.0.0"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert "node_id" in data.get("reason", "").lower()


def test_boot_status_endpoint():
    from fastapi.testclient import TestClient
    import memopt.control_plane.server as cp_mod

    with TestClient(cp_mod.app) as client:
        response = client.get(
            "/boot/status",
            headers={"X-Memopt-API-Key": cp_mod._API_KEY})
        assert response.status_code == 200
        data = response.json()
        required = ["total_nodes", "booted_nodes",
                    "current_version",
                    "version_distribution",
                    "failed_certs"]
        for key in required:
            assert key in data, f"Missing: {key}"


def test_boot_status_reflects_callbacks():
    """After a callback, status should list the version."""
    from fastapi.testclient import TestClient
    import memopt.control_plane.server as cp_mod

    with TestClient(cp_mod.app) as client:
        client.post("/boot/callback", json={
            "node_id":           "status-test-01",
            "image_version":     "v2.0.0",
            "cert_status":       "PASSED",
            "gpu_count":         8,
            "boot_time_seconds": 42.0,
        })

        response = client.get(
            "/boot/status",
            headers={"X-Memopt-API-Key": cp_mod._API_KEY})
        data = response.json()

        assert data["total_nodes"] >= 1
        assert "v2.0.0" in data["version_distribution"]


def test_boot_status_rejects_missing_auth():
    from fastapi.testclient import TestClient
    import memopt.control_plane.server as cp_mod

    with TestClient(cp_mod.app) as client:
        response = client.get("/boot/status")
        assert response.status_code == 401


# ══════════════════════════════════════════════════════════════════════
#  Database methods (isolated in-memory backend)
# ══════════════════════════════════════════════════════════════════════


def test_database_record_boot_event():
    from memopt.control_plane.database import (
        ControlPlaneDB, SQLiteBackend)

    backend = SQLiteBackend(":memory:")
    db = ControlPlaneDB(backend=backend)
    db.init()

    db.record_boot_event(
        node_id="db-test-node",
        image_version="v1.0.0",
        cert_status="PASSED",
        gpu_count=8,
        boot_time_seconds=55.3)

    status = db.get_boot_status()
    assert status["total_nodes"] == 1
    assert status["booted_nodes"] == 1
    assert status["current_version"] == "v1.0.0"
    assert "v1.0.0" in status["version_distribution"]
    assert status["version_distribution"]["v1.0.0"] == 1
    assert status["failed_certs"] == 0
    backend.close()


def test_database_latest_boot_per_node():
    """get_boot_status uses only the latest boot per node."""
    from memopt.control_plane.database import (
        ControlPlaneDB, SQLiteBackend)

    backend = SQLiteBackend(":memory:")
    db = ControlPlaneDB(backend=backend)
    db.init()

    # Two reboots of the same node, different versions
    db.record_boot_event(
        "n1", "v1.0.0", "PASSED", 8, 10.0)
    db.record_boot_event(
        "n1", "v2.0.0", "PASSED", 8, 12.0)
    # Separate node
    db.record_boot_event(
        "n2", "v2.0.0", "FAILED", 4, 20.0)

    status = db.get_boot_status()
    assert status["total_nodes"] == 2
    assert status["version_distribution"]["v2.0.0"] == 2
    assert "v1.0.0" not in status["version_distribution"]
    assert status["failed_certs"] == 1
    backend.close()


def test_database_get_node_by_mac_not_found():
    from memopt.control_plane.database import (
        ControlPlaneDB, SQLiteBackend)

    backend = SQLiteBackend(":memory:")
    db = ControlPlaneDB(backend=backend)
    db.init()

    assert db.get_node_by_mac("aa:bb:cc:dd:ee:ff") is None
    backend.close()


def test_database_get_boot_status_empty():
    """With no boot events recorded."""
    from memopt.control_plane.database import (
        ControlPlaneDB, SQLiteBackend)

    backend = SQLiteBackend(":memory:")
    db = ControlPlaneDB(backend=backend)
    db.init()

    status = db.get_boot_status()
    assert status["total_nodes"] == 0
    assert status["current_version"] == "unknown"
    assert status["version_distribution"] == {}
    backend.close()


def test_database_migration_003_adds_nodes_columns():
    """New nodes columns present after init."""
    from memopt.control_plane.database import (
        ControlPlaneDB, SQLiteBackend)

    backend = SQLiteBackend(":memory:")
    db = ControlPlaneDB(backend=backend)
    db.init()

    rows = backend.fetchall("PRAGMA table_info(nodes)")
    cols = {r["name"] for r in rows}
    for expected in (
            "mac_address", "image_version",
            "rack", "pod", "region"):
        assert expected in cols, \
            f"migration 003 did not add column: {expected}"
    backend.close()


def test_database_init_is_idempotent():
    """Calling init() twice does not fail (duplicate-column tolerance)."""
    from memopt.control_plane.database import (
        ControlPlaneDB, SQLiteBackend)

    backend = SQLiteBackend(":memory:")
    db = ControlPlaneDB(backend=backend)
    db.init()
    db.init()  # must not raise
    backend.close()


# ══════════════════════════════════════════════════════════════════════
#  iPXE script sanity
# ══════════════════════════════════════════════════════════════════════


def test_ipxe_script_valid():
    repo_root = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))
    path = os.path.join(
        repo_root, "deploy/pxe/ipxe_boot.script")
    with open(path) as f:
        content = f.read()
    assert "#!ipxe" in content
    assert "boot-server" in content
    assert "chain" in content
    assert "kernel" in content
    assert "boot" in content


def test_node_config_template_exists():
    repo_root = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))
    path = os.path.join(
        repo_root, "deploy/pxe/node_config.json.template")
    with open(path) as f:
        content = f.read()
    # Template placeholders present
    for placeholder in (
            "{{NODE_ID}}", "{{MAC_ADDRESS}}",
            "{{IMAGE_VERSION}}",
            "{{BOOT_SERVER_URL}}"):
        assert placeholder in content, \
            f"Missing placeholder: {placeholder}"
