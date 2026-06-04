"""
Tests for image version registry + rollback mechanism.

HTTP tests use `with TestClient(app) as client:` so the control
plane's startup event fires and sets the module-level API key.
DB tests use an in-memory backend for isolation.
"""
import os
import subprocess



# ══════════════════════════════════════════════════════════════════════
#  HTTP — image registry
# ══════════════════════════════════════════════════════════════════════


def test_register_image_version():
    from fastapi.testclient import TestClient
    import memopt.control_plane.server as cp_mod

    with TestClient(cp_mod.app) as client:
        response = client.post(
            "/api/v1/images",
            json={
                "version": "v1.0.0-test",
                "git_commit": "abc1234",
                "build_date": "2026-04-13T00:00:00Z",
                "cuda_version": "12.4",
                "sm_targets": "86;90;100",
                "image_url": "registry/memopt:v1.0.0-test",
                "digest": "sha256:abc",
            },
            headers={"X-Memopt-API-Key": cp_mod._API_KEY})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "v1.0.0-test"


def test_register_image_missing_version():
    from fastapi.testclient import TestClient
    import memopt.control_plane.server as cp_mod

    with TestClient(cp_mod.app) as client:
        response = client.post(
            "/api/v1/images",
            json={"git_commit": "abc1234"},
            headers={"X-Memopt-API-Key": cp_mod._API_KEY})
        assert response.status_code == 400


def test_list_image_versions():
    from fastapi.testclient import TestClient
    import memopt.control_plane.server as cp_mod

    with TestClient(cp_mod.app) as client:
        response = client.get(
            "/api/v1/images",
            headers={"X-Memopt-API-Key": cp_mod._API_KEY})
        assert response.status_code == 200
        data = response.json()
        assert "versions" in data
        assert "total" in data
        assert "stable_version" in data
        assert isinstance(data["versions"], list)


def test_mark_version_stable():
    from fastapi.testclient import TestClient
    import memopt.control_plane.server as cp_mod

    with TestClient(cp_mod.app) as client:
        client.post(
            "/api/v1/images",
            json={
                "version": "v2.0.0-stable",
                "git_commit": "def5678",
                "build_date": "2026-04-13T00:00:00Z",
                "cuda_version": "12.4",
                "sm_targets": "86;90;100",
                "image_url": "registry/memopt:v2.0.0",
            },
            headers={"X-Memopt-API-Key": cp_mod._API_KEY})

        response = client.post(
            "/api/v1/images/v2.0.0-stable/stable",
            headers={"X-Memopt-API-Key": cp_mod._API_KEY})
        assert response.status_code == 200
        assert response.json()["is_stable"] is True

        # Listing should now show v2.0.0-stable as stable_version
        listing = client.get(
            "/api/v1/images",
            headers={"X-Memopt-API-Key": cp_mod._API_KEY}).json()
        assert listing["stable_version"] == "v2.0.0-stable"


def test_mark_version_deprecated():
    from fastapi.testclient import TestClient
    import memopt.control_plane.server as cp_mod

    with TestClient(cp_mod.app) as client:
        client.post(
            "/api/v1/images",
            json={
                "version": "v0.5.0-old",
                "git_commit": "deadbef",
                "build_date": "2025-01-01T00:00:00Z",
                "cuda_version": "12.2",
                "sm_targets": "80",
                "image_url": "registry/memopt:v0.5.0",
            },
            headers={"X-Memopt-API-Key": cp_mod._API_KEY})

        response = client.post(
            "/api/v1/images/v0.5.0-old/deprecated",
            headers={"X-Memopt-API-Key": cp_mod._API_KEY})
        assert response.status_code == 200
        assert response.json()["is_deprecated"] is True


def test_list_nodes_on_version():
    from fastapi.testclient import TestClient
    import memopt.control_plane.server as cp_mod

    with TestClient(cp_mod.app) as client:
        # Record a boot on v3.0.0
        client.post("/boot/callback", json={
            "node_id":           "nodes-test-01",
            "image_version":     "v3.0.0",
            "cert_status":       "PASSED",
            "gpu_count":         4,
            "boot_time_seconds": 10.0,
        })

        response = client.get(
            "/api/v1/images/v3.0.0/nodes",
            headers={"X-Memopt-API-Key": cp_mod._API_KEY})
        assert response.status_code == 200
        data = response.json()
        assert data["version"] == "v3.0.0"
        assert "nodes-test-01" in data["nodes"]


# ══════════════════════════════════════════════════════════════════════
#  HTTP — rollback
# ══════════════════════════════════════════════════════════════════════


def test_rollback_records_intent():
    from fastapi.testclient import TestClient
    import memopt.control_plane.server as cp_mod

    with TestClient(cp_mod.app) as client:
        # Register target version
        client.post(
            "/api/v1/images",
            json={
                "version": "v0.9.0",
                "git_commit": "old123",
                "build_date": "2025-12-01T00:00:00Z",
                "cuda_version": "12.4",
                "sm_targets": "86;90;100",
                "image_url": "registry/memopt:v0.9.0",
            },
            headers={"X-Memopt-API-Key": cp_mod._API_KEY})

        response = client.post(
            "/api/v1/nodes/rollback-test-node/rollback",
            json={
                "target_version": "v0.9.0",
                "reason": "regression test",
            },
            headers={"X-Memopt-API-Key": cp_mod._API_KEY})

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "rollback_scheduled"
        assert data["target_version"] == "v0.9.0"


def test_rollback_unknown_version_404():
    from fastapi.testclient import TestClient
    import memopt.control_plane.server as cp_mod

    with TestClient(cp_mod.app) as client:
        response = client.post(
            "/api/v1/nodes/some-node/rollback",
            json={
                "target_version": "v999.999.999",
                "reason": "test",
            },
            headers={"X-Memopt-API-Key": cp_mod._API_KEY})
        assert response.status_code == 404


def test_rollback_missing_target_version_400():
    from fastapi.testclient import TestClient
    import memopt.control_plane.server as cp_mod

    with TestClient(cp_mod.app) as client:
        response = client.post(
            "/api/v1/nodes/some-node/rollback",
            json={"reason": "no target"},
            headers={"X-Memopt-API-Key": cp_mod._API_KEY})
        assert response.status_code == 400


# ══════════════════════════════════════════════════════════════════════
#  Database-level tests (isolated in-memory backend)
# ══════════════════════════════════════════════════════════════════════


def test_database_register_and_stable():
    from memopt.control_plane.database import (
        ControlPlaneDB, SQLiteBackend)

    backend = SQLiteBackend(":memory:")
    db = ControlPlaneDB(backend=backend)
    db.init()

    db.register_image_version(
        version="v1.0.0",
        git_commit="abc1234",
        build_date="2026-04-13T00:00:00Z",
        cuda_version="12.4",
        sm_targets="86;90;100",
        image_url="registry/memopt:v1.0.0")

    db.mark_version_stable("v1.0.0")

    assert db.get_stable_version() == "v1.0.0"
    versions = db.list_image_versions()
    assert len(versions) == 1
    assert versions[0]["version"] == "v1.0.0"
    assert versions[0]["is_stable"] == 1
    backend.close()


def test_database_stable_excludes_deprecated():
    """A version that is both stable AND deprecated is not returned."""
    from memopt.control_plane.database import (
        ControlPlaneDB, SQLiteBackend)

    backend = SQLiteBackend(":memory:")
    db = ControlPlaneDB(backend=backend)
    db.init()

    db.register_image_version(
        "v1.0.0", "a", "2026-04-13", "12.4",
        "86", "url")
    db.mark_version_stable("v1.0.0")
    db.mark_version_deprecated("v1.0.0")

    assert db.get_stable_version() is None
    backend.close()


def test_database_version_node_count():
    from memopt.control_plane.database import (
        ControlPlaneDB, SQLiteBackend)

    backend = SQLiteBackend(":memory:")
    db = ControlPlaneDB(backend=backend)
    db.init()

    db.record_boot_event("n1", "v1", "PASSED", 8, 10.0)
    db.record_boot_event("n2", "v1", "PASSED", 8, 10.0)
    db.record_boot_event("n3", "v2", "PASSED", 8, 10.0)

    assert db.get_version_node_count("v1") == 2
    assert db.get_version_node_count("v2") == 1
    assert db.get_version_node_count("v-missing") == 0
    backend.close()


def test_database_rollback_intent_retrievable():
    from memopt.control_plane.database import (
        ControlPlaneDB, SQLiteBackend)

    backend = SQLiteBackend(":memory:")
    db = ControlPlaneDB(backend=backend)
    db.init()

    db.record_rollback_intent(
        "n1", "v0.9.0", "regression")
    pending = db.get_pending_rollback("n1")
    assert pending is not None
    assert pending["target_version"] == "v0.9.0"
    assert pending["reason"] == "regression"
    assert pending["status"] == "pending"

    # Unknown node returns None
    assert db.get_pending_rollback("nobody") is None
    backend.close()


def test_database_migrations_all_run():
    """All 5 migrations run; expected tables present."""
    from memopt.control_plane.database import (
        ControlPlaneDB, SQLiteBackend)

    backend = SQLiteBackend(":memory:")
    db = ControlPlaneDB(backend=backend)
    db.init()

    rows = backend.fetchall(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' "
        "ORDER BY name")
    table_names = [r["name"] for r in rows]

    expected = [
        "boot_events", "events", "image_versions",
        "metrics", "migrations", "nodes", "pods",
        "rollback_intents",
    ]
    for table in expected:
        assert table in table_names, \
            f"Missing table: {table}"
    backend.close()


# ══════════════════════════════════════════════════════════════════════
#  Rollback affects /boot/config
# ══════════════════════════════════════════════════════════════════════


def test_boot_config_honors_pending_rollback():
    """
    If a node has a pending rollback, /boot/config serves the
    rollback target as the image_version.
    """
    from fastapi.testclient import TestClient
    import memopt.control_plane.server as cp_mod

    with TestClient(cp_mod.app) as client:
        # Register target version
        client.post(
            "/api/v1/images",
            json={
                "version": "v1.0.0-rollback-target",
                "git_commit": "cafebabe",
                "build_date": "2026-04-13T00:00:00Z",
                "cuda_version": "12.4",
                "sm_targets": "86;90;100",
                "image_url": "registry/memopt:v1.0.0-rt",
            },
            headers={"X-Memopt-API-Key": cp_mod._API_KEY})

        # Request rollback on a MAC-identified node (node_id = mac)
        mac = "bb:cc:dd:ee:ff:01"
        client.post(
            f"/api/v1/nodes/{mac}/rollback",
            json={
                "target_version": "v1.0.0-rollback-target",
                "reason": "verify override",
            },
            headers={"X-Memopt-API-Key": cp_mod._API_KEY})

        # Boot config should now serve the rollback target
        config = client.get(f"/boot/config/{mac}").json()
        assert config["image_version"] == \
            "v1.0.0-rollback-target"


# ══════════════════════════════════════════════════════════════════════
#  Shell script syntax
# ══════════════════════════════════════════════════════════════════════


def test_register_script_syntax():
    repo_root = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))
    result = subprocess.run(
        ["bash", "-n",
         os.path.join(repo_root, "scripts/register_image.sh")],
        capture_output=True, text=True)
    assert result.returncode == 0, \
        f"Syntax error: {result.stderr}"


def test_register_script_help():
    repo_root = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))
    result = subprocess.run(
        ["bash",
         os.path.join(repo_root, "scripts/register_image.sh"),
         "--help"],
        capture_output=True, text=True)
    assert result.returncode == 0
    assert "Usage" in result.stdout or "usage" in result.stdout


def test_register_script_requires_version():
    """Missing --version exits 1."""
    repo_root = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))
    result = subprocess.run(
        ["bash",
         os.path.join(repo_root, "scripts/register_image.sh")],
        capture_output=True, text=True)
    assert result.returncode == 1
    assert "missing" in (result.stdout + result.stderr).lower()
