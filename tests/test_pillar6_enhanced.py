"""
Tests for Pillar 6 enhancements:
  - Extended certification ops (7 total)
  - SLACertificate generation
  - GPU_SPECS_AUDIT honest provenance
  - Node healthy endpoint for LB health checks
  - /hardware/specs endpoint

All tests run without GPU. Attention test skips on CPU.
"""
import json
import os
import tempfile
import time
from unittest import mock

import pytest


# ── Extended certification ──────────────────────────────────────────


def test_certification_has_seven_ops():
    from memopt.kernels.certification import run_certification

    cert = run_certification("test_node")

    op_names = [r.name for r in cert.correctness_tests]

    expected_ops = [
        "rope", "layer_norm_residual", "scaled_softmax",
        "matmul", "embedding_lookup", "attention", "layer_norm",
    ]

    for op in expected_ops:
        assert op in op_names, f"Missing op in certification: {op}"


def test_certification_all_pass_on_cpu():
    from memopt.kernels.certification import run_certification

    cert = run_certification("test_node")

    for r in cert.correctness_tests:
        if r.name == "attention" and "Skipped" in (r.note or ""):
            continue  # GPU-only skip
        assert r.passed, f"Op {r.name} ({r.dtype}) failed on CPU: {r.note}"


def test_certification_has_certificate_hash():
    from memopt.kernels.certification import run_certification

    cert = run_certification("test_node")

    assert cert.certificate_hash is not None
    assert len(cert.certificate_hash) == 64
    assert all(c in "0123456789abcdef" for c in cert.certificate_hash)


def test_certification_matmul_passes():
    from memopt.kernels.certification import run_certification

    cert = run_certification("test_node")
    matmul_results = [r for r in cert.correctness_tests if r.name == "matmul"]
    assert len(matmul_results) >= 1
    assert matmul_results[0].passed


def test_certification_embedding_passes():
    from memopt.kernels.certification import run_certification

    cert = run_certification("test_node")
    embed_results = [r for r in cert.correctness_tests
                     if r.name == "embedding_lookup"]
    assert len(embed_results) >= 1
    assert embed_results[0].passed


def test_certification_layer_norm_standalone_passes():
    from memopt.kernels.certification import run_certification

    cert = run_certification("test_node")
    ln_results = [r for r in cert.correctness_tests if r.name == "layer_norm"]
    assert len(ln_results) >= 1
    assert ln_results[0].passed


# ── SLACertificate ──────────────────────────────────────────────────


def test_sla_certificate_empty_on_no_history():
    from memopt.kernels.certification import SLACertificate

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cert_history.json")
        old = os.environ.get("MEMOPT_CERT_HISTORY_PATH")
        os.environ["MEMOPT_CERT_HISTORY_PATH"] = path
        try:
            sla = SLACertificate(node_id="test_node", period_days=30)
            cert = sla.generate()

            assert cert["total_runs"] == 0
            assert cert["hardware_correctness_pct"] is None
            assert len(cert["honest_notes"]) > 0
        finally:
            if old is not None:
                os.environ["MEMOPT_CERT_HISTORY_PATH"] = old
            else:
                os.environ.pop("MEMOPT_CERT_HISTORY_PATH", None)


def test_sla_certificate_from_history():
    from memopt.kernels.certification import SLACertificate

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cert_history.json")
        old = os.environ.get("MEMOPT_CERT_HISTORY_PATH")
        os.environ["MEMOPT_CERT_HISTORY_PATH"] = path
        try:
            # Write fake history
            now = time.time()
            history = [
                {
                    "node_id": "test_node",
                    "timestamp": now - i * 3600,
                    "all_passed": True,
                    "bandwidth_pct_of_peak": 54.0,
                    "drift_detected": False,
                    "cert_hash": f"hash{i:06d}",
                }
                for i in range(10)
            ]
            with open(path, "w") as f:
                json.dump(history, f)

            sla = SLACertificate(node_id="test_node", period_days=30)
            cert = sla.generate()

            assert cert["total_runs"] == 10
            assert cert["passed_runs"] == 10
            assert cert["hardware_correctness_pct"] == 100.0
            assert cert["avg_bandwidth_pct_of_peak"] == pytest.approx(54.0)
            assert cert["certificate_hash"] is not None
        finally:
            if old is not None:
                os.environ["MEMOPT_CERT_HISTORY_PATH"] = old
            else:
                os.environ.pop("MEMOPT_CERT_HISTORY_PATH", None)


def test_sla_certificate_honest_notes_present():
    from memopt.kernels.certification import SLACertificate

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cert_history.json")
        old = os.environ.get("MEMOPT_CERT_HISTORY_PATH")
        os.environ["MEMOPT_CERT_HISTORY_PATH"] = path
        try:
            sla = SLACertificate("node1")
            cert = sla.generate()
            notes = cert.get("honest_notes", [])
            assert len(notes) > 0
        finally:
            if old is not None:
                os.environ["MEMOPT_CERT_HISTORY_PATH"] = old
            else:
                os.environ.pop("MEMOPT_CERT_HISTORY_PATH", None)


def test_sla_certificate_unsigned_without_key():
    from memopt.kernels.certification import SLACertificate

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cert_history.json")
        now = time.time()
        history = [{
            "node_id": "n1", "timestamp": now,
            "all_passed": True,
            "bandwidth_pct_of_peak": 50.0,
            "drift_detected": False,
            "cert_hash": "abc",
        }]
        with open(path, "w") as f:
            json.dump(history, f)

        old_key = os.environ.get("MEMOPT_SIGNING_KEY")
        old_path = os.environ.get("MEMOPT_CERT_HISTORY_PATH")
        try:
            os.environ.pop("MEMOPT_SIGNING_KEY", None)
            os.environ["MEMOPT_CERT_HISTORY_PATH"] = path

            sla = SLACertificate("n1")
            cert = sla.generate()
            assert cert["signature_status"] == "unsigned"
        finally:
            if old_key is not None:
                os.environ["MEMOPT_SIGNING_KEY"] = old_key
            if old_path is not None:
                os.environ["MEMOPT_CERT_HISTORY_PATH"] = old_path
            else:
                os.environ.pop("MEMOPT_CERT_HISTORY_PATH", None)


def test_append_run_creates_history():
    from memopt.kernels.certification import SLACertificate, run_certification

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cert_history.json")

        cert = run_certification("test_node")

        SLACertificate.append_run(
            history_path=path,
            node_id="test_node",
            cert=cert,
            drift_detected=False)

        assert os.path.exists(path)
        with open(path) as f:
            history = json.load(f)
        assert len(history) == 1
        assert history[0]["node_id"] == "test_node"
        assert "timestamp" in history[0]
        assert "all_passed" in history[0]


# ── GPU_SPECS_AUDIT ─────────────────────────────────────────────────


def test_gpu_specs_audit_entry_exists():
    from memopt.profiler.hardware_counters import GPU_SPECS_AUDIT

    assert len(GPU_SPECS_AUDIT) > 0

    for name, audit in GPU_SPECS_AUDIT.items():
        assert "datasheet_bw_gbps" in audit
        assert "measured_bw_gbps" in audit
        assert "notes" in audit


def test_get_spec_with_audit_known_device():
    from memopt.profiler.hardware_counters import get_spec_with_audit, GPU_SPECS

    if not GPU_SPECS:
        pytest.skip("No GPU specs defined")

    name = next(iter(GPU_SPECS))
    result = get_spec_with_audit(name)
    assert result["found"] is True
    assert "measured" in result
    assert "source" in result


def test_get_spec_with_audit_unknown_device():
    from memopt.profiler.hardware_counters import get_spec_with_audit

    result = get_spec_with_audit("NONEXISTENT GPU 9999")
    assert result["found"] is False
    assert "note" in result


def test_gpu_specs_all_have_source_field():
    """Every GPUSpec should document its source."""
    from memopt.profiler.hardware_counters import GPU_SPECS

    for name, spec in GPU_SPECS.items():
        # source defaults to "" in older entries
        assert hasattr(spec, "source"), f"{name} missing source field"
        assert hasattr(spec, "measured"), f"{name} missing measured field"


# ── Hardware specs endpoint ─────────────────────────────────────────


def test_hardware_specs_endpoint():
    from fastapi.testclient import TestClient
    import memopt.api.server as api_srv

    old = os.environ.get("MEMOPT_API_KEY")
    os.environ["MEMOPT_API_KEY"] = "test_key"
    api_srv._API_KEY = "test_key"
    try:
        client = TestClient(api_srv.app)
        r = client.get(
            "/hardware/specs",
            headers={"x-memopt-api-key": "test_key"})
        assert r.status_code == 200
        data = r.json()

        assert "specs" in data
        assert "total" in data
        assert "measured_count" in data
        assert "note" in data

        measured = sum(
            1 for v in data["specs"].values() if v.get("measured"))
        assert data["measured_count"] == measured
    finally:
        if old is not None:
            os.environ["MEMOPT_API_KEY"] = old
        else:
            os.environ.pop("MEMOPT_API_KEY", None)


# ── Node healthy endpoint ──────────────────────────────────────────


def test_node_healthy_unknown_returns_404():
    from fastapi.testclient import TestClient
    from memopt.control_plane.server import app
    client = TestClient(app)

    r = client.get("/api/v1/nodes/nonexistent_node/healthy")
    assert r.status_code == 404
    data = r.json()
    assert data["healthy"] is False
