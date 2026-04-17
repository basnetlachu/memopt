"""
Pillar 6 — Silicon Certification Suite tests.
All pass without CUDA, without ANTHROPIC_API_KEY, without triton.
Runs on CPU (CI / local dev) and on GPU (A100 / H100).
"""
import json
import os
import tempfile
import time
import pytest

from memopt.kernels.certification import (
    SiliconCertificate,
    TestResult,
    ThroughputResult,
    _get_hardware_info,
    _theoretical_peak_gb_s,
    _run_correctness_tests,
    _run_throughput_tests,
    run_certification,
    _save_certificate,
    verify_certificate,
    _canonical_json,
    _sign,
    _TOLERANCES,
)


# ── _get_hardware_info ────────────────────────────────────────────────────────

def test_hardware_info_returns_dict():
    info = _get_hardware_info()
    assert isinstance(info, dict)
    for key in ("device_name", "compute_cap", "driver_version", "cuda_version"):
        assert key in info
    assert isinstance(info["device_name"], str)
    assert len(info["device_name"]) > 0


def test_hardware_info_device_name_not_empty():
    info = _get_hardware_info()
    assert info["device_name"]   # "CPU" or real GPU name


# ── _theoretical_peak_gb_s ────────────────────────────────────────────────────

def test_theoretical_peak_returns_float():
    peak = _theoretical_peak_gb_s("CPU")
    assert isinstance(peak, float)
    assert peak >= 0.0


def test_theoretical_peak_known_gpu():
    """H100 is in GPU_SPECS — should not use probe fallback."""
    try:
        from memopt.profiler.hardware_counters import GPU_SPECS
        assert "H100" in GPU_SPECS
        peak = _theoretical_peak_gb_s("H100")
        assert peak > 0.0
    except ImportError:
        pytest.skip("hardware_counters not available")


# ── _TOLERANCES ───────────────────────────────────────────────────────────────

def test_tolerances_cover_required_dtypes():
    for dtype in ("float16", "bfloat16", "float32", "float64"):
        assert dtype in _TOLERANCES
        atol, rtol = _TOLERANCES[dtype]
        assert atol > 0
        assert rtol > 0


# ── run_certification (CPU path) ──────────────────────────────────────────────

def test_run_certification_returns_silicon_certificate():
    cert = run_certification(node_id="test-node")
    assert isinstance(cert, SiliconCertificate)


def test_run_certification_never_raises():
    """run_certification must not raise regardless of environment."""
    try:
        run_certification()
    except Exception as e:
        pytest.fail(f"run_certification raised: {e}")


def test_certification_has_correctness_tests():
    cert = run_certification()
    assert isinstance(cert.correctness_tests, list)
    assert len(cert.correctness_tests) > 0
    for t in cert.correctness_tests:
        assert isinstance(t, TestResult)
        assert t.name in (
            "rope", "layer_norm_residual", "scaled_softmax",
            "matmul", "embedding_lookup", "attention", "layer_norm")


def test_certification_has_throughput_tests():
    cert = run_certification()
    assert isinstance(cert.throughput_tests, list)
    assert len(cert.throughput_tests) > 0
    for t in cert.throughput_tests:
        assert isinstance(t, ThroughputResult)


def test_certification_all_passed_is_bool():
    cert = run_certification()
    assert isinstance(cert.all_passed, bool)


def test_certification_certificate_hash_nonempty():
    cert = run_certification()
    assert isinstance(cert.certificate_hash, str)
    assert len(cert.certificate_hash) == 64   # SHA-256 hex


# ── signing ───────────────────────────────────────────────────────────────────

def test_unsigned_when_no_key(monkeypatch):
    monkeypatch.delenv("MEMOPT_SIGNING_KEY", raising=False)
    # Reset module-level constant
    import memopt.kernels.certification as mod
    old_key = mod._SIGNING_KEY
    mod._SIGNING_KEY = ""
    try:
        cert = run_certification()
        assert cert.signature_status == "unsigned"
        assert cert.signature is None
    finally:
        mod._SIGNING_KEY = old_key


def test_signed_when_key_set(monkeypatch):
    import memopt.kernels.certification as mod
    old_key = mod._SIGNING_KEY
    mod._SIGNING_KEY = "test-secret-key"
    try:
        cert = run_certification(node_id="signed-node")
        assert cert.signature_status == "signed"
        assert cert.signing_algorithm == "HMAC-SHA256"
        assert cert.signature is not None
        assert len(cert.signature) == 64
    finally:
        mod._SIGNING_KEY = old_key


# ── _save_certificate + verify_certificate ────────────────────────────────────

def test_save_certificate_creates_file():
    cert = run_certification()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _save_certificate(cert, out_dir=tmpdir)
        assert path != ""
        assert os.path.isfile(path)


def test_save_certificate_valid_json():
    cert = run_certification()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _save_certificate(cert, out_dir=tmpdir)
        with open(path) as f:
            doc = json.load(f)
        assert "version" in doc
        assert "correctness_tests" in doc
        assert "throughput_tests" in doc


def test_verify_certificate_valid_signature():
    import memopt.kernels.certification as mod
    old_key = mod._SIGNING_KEY
    key = "verify-test-key-42"
    mod._SIGNING_KEY = key
    try:
        cert = run_certification()
        from dataclasses import asdict
        cert_dict = asdict(cert)
        assert verify_certificate(cert_dict, key)
    finally:
        mod._SIGNING_KEY = old_key


def test_verify_certificate_tamper_detected():
    import memopt.kernels.certification as mod
    old_key = mod._SIGNING_KEY
    key = "tamper-test-key"
    mod._SIGNING_KEY = key
    try:
        cert = run_certification()
        from dataclasses import asdict
        cert_dict = asdict(cert)
        # Tamper with a field
        cert_dict["node_id"] = "attacker-node"
        assert not verify_certificate(cert_dict, key)
    finally:
        mod._SIGNING_KEY = old_key


def test_verify_unsigned_returns_false():
    import memopt.kernels.certification as mod
    old_key = mod._SIGNING_KEY
    mod._SIGNING_KEY = ""
    try:
        cert = run_certification()
        from dataclasses import asdict
        cert_dict = asdict(cert)
        assert not verify_certificate(cert_dict, "any-key")
    finally:
        mod._SIGNING_KEY = old_key
