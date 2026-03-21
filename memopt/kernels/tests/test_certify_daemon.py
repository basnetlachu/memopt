"""
Tests for the continuous certification daemon.
All pass without GPU, without signing key.
"""
import time
import pytest
from memopt.kernels.certify_daemon import CertifyDaemon


def test_daemon_starts_and_stops():
    d = CertifyDaemon(interval_hours=999, on_startup=False)
    d.start()
    time.sleep(0.05)
    assert d._thread is not None
    assert d._thread.is_alive()
    d.stop()


def test_daemon_runs_certification_on_startup():
    d = CertifyDaemon(interval_hours=999, on_startup=True)
    d.start()
    # Give it time to complete one run
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if d.last_result() is not None:
            break
        time.sleep(0.1)
    d.stop()
    result = d.last_result()
    assert result is not None, "Daemon did not produce a result"
    assert "all_passed" in result


def test_daemon_writes_status_file(tmp_path):
    status_file = tmp_path / "node_status.json"
    d = CertifyDaemon(
        interval_hours=999,
        on_startup=True,
        status_path=status_file,
    )
    d.start()
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if status_file.exists():
            break
        time.sleep(0.1)
    d.stop()
    import json
    assert status_file.exists(), "Status file was not written"
    data = json.loads(status_file.read_text())
    assert "certified"  in data
    assert "checked_at" in data


def test_alert_callback_fires_on_failure():
    """
    Simulate a certification failure by monkey-patching run_certification.
    Alert callback must fire exactly once.
    """
    import unittest.mock as mock
    from dataclasses import asdict

    alerted = []
    d = CertifyDaemon(
        interval_hours=999,
        on_startup=False,
        alert_callback=lambda r: alerted.append(r),
    )

    def fake_cert(node_id=""):
        from memopt.kernels.certification import (
            SiliconCertificate, TestResult, ThroughputResult,
        )
        return SiliconCertificate(
            version="1.0", issued_at=time.time(), node_id=node_id,
            device_name="CPU", compute_cap="", driver_version="",
            cuda_version="",
            correctness_tests=[
                TestResult(
                    name="fake_test", dtype="float32", passed=False,
                    max_err=1.0, atol=0.001, rtol=0.001, note="forced fail"
                ),
            ],
            throughput_tests=[],
            all_passed=False,
            signature=None, signature_status="unsigned",
            signing_algorithm="none",
            certificate_hash="fake",
        )

    with mock.patch(
        "memopt.kernels.certification.run_certification",
        side_effect=fake_cert
    ):
        d._run_once()

    # Certification failed, alert should have fired
    assert not d.is_certified()
    assert len(alerted) == 1


def test_last_result_initially_none():
    d = CertifyDaemon(interval_hours=999, on_startup=False)
    assert d.last_result() is None
