"""
Drift detector hardening tests — baseline, drift detection, certify daemon.
"""
import os
import tempfile
import time
from pathlib import Path

import pytest

from memopt.kernels.drift_detector import DriftDetector


def test_drift_detector_baseline_none_when_insufficient():
    """baseline_avg() returns None when < BASELINE_N measurements."""
    with tempfile.TemporaryDirectory() as td:
        d = DriftDetector(
            node_id="test",
            history_path=Path(td) / "drift.json")
        # Fresh detector has 0 measurements
        assert d.baseline_avg() is None


def test_drift_detector_not_drifted_with_no_data():
    """is_drifted() returns False when insufficient data."""
    with tempfile.TemporaryDirectory() as td:
        d = DriftDetector(
            node_id="test",
            history_path=Path(td) / "drift.json")
        assert d.is_drifted() is False


def test_drift_detector_records_measurements():
    """record() should accumulate measurements."""
    with tempfile.TemporaryDirectory() as td:
        d = DriftDetector(
            node_id="test",
            history_path=Path(td) / "drift.json")
        for i in range(10):
            d.record(80.0 + i)
        assert len(d._measurements) >= 10


def test_drift_detector_reset_clears_all():
    """reset() should clear all measurements."""
    with tempfile.TemporaryDirectory() as td:
        d = DriftDetector(
            node_id="test",
            history_path=Path(td) / "drift.json")
        for i in range(10):
            d.record(80.0)
        d.reset()
        assert len(d._measurements) == 0


def test_drift_detected_on_large_drop():
    """Drift should be detected when rolling avg drops significantly."""
    with tempfile.TemporaryDirectory() as td:
        d = DriftDetector(
            node_id="test",
            history_path=Path(td) / "drift.json")
        # Baseline: 7 measurements at 80%
        for _ in range(7):
            d.record(80.0)
        # Rolling: 3 measurements at 60% (25% drop > 5% threshold)
        for _ in range(3):
            d.record(60.0)
        assert d.is_drifted() is True


def test_certify_daemon_instantiates():
    """CertifyDaemon can be created without crashing."""
    from memopt.kernels.certify_daemon import CertifyDaemon
    daemon = CertifyDaemon(node_id="test-node")
    assert daemon is not None
    assert daemon.is_certified() is False


def test_certify_now_runs_immediately():
    """certify_now() should trigger _run_once() synchronously."""
    from unittest.mock import patch, MagicMock
    from memopt.kernels.certify_daemon import CertifyDaemon

    daemon = CertifyDaemon(node_id="test-node")

    with patch.object(daemon, '_run_once') as mock_run:
        daemon.certify_now("test-trigger")
        mock_run.assert_called_once()
