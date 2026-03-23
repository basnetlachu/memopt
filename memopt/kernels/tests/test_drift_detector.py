"""
Tests for DriftDetector.
All pass without GPU, without signing key.
"""
import pytest
from pathlib import Path
from memopt.kernels.drift_detector import DriftDetector


@pytest.fixture
def detector(tmp_path):
    path = tmp_path / "drift_history.json"
    return DriftDetector(
        node_id="test-node",
        threshold_pct=5.0,
        baseline_n=5,
        rolling_n=3,
        history_path=path,
    )


def test_no_drift_without_enough_data(detector):
    detector.record(55.0)
    detector.record(54.0)
    assert not detector.is_drifted()
    assert detector.baseline_avg() is None
    assert detector.rolling_avg()  is None


def test_baseline_computed_after_n_measurements(detector):
    for pct in [55.0, 54.0, 55.5, 54.5, 55.0]:
        detector.record(pct)
    assert detector.baseline_avg() is not None
    assert abs(detector.baseline_avg() - 54.8) < 0.5


def test_no_drift_when_stable(detector):
    # Baseline
    for _ in range(5):
        detector.record(55.0)
    # Rolling — same as baseline
    for _ in range(3):
        detector.record(55.0)
    assert not detector.is_drifted()
    assert detector.drift_pct() < 1.0


def test_drift_detected_on_drop(detector):
    # Establish baseline at 55%
    for _ in range(5):
        detector.record(55.0)
    # Simulate hardware degradation — drop to 45%
    for _ in range(3):
        detector.record(45.0)
    assert detector.is_drifted()
    drift = detector.drift_pct()
    assert drift > 5.0   # more than threshold


def test_no_drift_on_small_variation(detector):
    for _ in range(5):
        detector.record(55.0)
    # 3% drop — below 5% threshold
    for _ in range(3):
        detector.record(53.4)
    assert not detector.is_drifted()


def test_reset_clears_history(detector):
    for _ in range(5):
        detector.record(55.0)
    assert detector.baseline_avg() is not None
    detector.reset()
    assert detector.baseline_avg() is None
    assert len(detector._measurements) == 0


def test_skips_invalid_bw_pct(detector):
    detector.record(None)
    detector.record(0.0)
    detector.record(-1.0)
    assert len(detector._measurements) == 0


def test_persists_and_loads(tmp_path):
    path = tmp_path / "drift.json"
    d1   = DriftDetector("n", 5.0, 5, 3, path)
    for pct in [55.0, 54.0, 55.5, 54.5, 55.0]:
        d1.record(pct)
    # Create new instance — should load from disk
    d2 = DriftDetector("n", 5.0, 5, 3, path)
    assert len(d2._measurements) == 5
    assert d2.baseline_avg() is not None


def test_stats_has_required_keys(detector):
    s = detector.stats()
    for k in ("node_id", "n_measurements", "baseline_avg",
              "rolling_avg", "drift_pct", "is_drifted",
              "threshold_pct", "baseline_n", "rolling_n"):
        assert k in s, f"Missing: {k}"


def test_drift_pct_none_before_baseline(detector):
    detector.record(55.0)
    assert detector.drift_pct() is None


def test_drift_pct_negative_on_improvement(detector):
    for _ in range(5):
        detector.record(50.0)
    # Hardware got faster (unlikely but possible after driver update)
    for _ in range(3):
        detector.record(60.0)
    pct = detector.drift_pct()
    assert pct is not None
    assert pct < 0   # negative = improved


def test_certify_daemon_starts_and_stops():
    from memopt.kernels.certify_daemon import CertifyDaemon
    import time, tempfile
    with tempfile.TemporaryDirectory() as d:
        status = Path(d) / "status.json"
        daemon = CertifyDaemon(
            interval_hours=999,
            on_startup=False,
            status_path=status,
        )
        daemon.start()
        time.sleep(0.05)
        assert daemon._thread.is_alive()
        daemon.stop()


def test_certify_daemon_has_drift_stats():
    from memopt.kernels.certify_daemon import CertifyDaemon
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        daemon = CertifyDaemon(
            interval_hours=999,
            on_startup=False,
            status_path=Path(d) / "s.json",
        )
        s = daemon.drift_stats()
        assert "is_drifted"     in s
        assert "baseline_avg"   in s
        assert "drift_pct"      in s
