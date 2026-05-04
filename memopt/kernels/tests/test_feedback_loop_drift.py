"""
Drift re-synthesis tests — split from main repo's test_feedback_loop.py
when the trust/compliance pillars were extracted to memopt-trust.

These cover make_drift_resynthesis_callback and CertifyDaemon's default
alert wiring. They pass without GPU, without ANTHROPIC_API_KEY.
"""
from unittest.mock import MagicMock


def test_make_drift_resynthesis_callback_returns_callable():
    from memopt.kernels.certify_daemon import (
        make_drift_resynthesis_callback
    )
    cb = make_drift_resynthesis_callback()
    assert callable(cb)


def test_callback_skips_non_drift_alerts():
    """Callback does nothing when alert is cert failure, not drift."""
    from memopt.kernels.certify_daemon import (
        make_drift_resynthesis_callback
    )

    mock_gen = MagicMock()
    cb = make_drift_resynthesis_callback(jit_generator=mock_gen)

    cb({"chain_valid": False, "certified": False})

    mock_gen.generate.assert_not_called()


def test_callback_triggers_resynthesis_on_drift():
    """When drift_detected=True, generate() is called for each kernel."""
    from memopt.kernels.certify_daemon import (
        make_drift_resynthesis_callback
    )

    mock_gen   = MagicMock()
    mock_cache = MagicMock()
    mock_cache.get_metadata.return_value = {
        "speedup": 2.1,
        "is_memory_bound": True,
    }

    mock_result        = MagicMock()
    mock_result.speedup = 2.0
    mock_gen.generate.return_value = mock_result

    cb = make_drift_resynthesis_callback(
        jit_generator = mock_gen,
        kernel_cache  = mock_cache,
    )

    cb({
        "drift_detected": True,
        "drift_pct":      8.5,
    })

    assert mock_gen.generate.call_count == 3


def test_callback_never_raises():
    """Callback must never raise even if everything fails."""
    from memopt.kernels.certify_daemon import (
        make_drift_resynthesis_callback
    )

    mock_gen = MagicMock()
    mock_gen.generate.side_effect = RuntimeError("API down")

    cb = make_drift_resynthesis_callback(jit_generator=mock_gen)

    cb({"drift_detected": True, "drift_pct": 10.0})


def test_certify_daemon_has_default_callback():
    """CertifyDaemon without alert_callback has drift re-synthesis."""
    import tempfile
    from pathlib import Path
    from memopt.kernels.certify_daemon import CertifyDaemon

    with tempfile.TemporaryDirectory() as d:
        daemon = CertifyDaemon(
            interval_hours = 999,
            on_startup     = False,
            status_path    = Path(d) / "status.json",
        )
        assert daemon._alert is not None
        assert callable(daemon._alert)


def test_certify_daemon_custom_callback_not_overridden():
    """Custom alert_callback is never replaced by default."""
    import tempfile
    from pathlib import Path
    from memopt.kernels.certify_daemon import CertifyDaemon

    custom_cb = MagicMock()
    with tempfile.TemporaryDirectory() as d:
        daemon = CertifyDaemon(
            interval_hours = 999,
            on_startup     = False,
            status_path    = Path(d) / "status.json",
            alert_callback = custom_cb,
        )
        assert daemon._alert is custom_cb
