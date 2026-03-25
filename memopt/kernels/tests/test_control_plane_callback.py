"""
Tests for make_control_plane_callback.
All pass without GPU, without API key, without control plane running.
"""
import pytest
from unittest.mock import MagicMock, patch


def test_make_control_plane_callback_returns_callable():
    from memopt.kernels.certify_daemon import make_control_plane_callback
    cb = make_control_plane_callback()
    assert callable(cb)


def test_callback_skips_non_drift_alerts():
    """On cert failure (no drift), no HTTP call and no re-synthesis."""
    from memopt.kernels.certify_daemon import make_control_plane_callback

    mock_gen = MagicMock()
    cb = make_control_plane_callback(
        control_plane_url="http://localhost:9999",
        jit_generator=mock_gen,
    )

    with patch("urllib.request.urlopen") as mock_urlopen:
        cb({"certified": False, "all_passed": False})

    mock_urlopen.assert_not_called()


def test_callback_posts_on_drift():
    """On drift detection, HTTP POST is sent to control plane."""
    from memopt.kernels.certify_daemon import make_control_plane_callback

    mock_gen = MagicMock()
    cb = make_control_plane_callback(
        control_plane_url="http://localhost:9999",
        jit_generator=mock_gen,
    )

    with patch("urllib.request.urlopen") as mock_urlopen:
        cb({"drift_detected": True, "drift_pct": 8.5})

    mock_urlopen.assert_called_once()
    call_args = mock_urlopen.call_args
    req = call_args[0][0]
    assert "/api/v1/nodes/" in req.full_url
    assert req.method == "POST"


def test_callback_never_raises_on_http_failure():
    """If control plane is unreachable, callback logs but does not raise."""
    from memopt.kernels.certify_daemon import make_control_plane_callback

    mock_gen = MagicMock()
    cb = make_control_plane_callback(
        control_plane_url="http://unreachable:9999",
        jit_generator=mock_gen,
    )

    with patch("urllib.request.urlopen", side_effect=ConnectionError("refused")):
        # Must not raise
        cb({"drift_detected": True, "drift_pct": 10.0})


def test_callback_skips_post_when_no_url():
    """When control_plane_url is empty, no HTTP call is made."""
    from memopt.kernels.certify_daemon import make_control_plane_callback

    mock_gen = MagicMock()
    cb = make_control_plane_callback(
        control_plane_url="",
        jit_generator=mock_gen,
    )

    with patch("urllib.request.urlopen") as mock_urlopen:
        cb({"drift_detected": True, "drift_pct": 5.0})

    mock_urlopen.assert_not_called()


def test_callback_still_runs_resynthesis_on_drift():
    """Even when control plane POST fails, re-synthesis still runs."""
    from memopt.kernels.certify_daemon import make_control_plane_callback

    mock_gen = MagicMock()
    mock_cache = MagicMock()
    mock_cache.get_metadata.return_value = {"speedup": 2.0}
    mock_gen.generate.return_value = MagicMock(speedup=1.9)

    cb = make_control_plane_callback(
        control_plane_url="http://localhost:9999",
        jit_generator=mock_gen,
        kernel_cache=mock_cache,
    )

    with patch("urllib.request.urlopen", side_effect=ConnectionError("refused")):
        cb({"drift_detected": True, "drift_pct": 8.0})

    # Re-synthesis should have been called for 3 active kernels
    assert mock_gen.generate.call_count == 3
