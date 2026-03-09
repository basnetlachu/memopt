"""
Tests for large-model rollback safety in optimization_executor.py.

Verifies safe_copy() strategy selection, _estimate_param_bytes(),
_get_available_cpu_ram_bytes(), and rollback_available on OptimizationResult.
"""
import pytest
import torch
import torch.nn as nn
from unittest.mock import patch

from memopt.phase3.optimization_executor import (
    _estimate_param_bytes,
    _get_available_cpu_ram_bytes,
    safe_copy,
    SNAPSHOT_RAM_SAFETY_MARGIN,
    OptimizationResult,
    _restore_snapshot,
)


# ── _estimate_param_bytes ─────────────────────────────────────────────────────

def test_estimate_param_bytes_linear():
    m = nn.Linear(64, 64)  # 64*64 + 64 = 4160 params, float32 → 4*4160 = 16640 bytes
    expected = sum(p.numel() * p.element_size() for p in m.parameters())
    assert _estimate_param_bytes(m) == expected
    assert _estimate_param_bytes(m) > 0


def test_estimate_param_bytes_zero_for_no_params():
    m = nn.Sequential()  # no parameters
    assert _estimate_param_bytes(m) == 0


def test_estimate_param_bytes_scales_with_size():
    small = nn.Linear(32, 32)
    large = nn.Linear(256, 256)
    assert _estimate_param_bytes(large) > _estimate_param_bytes(small)


# ── _get_available_cpu_ram_bytes ──────────────────────────────────────────────

def test_get_available_cpu_ram_positive():
    avail = _get_available_cpu_ram_bytes()
    assert avail > 0


def test_get_available_cpu_ram_at_least_1gb():
    """Expect at least 1 GB on any development machine."""
    avail = _get_available_cpu_ram_bytes()
    assert avail >= 1 * 1024 ** 3


def test_get_available_cpu_ram_fallback_on_no_psutil(monkeypatch):
    """When psutil is absent the function still returns a positive value."""
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("mocked absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    avail = _get_available_cpu_ram_bytes()
    assert avail > 0


# ── safe_copy strategy selection ──────────────────────────────────────────────

def test_safe_copy_state_dict_for_small_model():
    """Small model easily fits in RAM → state_dict strategy."""
    m = nn.Linear(64, 64)
    snap, strategy = safe_copy(m)
    assert strategy == "state_dict"
    assert snap is not None
    assert "weight" in snap
    assert "bias" in snap


def test_safe_copy_snapshot_is_cpu_tensors():
    """All snapshot tensors must be on CPU."""
    m = nn.Linear(128, 128)
    snap, strategy = safe_copy(m)
    assert strategy == "state_dict"
    for k, v in snap.items():
        assert v.device.type == "cpu", f"Tensor '{k}' not on CPU"


def test_safe_copy_in_place_when_ram_tight():
    """Simulate tight RAM: safe_copy falls back to in_place."""
    m = nn.Linear(64, 64)
    # Pretend only 100 bytes available — far less than model
    with patch(
        "memopt.phase3.optimization_executor._get_available_cpu_ram_bytes",
        return_value=100,
    ):
        snap, strategy = safe_copy(m)
    assert strategy in ("in_place", "failed")
    assert snap is None


def test_safe_copy_failed_when_no_ram():
    """Simulate zero available RAM → failed strategy."""
    m = nn.Linear(64, 64)
    with patch(
        "memopt.phase3.optimization_executor._get_available_cpu_ram_bytes",
        return_value=0,
    ):
        snap, strategy = safe_copy(m)
    assert strategy == "failed"
    assert snap is None


def test_safe_copy_roundtrip_restore():
    """state_dict snapshot can be restored exactly."""
    m = nn.Linear(32, 32)
    orig_weight = m.weight.data.clone()
    snap, strategy = safe_copy(m)
    assert strategy == "state_dict"

    # Corrupt weights
    m.weight.data.fill_(99.0)
    assert not torch.allclose(m.weight.data, orig_weight)

    # Restore
    _restore_snapshot(m, snap)
    assert torch.allclose(m.weight.data.cpu(), orig_weight.cpu())


# ── rollback_available on OptimizationResult ─────────────────────────────────

def test_optimization_result_rollback_available_default():
    """rollback_available defaults to True."""
    r = OptimizationResult(
        success=True,
        baseline_time_ms=10.0,
        optimized_time_ms=8.0,
        speedup_pct=20.0,
        regression_detected=False,
        error_message=None,
    )
    assert r.rollback_available is True


def test_optimization_result_rollback_available_false():
    """rollback_available=False can be set explicitly."""
    r = OptimizationResult(
        success=False,
        baseline_time_ms=10.0,
        optimized_time_ms=10.0,
        speedup_pct=0.0,
        regression_detected=False,
        error_message="no RAM",
        rollback_available=False,
    )
    assert r.rollback_available is False


# ── SNAPSHOT_RAM_SAFETY_MARGIN ────────────────────────────────────────────────

def test_margin_between_zero_and_one():
    assert 0 < SNAPSHOT_RAM_SAFETY_MARGIN < 1
