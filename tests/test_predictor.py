"""
Tests for the PredictivePredictor.

All tests are pure-Python — no GPU, no torch, no pynvml required.
Telemetry is injected directly into predictor.telemetry_history.
"""

import time

import pytest

from memopt.fleet.predictor import (
    GPUTelemetry,
    PredictivePredictor,
    PredictionSignal,
    _linear_trend,
    _clamp,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_telemetry(gpu_index: int = 0,
                    memory_pressure_pct: float = 0.40,
                    gpu_util_pct: float = 70.0,
                    memory_util_pct: float = 30.0,
                    power_watts: float = 200.0,
                    temperature_c: float = 65.0,
                    ts: float = None) -> GPUTelemetry:
    """Create a synthetic telemetry sample with sensible defaults."""
    return GPUTelemetry(
        timestamp=ts if ts is not None else time.time(),
        gpu_index=gpu_index,
        gpu_util_pct=gpu_util_pct,
        memory_used_mb=int(memory_pressure_pct * 80_000),
        memory_total_mb=80_000,
        memory_util_pct=memory_util_pct,
        power_watts=power_watts,
        temperature_c=temperature_c,
        sm_clock_mhz=1410,
        memory_clock_mhz=1215,
        memory_pressure_pct=memory_pressure_pct,
    )


def _inject(predictor: PredictivePredictor,
            samples: list,
            gpu_index: int = 0):
    """Inject a list of GPUTelemetry samples directly into the predictor."""
    for t in samples:
        predictor._append_telemetry(gpu_index, t)


# ── Test 1: Low memory pressure → low risk ────────────────────────────────

def test_low_memory_pressure_gives_low_risk():
    """
    Stable 40% memory pressure (well below the 75% warning threshold)
    must produce a risk score below 0.3 and no migration trigger.
    """
    p = PredictivePredictor()
    now = time.time()
    samples = [
        _make_telemetry(memory_pressure_pct=0.40, ts=now - (9 - i) * 5)
        for i in range(10)
    ]
    _inject(p, samples)

    signal = p.predict(gpu_index=0)

    assert signal is not None
    assert signal.migration_risk_score < 0.3, (
        f"Expected risk < 0.3, got {signal.migration_risk_score:.3f}"
    )
    assert signal.should_migrate_now is False


# ── Test 2: High rising memory → migration triggered ──────────────────────

def test_high_rising_memory_triggers_migration():
    """
    Memory pressure rising from 70% to 89% over the observation window
    must produce should_migrate_now = True (risk >= 0.70).
    """
    p = PredictivePredictor()
    now = time.time()
    # 10 samples rising linearly from 0.70 → 0.89
    pressures = [0.70 + i * (0.19 / 9) for i in range(10)]
    samples = [
        _make_telemetry(memory_pressure_pct=pressures[i], ts=now - (9 - i) * 5)
        for i in range(10)
    ]
    _inject(p, samples)

    signal = p.predict(gpu_index=0)

    assert signal is not None
    assert signal.should_migrate_now is True, (
        f"Expected migration trigger, risk={signal.migration_risk_score:.3f}"
    )
    assert signal.migration_risk_score >= 0.70


# ── Test 3: Stable high memory (no trend) → no migration ─────────────────

def test_stable_high_memory_no_migration():
    """
    Memory stable at 78% (above warning, below critical, flat trend)
    must NOT trigger migration — no upward trend to amplify the danger.
    """
    p = PredictivePredictor()
    now = time.time()
    samples = [
        _make_telemetry(memory_pressure_pct=0.78, ts=now - (9 - i) * 5)
        for i in range(10)
    ]
    _inject(p, samples)

    signal = p.predict(gpu_index=0)

    assert signal is not None
    assert signal.should_migrate_now is False, (
        f"Stable 78% should not trigger migration, risk={signal.migration_risk_score:.3f}"
    )


# ── Test 4: Linear trend helper ───────────────────────────────────────────

def test_linear_trend_calculation():
    """
    _linear_trend() must return ~0 for a flat series,
    a positive slope for a rising series, and negative for falling.
    """
    flat    = [0.5] * 10
    rising  = [0.1 * i for i in range(10)]   # 0.0, 0.1, ..., 0.9
    falling = [0.9 - 0.1 * i for i in range(10)]  # 0.9, 0.8, ..., 0.0

    trend_flat    = _linear_trend(flat)
    trend_rising  = _linear_trend(rising)
    trend_falling = _linear_trend(falling)

    assert abs(trend_flat) < 1e-9, f"Flat trend should be ~0, got {trend_flat}"
    assert trend_rising  > 0, f"Rising trend should be positive, got {trend_rising}"
    assert trend_falling < 0, f"Falling trend should be negative, got {trend_falling}"

    # Edge cases
    assert _linear_trend([]) == 0.0
    assert _linear_trend([0.5]) == 0.0


# ── Test 5: Risk score is always in [0, 1] ────────────────────────────────

def test_risk_score_between_0_and_1():
    """
    predict() must always return a risk_score in [0.0, 1.0]
    regardless of extreme input values.
    """
    p = PredictivePredictor()
    now = time.time()

    extreme_cases = [
        # Very low everything
        {"memory_pressure_pct": 0.01, "memory_util_pct": 1.0, "temperature_c": 20.0},
        # Maximum memory pressure
        {"memory_pressure_pct": 0.99, "memory_util_pct": 99.0, "temperature_c": 90.0},
        # Right at warning thresholds
        {"memory_pressure_pct": 0.75, "memory_util_pct": 80.0, "temperature_c": 80.0},
        # Right at critical thresholds
        {"memory_pressure_pct": 0.88, "memory_util_pct": 92.0, "temperature_c": 83.0},
    ]

    for kwargs in extreme_cases:
        samples = [
            _make_telemetry(ts=now - (4 - i) * 5, **kwargs)
            for i in range(5)
        ]
        _inject(p, samples, gpu_index=0)
        # Clear history between cases
        p.telemetry_history[0].clear()
        _inject(p, samples, gpu_index=0)

        signal = p.predict(gpu_index=0)
        assert signal is not None
        assert 0.0 <= signal.migration_risk_score <= 1.0, (
            f"Risk {signal.migration_risk_score:.4f} out of [0, 1] for {kwargs}"
        )

    # Also verify the _clamp helper directly
    assert _clamp(-5.0) == 0.0
    assert _clamp(5.0)  == 1.0
    assert _clamp(0.5)  == pytest.approx(0.5)
