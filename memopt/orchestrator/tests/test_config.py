"""Tests for OrchestratorConfig (orchestrator v1 Commit 4; design
§2.3.6, S0.5/TV5). 8 tests per §3.1.6."""
from __future__ import annotations

import dataclasses
import os

import pytest

from memopt.orchestrator.config import OrchestratorConfig


@pytest.fixture
def _env_clean(monkeypatch):
    for k in (
        "MEMOPT_ORCH_CYCLE_MS",
        "MEMOPT_ORCH_QUEUE_CAP",
        "MEMOPT_ORCH_MAX_TRANSITIONS",
        "MEMOPT_EVICT_HIGH",
        "MEMOPT_EVICT_LOW",
    ):
        monkeypatch.delenv(k, raising=False)
    yield


def test_default_config_has_documented_values(_env_clean):
    cfg = OrchestratorConfig.from_env()
    assert cfg.cycle_period_ms == 50
    assert cfg.event_queue_capacity == 16384
    assert cfg.predictor_max_transitions == 100_000
    assert cfg.predictor_min_confidence == 0.3
    assert cfg.built_in_policy is True
    assert cfg.lru_high_watermark == 0.90
    assert cfg.lru_low_watermark == 0.75


def test_env_overrides_cycle_period(_env_clean, monkeypatch):
    monkeypatch.setenv("MEMOPT_ORCH_CYCLE_MS", "37")
    cfg = OrchestratorConfig.from_env()
    assert cfg.cycle_period_ms == 37


def test_env_overrides_queue_capacity(_env_clean, monkeypatch):
    monkeypatch.setenv("MEMOPT_ORCH_QUEUE_CAP", "256")
    cfg = OrchestratorConfig.from_env()
    assert cfg.event_queue_capacity == 256


def test_env_overrides_max_transitions(_env_clean, monkeypatch):
    monkeypatch.setenv("MEMOPT_ORCH_MAX_TRANSITIONS", "9999")
    cfg = OrchestratorConfig.from_env()
    assert cfg.predictor_max_transitions == 9999


def test_env_overrides_evict_thresholds_shared_with_vmm(
    _env_clean, monkeypatch
):
    # Set shared env vars; both TierManager and OrchestratorConfig must
    # read 0.85 (S0.5 / TV5 contract).
    monkeypatch.setenv("MEMOPT_EVICT_HIGH", "0.85")
    monkeypatch.setenv("MEMOPT_EVICT_LOW", "0.65")
    cfg = OrchestratorConfig.from_env()
    assert cfg.lru_high_watermark == 0.85
    assert cfg.lru_low_watermark == 0.65

    # TierManager parity: read the same env vars directly the way the VMM
    # does (tier_manager.py:72-75) — no shared state, just same env source.
    high_via_vmm = float(os.environ["MEMOPT_EVICT_HIGH"])
    low_via_vmm = float(os.environ["MEMOPT_EVICT_LOW"])
    assert cfg.lru_high_watermark == high_via_vmm
    assert cfg.lru_low_watermark == low_via_vmm


def test_env_invalid_falls_back_to_default(_env_clean, monkeypatch):
    # Garbage values fall back, no exception raised.
    monkeypatch.setenv("MEMOPT_ORCH_CYCLE_MS", "notanint")
    monkeypatch.setenv("MEMOPT_EVICT_HIGH", "0.5")  # low >= high → invalid
    monkeypatch.setenv("MEMOPT_EVICT_LOW", "0.6")
    cfg = OrchestratorConfig.from_env()
    assert cfg.cycle_period_ms == 50
    # invalid relationship reverts BOTH watermarks to defaults.
    assert cfg.lru_high_watermark == 0.90
    assert cfg.lru_low_watermark == 0.75


def test_config_is_frozen_dataclass():
    cfg = OrchestratorConfig()
    assert dataclasses.is_dataclass(cfg)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.cycle_period_ms = 1  # type: ignore[misc]


def test_built_in_policy_can_be_disabled():
    cfg = OrchestratorConfig(built_in_policy=False)
    assert cfg.built_in_policy is False
