"""
Tests for the StaticGraphAnalyzer (pre-flight engine).

All tests are pure-Python — no GPU, no torch.cuda required.
torch.nn CPU tensors are used for model structure only.
"""

import pytest
import torch.nn as nn

from memopt.profiler.graph_analyzer import (
    StaticGraphAnalyzer,
    Bottleneck,
    LayerType,
)


# ── Test 1: Linear layers detected ───────────────────────────────────────────

def test_analyzer_detects_linear_layers():
    """
    A sequential model with two Linear layers must produce at least 2
    LayerAnalysis entries, a nonzero total_params, and finish in under 5 s.
    """
    model = nn.Sequential(nn.Linear(512, 512), nn.Linear(512, 256))
    analyzer = StaticGraphAnalyzer()
    result = analyzer.analyze(model, gpu_name="A100", ridge_point=156.0)

    assert len(result.layer_analyses) >= 2
    assert result.total_params > 0
    assert result.analysis_time_ms < 5000


# ── Test 2: Memory-bound bottleneck ──────────────────────────────────────────

def test_memory_bound_model_gets_correct_bottleneck():
    """
    A wide Linear(4096, 4096) at batch=1 seq=512 is always memory-bound
    (arithmetic intensity well below A100 ridge of 156 FLOPS/byte).
    """
    model = nn.Linear(4096, 4096)
    analyzer = StaticGraphAnalyzer()
    result = analyzer.analyze(model, gpu_name="A100", ridge_point=156.0)

    assert result.overall_bottleneck == Bottleneck.MEMORY_BANDWIDTH


# ── Test 3: Preflight config has required fields ──────────────────────────────

def test_preflight_config_has_required_fields():
    """
    preflight_config must always contain the three minimum keys needed
    by AutoMigrationEngine: recommended_backend, flash_attention_version, dtype.
    """
    model = nn.Linear(512, 512)
    analyzer = StaticGraphAnalyzer()
    result = analyzer.analyze(model, gpu_name="H100", ridge_point=590.0)
    config = result.preflight_config

    assert "recommended_backend" in config
    assert "flash_attention_version" in config
    assert "dtype" in config


# ── Test 4: H100 → flash_attention_3 ─────────────────────────────────────────

def test_h100_gets_flash_attention_3():
    """
    When gpu_name contains 'H100', preflight_config must recommend
    flash_attention_3 (Hopper-specific).
    """
    model = nn.Linear(512, 512)
    analyzer = StaticGraphAnalyzer()
    result = analyzer.analyze(model, gpu_name="H100 SXM", ridge_point=590.0)

    assert result.preflight_config["flash_attention_version"] == "flash_attention_3"


# ── Test 5: A100 → flash_attention_2 ─────────────────────────────────────────

def test_a100_gets_flash_attention_2():
    """
    When gpu_name contains 'A100', preflight_config must recommend
    flash_attention_2 (Ampere).
    """
    model = nn.Linear(512, 512)
    analyzer = StaticGraphAnalyzer()
    result = analyzer.analyze(model, gpu_name="A100-SXM4", ridge_point=156.0)

    assert result.preflight_config["flash_attention_version"] == "flash_attention_2"


# ── Test 6: format_report contains key sections ───────────────────────────────

def test_format_report_contains_key_sections():
    """
    format_report() output must contain all three named sections so the
    CLI and API can present actionable output.
    """
    model = nn.Linear(512, 512)
    analyzer = StaticGraphAnalyzer()
    result = analyzer.analyze(model, gpu_name="A100", ridge_point=156.0)
    report = analyzer.format_report(result)

    assert "PRE-FLIGHT GRAPH ANALYSIS" in report
    assert "ESTIMATED SPEEDUP" in report
    assert "PRE-FLIGHT CONFIG" in report
