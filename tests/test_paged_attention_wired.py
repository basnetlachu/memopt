"""
Tests verifying that PagedKVCache is wired into ContinuousBatchingEngine.

Inline model only — no external model names.
"""
import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

from memopt.serving import (
    ContinuousBatchingEngine,
    PagedKVCache,
    BatchingConfig,
)


class TinyLM(nn.Module):
    """Minimal decoder-style LM with a HF-style config."""

    def __init__(self, vocab_size: int = 256, hidden: int = 64):
        super().__init__()
        self.config = SimpleNamespace(
            num_hidden_layers=2,
            num_attention_heads=4,
            hidden_size=hidden,
        )
        self.embed = nn.Embedding(vocab_size, hidden)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        return self.fc(self.embed(input_ids))


# ── Import surface ────────────────────────────────────────────────────────────

def test_paged_kv_cache_importable_from_serving():
    """PagedKVCache must be importable from memopt.serving."""
    assert PagedKVCache is not None


# ── use_paged_attention=False (default) ───────────────────────────────────────

def test_engine_default_no_paged_cache():
    """By default, paged_kv_cache is None."""
    m = TinyLM()
    engine = ContinuousBatchingEngine(m, BatchingConfig(max_batch_size=2))
    assert engine.paged_kv_cache is None


# ── use_paged_attention=True ──────────────────────────────────────────────────

def test_engine_use_paged_attention_creates_cache():
    """use_paged_attention=True must create a PagedKVCache instance."""
    m = TinyLM()
    engine = ContinuousBatchingEngine(
        m, BatchingConfig(max_batch_size=2), use_paged_attention=True
    )
    assert isinstance(engine.paged_kv_cache, PagedKVCache)


def test_paged_cache_num_layers_from_model_config():
    """PagedKVCache.num_layers must match model.config.num_hidden_layers."""
    m = TinyLM()
    engine = ContinuousBatchingEngine(
        m, BatchingConfig(max_batch_size=2), use_paged_attention=True
    )
    assert engine.paged_kv_cache.num_layers == m.config.num_hidden_layers


def test_paged_cache_num_heads_from_model_config():
    """PagedKVCache.num_heads must match model.config.num_attention_heads."""
    m = TinyLM()
    engine = ContinuousBatchingEngine(
        m, BatchingConfig(max_batch_size=2), use_paged_attention=True
    )
    assert engine.paged_kv_cache.num_heads == m.config.num_attention_heads


def test_paged_cache_head_dim_correct():
    """head_dim = hidden_size // num_attention_heads."""
    m = TinyLM(hidden=64)  # 64 // 4 = 16
    engine = ContinuousBatchingEngine(
        m, BatchingConfig(max_batch_size=2), use_paged_attention=True
    )
    assert engine.paged_kv_cache.head_dim == 16


def test_paged_cache_num_blocks_scales_with_batch_size():
    """num_blocks = max_batch_size * 64."""
    m = TinyLM()
    engine = ContinuousBatchingEngine(
        m, BatchingConfig(max_batch_size=4), use_paged_attention=True
    )
    assert engine.paged_kv_cache.num_blocks == 4 * 64


def test_engine_without_model_config_does_not_crash():
    """Model without .config falls back to defaults — must not crash."""

    class BareModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(64, 64)

        def forward(self, x):
            return self.fc(x)

    m = BareModel()
    engine = ContinuousBatchingEngine(
        m, BatchingConfig(max_batch_size=1), use_paged_attention=True
    )
    assert isinstance(engine.paged_kv_cache, PagedKVCache)
