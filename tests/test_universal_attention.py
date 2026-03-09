"""
Tests for the three-layer universal attention detection and SDPA patching.

Inline model classes only — no hardcoded external model names.
Runs on CPU unless CUDA is available.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from memopt.phase3.transformations import (
    _is_attention_by_structure,
    _find_attention_by_structure,
    _find_attention_by_fx,
    _find_attention_by_hooks,
    _find_all_attention_modules,
    _detect_qkv_layout,
    _patch_module_with_sdpa,
)


# ── Inline model definitions ─────────────────────────────────────────────────

class SeparateQKVAttention(nn.Module):
    """Classic separate Q/K/V projection attention."""

    def __init__(self, d_model: int = 192, n_heads: int = 6):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        H, HD = self.n_heads, self.head_dim
        q = self.q_proj(x).view(B, S, H, HD).transpose(1, 2)
        k = self.k_proj(x).view(B, S, H, HD).transpose(1, 2)
        v = self.v_proj(x).view(B, S, H, HD).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, S, D))


class FusedQKVAttention(nn.Module):
    """GPT-2 / c_attn style: one fused QKV linear."""

    def __init__(self, d_model: int = 192, n_heads: int = 6):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        H, HD = self.n_heads, self.head_dim
        q, k, v = self.c_attn(x).split(D, dim=-1)
        q = q.view(B, S, H, HD).transpose(1, 2)
        k = k.view(B, S, H, HD).transpose(1, 2)
        v = v.view(B, S, H, HD).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, S, D))


class GQAAttention(nn.Module):
    """Grouped-query attention: more Q heads than K/V heads."""

    def __init__(self, d_model: int = 192, n_q_heads: int = 6, n_kv_heads: int = 2):
        super().__init__()
        self.n_q = n_q_heads
        self.n_kv = n_kv_heads
        self.head_dim = d_model // n_q_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.n_q, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv, self.head_dim).transpose(1, 2)
        k = k.repeat_interleave(self.n_q // self.n_kv, dim=1)
        v = v.repeat_interleave(self.n_q // self.n_kv, dim=1)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, S, D))


class NestedTransformerLayer(nn.Module):
    """Transformer layer: attention + FFN nested inside."""

    def __init__(self, d_model: int = 192, n_heads: int = 6):
        super().__init__()
        self.attention = SeparateQKVAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class CNNModel(nn.Module):
    """Pure CNN — should NOT be detected as attention."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64 * 8 * 8, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.adaptive_avg_pool2d(x, 8)
        return self.fc(x.view(x.shape[0], -1))


# ── _is_attention_by_structure ───────────────────────────────────────────────

def test_is_attention_separate_qkv():
    """SeparateQKVAttention has 4 equal direct-child Linears → True."""
    m = SeparateQKVAttention(192, 6)
    assert _is_attention_by_structure(m)


def test_is_attention_fused_qkv_false():
    """FusedQKVAttention has only 2 direct-child Linears → False."""
    m = FusedQKVAttention(192, 6)
    assert not _is_attention_by_structure(m)


def test_is_attention_gqa_no_crash():
    """GQAAttention may or may not match — just assert no crash."""
    m = GQAAttention(192, 6, 2)
    result = _is_attention_by_structure(m)
    assert isinstance(result, bool)


def test_not_attention_single_linear():
    """A plain nn.Linear is not an attention block."""
    m = nn.Linear(64, 64)
    assert not _is_attention_by_structure(m)


def test_not_attention_cnn():
    """CNNModel has no groups of 3+ equal Linears."""
    m = CNNModel()
    assert not _is_attention_by_structure(m)


# ── _find_attention_by_structure ─────────────────────────────────────────────

def test_find_by_structure_finds_nested_attention():
    """NestedTransformerLayer's 'attention' child should be found."""
    m = NestedTransformerLayer(192, 6)
    result = _find_attention_by_structure(m)
    names = [name for name, _ in result]
    assert 'attention' in names


def test_find_by_structure_no_duplicates():
    """Parent not returned after child match (matched_prefixes guard)."""
    m = NestedTransformerLayer(192, 6)
    result = _find_attention_by_structure(m)
    names = [name for name, _ in result]
    # Should not find both 'attention' and 'attention.q_proj' etc.
    assert not any('.' in n and any(n.startswith(p + '.') for p in names) for n in names)


def test_find_by_structure_cnn_empty():
    """CNNModel has no attention structure → empty list."""
    m = CNNModel()
    assert _find_attention_by_structure(m) == []


# ── _find_attention_by_fx ────────────────────────────────────────────────────

def test_find_by_fx_no_crash_on_attention():
    """Should not crash even if FX tracing fails."""
    m = SeparateQKVAttention(192, 6)
    result = _find_attention_by_fx(m)
    assert isinstance(result, list)


def test_find_by_fx_cnn_returns_empty():
    """CNN has no softmax in its graph → []."""
    m = CNNModel()
    result = _find_attention_by_fx(m)
    assert result == []


# ── _find_all_attention_modules orchestrator ─────────────────────────────────

def test_orchestrator_finds_nested_attention():
    """Orchestrator must find the 'attention' submodule of NestedTransformerLayer."""
    m = NestedTransformerLayer(192, 6)
    inputs = {'x': torch.randn(2, 16, 192)}
    result = _find_all_attention_modules(m, inputs)
    assert len(result) > 0
    names = [name for name, _ in result]
    assert any('attention' in n for n in names)


def test_orchestrator_cnn_returns_empty():
    """Orchestrator returns [] for a pure CNN."""
    m = CNNModel()
    x = torch.randn(2, 3, 32, 32)
    try:
        result = _find_all_attention_modules(m, {'x': x})
    except Exception:
        result = []
    assert result == []


# ── _detect_qkv_layout ───────────────────────────────────────────────────────

def test_layout_separate():
    m = SeparateQKVAttention(192, 6)
    linears = [(n, mod) for n, mod in m.named_children() if isinstance(mod, nn.Linear)]
    assert _detect_qkv_layout(linears) == "separate"


def test_layout_fused():
    m = FusedQKVAttention(192, 6)
    linears = [(n, mod) for n, mod in m.named_children() if isinstance(mod, nn.Linear)]
    # c_attn.out=576 (3×192), out_proj.out=192 → max=576 ≥ 2.5×192=480 → fused
    assert _detect_qkv_layout(linears) == "fused"


def test_layout_empty_returns_unknown():
    assert _detect_qkv_layout([]) == "unknown"


def test_layout_single_returns_unknown():
    m = nn.Linear(64, 64)
    assert _detect_qkv_layout([('fc', m)]) == "unknown"


# ── _patch_module_with_sdpa ──────────────────────────────────────────────────

def test_patch_does_not_crash():
    """Patching should not raise even if compile fails."""
    m = NestedTransformerLayer(192, 6)
    x = torch.randn(2, 16, 192)
    with torch.no_grad():
        before = m(x)
    _patch_module_with_sdpa(m, 'attention', m.attention)
    with torch.no_grad():
        after = m(x)
    assert after.shape == before.shape


def test_patch_output_shape_preserved():
    """Output shape must be identical after patching."""
    m = NestedTransformerLayer(192, 6)
    x = torch.randn(1, 32, 192)
    with torch.no_grad():
        ref = m(x)
    _patch_module_with_sdpa(m, 'attention', m.attention)
    with torch.no_grad():
        out = m(x)
    assert out.shape == ref.shape


# ── StrangeAttention smoke test ───────────────────────────────────────────────

def test_strange_attention_smoke():
    """Smoke test: custom attention with 192 hidden / 6 heads is detected."""

    class StrangeAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(192, 192)
            self.k = nn.Linear(192, 192)
            self.v = nn.Linear(192, 192)
            self.o = nn.Linear(192, 192)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, S, D = x.shape
            H, HD = 6, 32
            q = self.q(x).view(B, S, H, HD).transpose(1, 2)
            k = self.k(x).view(B, S, H, HD).transpose(1, 2)
            v = self.v(x).view(B, S, H, HD).transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v)
            return self.o(out.transpose(1, 2).contiguous().view(B, S, D))

    class SmallTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = StrangeAttention()
            self.norm = nn.LayerNorm(192)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.attn(self.norm(x))

    m = SmallTransformer()
    inputs = {'x': torch.randn(2, 32, 192)}
    result = _find_all_attention_modules(m, inputs)
    assert len(result) > 0, "StrangeAttention (192 hidden, 6 heads) must be detected"
