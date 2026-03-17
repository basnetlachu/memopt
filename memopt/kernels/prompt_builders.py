"""
Production prompt builders for fused Triton kernel synthesis.

These functions generate detailed, op-specific prompts for the Claude API
(via JITGenerator) to synthesise Triton kernels that replace common
bottleneck operation sequences with single-pass fused kernels.

Moved from memopt/kernels/tests/test_pillar3_fused.py — prompt logic
belongs in production code, not in test modules.
"""
from __future__ import annotations
import math


def _make_rope_prompt(seq_len, n_heads, head_dim, dtype, hw) -> str:
    """Build a synthesis prompt for a fused Rotary Position Embedding kernel."""
    half = head_dim // 2
    return f"""Write a fused Triton kernel for Rotary Position Embedding (RoPE).
Return EXACTLY this code — do not change the structure, only verify it is correct:

import triton
import triton.language as tl
import torch

@triton.jit
def rope_kernel(
    xq_ptr, xk_ptr, cos_ptr, sin_ptr,
    xq_rot_ptr, xk_rot_ptr,
    n_heads, stride_s, stride_h, stride_cos_s,
):
    pid = tl.program_id(0)
    s = pid // n_heads
    h = pid % n_heads
    base = s * stride_s + h * stride_h
    cos_base = s * stride_cos_s
    d1 = tl.arange(0, {half})
    d2 = tl.arange({half}, {head_dim})
    xq1 = tl.load(xq_ptr + base + d1)
    xq2 = tl.load(xq_ptr + base + d2)
    xk1 = tl.load(xk_ptr + base + d1)
    xk2 = tl.load(xk_ptr + base + d2)
    cos1 = tl.load(cos_ptr + cos_base + d1)
    cos2 = tl.load(cos_ptr + cos_base + d2)
    sin1 = tl.load(sin_ptr + cos_base + d1)
    sin2 = tl.load(sin_ptr + cos_base + d2)
    tl.store(xq_rot_ptr + base + d1, xq1 * cos1 + (-xq2) * sin1)
    tl.store(xq_rot_ptr + base + d2, xq2 * cos2 + xq1 * sin2)
    tl.store(xk_rot_ptr + base + d1, xk1 * cos1 + (-xk2) * sin1)
    tl.store(xk_rot_ptr + base + d2, xk2 * cos2 + xk1 * sin2)

def run_kernel(xq, xk, cos, sin):
    seq_len, n_heads, head_dim = xq.shape
    xq_rot = torch.empty_like(xq)
    xk_rot = torch.empty_like(xk)
    grid = (seq_len * n_heads,)
    rope_kernel[grid](
        xq, xk, cos, sin, xq_rot, xk_rot,
        n_heads, xq.stride(0), xq.stride(1), cos.stride(0),
    )
    return xq_rot, xk_rot

Return only valid Python source starting with `import triton`. No markdown. No prose."""


def _make_ln_residual_prompt(seq_len, hidden, dtype, hw) -> str:
    """Build a synthesis prompt for a fused layer-norm + residual-add kernel."""
    return f"""
Write a fused Triton kernel that performs residual add + layer norm
in a single GPU kernel pass.

The unfused PyTorch path does:
  1. added  = x + residual        (read x, residual → write added to HBM)
  2. normed = layer_norm(added)   (read added from HBM → write normed)

Your fused kernel must:
- Accept: x [seq_len={seq_len}, hidden={hidden}] dtype={dtype},
          residual [same shape],
          weight [hidden] (layer norm scale),
          bias [hidden] (layer norm shift)
- Compute: mean and variance of (x + residual) in registers
- Apply layer norm without writing (x + residual) to HBM
- Return: normed tensor of shape [seq_len, hidden] dtype={dtype}
- Use one threadblock per row (one token per block)
- BLOCK_SIZE must be >= hidden={hidden}, use next power of 2

Hardware: {hw}

The run_kernel signature must be exactly:
def run_kernel(x, residual, weight, bias):
    ...
    return normed

Return only valid Python source starting with `import triton`.
No markdown. No explanation. No comments outside the code.
""".strip()


def _make_softmax_scale_prompt(batch, n_heads, seq_len, dtype, hw) -> str:
    """Build a synthesis prompt for a fused scaled-softmax kernel."""
    scale = 1.0 / math.sqrt(seq_len)
    return f"""
Write a fused Triton kernel for scaled softmax used in attention.

The unfused path does:
  1. scaled = scores * {scale:.6f}   (read scores → write scaled to HBM)
  2. out    = softmax(scaled, dim=-1) (read scaled → write out)

Your fused kernel must:
- Accept: scores [{batch}, {n_heads}, {seq_len}, {seq_len}] dtype={dtype},
          scale float = {scale:.6f}
- Compute online softmax (max-then-exp trick for numerical stability)
  in a single pass without materialising the scaled intermediate tensor
- Return: attention weights of same shape, dtype={dtype}
- One threadblock per row of the last dimension (one query token)
- Use BLOCK_SIZE={min(seq_len, 128)} for the key dimension

Hardware: {hw}

The run_kernel signature must be exactly:
def run_kernel(scores, scale):
    ...
    return attention_weights

Return only valid Python source starting with `import triton`.
No markdown. No explanation. No comments outside the code.
""".strip()
