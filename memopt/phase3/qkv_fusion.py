"""
memopt.phase3.qkv_fusion
==========================

QKV projection fusion for transformer models.

Merges the separate Q, K, V linear projections into a single batched matmul,
reducing:
  - HBM reads: 3 weight matrix loads → 1 concatenated load  (33% fewer reads)
  - Kernel launch overhead: 3 GEMM calls → 1

This is most effective for memory-bound transformers where the attention
projection weights dominate HBM traffic.

Naming conventions supported:
  q_proj / k_proj / v_proj        (LLaMA, Mistral, Falcon, Phi)
  query / key / value             (some HF BERT variants)
  query_key_value                 (already fused — skip)
  self.query / self.key / self.value

The fused module (FusedQKVLinear) stores one concatenated weight matrix and
splits the output at forward time — identical numerical output to 3 separate
matmuls.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger("memopt.phase3.qkv_fusion")

# ---------------------------------------------------------------------------
# Candidate triplet patterns (in priority order, matched left-to-right)
# ---------------------------------------------------------------------------

_QKV_PATTERNS: List[Tuple[str, str, str]] = [
    ("q_proj",   "k_proj",   "v_proj"),    # LLaMA, Mistral, Falcon
    ("query",    "key",      "value"),      # BERT-style
    ("q",        "k",        "v"),          # generic short names
    ("wq",       "wk",       "wv"),         # some custom transformers
    ("Q",        "K",        "V"),          # uppercase (some research code)
]


# ---------------------------------------------------------------------------
# FusedQKVLinear — the replacement module
# ---------------------------------------------------------------------------

class FusedQKVLinear(nn.Module):
    """
    A single linear layer that replaces three separate Q, K, V projections.

    Weight layout: [q_out | k_out | v_out] stacked along dim=0.
    Forward pass:  one F.linear call, then split into q, k, v.

    Attributes:
        q_out_features, k_out_features, v_out_features: individual output dims.
        in_features:  shared input dimension.
    """

    def __init__(
        self,
        q_linear: nn.Linear,
        k_linear: nn.Linear,
        v_linear: nn.Linear,
    ) -> None:
        super().__init__()

        self.in_features      = q_linear.in_features
        self.q_out_features   = q_linear.out_features
        self.k_out_features   = k_linear.out_features
        self.v_out_features   = v_linear.out_features
        total_out             = self.q_out_features + self.k_out_features + self.v_out_features

        has_bias = (q_linear.bias is not None)

        # Single weight matrix: stack Q, K, V row-wise
        fused_weight = torch.cat(
            [q_linear.weight.data,
             k_linear.weight.data,
             v_linear.weight.data],
            dim=0,
        )  # shape: (total_out, in_features)

        self.weight = nn.Parameter(fused_weight)

        if has_bias:
            fused_bias = torch.cat(
                [q_linear.bias.data,
                 k_linear.bias.data,
                 v_linear.bias.data],
                dim=0,
            )
            self.bias = nn.Parameter(fused_bias)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run one fused matmul and split into (q, k, v).

        Returns:
            (q, k, v) — same shapes as the original separate projections.
        """
        # One GEMM: (batch, seq, in) × (total_out, in).T → (batch, seq, total_out)
        qkv = F.linear(x, self.weight, self.bias)

        q, k, v = torch.split(
            qkv,
            [self.q_out_features, self.k_out_features, self.v_out_features],
            dim=-1,
        )
        return q, k, v

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, "
            f"q_out={self.q_out_features}, "
            f"k_out={self.k_out_features}, "
            f"v_out={self.v_out_features}"
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply_qkv_fusion(model: nn.Module) -> Tuple[nn.Module, int]:
    """
    Fuse Q/K/V projection triplets throughout *model*.

    Scans for sibling nn.Linear modules with QKV-naming patterns in each
    parent module.  Replaces each matching triplet with a FusedQKVLinear.

    A FusedQKVLinear wraps the combined weight and splits output at forward
    time — numerical output is bit-identical to the original three matmuls.

    Note: The attention forward code must be updated to unpack the tuple
    returned by FusedQKVLinear.  We do NOT modify attention forward methods
    here — the agent's benchmark will detect if the fusion broke forward
    pass (exception → ROLLED_BACK).

    For models whose attention modules directly call self.q_proj(x),
    self.k_proj(x), self.v_proj(x) separately, this fusion is incompatible
    (output signature change from Tensor to Tuple[Tensor,Tensor,Tensor]).
    In that case, the forward will raise → agent rolls back.

    A safer approach for production: override only models whose attention
    class is known to support fused QKV (e.g. custom attention layers you
    control).  The agent treats this as an experiment: benchmark → commit or
    rollback.

    Args:
        model: nn.Module to modify in-place.

    Returns:
        (model, num_fusions)  — num_fusions is 0 if nothing was fused.
    """
    targets = _find_qkv_targets(model)

    if not targets:
        log.info("qkv_fusion: no fusable QKV triplets found")
        return model, 0

    fused_count = 0
    for parent_path, parent_module, q_name, k_name, v_name in targets:
        try:
            n = _fuse_qkv_in_module(parent_module, q_name, k_name, v_name)
            fused_count += n
            log.info(
                "qkv_fusion: fused %s.(%s,%s,%s) → FusedQKVLinear",
                parent_path or "<root>", q_name, k_name, v_name,
            )
        except Exception as exc:
            log.debug(
                "qkv_fusion: failed to fuse %s.(%s,%s,%s) — %s",
                parent_path, q_name, k_name, v_name, exc,
            )

    log.info("qkv_fusion: total fusions=%d", fused_count)
    return model, fused_count


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _find_qkv_targets(
    model: nn.Module,
) -> List[Tuple[str, nn.Module, str, str, str]]:
    """
    Walk model, find parent modules that contain a QKV triplet.

    Returns list of (parent_path, parent_module, q_name, k_name, v_name).
    Does NOT recurse into already-fused modules.
    """
    results: List[Tuple[str, nn.Module, str, str, str]] = []
    seen_parents: Set[int] = set()

    for parent_path, parent_module in model.named_modules():
        if id(parent_module) in seen_parents:
            continue

        # Get direct children that are nn.Linear
        linear_children: Dict[str, nn.Linear] = {
            name: child
            for name, child in parent_module.named_children()
            if isinstance(child, nn.Linear)
        }

        if not linear_children:
            continue

        # Check if already has a FusedQKVLinear (skip if already fused)
        already_fused = any(
            isinstance(child, FusedQKVLinear)
            for child in parent_module.children()
        )
        if already_fused:
            continue

        # Also skip if query_key_value exists (already fused externally)
        if "query_key_value" in {name for name, _ in parent_module.named_children()}:
            continue

        for q_name, k_name, v_name in _QKV_PATTERNS:
            if (q_name in linear_children and
                    k_name in linear_children and
                    v_name in linear_children):

                q_lin = linear_children[q_name]
                k_lin = linear_children[k_name]
                v_lin = linear_children[v_name]

                # Sanity: all must share same in_features
                if not (q_lin.in_features == k_lin.in_features == v_lin.in_features):
                    log.debug(
                        "qkv_fusion: %s has mismatched in_features — skip",
                        parent_path,
                    )
                    continue

                # Sanity: all must have same bias presence
                q_has_bias = q_lin.bias is not None
                k_has_bias = k_lin.bias is not None
                v_has_bias = v_lin.bias is not None
                if not (q_has_bias == k_has_bias == v_has_bias):
                    log.debug(
                        "qkv_fusion: %s has mixed bias config — skip",
                        parent_path,
                    )
                    continue

                results.append((parent_path, parent_module, q_name, k_name, v_name))
                seen_parents.add(id(parent_module))
                break  # one pattern per parent

    return results


def _fuse_qkv_in_module(
    parent: nn.Module,
    q_name: str,
    k_name: str,
    v_name: str,
) -> int:
    """
    Replace the (q, k, v) triplet in *parent* with a single FusedQKVLinear.

    Sets `parent.{q_name}` to the fused module and deletes `{k_name}` and
    `{v_name}` from parent.

    Returns 1 on success.
    """
    q_lin = getattr(parent, q_name)
    k_lin = getattr(parent, k_name)
    v_lin = getattr(parent, v_name)

    fused = FusedQKVLinear(q_lin, k_lin, v_lin)

    # Move to same device / dtype as Q
    device = q_lin.weight.device
    dtype  = q_lin.weight.dtype
    fused  = fused.to(device=device, dtype=dtype)

    # Replace q_name with the fused module
    setattr(parent, q_name, fused)

    # Remove k and v from parent (they are now inside FusedQKVLinear)
    try:
        delattr(parent, k_name)
    except AttributeError:
        pass

    try:
        delattr(parent, v_name)
    except AttributeError:
        pass

    # Register k_name and v_name as None so state_dict doesn't break
    # existing code that might access parent.k_proj (optional safety net)
    # We don't do this by default — it would pollute the module namespace.

    return 1
