"""
Utilities for true parallel batch processing.

This module provides functions for padding, masking, and managing
multiple sequences in parallel batches.
"""

import torch
from typing import List, Tuple, Optional


def pad_sequences(
    sequences: List[torch.Tensor],
    padding_value: int = 0,
    padding_side: str = "left"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad a list of sequences to the same length.

    Args:
        sequences: List of 1D tensors [seq_len]
        padding_value: Value to use for padding
        padding_side: "left" or "right" padding

    Returns:
        Tuple of (padded_sequences, attention_mask)
        - padded_sequences: [batch_size, max_seq_len]
        - attention_mask: [batch_size, max_seq_len] (1 for real tokens, 0 for padding)
    """
    batch_size = len(sequences)
    max_len = max(seq.shape[0] for seq in sequences)

    # Create padded tensor
    padded = torch.full(
        (batch_size, max_len),
        padding_value,
        dtype=sequences[0].dtype,
        device=sequences[0].device
    )

    # Create attention mask
    attention_mask = torch.zeros(
        (batch_size, max_len),
        dtype=torch.long,
        device=sequences[0].device
    )

    # Fill in sequences and mask
    for i, seq in enumerate(sequences):
        seq_len = seq.shape[0]

        if padding_side == "left":
            # Left padding (for causal LM)
            padded[i, -seq_len:] = seq
            attention_mask[i, -seq_len:] = 1
        else:
            # Right padding
            padded[i, :seq_len] = seq
            attention_mask[i, :seq_len] = 1

    return padded, attention_mask


def create_causal_mask(
    batch_size: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Create causal attention mask for autoregressive generation.

    Args:
        batch_size: Number of sequences
        seq_len: Sequence length
        device: Device to create mask on
        dtype: Data type for mask

    Returns:
        Causal mask [batch_size, 1, seq_len, seq_len]
        Values: 0 for allowed attention, -inf for masked
    """
    # Create causal mask (lower triangular)
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=dtype) * float('-inf'),
        diagonal=1
    )

    # Expand for batch and heads
    mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
    mask = mask.expand(batch_size, 1, seq_len, seq_len)

    return mask


def combine_masks(
    attention_mask: torch.Tensor,
    causal_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Combine padding mask with causal mask.

    Args:
        attention_mask: Padding mask [batch_size, seq_len]
        causal_mask: Causal mask [batch_size, 1, seq_len, seq_len]

    Returns:
        Combined mask [batch_size, 1, seq_len, seq_len]
    """
    batch_size, seq_len = attention_mask.shape

    # Convert attention_mask to 4D: [batch_size, 1, 1, seq_len]
    # This broadcasts across query positions
    mask_4d = attention_mask.unsqueeze(1).unsqueeze(2)

    # Expand to [batch_size, 1, seq_len, seq_len]
    mask_4d = mask_4d.expand(batch_size, 1, seq_len, seq_len)

    # Convert to float and set padding to -inf
    combined = torch.where(
        mask_4d == 1,
        torch.zeros_like(mask_4d, dtype=torch.float32),
        torch.ones_like(mask_4d, dtype=torch.float32) * float('-inf')
    )

    # Add causal mask if provided
    if causal_mask is not None:
        combined = combined + causal_mask

    return combined


def update_attention_mask(
    attention_mask: torch.Tensor,
    new_tokens: torch.Tensor
) -> torch.Tensor:
    """
    Update attention mask when new tokens are generated.

    Args:
        attention_mask: Current mask [batch_size, seq_len]
        new_tokens: New tokens [batch_size, 1]

    Returns:
        Updated mask [batch_size, seq_len + 1]
    """
    batch_size = attention_mask.shape[0]

    # Add mask for new token (always 1, not padding)
    new_mask = torch.ones(
        (batch_size, 1),
        dtype=attention_mask.dtype,
        device=attention_mask.device
    )

    return torch.cat([attention_mask, new_mask], dim=1)


def batch_select(
    tensor: torch.Tensor,
    indices: torch.Tensor
) -> torch.Tensor:
    """
    Select elements from tensor using per-batch indices.

    Args:
        tensor: Input tensor [batch_size, ...]
        indices: Boolean mask [batch_size] or integer indices

    Returns:
        Selected elements
    """
    if indices.dtype == torch.bool:
        return tensor[indices]
    else:
        return tensor[indices]


def get_unfinished_sequences(finished: torch.Tensor) -> torch.Tensor:
    """
    Get indices of unfinished sequences.

    Args:
        finished: Boolean tensor [batch_size]

    Returns:
        Integer indices of unfinished sequences
    """
    return torch.where(~finished)[0]
