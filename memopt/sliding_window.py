"""
Sliding Window Attention for Trillion-Token Scale

Converts unbounded context growth O(n²) into bounded O(n·W) where W is window size.

This is CRITICAL for trillion-token deployment:
- Prevents memory explosion
- Maintains constant memory footprint
- Enables infinite-length generation

Production implementation:
- Automatically evicts old KV blocks
- Maintains correctness via configurable window size
- Integrates with paged KV cache
"""

import torch
from typing import Optional, Tuple


class SlidingWindowManager:
    """
    Manages sliding window for attention to keep memory bounded.

    Key insight: For trillion-token generation, we CANNOT keep full context.
    Solution: Keep only last W tokens, evict old KV blocks.

    This gives:
    - Memory: O(W) instead of O(n)
    - Attention: O(n·W) instead of O(n²)
    - Generation: Unbounded length with bounded resources
    """

    def __init__(
        self,
        window_size: int = 4096,
        min_window_size: int = 1024,
        enable_dynamic_window: bool = True
    ):
        """
        Args:
            window_size: Maximum tokens to keep in context
            min_window_size: Minimum window size (safety)
            enable_dynamic_window: Whether to dynamically adjust window based on load
        """
        self.window_size = window_size
        self.min_window_size = min_window_size
        self.max_window_size = window_size
        self.enable_dynamic_window = enable_dynamic_window

        # Tracking
        self.total_tokens_processed = 0
        self.total_evictions = 0
        self.total_windows_slid = 0

    def should_apply_window(self, seq_length: int) -> bool:
        """
        Decide if sliding window should be applied.

        Args:
            seq_length: Current sequence length

        Returns:
            True if window should be applied
        """
        return seq_length > self.window_size

    def apply_window(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Apply sliding window to inputs.

        Keeps only last W tokens, discards older tokens.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len] or None
            position_ids: [batch, seq_len] or None

        Returns:
            Tuple of (windowed_input_ids, windowed_attention_mask, windowed_position_ids)
        """
        batch_size, seq_len = input_ids.shape

        if seq_len <= self.window_size:
            # No windowing needed
            return input_ids, attention_mask, position_ids

        # Calculate how much to keep
        start_idx = seq_len - self.window_size

        # Apply window
        windowed_input_ids = input_ids[:, start_idx:]

        windowed_attention_mask = None
        if attention_mask is not None:
            windowed_attention_mask = attention_mask[:, start_idx:]

        windowed_position_ids = None
        if position_ids is not None:
            windowed_position_ids = position_ids[:, start_idx:]

        # Track metrics
        self.total_evictions += (seq_len - self.window_size)
        self.total_windows_slid += 1
        self.total_tokens_processed += seq_len

        return windowed_input_ids, windowed_attention_mask, windowed_position_ids

    def get_kv_indices_to_evict(
        self,
        current_length: int,
        kv_cache_length: int
    ) -> Optional[Tuple[int, int]]:
        """
        Get which KV cache indices should be evicted.

        Args:
            current_length: Current sequence length
            kv_cache_length: Length of KV cache

        Returns:
            Tuple of (start_idx, end_idx) to evict, or None if no eviction needed
        """
        if current_length <= self.window_size:
            return None

        # Evict oldest tokens
        num_to_evict = current_length - self.window_size
        return (0, num_to_evict)

    def adjust_window_size(self, memory_pressure: float):
        """
        Dynamically adjust window size based on memory pressure.

        Production rule:
        - High memory pressure → reduce window
        - Low memory pressure → increase window (up to max)

        Args:
            memory_pressure: 0.0 to 1.0
        """
        if not self.enable_dynamic_window:
            return

        if memory_pressure > 0.85:
            # Critical memory pressure - reduce window aggressively
            self.window_size = max(
                self.min_window_size,
                int(self.window_size * 0.75)
            )
        elif memory_pressure > 0.70:
            # High memory pressure - reduce window moderately
            self.window_size = max(
                self.min_window_size,
                int(self.window_size * 0.90)
            )
        elif memory_pressure < 0.50:
            # Low memory pressure - can increase window
            self.window_size = min(
                self.max_window_size,
                int(self.window_size * 1.1)
            )

    def get_stats(self) -> dict:
        """Get sliding window statistics for monitoring."""
        return {
            'current_window_size': self.window_size,
            'min_window_size': self.min_window_size,
            'max_window_size': self.max_window_size,
            'total_tokens_processed': self.total_tokens_processed,
            'total_evictions': self.total_evictions,
            'total_windows_slid': self.total_windows_slid,
            'eviction_rate': self.total_evictions / max(1, self.total_tokens_processed)
        }

    def reset(self):
        """Reset for new request."""
        self.window_size = self.max_window_size
        # Don't reset counters - these are fleet-wide metrics


def apply_sliding_window_to_kv_cache(
    kv_cache,
    window_manager: SlidingWindowManager,
    seq_id: int
):
    """
    Apply sliding window eviction to KV cache.

    This integrates sliding window with paged KV cache system.

    Args:
        kv_cache: PagedKVCache instance
        window_manager: SlidingWindowManager instance
        seq_id: Sequence ID to apply window to
    """
    # Get current sequence length from KV cache
    if not hasattr(kv_cache, 'sequence_blocks') or seq_id not in kv_cache.sequence_blocks:
        return

    seq_info = kv_cache.sequence_blocks[seq_id]
    current_length = seq_info['current_length']

    # Check if we need to evict
    eviction_range = window_manager.get_kv_indices_to_evict(
        current_length=current_length,
        kv_cache_length=current_length
    )

    if eviction_range is None:
        return

    start_idx, end_idx = eviction_range

    # Evict blocks corresponding to old tokens
    # This is production-critical for trillion-token scale
    blocks_to_evict = []
    block_size = kv_cache.block_size

    # Calculate which blocks contain the tokens to evict
    start_block = start_idx // block_size
    end_block = (end_idx - 1) // block_size

    for block_idx in range(start_block, end_block + 1):
        if block_idx < len(seq_info['blocks']):
            block_id = seq_info['blocks'][block_idx]
            blocks_to_evict.append(block_id)

    # Evict the blocks
    for block_id in blocks_to_evict:
        if block_id in kv_cache.allocated_blocks:
            kv_cache.allocated_blocks.remove(block_id)
            kv_cache.free_blocks.append(block_id)

    # Update sequence info
    seq_info['blocks'] = seq_info['blocks'][end_block + 1:]
    seq_info['current_length'] = current_length - end_idx

    # Update window manager stats
    window_manager.total_evictions += end_idx
    window_manager.total_windows_slid += 1
