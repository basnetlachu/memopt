"""
Adaptive Speculation Controller for Production Hyperscale Inference

This controller dynamically adjusts or disables speculative decoding based on:
- Rolling acceptance rate
- Context length
- Fleet-wide performance metrics

Key insight: Speculative decoding CANNOT maintain constant speedup as context grows.
This is fundamental, not a bug. Therefore, we make it adaptive and opportunistic.
"""

import numpy as np
from typing import Optional, Dict
from collections import deque


class AdaptiveSpeculationController:
    """
    Adaptively controls speculative decoding parameters based on runtime metrics.

    Production rules:
    1. Increase K when acceptance > 80% (speculation is cheap)
    2. Decrease K when acceptance < 60% (overhead dominates)
    3. Disable entirely when:
       - Context length > threshold (e.g. 4k-8k tokens)
       - Acceptance remains low for N consecutive steps
       - Memory pressure is high

    This converts speculative decoding from mandatory to opportunistic.
    """

    def __init__(
        self,
        initial_k: int = 4,
        min_k: int = 1,
        max_k: int = 8,
        high_acceptance_threshold: float = 0.80,
        low_acceptance_threshold: float = 0.60,
        context_disable_threshold: int = 8192,
        window_size: int = 100,
        disable_after_low_steps: int = 10
    ):
        """
        Args:
            initial_k: Initial number of speculative tokens
            min_k: Minimum K (below this, disable speculation)
            max_k: Maximum K
            high_acceptance_threshold: Increase K above this acceptance rate
            low_acceptance_threshold: Decrease K below this acceptance rate
            context_disable_threshold: Disable speculation above this context length
            window_size: Rolling window for acceptance rate calculation
            disable_after_low_steps: Disable after N consecutive low-acceptance steps
        """
        self.current_k = initial_k
        self.min_k = min_k
        self.max_k = max_k
        self.high_threshold = high_acceptance_threshold
        self.low_threshold = low_acceptance_threshold
        self.context_threshold = context_disable_threshold
        self.window_size = window_size
        self.disable_after_low_steps = disable_after_low_steps

        # Rolling metrics
        self.acceptance_window = deque(maxlen=window_size)
        self.low_acceptance_streak = 0

        # State
        self.speculation_enabled = True
        self.total_steps = 0
        self.total_accepted = 0
        self.total_rejected = 0

        # Fleet metrics (for trillion-token scale)
        self.tokens_served = 0
        self.tokens_via_speculation = 0

    def should_speculate(self, context_length: int, memory_pressure: float = 0.0) -> bool:
        """
        Decide whether to use speculative decoding for this step.

        Args:
            context_length: Current context length in tokens
            memory_pressure: Memory usage ratio (0.0 to 1.0)

        Returns:
            True if speculation should be used, False otherwise
        """
        # Hard cutoff: disable if context too long
        if context_length > self.context_threshold:
            return False

        # Hard cutoff: disable if memory pressure high
        if memory_pressure > 0.85:
            return False

        # Hard cutoff: disable if streak of low acceptance
        if self.low_acceptance_streak >= self.disable_after_low_steps:
            return False

        # Check if speculation is enabled
        if not self.speculation_enabled:
            return False

        # Check if K is below minimum
        if self.current_k < self.min_k:
            return False

        return True

    def get_num_speculative_tokens(self) -> int:
        """Get current K value."""
        return self.current_k

    def update(
        self,
        num_accepted: int,
        num_proposed: int,
        context_length: int
    ) -> Dict[str, any]:
        """
        Update controller based on latest speculation result.

        Args:
            num_accepted: Number of tokens accepted
            num_proposed: Number of tokens proposed
            context_length: Current context length

        Returns:
            Dict with metrics and actions taken
        """
        self.total_steps += 1
        self.total_accepted += num_accepted
        self.total_rejected += (num_proposed - num_accepted)
        self.tokens_served += 1  # Simplified - actual token count may vary

        # Calculate acceptance rate for this step
        acceptance_rate = num_accepted / num_proposed if num_proposed > 0 else 0.0
        self.acceptance_window.append(acceptance_rate)

        # Calculate rolling average acceptance rate
        avg_acceptance = np.mean(self.acceptance_window) if len(self.acceptance_window) > 0 else 0.0

        actions = []

        # Track low acceptance streak
        if avg_acceptance < self.low_threshold:
            self.low_acceptance_streak += 1
        else:
            self.low_acceptance_streak = 0

        # Adaptive K adjustment
        if len(self.acceptance_window) >= min(10, self.window_size // 10):
            # Only adjust after collecting some data

            if avg_acceptance > self.high_threshold and self.current_k < self.max_k:
                # High acceptance - increase speculation aggressiveness
                old_k = self.current_k
                self.current_k = min(self.current_k + 1, self.max_k)
                actions.append(f"Increased K: {old_k} -> {self.current_k}")
                self.tokens_via_speculation += num_accepted

            elif avg_acceptance < self.low_threshold and self.current_k > self.min_k:
                # Low acceptance - reduce speculation overhead
                old_k = self.current_k
                self.current_k = max(self.current_k - 1, self.min_k)
                actions.append(f"Decreased K: {old_k} -> {self.current_k}")

                # Check if we should disable entirely
                if self.current_k < self.min_k or self.low_acceptance_streak >= self.disable_after_low_steps:
                    self.speculation_enabled = False
                    actions.append("DISABLED speculation (low acceptance streak)")

        # Context-based disabling (production safety)
        if context_length > self.context_threshold:
            if self.speculation_enabled:
                self.speculation_enabled = False
                actions.append(f"DISABLED speculation (context {context_length} > {self.context_threshold})")
        else:
            # Re-enable if context drops and acceptance was good
            if not self.speculation_enabled and avg_acceptance > self.high_threshold:
                self.speculation_enabled = True
                self.current_k = self.initial_k
                actions.append("RE-ENABLED speculation (context OK, acceptance good)")

        return {
            'current_k': self.current_k,
            'enabled': self.speculation_enabled,
            'avg_acceptance': avg_acceptance,
            'low_streak': self.low_acceptance_streak,
            'actions': actions,
            'fleet_metrics': {
                'tokens_served': self.tokens_served,
                'tokens_via_speculation': self.tokens_via_speculation,
                'speculation_efficiency': self.tokens_via_speculation / max(1, self.tokens_served)
            }
        }

    def get_stats(self) -> Dict[str, any]:
        """Get comprehensive statistics for monitoring."""
        overall_acceptance = self.total_accepted / max(1, self.total_accepted + self.total_rejected)

        return {
            'current_k': self.current_k,
            'enabled': self.speculation_enabled,
            'total_steps': self.total_steps,
            'overall_acceptance_rate': overall_acceptance,
            'recent_acceptance_rate': np.mean(self.acceptance_window) if len(self.acceptance_window) > 0 else 0.0,
            'low_acceptance_streak': self.low_acceptance_streak,
            'tokens_served': self.tokens_served,
            'tokens_via_speculation': self.tokens_via_speculation,
            'speculation_hit_rate': self.tokens_via_speculation / max(1, self.tokens_served)
        }

    def reset(self):
        """Reset controller state (for new request/session)."""
        self.acceptance_window.clear()
        self.low_acceptance_streak = 0
        self.speculation_enabled = True
        self.current_k = self.initial_k
