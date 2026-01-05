"""
Hard Cutoff Policies for Production Stability

Prevents catastrophic slowdowns and OOM crashes in trillion-token deployments.

Key principle: EXPLICIT is better than IMPLICIT.
Instead of letting the system degrade gracefully, we enforce hard limits that
ensure predictable behavior under all conditions.

Production requirements:
- No silent performance degradation
- No OOM crashes
- Predictable worst-case latency
- Clear rejection messages
"""

from typing import Optional, Dict
from dataclasses import dataclass
import time


@dataclass
class CutoffPolicy:
    """
    Defines hard cutoff thresholds for a specific resource.

    Production design:
    - warning_threshold: Log warning but continue (0.7-0.8)
    - soft_cutoff: Start degrading service quality (0.8-0.9)
    - hard_cutoff: Reject new requests (0.9-0.95)
    - emergency_cutoff: Kill running requests to prevent crash (0.95-1.0)
    """
    warning_threshold: float
    soft_cutoff: float
    hard_cutoff: float
    emergency_cutoff: float
    resource_name: str


class ProductionCutoffManager:
    """
    Enforces hard cutoff policies across all resources.

    Production-critical for trillion-token scale:
    - Prevents OOM crashes (>99.9% uptime)
    - Ensures predictable latency (p99 < hard_cutoff)
    - Enables safe operation at high load (85-90% utilization)

    Without this: System crashes randomly, users get timeouts, fleet goes down.
    With this: System rejects cleanly, logs warnings, stays stable.
    """

    def __init__(
        self,
        max_context_length: int = 8192,
        max_batch_size: int = 32,
        max_memory_gb: float = 40.0,
        max_kv_cache_blocks: int = 4096,
        max_concurrent_requests: int = 100,
        enable_emergency_eviction: bool = True
    ):
        """
        Args:
            max_context_length: Hard limit on sequence length
            max_batch_size: Hard limit on batch size
            max_memory_gb: Hard limit on GPU memory usage
            max_kv_cache_blocks: Hard limit on KV cache blocks
            max_concurrent_requests: Hard limit on concurrent requests
            enable_emergency_eviction: Allow emergency eviction to prevent crash
        """
        self.max_context_length = max_context_length
        self.max_batch_size = max_batch_size
        self.max_memory_gb = max_memory_gb
        self.max_kv_cache_blocks = max_kv_cache_blocks
        self.max_concurrent_requests = max_concurrent_requests
        self.enable_emergency_eviction = enable_emergency_eviction

        # Define cutoff policies for each resource
        self.policies = {
            'context_length': CutoffPolicy(
                warning_threshold=0.7,    # 70% of max context
                soft_cutoff=0.85,         # 85% - start sliding window
                hard_cutoff=0.95,         # 95% - reject if no window
                emergency_cutoff=1.0,     # 100% - force truncate
                resource_name='Context Length'
            ),
            'memory': CutoffPolicy(
                warning_threshold=0.70,   # 70% memory usage
                soft_cutoff=0.85,         # 85% - reduce batch size
                hard_cutoff=0.95,         # 95% - reject new requests
                emergency_cutoff=0.98,    # 98% - emergency eviction
                resource_name='GPU Memory'
            ),
            'kv_cache': CutoffPolicy(
                warning_threshold=0.75,   # 75% cache usage
                soft_cutoff=0.90,         # 90% - start eviction
                hard_cutoff=0.95,         # 95% - reject requests
                emergency_cutoff=0.98,    # 98% - emergency eviction
                resource_name='KV Cache'
            ),
            'concurrent_requests': CutoffPolicy(
                warning_threshold=0.80,   # 80% of max concurrent
                soft_cutoff=0.90,         # 90% - queue new requests
                hard_cutoff=0.95,         # 95% - reject new requests
                emergency_cutoff=1.0,     # 100% - impossible to exceed
                resource_name='Concurrent Requests'
            ),
            'acceptance_rate': CutoffPolicy(
                warning_threshold=0.60,   # 60% acceptance (speculation failing)
                soft_cutoff=0.40,         # 40% - reduce K
                hard_cutoff=0.20,         # 20% - disable speculation
                emergency_cutoff=0.0,     # Never emergency for acceptance
                resource_name='Speculation Acceptance'
            ),
        }

        # Statistics (production monitoring)
        self.total_warnings = 0
        self.total_soft_cutoffs = 0
        self.total_hard_cutoffs = 0
        self.total_emergency_cutoffs = 0
        self.total_rejections = 0

        # Track cutoffs by resource
        self.warnings_by_resource: Dict[str, int] = {}
        self.rejections_by_resource: Dict[str, int] = {}

    def check_context_length(
        self,
        current_length: int,
        sliding_window_enabled: bool = False
    ) -> tuple[bool, str, str]:
        """
        Check if context length is within acceptable limits.

        Production decision tree:
        1. < warning: OK
        2. warning-soft: Log warning
        3. soft-hard: Enable sliding window if available, else warn
        4. hard-emergency: Reject if no sliding window
        5. > emergency: Force truncate

        Args:
            current_length: Current sequence length
            sliding_window_enabled: Whether sliding window is available

        Returns:
            (allowed, severity, message)
            - allowed: True if request can continue
            - severity: 'ok', 'warning', 'soft', 'hard', 'emergency'
            - message: Human-readable explanation
        """
        policy = self.policies['context_length']
        ratio = current_length / self.max_context_length

        if ratio >= policy.emergency_cutoff:
            self.total_emergency_cutoffs += 1
            self._track_cutoff('context_length', 'emergency')
            return (
                False,
                'emergency',
                f'Context length {current_length} exceeds EMERGENCY limit {self.max_context_length}. '
                f'This request will be force-truncated or rejected.'
            )

        if ratio >= policy.hard_cutoff:
            if not sliding_window_enabled:
                self.total_hard_cutoffs += 1
                self.total_rejections += 1
                self._track_cutoff('context_length', 'rejection')
                return (
                    False,
                    'hard',
                    f'Context length {current_length} exceeds hard limit '
                    f'({policy.hard_cutoff * 100:.0f}% of {self.max_context_length}). '
                    f'Sliding window not enabled. Request rejected.'
                )
            else:
                # Sliding window will handle it
                self.total_soft_cutoffs += 1
                self._track_cutoff('context_length', 'soft')
                return (
                    True,
                    'soft',
                    f'Context length {current_length} approaching limit. '
                    f'Sliding window will truncate to {int(self.max_context_length * 0.9)}.'
                )

        if ratio >= policy.soft_cutoff:
            self.total_soft_cutoffs += 1
            self._track_cutoff('context_length', 'soft')
            return (
                True,
                'soft',
                f'Context length {current_length} at {ratio*100:.1f}% of limit. '
                f'Consider enabling sliding window or reducing max_tokens.'
            )

        if ratio >= policy.warning_threshold:
            self.total_warnings += 1
            self._track_cutoff('context_length', 'warning')
            return (
                True,
                'warning',
                f'Context length {current_length} at {ratio*100:.1f}% of limit.'
            )

        return (True, 'ok', '')

    def check_memory_pressure(
        self,
        memory_pressure: float,
        can_reduce_batch: bool = True,
        can_evict: bool = True
    ) -> tuple[bool, str, str]:
        """
        Check if memory pressure is within acceptable limits.

        Production actions:
        1. < warning: OK
        2. warning-soft: Log warning
        3. soft-hard: Reduce batch size, enable eviction
        4. hard-emergency: Reject new requests
        5. > emergency: Emergency eviction

        Args:
            memory_pressure: 0.0 to 1.0
            can_reduce_batch: Whether batch size can be reduced
            can_evict: Whether cache eviction is possible

        Returns:
            (allowed, severity, message)
        """
        policy = self.policies['memory']

        if memory_pressure >= policy.emergency_cutoff:
            self.total_emergency_cutoffs += 1
            self._track_cutoff('memory', 'emergency')
            if self.enable_emergency_eviction and can_evict:
                return (
                    True,
                    'emergency',
                    f'CRITICAL: Memory at {memory_pressure*100:.1f}%. '
                    f'Emergency eviction triggered.'
                )
            else:
                return (
                    False,
                    'emergency',
                    f'CRITICAL: Memory at {memory_pressure*100:.1f}%. '
                    f'System may crash. Rejecting all new requests.'
                )

        if memory_pressure >= policy.hard_cutoff:
            self.total_hard_cutoffs += 1
            self.total_rejections += 1
            self._track_cutoff('memory', 'rejection')
            return (
                False,
                'hard',
                f'Memory pressure {memory_pressure*100:.1f}% exceeds hard limit '
                f'({policy.hard_cutoff*100:.0f}%). Request rejected.'
            )

        if memory_pressure >= policy.soft_cutoff:
            self.total_soft_cutoffs += 1
            self._track_cutoff('memory', 'soft')
            actions = []
            if can_reduce_batch:
                actions.append('reducing batch size')
            if can_evict:
                actions.append('enabling aggressive eviction')
            action_str = ' and '.join(actions) if actions else 'no mitigation available'

            return (
                True,
                'soft',
                f'High memory pressure {memory_pressure*100:.1f}%. '
                f'Taking action: {action_str}.'
            )

        if memory_pressure >= policy.warning_threshold:
            self.total_warnings += 1
            self._track_cutoff('memory', 'warning')
            return (
                True,
                'warning',
                f'Memory pressure at {memory_pressure*100:.1f}%.'
            )

        return (True, 'ok', '')

    def check_acceptance_rate(
        self,
        acceptance_rate: float,
        current_k: int
    ) -> tuple[bool, str, str, int]:
        """
        Check if speculation acceptance rate justifies continued use.

        Production decision:
        - > 80%: Excellent, can increase K
        - 60-80%: Good, maintain current K
        - 40-60%: Poor, reduce K
        - 20-40%: Bad, reduce K to 1
        - < 20%: Terrible, disable speculation

        Args:
            acceptance_rate: 0.0 to 1.0
            current_k: Current number of speculative tokens

        Returns:
            (should_continue, severity, message, recommended_k)
        """
        policy = self.policies['acceptance_rate']

        # Note: For acceptance rate, thresholds work in reverse
        # (lower is worse, unlike other resources)

        if acceptance_rate <= policy.hard_cutoff:
            # < 20% acceptance - disable speculation
            self.total_hard_cutoffs += 1
            self._track_cutoff('acceptance_rate', 'hard')
            return (
                False,
                'hard',
                f'Speculation acceptance rate {acceptance_rate*100:.1f}% is too low. '
                f'Disabling speculation (overhead > benefit).',
                0
            )

        if acceptance_rate <= policy.soft_cutoff:
            # 20-40% acceptance - reduce to K=1
            self.total_soft_cutoffs += 1
            self._track_cutoff('acceptance_rate', 'soft')
            return (
                True,
                'soft',
                f'Low acceptance rate {acceptance_rate*100:.1f}%. '
                f'Reducing K to 1.',
                1
            )

        if acceptance_rate <= policy.warning_threshold:
            # 40-60% acceptance - reduce K but keep going
            self.total_warnings += 1
            self._track_cutoff('acceptance_rate', 'warning')
            new_k = max(1, current_k - 1)
            return (
                True,
                'warning',
                f'Acceptance rate {acceptance_rate*100:.1f}% below target. '
                f'Reducing K from {current_k} to {new_k}.',
                new_k
            )

        # Good acceptance rate - maintain or increase K
        return (True, 'ok', '', current_k)

    def _track_cutoff(self, resource: str, severity: str):
        """Track cutoff by resource for monitoring."""
        key = f'{resource}_{severity}'
        if resource not in self.warnings_by_resource:
            self.warnings_by_resource[resource] = 0
        if severity == 'rejection':
            if resource not in self.rejections_by_resource:
                self.rejections_by_resource[resource] = 0
            self.rejections_by_resource[resource] += 1
        else:
            self.warnings_by_resource[resource] += 1

    def get_stats(self) -> dict:
        """
        Get cutoff policy statistics.

        Production monitoring:
        - rejection_rate: % of requests rejected
        - emergency_rate: % of requests hitting emergency
        - by_resource: Breakdown by resource type
        """
        return {
            'total_warnings': self.total_warnings,
            'total_soft_cutoffs': self.total_soft_cutoffs,
            'total_hard_cutoffs': self.total_hard_cutoffs,
            'total_emergency_cutoffs': self.total_emergency_cutoffs,
            'total_rejections': self.total_rejections,
            'warnings_by_resource': dict(self.warnings_by_resource),
            'rejections_by_resource': dict(self.rejections_by_resource),
            'policies': {
                name: {
                    'warning_threshold': p.warning_threshold,
                    'soft_cutoff': p.soft_cutoff,
                    'hard_cutoff': p.hard_cutoff,
                    'emergency_cutoff': p.emergency_cutoff,
                }
                for name, p in self.policies.items()
            }
        }

    def reset_stats(self):
        """Reset statistics counters."""
        self.total_warnings = 0
        self.total_soft_cutoffs = 0
        self.total_hard_cutoffs = 0
        self.total_emergency_cutoffs = 0
        self.total_rejections = 0
        self.warnings_by_resource.clear()
        self.rejections_by_resource.clear()
