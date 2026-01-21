"""
Load balancing strategies for routing requests to workers.
"""
import random
from typing import List, Optional
from abc import ABC, abstractmethod
from production.registry import WorkerInfo
import logging

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(ABC):
    """Base class for load balancing strategies."""

    @abstractmethod
    def select_worker(self, workers: List[WorkerInfo], session_id: Optional[str] = None) -> Optional[WorkerInfo]:
        """
        Select a worker for the given request.

        Args:
            workers: List of available workers
            session_id: Optional session ID for affinity

        Returns:
            Selected worker or None if no workers available
        """
        pass


class RoundRobinStrategy(LoadBalancingStrategy):
    """Simple round-robin load balancing."""

    def __init__(self):
        self.counter = 0

    def select_worker(self, workers: List[WorkerInfo], session_id: Optional[str] = None) -> Optional[WorkerInfo]:
        if not workers:
            return None

        worker = workers[self.counter % len(workers)]
        self.counter += 1
        return worker


class LeastLoadedStrategy(LoadBalancingStrategy):
    """
    Select the worker with lowest load score.

    Load score considers:
    - Queue depth (70%)
    - Current batch size (30%)
    """

    def select_worker(self, workers: List[WorkerInfo], session_id: Optional[str] = None) -> Optional[WorkerInfo]:
        if not workers:
            return None

        # Select worker with minimum load score
        return min(workers, key=lambda w: w.load_score)


class CostAwareStrategy(LoadBalancingStrategy):
    """
    Cost-aware routing that considers both load and throughput.

    Prefers workers with:
    - Lower queue depth
    - Higher tokens/second (more capable)
    - Lower p95 latency
    """

    def select_worker(self, workers: List[WorkerInfo], session_id: Optional[str] = None) -> Optional[WorkerInfo]:
        if not workers:
            return None

        def cost_score(worker: WorkerInfo) -> float:
            """
            Calculate cost score. Lower is better.

            Factors:
            - Queue depth (higher = worse)
            - Throughput (higher = better, so inverse)
            - Latency (higher = worse)
            """
            # Normalize queue (0-1 range, max 128)
            queue_cost = min(worker.queue_depth / 128.0, 1.0)

            # Throughput cost (inverse, normalized assuming max 3000 tok/s)
            throughput_cost = 1.0 - min(worker.tokens_per_second / 3000.0, 1.0)

            # Latency cost (normalized assuming max 1000ms)
            latency_cost = min(worker.p95_latency_ms / 1000.0, 1.0)

            # Weighted combination
            return (queue_cost * 0.5) + (throughput_cost * 0.3) + (latency_cost * 0.2)

        return min(workers, key=cost_score)


class RandomStrategy(LoadBalancingStrategy):
    """Random worker selection (useful for testing)."""

    def select_worker(self, workers: List[WorkerInfo], session_id: Optional[str] = None) -> Optional[WorkerInfo]:
        if not workers:
            return None

        return random.choice(workers)


class PowerOfTwoChoicesStrategy(LoadBalancingStrategy):
    """
    Power of Two Choices algorithm.

    Randomly select 2 workers and choose the least loaded one.
    Provides good load balancing with minimal overhead.
    """

    def select_worker(self, workers: List[WorkerInfo], session_id: Optional[str] = None) -> Optional[WorkerInfo]:
        if not workers:
            return None

        if len(workers) == 1:
            return workers[0]

        # Sample 2 random workers
        sample = random.sample(workers, min(2, len(workers)))

        # Return the one with lower load
        return min(sample, key=lambda w: w.load_score)


def get_strategy(strategy_name: str) -> LoadBalancingStrategy:
    """Get load balancing strategy by name."""
    strategies = {
        "round_robin": RoundRobinStrategy,
        "least_loaded": LeastLoadedStrategy,
        "cost_aware": CostAwareStrategy,
        "random": RandomStrategy,
        "power_of_two": PowerOfTwoChoicesStrategy,
    }

    strategy_cls = strategies.get(strategy_name.lower())
    if not strategy_cls:
        logger.warning(f"Unknown strategy '{strategy_name}', using least_loaded")
        strategy_cls = LeastLoadedStrategy

    return strategy_cls()
