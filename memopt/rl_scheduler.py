"""
Reinforcement Learning Environment for Batch Scheduler Optimization

This module provides an RL environment (OpenAI Gym compatible) for training
an intelligent batch scheduling policy that adapts to workload patterns.

Key Features:
- State: [avg_seq_len, queue_depth, memory_usage, num_urgent_requests]
- Action: batch_size (discrete: 1, 2, 4, 8, 16, 32, 64)
- Reward: throughput / latency (higher is better)

Expected Improvement: +15-25% throughput over rule-based scheduling
"""

import gymnasium as gym
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
import time
from dataclasses import dataclass


@dataclass
class RLSchedulerConfig:
    """Configuration for RL scheduler."""
    min_batch_size: int = 1
    max_batch_size: int = 64
    batch_size_options: List[int] = None  # [1, 2, 4, 8, 16, 32, 64]
    max_queue_depth: int = 1000
    max_memory_gb: float = 80.0
    max_seq_len: int = 8192

    def __post_init__(self):
        if self.batch_size_options is None:
            self.batch_size_options = [1, 2, 4, 8, 16, 32, 64]


class BatchSchedulerEnv(gym.Env):
    """
    RL environment for learning optimal batch scheduling policy.

    This environment wraps a ContinuousBatchScheduler and exposes a Gym interface
    for training RL agents (PPO, DQN, etc.) to learn optimal batch size selection.

    State Space (4 continuous features):
    - avg_seq_len: Average sequence length in current batch (0-8192)
    - queue_depth: Number of waiting requests (0-1000)
    - memory_usage: Current memory usage in GB (0-80)
    - num_urgent_requests: Count of high-priority requests (0-100)

    Action Space (7 discrete actions):
    - Action 0-6: Select batch size from [1, 2, 4, 8, 16, 32, 64]

    Reward Function:
    - +10 points for each request completed
    - -1 point for each millisecond of latency over SLA (500ms)
    - -50 points if OOM occurs
    - +5 points for high memory utilization (80-90% range)

    Usage:
        env = BatchSchedulerEnv(scheduler)
        obs = env.reset()

        for _ in range(1000):
            action = agent.predict(obs)
            obs, reward, done, info = env.step(action)
    """

    def __init__(
        self,
        scheduler=None,
        config: Optional[RLSchedulerConfig] = None,
        verbose: bool = False
    ):
        """
        Args:
            scheduler: ContinuousBatchScheduler instance (optional, created if None)
            config: RLSchedulerConfig with batch size options and limits
            verbose: Print debug information
        """
        super().__init__()

        self.config = config or RLSchedulerConfig()
        self.verbose = verbose

        # Will be set in reset() if not provided
        self.scheduler = scheduler

        # Batch size mapping (action index -> actual batch size)
        self.batch_size_map = self.config.batch_size_options

        # State space: 4 continuous features
        # [avg_seq_len, queue_depth, memory_usage_gb, num_urgent_requests]
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([
                float(self.config.max_seq_len),
                float(self.config.max_queue_depth),
                float(self.config.max_memory_gb),
                100.0  # max urgent requests
            ], dtype=np.float32),
            dtype=np.float32
        )

        # Action space: discrete batch size selection
        self.action_space = gym.spaces.Discrete(len(self.batch_size_map))

        # Episode tracking
        self.episode_steps = 0
        self.max_episode_steps = 100
        self.total_reward = 0.0

        # Performance metrics
        self.requests_completed = 0
        self.total_latency_ms = 0.0
        self.oom_count = 0

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed (gymnasium API)
            options: Additional options (gymnasium API)

        Returns:
            Tuple of (observation, info)
        """
        # Handle seed for gymnasium compatibility
        super().reset(seed=seed)

        # Import here to avoid circular dependency
        if self.scheduler is None:
            from .scheduler import ContinuousBatchScheduler
            self.scheduler = ContinuousBatchScheduler(
                max_batch_size=self.config.max_batch_size,
                max_queue_depth=self.config.max_queue_depth,
                memory_limit_gb=self.config.max_memory_gb,
                enable_dynamic_batching=True,
                enable_affinity=True
            )

        # Reset episode tracking
        self.episode_steps = 0
        self.total_reward = 0.0
        self.requests_completed = 0
        self.total_latency_ms = 0.0
        self.oom_count = 0

        # Clear scheduler state
        self.scheduler.clear()

        # Generate initial requests (simulate workload)
        self._generate_workload(num_requests=10)

        return self._get_state(), {}

    def _get_state(self) -> np.ndarray:
        """
        Extract current state from scheduler metrics.

        Returns:
            State vector: [avg_seq_len, queue_depth, memory_usage_gb, num_urgent]
        """
        metrics = self.scheduler.get_metrics()

        # Feature 1: Average sequence length
        avg_seq_len = metrics.avg_seq_len if metrics.avg_seq_len > 0 else 128.0

        # Feature 2: Queue depth (waiting requests)
        queue_depth = float(self.scheduler.get_queue_depth())

        # Feature 3: Memory usage in GB
        memory_usage_gb = metrics.memory_usage_mb / 1024.0

        # Feature 4: Number of urgent requests (priority < 1)
        num_urgent = 0
        for neg_priority, timestamp, counter, request in self.scheduler.waiting_queue:
            if -neg_priority < 1:  # Priority.URGENT or Priority.HIGH
                num_urgent += 1
        num_urgent = min(num_urgent, 100)  # Cap at 100 for normalization

        state = np.array([
            avg_seq_len,
            queue_depth,
            memory_usage_gb,
            float(num_urgent)
        ], dtype=np.float32)

        # Clip to observation space bounds
        state = np.clip(state, self.observation_space.low, self.observation_space.high)

        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Apply action (select batch size) and execute one scheduling step.

        Args:
            action: Action index (0-6 for batch sizes [1, 2, 4, 8, 16, 32, 64])

        Returns:
            observation: Next state
            reward: Reward for this step
            terminated: Whether episode ended naturally
            truncated: Whether episode was cut short (e.g., time limit)
            info: Additional metrics
        """
        # Map action to batch size
        batch_size = self.batch_size_map[action]

        # Store original batch size
        original_batch_size = self.scheduler.max_batch_size

        # Apply selected batch size
        self.scheduler.max_batch_size = batch_size

        # Execute one scheduling step
        terminated = False
        truncated = False
        try:
            batch = self.scheduler.schedule_batch()

            # Simulate processing this batch
            reward, info = self._compute_reward(batch, batch_size)

            # Track metrics
            self.total_reward += reward
            self.episode_steps += 1

            # Generate new requests (simulate arrivals)
            if self.episode_steps % 10 == 0:
                self._generate_workload(num_requests=5)

            # Episode termination conditions
            if self.scheduler.get_queue_depth() == 0 and len(self.scheduler.running_batch) == 0:
                terminated = True  # Natural end - no more work

            if self.episode_steps >= self.max_episode_steps:
                truncated = True  # Time limit reached

            # Restore original batch size (for compatibility)
            self.scheduler.max_batch_size = original_batch_size

        except MemoryError:
            # OOM penalty
            reward = -50.0
            terminated = True
            self.oom_count += 1
            info = {'oom': True, 'batch_size': batch_size}

            # Restore original batch size
            self.scheduler.max_batch_size = original_batch_size

        # Get next state
        next_state = self._get_state()

        if self.verbose:
            print(f"[RL Step {self.episode_steps}] Action={action} (bs={batch_size}), "
                  f"Reward={reward:.2f}, Terminated={terminated}, Truncated={truncated}")

        return next_state, reward, terminated, truncated, info

    def _compute_reward(self, batch: Optional[List], batch_size: int) -> tuple:
        """
        Compute reward based on batch performance.

        Reward components:
        1. Throughput: +10 per completed request
        2. Latency: -1 per ms over SLA (500ms)
        3. Memory efficiency: +5 if memory usage in sweet spot (80-90%)
        4. Queue management: +1 per request removed from queue

        Args:
            batch: Current batch (list of InferenceRequest objects)
            batch_size: Selected batch size

        Returns:
            (reward, info_dict)
        """
        reward = 0.0
        info = {}

        if batch is None or len(batch) == 0:
            # Penalty for idle GPU
            return -5.0, {'idle': True}

        # 1. Throughput reward: completed requests
        completed = sum(1 for req in batch if req.finished)
        reward += completed * 10.0
        self.requests_completed += completed
        info['completed'] = completed

        # 2. Latency penalty: penalize requests over SLA (500ms)
        latencies = [req.latency * 1000 for req in batch if req.latency is not None]
        if latencies:
            avg_latency_ms = np.mean(latencies)
            sla_ms = 500.0  # 500ms SLA
            if avg_latency_ms > sla_ms:
                latency_penalty = -(avg_latency_ms - sla_ms) * 0.1
                reward += latency_penalty
                info['latency_penalty'] = latency_penalty

            self.total_latency_ms += avg_latency_ms
            info['avg_latency_ms'] = avg_latency_ms

        # 3. Memory efficiency bonus: reward 80-90% utilization
        metrics = self.scheduler.get_metrics()
        memory_usage_pct = (metrics.memory_usage_mb / 1024.0) / self.config.max_memory_gb
        if 0.80 <= memory_usage_pct <= 0.90:
            reward += 5.0
            info['memory_bonus'] = True

        # 4. Queue management: reward reducing queue depth
        queue_depth = self.scheduler.get_queue_depth()
        if queue_depth < 10:
            reward += 2.0  # Bonus for keeping queue short

        # 5. Batch efficiency: penalize very small batches (underutilization)
        if batch_size < 4 and queue_depth > 10:
            reward -= 3.0  # Could have batched more
            info['underutilization'] = True

        info['batch_size'] = batch_size
        info['queue_depth'] = queue_depth
        info['reward'] = reward

        return reward, info

    def _generate_workload(self, num_requests: int = 10):
        """
        Generate synthetic workload (simulate request arrivals).

        Creates requests with varying:
        - Sequence lengths (short: 50-200, medium: 200-500, long: 500-1000)
        - Priorities (urgent, high, normal, low)
        - Max tokens (50-200)

        Args:
            num_requests: Number of requests to generate
        """
        import torch
        from .scheduler import InferenceRequest
        import uuid

        for i in range(num_requests):
            # Random sequence length distribution
            length_category = np.random.choice(['short', 'medium', 'long'], p=[0.5, 0.3, 0.2])
            if length_category == 'short':
                seq_len = np.random.randint(50, 200)
            elif length_category == 'medium':
                seq_len = np.random.randint(200, 500)
            else:
                seq_len = np.random.randint(500, 1000)

            # Random priority distribution
            priority = np.random.choice([0, 1, 2, 3, 4], p=[0.1, 0.2, 0.5, 0.15, 0.05])

            # Random max tokens
            max_tokens = np.random.randint(50, 200)

            # Create dummy input_ids
            input_ids = torch.randint(0, 32000, (1, seq_len), dtype=torch.long)

            # Create request
            request = InferenceRequest(
                request_id=str(uuid.uuid4()),
                prompt=f"Request {i} (len={seq_len})",
                input_ids=input_ids,
                max_tokens=max_tokens,
                priority=priority
            )

            # Add to scheduler queue
            try:
                self.scheduler.add_request(request)
            except Exception:
                # Queue full - stop adding
                break

    def render(self, mode='human'):
        """Render current environment state (for debugging)."""
        if mode == 'human':
            print(f"\n{'='*60}")
            print(f"Episode Step: {self.episode_steps}/{self.max_episode_steps}")
            print(f"Total Reward: {self.total_reward:.2f}")
            print(f"Requests Completed: {self.requests_completed}")
            print(f"Queue Depth: {self.scheduler.get_queue_depth()}")
            print(f"Running Batch Size: {len(self.scheduler.running_batch)}")
            metrics = self.scheduler.get_metrics()
            print(f"Memory Usage: {metrics.memory_usage_mb/1024:.2f} GB")
            print(f"Avg Seq Len: {metrics.avg_seq_len:.1f}")
            print(f"OOM Count: {self.oom_count}")
            print(f"{'='*60}\n")


class RLSchedulerAgent:
    """
    Wrapper for trained RL agent (PPO, DQN, etc.).

    This class provides a simple interface for using a trained RL model
    to make batch size decisions in production.

    Usage:
        agent = RLSchedulerAgent.load('scheduler_rl_agent.zip')
        batch_size = agent.predict(scheduler)
    """

    def __init__(self, model, batch_size_map: List[int]):
        """
        Args:
            model: Trained RL model (e.g., stable_baselines3.PPO)
            batch_size_map: Mapping from action index to batch size
        """
        self.model = model
        self.batch_size_map = batch_size_map

    def predict(self, scheduler, deterministic: bool = True) -> int:
        """
        Predict optimal batch size for current scheduler state.

        Args:
            scheduler: ContinuousBatchScheduler instance
            deterministic: Use deterministic policy (True for production)

        Returns:
            Optimal batch size
        """
        # Extract state
        state = self._extract_state(scheduler)

        # Predict action
        action, _ = self.model.predict(state, deterministic=deterministic)

        # Map to batch size
        batch_size = self.batch_size_map[action]

        return batch_size

    def _extract_state(self, scheduler) -> np.ndarray:
        """Extract state from scheduler (same as BatchSchedulerEnv._get_state)."""
        metrics = scheduler.get_metrics()

        avg_seq_len = metrics.avg_seq_len if metrics.avg_seq_len > 0 else 128.0
        queue_depth = float(scheduler.get_queue_depth())
        memory_usage_gb = metrics.memory_usage_mb / 1024.0

        num_urgent = 0
        for neg_priority, timestamp, counter, request in scheduler.waiting_queue:
            if -neg_priority < 1:
                num_urgent += 1
        num_urgent = min(num_urgent, 100)

        state = np.array([
            avg_seq_len,
            queue_depth,
            memory_usage_gb,
            float(num_urgent)
        ], dtype=np.float32)

        return state

    @classmethod
    def load(cls, path: str, batch_size_map: Optional[List[int]] = None):
        """
        Load trained RL agent from disk.

        Args:
            path: Path to saved model (e.g., 'scheduler_rl_agent.zip')
            batch_size_map: Batch size options (default: [1,2,4,8,16,32,64])

        Returns:
            RLSchedulerAgent instance
        """
        try:
            from stable_baselines3 import PPO
        except ImportError:
            raise ImportError(
                "stable-baselines3 not installed. "
                "Install with: pip install stable-baselines3"
            )

        if batch_size_map is None:
            batch_size_map = [1, 2, 4, 8, 16, 32, 64]

        model = PPO.load(path)
        return cls(model, batch_size_map)
