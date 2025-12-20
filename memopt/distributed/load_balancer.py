"""
Load balancer for multi-GPU inference

PURPOSE: Distribute requests optimally across GPUs
WHY: Maximize throughput and minimize latency
STRATEGIES: Round-robin, least-loaded, adaptive
"""

import threading
import time
from typing import Dict, Optional
from collections import deque

from memopt.monitoring.logger import get_logger

logger = get_logger(__name__)


class LoadBalancer:
    """
    Load balancer for distributing work across GPUs
    
    Strategies:
    - round_robin: Simple round-robin distribution
    - least_loaded: Send to GPU with fewest active requests
    - adaptive: Learn from past performance
    """
    
    def __init__(
        self,
        num_workers: int,
        strategy: str = "round_robin"
    ):
        """
        Initialize load balancer
        
        Args:
            num_workers: Number of workers (GPUs)
            strategy: Load balancing strategy
        """
        self.num_workers = num_workers
        self.strategy = strategy
        
        # Round-robin state
        self.current_worker = 0
        self.rr_lock = threading.Lock()
        
        # Least-loaded state
        self.active_requests = [0] * num_workers
        self.ll_lock = threading.Lock()
        
        # Adaptive state
        self.worker_times = [deque(maxlen=100) for _ in range(num_workers)]
        self.worker_failures = [0] * num_workers
        
        # Statistics
        self.total_requests = 0
        self.worker_request_counts = [0] * num_workers
        
        logger.info(f"Load balancer initialized: {strategy} strategy, {num_workers} workers")
    
    def get_next_worker(self) -> int:
        """
        Get next worker ID based on strategy
        
        Returns:
            Worker ID (GPU ID)
        """
        if self.strategy == "round_robin":
            return self._round_robin()
        elif self.strategy == "least_loaded":
            return self._least_loaded()
        elif self.strategy == "adaptive":
            return self._adaptive()
        else:
            logger.warning(f"Unknown strategy {self.strategy}, using round_robin")
            return self._round_robin()
    
    def _round_robin(self) -> int:
        """Round-robin worker selection"""
        with self.rr_lock:
            worker_id = self.current_worker
            self.current_worker = (self.current_worker + 1) % self.num_workers
            self.worker_request_counts[worker_id] += 1
            self.total_requests += 1
            return worker_id
    
    def _least_loaded(self) -> int:
        """Select worker with fewest active requests"""
        with self.ll_lock:
            # Find worker with minimum active requests
            worker_id = min(
                range(self.num_workers),
                key=lambda i: self.active_requests[i]
            )
            
            self.active_requests[worker_id] += 1
            self.worker_request_counts[worker_id] += 1
            self.total_requests += 1
            
            logger.debug(f"Selected worker {worker_id} (active: {self.active_requests[worker_id]})")
            return worker_id
    
    def _adaptive(self) -> int:
        """
        Adaptive worker selection based on past performance
        
        Considers:
        - Average completion time
        - Failure rate
        - Current load
        """
        with self.ll_lock:
            scores = []
            
            for i in range(self.num_workers):
                # Calculate average time (lower is better)
                if len(self.worker_times[i]) > 0:
                    avg_time = sum(self.worker_times[i]) / len(self.worker_times[i])
                else:
                    avg_time = 1.0  # Default
                
                # Failure penalty (more failures = higher penalty)
                failure_penalty = 1.0 + (self.worker_failures[i] * 0.1)
                
                # Load penalty (more active requests = higher penalty)
                load_penalty = 1.0 + (self.active_requests[i] * 0.2)
                
                # Combined score (lower is better)
                score = avg_time * failure_penalty * load_penalty
                scores.append(score)
            
            # Select worker with best (lowest) score
            worker_id = min(range(self.num_workers), key=lambda i: scores[i])
            
            self.active_requests[worker_id] += 1
            self.worker_request_counts[worker_id] += 1
            self.total_requests += 1
            
            logger.debug(f"Adaptive selected worker {worker_id} (score: {scores[worker_id]:.3f})")
            return worker_id
    
    def report_completion(self, worker_id: int, completion_time: float):
        """
        Report request completion
        
        Args:
            worker_id: Worker that completed the request
            completion_time: Time taken in seconds
        """
        with self.ll_lock:
            # Decrease active request count
            if self.active_requests[worker_id] > 0:
                self.active_requests[worker_id] -= 1
            
            # Record completion time for adaptive strategy
            self.worker_times[worker_id].append(completion_time)
    
    def report_failure(self, worker_id: int):
        """
        Report request failure
        
        Args:
            worker_id: Worker that failed the request
        """
        with self.ll_lock:
            # Decrease active request count
            if self.active_requests[worker_id] > 0:
                self.active_requests[worker_id] -= 1
            
            # Record failure
            self.worker_failures[worker_id] += 1
            
            logger.warning(f"Worker {worker_id} failure recorded (total: {self.worker_failures[worker_id]})")
    
    def get_stats(self) -> Dict:
        """Get load balancing statistics"""
        with self.ll_lock:
            stats = {
                'strategy': self.strategy,
                'total_requests': self.total_requests,
                'workers': []
            }
            
            for i in range(self.num_workers):
                worker_stats = {
                    'worker_id': i,
                    'requests_handled': self.worker_request_counts[i],
                    'active_requests': self.active_requests[i],
                    'failures': self.worker_failures[i]
                }
                
                # Add timing stats if available
                if len(self.worker_times[i]) > 0:
                    worker_stats['avg_time'] = sum(self.worker_times[i]) / len(self.worker_times[i])
                    worker_stats['min_time'] = min(self.worker_times[i])
                    worker_stats['max_time'] = max(self.worker_times[i])
                
                stats['workers'].append(worker_stats)
            
            return stats
    
    def reset_stats(self):
        """Reset all statistics"""
        with self.ll_lock:
            self.total_requests = 0
            self.worker_request_counts = [0] * self.num_workers
            self.active_requests = [0] * self.num_workers
            self.worker_times = [deque(maxlen=100) for _ in range(self.num_workers)]
            self.worker_failures = [0] * self.num_workers
        
        logger.info("Load balancer statistics reset")