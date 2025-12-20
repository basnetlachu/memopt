"""
Multi-GPU support for MemOpt

PURPOSE: Distribute inference across multiple GPUs
WHY: Scale to datacenter workloads
WHEN: Customer has 10+ GPUs

Features:
- Model parallelism (split model across GPUs)
- Data parallelism (different requests on different GPUs)
- Automatic load balancing
- GPU health monitoring

Usage:
    from memopt.distributed import MultiGPUManager
    
    manager = MultiGPUManager(
        model_name="gpt2-xl",
        num_gpus=4
    )
    
    output = manager.generate("Hello", max_tokens=100)
"""

import torch
from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import time

from memopt.core.model import OptimizedLLM
from memopt.monitoring.logger import get_logger
from memopt.utils.errors import DistributedError
from .load_balancer import LoadBalancer

logger = get_logger(__name__)


class MultiGPUManager:
    """
    Manages multiple GPU instances for parallel inference
    
    Strategies:
    - round_robin: Distribute requests evenly
    - least_loaded: Send to GPU with lowest load
    - dedicated: Dedicate GPUs to specific request types
    """
    
    def __init__(
        self,
        model_name: str,
        num_gpus: Optional[int] = None,
        optimization_level: str = "high",
        strategy: str = "round_robin"
    ):
        """
        Initialize multi-GPU manager
        
        Args:
            model_name: Model to load
            num_gpus: Number of GPUs to use (None = use all available)
            optimization_level: Optimization level
            strategy: Load balancing strategy
        """
        logger.info("="*70)
        logger.info("Initializing Multi-GPU Manager")
        logger.info("="*70)
        
        # Detect available GPUs
        if not torch.cuda.is_available():
            raise DistributedError("CUDA not available")
        
        available_gpus = torch.cuda.device_count()
        
        if num_gpus is None:
            num_gpus = available_gpus
        elif num_gpus > available_gpus:
            logger.warning(
                f"Requested {num_gpus} GPUs but only {available_gpus} available. "
                f"Using {available_gpus} GPUs."
            )
            num_gpus = available_gpus
        
        self.num_gpus = num_gpus
        self.model_name = model_name
        self.optimization_level = optimization_level
        
        logger.info(f"Available GPUs: {available_gpus}")
        logger.info(f"Using GPUs: {num_gpus}")
        logger.info(f"Strategy: {strategy}")
        
        # Initialize models on each GPU
        self.models: List[OptimizedLLM] = []
        self._load_models()
        
        # Initialize load balancer
        self.load_balancer = LoadBalancer(
            num_workers=num_gpus,
            strategy=strategy
        )
        
        # Request queue
        self.request_queue = Queue()
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'gpu_requests': [0] * num_gpus,
            'gpu_times': [0.0] * num_gpus
        }
        self.stats_lock = threading.Lock()
        
        logger.info("✓ Multi-GPU manager initialized")
    
    def _load_models(self):
        """Load model on each GPU"""
        logger.info("Loading models on GPUs...")
        
        for gpu_id in range(self.num_gpus):
            logger.info(f"Loading model on GPU {gpu_id}...")
            
            try:
                model = OptimizedLLM(
                    model=self.model_name,
                    device=f"cuda:{gpu_id}",
                    optimization_level=self.optimization_level,
                    enable_profiling=True
                )
                self.models.append(model)
                logger.info(f"✓ Model loaded on GPU {gpu_id}")
                
            except Exception as e:
                logger.error(f"Failed to load model on GPU {gpu_id}: {e}")
                raise DistributedError(f"Failed to load model on GPU {gpu_id}: {e}")
        
        logger.info(f"✓ All {self.num_gpus} models loaded")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        **kwargs
    ) -> str:
        """
        Generate text using load-balanced GPU selection
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation arguments
            
        Returns:
            Generated text
        """
        # Select GPU
        gpu_id = self.load_balancer.get_next_worker()
        
        logger.debug(f"Routing request to GPU {gpu_id}")
        
        # Generate on selected GPU
        start_time = time.time()
        
        try:
            output = self.models[gpu_id].generate(
                prompt=prompt,
                max_tokens=max_tokens,
                **kwargs
            )
            
            generation_time = time.time() - start_time
            
            # Update statistics
            with self.stats_lock:
                self.stats['total_requests'] += 1
                self.stats['gpu_requests'][gpu_id] += 1
                self.stats['gpu_times'][gpu_id] += generation_time
            
            # Report load back to balancer
            self.load_balancer.report_completion(gpu_id, generation_time)
            
            return output
            
        except Exception as e:
            logger.error(f"Generation failed on GPU {gpu_id}: {e}")
            # Report failure
            self.load_balancer.report_failure(gpu_id)
            raise
    
    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 256,
        **kwargs
    ) -> List[str]:
        """
        Generate text for multiple prompts in parallel
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate per prompt
            **kwargs: Additional generation arguments
            
        Returns:
            List of generated texts (same order as input)
        """
        logger.info(f"Processing batch of {len(prompts)} prompts across {self.num_gpus} GPUs")
        
        results = [None] * len(prompts)
        
        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(
                    self.generate,
                    prompt,
                    max_tokens,
                    **kwargs
                ): idx
                for idx, prompt in enumerate(prompts)
            }
            
            # Collect results
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Batch item {idx} failed: {e}")
                    results[idx] = f"[ERROR: {str(e)}]"
        
        logger.info("✓ Batch processing complete")
        return results
    
    def get_gpu_stats(self) -> Dict:
        """Get statistics for all GPUs"""
        with self.stats_lock:
            stats = {
                'total_requests': self.stats['total_requests'],
                'gpus': []
            }
            
            for gpu_id in range(self.num_gpus):
                gpu_requests = self.stats['gpu_requests'][gpu_id]
                gpu_time = self.stats['gpu_times'][gpu_id]
                avg_time = gpu_time / gpu_requests if gpu_requests > 0 else 0
                
                # Get memory stats
                memory_stats = self.models[gpu_id].get_memory_stats()
                
                stats['gpus'].append({
                    'gpu_id': gpu_id,
                    'requests': gpu_requests,
                    'total_time': gpu_time,
                    'avg_time': avg_time,
                    'memory_allocated_gb': memory_stats['allocated_gb'],
                    'memory_utilization': memory_stats['utilization']
                })
            
            return stats
    
    def get_load_balance_stats(self) -> Dict:
        """Get load balancing statistics"""
        return self.load_balancer.get_stats()
    
    def reset_stats(self):
        """Reset all statistics"""
        with self.stats_lock:
            self.stats['total_requests'] = 0
            self.stats['gpu_requests'] = [0] * self.num_gpus
            self.stats['gpu_times'] = [0.0] * self.num_gpus
        
        self.load_balancer.reset_stats()
        logger.info("Statistics reset")