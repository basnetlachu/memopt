"""
MemOpt - GPU Memory Bandwidth Optimization Engine for LLM Inference

Reduces memory bandwidth usage by 40-60% through:
- INT8 KV cache quantization
- Page-based KV cache management
- Memory-efficient attention (FlashAttention-2)
- Continuous batching with memory awareness
- Real-time profiling and cost tracking

Usage:
    from memopt import OptimizedLLM
    
    model = OptimizedLLM(
        model="meta-llama/Llama-2-13b-hf",
        optimization_level="high"
    )
    
    response = model.generate("Your prompt here", max_tokens=512)
"""

__version__ = "0.1.0"
__author__ = "MemOpt Team"

from .model import OptimizedLLM
from .profiler import MemoryProfiler, ProfileStats

__all__ = [
    "OptimizedLLM",
    "MemoryProfiler",
    "ProfileStats"
]




