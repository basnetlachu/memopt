"""
Memopt - GPU Memory Bandwidth Optimization Engine for LLM Inference

Reduces memory bandwidth usage by 40-60% through:
- INT8 KV cache quantization
- Page-based KV cache management
- Memory-efficient attention (FlashAttention-2)
- Continuous batching with memory awareness
- Real-time profiling and cost tracking

On-Premise vLLM Plugin Usage:
    # Set environment variables before running vLLM
    export MEMOPT_ENABLED=1
    export MEMOPT_LICENSE_PATH=/etc/memopt/license.json

    # Run vLLM normally - MemOpt auto-attaches
    python -m vllm.entrypoints.openai.api_server --model /models/llama

Standalone Usage:
    from Memopt import OptimizedLLM

    model = OptimizedLLM(
        model="meta-llama/Llama-2-13b-hf",
        optimization_level="high"
    )

    response = model.generate("Your prompt here", max_tokens=512)
"""

__version__ = "0.1.0"
__author__ = "Memopt Team"

import os
from .model import OptimizedLLM
from .profiler import MemoryProfiler, ProfileStats

__all__ = [
    "OptimizedLLM",
    "MemoryProfiler",
    "ProfileStats",
    "enable_vllm"
]

# Auto-initialize vLLM plugin if enabled
if os.getenv("MEMOPT_ENABLED") == "1":
    try:
        from .vllm_plugin import initialize_plugin
        initialize_plugin()
    except Exception as e:
        # Fail silently unless strict mode
        if os.getenv("MEMOPT_STRICT") == "1":
            raise
        import warnings
        warnings.warn(f"MemOpt auto-initialization failed: {e}", UserWarning)

# Export enable_vllm for manual initialization
def enable_vllm():
    """
    Manually enable MemOpt vLLM plugin.

    Returns:
        bool: True if plugin enabled successfully
    """
    from .vllm_plugin import enable_vllm as _enable
    return _enable()
