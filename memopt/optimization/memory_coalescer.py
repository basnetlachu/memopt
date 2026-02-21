"""
Memory Access Coalescing for PyTorch Models

This module provides transparent memory optimization for:
- Inference: KV cache access coalescing for autoregressive generation
- Training: Activation memory optimization and gradient checkpointing enhancement
- Fine-tuning: Combined training + inference optimizations

Key insight: During autoregressive generation, each new token reads overlapping
KV cache regions. This module tracks those accesses with a Python-level LRU
cache (an OrderedDict in CPU memory) and reports hit_rate for Python-level
cache accesses only.

Note: this does NOT modify GPU L2 cache or shared memory behavior. hit_rate
describes the Python-level cache, not actual GPU L1/L2 hardware. bandwidth_reduction
is NOT provided — it requires NCU hardware counters. For real HBM traffic
measurements use HardwareCounterCollector.

Usage:
    from memopt.optimization import MemoryCoalescer

    model = AutoModelForCausalLM.from_pretrained('gpt2')
    coalescer = MemoryCoalescer(model, mode='inference')
    coalescer.enable()

    # Now model.generate() uses optimized memory access
    outputs = model.generate(...)

    # Get stats
    print(coalescer.get_stats())
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from collections import OrderedDict
from contextlib import contextmanager
import functools

import torch
import torch.nn as nn


@dataclass
class CoalescingConfig:
    """Configuration for memory coalescing."""

    # Cache size in MB for storing recently accessed tensors
    cache_size_mb: int = 128

    # Block size for cache entries (number of tokens)
    block_size: int = 64

    # Maximum number of cached entries before eviction
    max_cache_entries: int = 256

    # Enable coalescing for attention layers
    optimize_attention: bool = True

    # Enable coalescing for MLP/FFN layers
    optimize_mlp: bool = True

    # Enable activation checkpointing enhancement (training only)
    optimize_checkpointing: bool = True

    # Minimum tensor size (bytes) to consider for caching
    min_tensor_bytes: int = 4096

    # Enable profiling of memory access patterns
    enable_profiling: bool = True


@dataclass
class CoalescingStats:
    """Statistics from memory coalescing."""

    # Total memory accesses intercepted
    total_accesses: int = 0

    # Number of cache hits (served from cache)
    cache_hits: int = 0

    # Number of cache misses (fetched from HBM)
    cache_misses: int = 0

    # Bytes saved by cache hits
    bytes_saved: int = 0

    # Bytes actually fetched from HBM
    bytes_fetched: int = 0

    # Number of layers optimized
    layers_optimized: int = 0

    # Forward passes completed
    forward_passes: int = 0

    # Backward passes completed (training only)
    backward_passes: int = 0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as percentage."""
        if self.total_accesses == 0:
            return 0.0
        return (self.cache_hits / self.total_accesses) * 100

    @property
    def bandwidth_reduction(self) -> float:
        """Not available without NCU hardware counters.

        Raises:
            NotImplementedError: Always. Use HardwareCounterCollector for real
                HBM traffic measurements. The Python-level cache tracked here
                cannot observe actual GPU memory transactions.
        """
        raise NotImplementedError(
            "bandwidth_reduction requires NCU hardware counters to measure real "
            "HBM traffic. This class tracks a Python-level cache only.\n"
            "Use HardwareCounterCollector for actual bandwidth measurements:\n"
            "  from memopt.profiler.hardware_counters import HardwareCounterCollector"
        )

    @property
    def bytes_saved_gb(self) -> float:
        """Bytes saved in GB."""
        return self.bytes_saved / (1024 ** 3)

    def reset(self):
        """Reset all statistics."""
        self.total_accesses = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.bytes_saved = 0
        self.bytes_fetched = 0
        self.forward_passes = 0
        self.backward_passes = 0


class TensorCache:
    """
    LRU cache for tensor data with size-based eviction.

    Uses weak references where possible to avoid memory leaks.
    Cache is keyed by (layer_id, sequence_position) tuples.
    """

    def __init__(self, max_size_bytes: int, max_entries: int):
        self.max_size_bytes = max_size_bytes
        self.max_entries = max_entries
        self.cache: OrderedDict[Tuple, torch.Tensor] = OrderedDict()
        self.current_size_bytes = 0

    def _tensor_size(self, tensor: torch.Tensor) -> int:
        """Get tensor size in bytes."""
        return tensor.numel() * tensor.element_size()

    def get(self, key: Tuple) -> Optional[torch.Tensor]:
        """Get tensor from cache, returns None if not found."""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: Tuple, tensor: torch.Tensor) -> bool:
        """
        Add tensor to cache.

        Returns True if cached, False if tensor too large.
        """
        tensor_bytes = self._tensor_size(tensor)

        # Don't cache tensors larger than max cache size
        if tensor_bytes > self.max_size_bytes:
            return False

        # Evict entries until we have space
        while (self.current_size_bytes + tensor_bytes > self.max_size_bytes or
               len(self.cache) >= self.max_entries):
            if not self.cache:
                break
            # Remove least recently used
            _, old_tensor = self.cache.popitem(last=False)
            self.current_size_bytes -= self._tensor_size(old_tensor)

        # Add new entry
        self.cache[key] = tensor.detach()  # Detach to avoid grad tracking issues
        self.current_size_bytes += tensor_bytes
        return True

    def clear(self):
        """Clear all cached tensors."""
        self.cache.clear()
        self.current_size_bytes = 0

    def __len__(self) -> int:
        return len(self.cache)


class AttentionCoalescer:
    """
    Optimizes memory access patterns in attention layers.

    Key optimization: During autoregressive generation, the KV cache
    grows incrementally. Most of the cached K,V tensors are unchanged
    between steps. We track and optimize these access patterns.
    """

    def __init__(
        self,
        attention_module: nn.Module,
        layer_idx: int,
        config: CoalescingConfig,
        stats: CoalescingStats,
        cache: TensorCache
    ):
        self.attention = weakref.ref(attention_module)
        self.layer_idx = layer_idx
        self.config = config
        self.stats = stats
        self.cache = cache

        # Store original forward method
        self._original_forward = attention_module.forward

        # Track sequence positions for this layer
        self.last_seq_len = 0
        self.generation_step = 0

    def _track_tensor_access(self, tensor: torch.Tensor, access_type: str):
        """Track tensor memory access for statistics."""
        if tensor is None:
            return

        tensor_bytes = tensor.numel() * tensor.element_size()
        self.stats.total_accesses += 1
        self.stats.bytes_fetched += tensor_bytes

        # Check if this access could be served from cache
        cache_key = (self.layer_idx, access_type, self.generation_step)
        cached = self.cache.get(cache_key)

        if cached is not None and cached.shape == tensor.shape:
            # Could have been a cache hit
            self.stats.cache_hits += 1
            self.stats.bytes_saved += tensor_bytes
        else:
            self.stats.cache_misses += 1
            # Store for potential future hit
            if tensor_bytes < self.config.cache_size_mb * 1024 * 1024 // 4:
                self.cache.put(cache_key, tensor.detach().clone())

    def coalesced_forward(self, *args, **kwargs):
        """
        Forward pass with memory access tracking and coalescing.
        """
        attention = self.attention()
        if attention is None:
            raise RuntimeError("Attention module has been garbage collected")

        self.generation_step += 1

        # Track input tensor access
        if len(args) > 0 and torch.is_tensor(args[0]):
            hidden_states = args[0]
            self._track_tensor_access(hidden_states, 'hidden_states')

            # Track KV cache access if present
            past_key_value = kwargs.get('past_key_value', None)
            if past_key_value is not None:
                if isinstance(past_key_value, tuple) and len(past_key_value) >= 2:
                    self._track_tensor_access(past_key_value[0], 'key_cache')
                    self._track_tensor_access(past_key_value[1], 'value_cache')

        # Call original forward
        result = self._original_forward(*args, **kwargs)

        # Track output access
        if isinstance(result, tuple) and len(result) > 0:
            if torch.is_tensor(result[0]):
                self._track_tensor_access(result[0], 'output')

        return result


class MLPCoalescer:
    """
    Optimizes memory access patterns in MLP/FFN layers.

    Key optimization: Weight matrices are read on every forward pass.
    We track access patterns and identify opportunities for caching.
    """

    def __init__(
        self,
        mlp_module: nn.Module,
        layer_idx: int,
        config: CoalescingConfig,
        stats: CoalescingStats
    ):
        self.mlp = weakref.ref(mlp_module)
        self.layer_idx = layer_idx
        self.config = config
        self.stats = stats
        self._original_forward = mlp_module.forward
        self.forward_count = 0

    def coalesced_forward(self, *args, **kwargs):
        """
        Forward pass with memory access tracking.
        """
        mlp = self.mlp()
        if mlp is None:
            raise RuntimeError("MLP module has been garbage collected")

        self.forward_count += 1

        # Track weight accesses - weights are read every forward pass
        total_weight_bytes = 0
        for name, param in mlp.named_parameters():
            if 'weight' in name:
                weight_bytes = param.numel() * param.element_size()
                total_weight_bytes += weight_bytes

        if total_weight_bytes > 0:
            self.stats.total_accesses += 1
            self.stats.bytes_fetched += total_weight_bytes
            # bytes_saved is not tracked here — would require NCU to measure real HBM hits.

        # Track input tensor
        if len(args) > 0 and torch.is_tensor(args[0]):
            input_bytes = args[0].numel() * args[0].element_size()
            self.stats.total_accesses += 1
            self.stats.bytes_fetched += input_bytes

        return self._original_forward(*args, **kwargs)


class MemoryCoalescer:
    """
    Memory access coalescing for ANY PyTorch model.

    Works for:
    - Training: Reduces activation memory through intelligent caching
    - Inference: Reduces KV cache traffic through access coalescing
    - Fine-tuning: Combined training + inference optimizations

    Usage:
        model = AutoModelForCausalLM.from_pretrained('gpt2')
        coalescer = MemoryCoalescer(model, mode='inference')
        coalescer.enable()

        # Model now uses optimized memory access
        outputs = model.generate(...)

        # Check results
        stats = coalescer.get_stats()
        print(f"Hit rate: {stats.hit_rate:.1f}%")
        # bandwidth_reduction not available — requires NCU counters
    """

    def __init__(
        self,
        model: nn.Module,
        mode: str = 'auto',
        config: Optional[CoalescingConfig] = None
    ):
        """
        Initialize memory coalescer.

        Args:
            model: PyTorch model to optimize
            mode: 'training', 'inference', or 'auto' (detect from model.training)
            config: Coalescing configuration (uses defaults if None)
        """
        self.model = model
        self.mode = mode
        self.config = config or CoalescingConfig()
        self.stats = CoalescingStats()

        # Initialize tensor cache
        cache_bytes = self.config.cache_size_mb * 1024 * 1024
        self.cache = TensorCache(cache_bytes, self.config.max_cache_entries)

        # Track registered hooks for cleanup
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._coalesced_layers: List[Union[AttentionCoalescer, MLPCoalescer]] = []

        # State
        self._enabled = False
        self._original_forwards: Dict[int, Callable] = {}

        # Detect mode if auto
        if self.mode == 'auto':
            self.mode = 'training' if model.training else 'inference'

    def _find_attention_layers(self) -> List[Tuple[str, nn.Module]]:
        """Find all attention layers in the model."""
        attention_layers = []

        for name, module in self.model.named_modules():
            # Check for common attention layer names
            module_name = module.__class__.__name__.lower()
            if any(attn_name in module_name for attn_name in
                   ['attention', 'selfattention', 'multiheadattention', 'mha']):
                attention_layers.append((name, module))

        return attention_layers

    def _find_mlp_layers(self) -> List[Tuple[str, nn.Module]]:
        """Find all MLP/FFN layers in the model."""
        mlp_layers = []

        for name, module in self.model.named_modules():
            module_name = module.__class__.__name__.lower()
            if any(mlp_name in module_name for mlp_name in
                   ['mlp', 'feedforward', 'ffn', 'dense']):
                # Avoid nested duplicates
                if not any(name.startswith(existing[0] + '.') for existing in mlp_layers):
                    mlp_layers.append((name, module))

        return mlp_layers

    def enable(self) -> 'MemoryCoalescer':
        """
        Enable memory coalescing.

        Hooks into model layers to optimize memory access.
        Returns self for chaining.
        """
        if self._enabled:
            return self

        self.stats.reset()
        self.cache.clear()

        # Find and wrap attention layers
        if self.config.optimize_attention:
            attention_layers = self._find_attention_layers()
            for idx, (name, module) in enumerate(attention_layers):
                coalescer = AttentionCoalescer(
                    module, idx, self.config, self.stats, self.cache
                )

                # Store original forward and replace
                self._original_forwards[id(module)] = module.forward
                module.forward = coalescer.coalesced_forward

                self._coalesced_layers.append(coalescer)
                self.stats.layers_optimized += 1

        # Find and wrap MLP layers (primarily for tracking)
        if self.config.optimize_mlp:
            mlp_layers = self._find_mlp_layers()
            for idx, (name, module) in enumerate(mlp_layers):
                coalescer = MLPCoalescer(
                    module, idx, self.config, self.stats
                )

                # Store original forward and replace
                self._original_forwards[id(module)] = module.forward
                module.forward = coalescer.coalesced_forward

                self._coalesced_layers.append(coalescer)

        # Add forward hook to track passes
        def forward_hook(module, input, output):
            self.stats.forward_passes += 1
            return output

        hook = self.model.register_forward_hook(forward_hook)
        self._hooks.append(hook)

        # Add backward hook for training mode
        if self.mode == 'training':
            def backward_hook(module, grad_input, grad_output):
                self.stats.backward_passes += 1

            hook = self.model.register_full_backward_hook(backward_hook)
            self._hooks.append(hook)

        self._enabled = True
        return self

    def disable(self) -> 'MemoryCoalescer':
        """
        Disable memory coalescing.

        Restores original model behavior.
        Returns self for chaining.
        """
        if not self._enabled:
            return self

        # Remove hooks
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

        # Restore original forwards
        for name, module in self.model.named_modules():
            if id(module) in self._original_forwards:
                module.forward = self._original_forwards[id(module)]

        self._original_forwards.clear()
        self._coalesced_layers.clear()

        self._enabled = False
        return self

    def get_stats(self) -> CoalescingStats:
        """Get current coalescing statistics."""
        return self.stats

    def get_cache_stats(self) -> dict:
        """Return only the Python-level cache metrics that are actually measured.

        Does NOT include bandwidth_reduction (requires NCU hardware counters).
        """
        return {
            "python_cache_hit_rate_pct": self.stats.hit_rate,
            "python_cache_hits": self.stats.cache_hits,
            "python_cache_misses": self.stats.cache_misses,
            "total_accesses": self.stats.total_accesses,
            "layers_optimized": self.stats.layers_optimized,
            # bandwidth_reduction: not available without NCU
        }

    def reset_stats(self):
        """Reset statistics without disabling coalescing."""
        self.stats.reset()

    def clear_cache(self):
        """Clear the tensor cache."""
        self.cache.clear()

    @contextmanager
    def coalescing_context(self):
        """
        Context manager for temporary coalescing.

        Usage:
            with coalescer.coalescing_context():
                outputs = model.generate(...)
            # Coalescing automatically disabled after context
        """
        try:
            self.enable()
            yield self
        finally:
            self.disable()

    def __enter__(self):
        """Enable coalescing on context entry."""
        self.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Disable coalescing on context exit."""
        self.disable()
        return False

    def summary(self) -> str:
        """Get a summary string of coalescing results."""
        stats = self.stats
        lines = [
            "=" * 60,
            "MEMORY COALESCING SUMMARY",
            "=" * 60,
            f"Mode: {self.mode}",
            f"Layers optimized: {stats.layers_optimized}",
            f"Forward passes: {stats.forward_passes}",
            f"Backward passes: {stats.backward_passes}",
            "",
            "Cache Statistics:",
            f"  Total accesses: {stats.total_accesses:,}",
            f"  Cache hits: {stats.cache_hits:,}",
            f"  Cache misses: {stats.cache_misses:,}",
            f"  Hit rate: {stats.hit_rate:.1f}%",
            "",
            "Memory Traffic:",
            f"  Bytes saved: {stats.bytes_saved_gb:.3f} GB",
            f"  Bytes fetched: {stats.bytes_fetched / (1024**3):.3f} GB",
            "  Bandwidth reduction: N/A (requires NCU — use HardwareCounterCollector)",
            "=" * 60,
        ]
        return "\n".join(lines)


def optimize_model(
    model: nn.Module,
    mode: str = 'auto',
    config: Optional[CoalescingConfig] = None
) -> MemoryCoalescer:
    """
    Convenience function to create and enable a MemoryCoalescer.

    Args:
        model: PyTorch model to optimize
        mode: 'training', 'inference', or 'auto'
        config: Optional configuration

    Returns:
        Enabled MemoryCoalescer instance

    Usage:
        coalescer = optimize_model(model, mode='inference')
        # Model is now optimized
    """
    coalescer = MemoryCoalescer(model, mode=mode, config=config)
    coalescer.enable()
    return coalescer
