"""
Real-Time GPU Memory Profiler

Tracks and reports:
1. Memory bandwidth utilization
2. GPU compute utilization vs stall time
3. Tokens per second
4. Cost per 1M tokens
5. Before/after comparison

This is the sales enabler - customers need to see concrete numbers.
"""

import torch
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import json


@dataclass
class ProfileStats:
    """Statistics from profiling session."""
    
    # Performance metrics
    total_tokens_generated: int = 0
    total_time_seconds: float = 0.0
    tokens_per_second: float = 0.0
    latency_per_token_ms: float = 0.0
    
    # Memory metrics
    peak_memory_allocated_gb: float = 0.0
    peak_memory_reserved_gb: float = 0.0
    memory_bandwidth_utilization_pct: float = 0.0
    
    # GPU metrics
    gpu_utilization_pct: float = 0.0
    gpu_stall_pct: float = 0.0  # Estimated time waiting for memory
    
    # Cost metrics (estimates)
    cost_per_1m_tokens_usd: float = 0.0
    estimated_gpu_hours: float = 0.0
    
    # Cache metrics (from KV cache)
    kv_cache_memory_gb: float = 0.0
    kv_cache_hit_rate: float = 0.0
    memory_saved_by_quantization_gb: float = 0.0

    # Stage 3: Prefix sharing metrics
    num_cached_prefixes: int = 0
    prefix_sharing_enabled: bool = False
    total_prefix_hits: int = 0
    total_prefix_misses: int = 0

    # Batch metrics
    avg_batch_size: float = 0.0
    avg_sequence_length: float = 0.0

    # Stage 4: Dynamic batching metrics
    dynamic_batching_enabled: bool = False
    effective_batch_size: float = 0.0  # Auto-tuned batch size
    padding_tokens_saved: int = 0      # Tokens saved by smart grouping
    memory_efficiency_gain_pct: float = 0.0  # Memory saved by better estimation

    # Stage 5a: Model quantization metrics
    model_quantized: bool = False
    quantization_bits: int = 16  # 16 for FP16, 8 for INT8, 4 for INT4
    model_memory_savings_mb: float = 0.0
    model_memory_savings_pct: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class MemoryProfiler:
    """
    Real-time GPU memory and performance profiler.
    
    Tracks metrics during inference to provide before/after comparisons.
    """
    
    # GPU pricing (A100 80GB on AWS)
    GPU_HOURLY_COST_USD = {
        "a100-40gb": 4.0,  # Approximate
        "a100-80gb": 5.5,
        "h100": 8.0,
        "default": 5.0
    }
    
    def __init__(
        self,
        device: str = "cuda",
        gpu_type: str = "a100-80gb",
        enable_detailed_profiling: bool = False
    ):
        """
        Args:
            device: torch device
            gpu_type: GPU type for cost estimation
            enable_detailed_profiling: Enable CUDA profiler (adds overhead)
        """
        self.device = device
        self.gpu_type = gpu_type
        self.enable_detailed_profiling = enable_detailed_profiling
        
        # Counters
        self.tokens_generated = 0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
        # Memory tracking
        self.peak_memory_allocated = 0
        self.peak_memory_reserved = 0
        
        # Batch tracking
        self.batch_sizes: List[int] = []
        self.sequence_lengths: List[int] = []
        
        # KV cache tracking
        self.kv_cache_stats = None
        
        # CUDA events for precise timing
        if torch.cuda.is_available():
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        else:
            self.start_event = None
            self.end_event = None
    
    def start_profiling(self):
        """Start profiling session."""
        self.start_time = time.time()
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            if self.start_event:
                self.start_event.record()
    
    def end_profiling(self):
        """End profiling session."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            if self.end_event:
                self.end_event.record()
                torch.cuda.synchronize()
        
        self.end_time = time.time()
        
        if torch.cuda.is_available():
            self.peak_memory_allocated = torch.cuda.max_memory_allocated() / (1024**3)  # GB
            self.peak_memory_reserved = torch.cuda.max_memory_reserved() / (1024**3)  # GB
    
    def record_batch(self, batch_size: int, avg_seq_len: int):
        """Record batch statistics."""
        self.batch_sizes.append(batch_size)
        self.sequence_lengths.append(avg_seq_len)
    
    def record_tokens(self, num_tokens: int):
        """Record tokens generated."""
        self.tokens_generated += num_tokens
    
    def set_kv_cache_stats(self, cache_stats):
        """Set KV cache statistics."""
        self.kv_cache_stats = cache_stats
    
    def get_stats(self) -> ProfileStats:
        """
        Calculate and return profiling statistics.
        
        Returns:
            ProfileStats object with all metrics
        """
        stats = ProfileStats()
        
        # Time metrics
        if self.start_time and self.end_time:
            total_time = self.end_time - self.start_time
            stats.total_time_seconds = total_time
            
            if self.start_event and self.end_event:
                # More precise GPU time
                total_time = self.start_event.elapsed_time(self.end_event) / 1000.0  # ms to sec
                stats.total_time_seconds = total_time
        else:
            total_time = 0.0
        
        # Throughput metrics
        stats.total_tokens_generated = self.tokens_generated
        if total_time > 0 and self.tokens_generated > 0:
            stats.tokens_per_second = self.tokens_generated / total_time
            stats.latency_per_token_ms = (total_time / self.tokens_generated) * 1000
        else:
            stats.tokens_per_second = 0.0
            stats.latency_per_token_ms = 0.0
        
        # Memory metrics
        stats.peak_memory_allocated_gb = self.peak_memory_allocated
        stats.peak_memory_reserved_gb = self.peak_memory_reserved
        
        # Estimate memory bandwidth utilization
        # This is a rough estimate based on achieved throughput vs theoretical max
        # A100 80GB has ~2TB/s memory bandwidth
        # Each token decode roughly needs: model_size + KV_cache reads
        # For 13B model: ~26GB (FP16) or ~13GB (INT8)
        if stats.tokens_per_second > 0:
            bytes_per_token = 13e9  # 13GB for INT8 quantized 13B model
            achieved_bandwidth = stats.tokens_per_second * bytes_per_token
            theoretical_bandwidth = 2e12  # 2TB/s for A100
            stats.memory_bandwidth_utilization_pct = min(
                (achieved_bandwidth / theoretical_bandwidth) * 100, 100.0
            )
            
            # GPU stall estimate: if we're using 30% bandwidth, we're stalled 70% of time
            stats.gpu_stall_pct = 100.0 - stats.memory_bandwidth_utilization_pct
        
        # GPU utilization (rough estimate based on stalls)
        stats.gpu_utilization_pct = 100.0 - stats.gpu_stall_pct
        
        # Cost metrics
        gpu_hourly_cost = self.GPU_HOURLY_COST_USD.get(
            self.gpu_type,
            self.GPU_HOURLY_COST_USD["default"]
        )
        
        if total_time > 0:
            stats.estimated_gpu_hours = total_time / 3600.0
            
            # Cost per 1M tokens
            if stats.total_tokens_generated > 0:
                cost_for_run = stats.estimated_gpu_hours * gpu_hourly_cost
                stats.cost_per_1m_tokens_usd = (cost_for_run / stats.total_tokens_generated) * 1e6
        
        # KV cache metrics
        if self.kv_cache_stats:
            # Handle both dict and object formats
            if isinstance(self.kv_cache_stats, dict):
                stats.kv_cache_hit_rate = self.kv_cache_stats.get('hit_rate', 0.0) * 100
                stats.memory_saved_by_quantization_gb = self.kv_cache_stats.get('memory_saved_gb', 0.0)
            else:
                stats.kv_cache_hit_rate = self.kv_cache_stats.hit_rate * 100
                stats.memory_saved_by_quantization_gb = self.kv_cache_stats.memory_saved_gb

            # Estimate KV cache memory usage
            # For 13B model with 40 layers: ~160MB per 1K tokens with FP16
            # With INT8: ~40MB per 1K tokens
            if self.sequence_lengths:
                avg_seq_len = sum(self.sequence_lengths) / len(self.sequence_lengths)
                stats.kv_cache_memory_gb = (avg_seq_len / 1000.0) * 0.04  # 40MB per 1K tokens
        
        # Batch metrics
        if self.batch_sizes:
            stats.avg_batch_size = sum(self.batch_sizes) / len(self.batch_sizes)
        if self.sequence_lengths:
            stats.avg_sequence_length = sum(self.sequence_lengths) / len(self.sequence_lengths)
        
        return stats
    
    def print_stats(self, baseline_stats: Optional[ProfileStats] = None):
        """
        Print statistics in a human-readable format.
        
        Args:
            baseline_stats: Optional baseline stats for comparison
        """
        stats = self.get_stats()
        
        print("\n" + "="*70)
        print("Memopt PROFILING RESULTS")
        print("="*70)
        
        print("\n📊 PERFORMANCE METRICS")
        print(f"  Tokens generated:        {stats.total_tokens_generated:,}")
        print(f"  Total time:              {stats.total_time_seconds:.2f}s")
        print(f"  Throughput:              {stats.tokens_per_second:.1f} tokens/sec")
        print(f"  Latency per token:       {stats.latency_per_token_ms:.2f}ms")
        
        if baseline_stats:
            speedup = stats.tokens_per_second / max(baseline_stats.tokens_per_second, 1)
            print(f"  Speedup vs baseline:     {speedup:.2f}x")
        
        print("\n💾 MEMORY METRICS")
        print(f"  Peak allocated:          {stats.peak_memory_allocated_gb:.2f} GB")
        print(f"  Peak reserved:           {stats.peak_memory_reserved_gb:.2f} GB")
        print(f"  KV cache memory:         {stats.kv_cache_memory_gb:.2f} GB")
        print(f"  Memory saved (quant):    {stats.memory_saved_by_quantization_gb:.2f} GB")
        
        if baseline_stats:
            memory_reduction = (
                (baseline_stats.peak_memory_allocated_gb - stats.peak_memory_allocated_gb) /
                max(baseline_stats.peak_memory_allocated_gb, 1)
            ) * 100
            print(f"  Memory reduction:        {memory_reduction:.1f}%")
        
        print("\n🎮 GPU METRICS")
        print(f"  Memory bandwidth usage:  {stats.memory_bandwidth_utilization_pct:.1f}%")
        print(f"  GPU compute usage:       {stats.gpu_utilization_pct:.1f}%")
        print(f"  GPU stall time:          {stats.gpu_stall_pct:.1f}%")
        
        if baseline_stats:
            stall_reduction = baseline_stats.gpu_stall_pct - stats.gpu_stall_pct
            print(f"  Stall time reduction:    {stall_reduction:.1f}%")
        
        print("\n💰 COST METRICS")
        print(f"  GPU type:                {self.gpu_type}")
        print(f"  Cost per 1M tokens:      ${stats.cost_per_1m_tokens_usd:.2f}")
        print(f"  Estimated GPU hours:     {stats.estimated_gpu_hours:.4f}h")
        
        if baseline_stats:
            cost_savings = (
                (baseline_stats.cost_per_1m_tokens_usd - stats.cost_per_1m_tokens_usd) /
                max(baseline_stats.cost_per_1m_tokens_usd, 1)
            ) * 100
            daily_savings = (
                (baseline_stats.cost_per_1m_tokens_usd - stats.cost_per_1m_tokens_usd) *
                10000  # Assume 10B tokens/day
            )
            print(f"  Cost reduction:          {cost_savings:.1f}%")
            print(f"  Daily savings (10B tok): ${daily_savings:,.0f}")
            print(f"  Annual savings:          ${daily_savings * 365:,.0f}")
        
        print("\n📦 BATCH METRICS")
        print(f"  Avg batch size:          {stats.avg_batch_size:.1f}")
        print(f"  Avg sequence length:     {stats.avg_sequence_length:.0f}")

        if self.kv_cache_stats:
            print(f"  KV cache hit rate:       {stats.kv_cache_hit_rate:.1f}%")

        # Stage 4: Dynamic batching metrics
        if stats.dynamic_batching_enabled:
            print("\n🚀 STAGE 4 METRICS (Dynamic Batching)")
            print(f"  Auto-tuned batch size:   {stats.effective_batch_size:.1f}")
            print(f"  Padding tokens saved:    {stats.padding_tokens_saved:,}")
            print(f"  Memory efficiency gain:  {stats.memory_efficiency_gain_pct:.1f}%")

        # Stage 5a: Model quantization metrics
        if stats.model_quantized:
            print("\n⚡ STAGE 5a METRICS (Model Quantization)")
            print(f"  Quantization:            INT{stats.quantization_bits}")
            print(f"  Model memory savings:    {stats.model_memory_savings_mb:.1f} MB ({stats.model_memory_savings_pct:.1f}%)")

        print("\n" + "="*70 + "\n")
    
    def save_stats(self, filename: str, baseline_stats: Optional[ProfileStats] = None):
        """
        Save statistics to JSON file.
        
        Args:
            filename: Output filename
            baseline_stats: Optional baseline for comparison
        """
        stats = self.get_stats()
        
        output = {
            "optimized": stats.to_dict(),
        }
        
        if baseline_stats:
            output["baseline"] = baseline_stats.to_dict()
            output["comparison"] = {
                "speedup": stats.tokens_per_second / max(baseline_stats.tokens_per_second, 1),
                "memory_reduction_pct": (
                    (baseline_stats.peak_memory_allocated_gb - stats.peak_memory_allocated_gb) /
                    max(baseline_stats.peak_memory_allocated_gb, 1)
                ) * 100,
                "cost_reduction_pct": (
                    (baseline_stats.cost_per_1m_tokens_usd - stats.cost_per_1m_tokens_usd) /
                    max(baseline_stats.cost_per_1m_tokens_usd, 1)
                ) * 100,
                "stall_reduction_pct": baseline_stats.gpu_stall_pct - stats.gpu_stall_pct
            }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Stats saved to {filename}")


def compare_profiles(baseline: ProfileStats, optimized: ProfileStats):
    """
    Print a comparison of baseline vs optimized.
    
    Args:
        baseline: Baseline stats
        optimized: Optimized stats
    """
    print("\n" + "="*70)
    print("BASELINE vs OPTIMIZED COMPARISON")
    print("="*70)
    
    metrics = [
        ("Throughput (tok/s)", baseline.tokens_per_second, optimized.tokens_per_second, True),
        ("Latency (ms/tok)", baseline.latency_per_token_ms, optimized.latency_per_token_ms, False),
        ("Memory (GB)", baseline.peak_memory_allocated_gb, optimized.peak_memory_allocated_gb, False),
        ("GPU stall (%)", baseline.gpu_stall_pct, optimized.gpu_stall_pct, False),
        ("Cost/1M tok ($)", baseline.cost_per_1m_tokens_usd, optimized.cost_per_1m_tokens_usd, False),
    ]
    
    for name, base_val, opt_val, higher_better in metrics:
        if base_val == 0:
            continue
        
        if higher_better:
            improvement = (opt_val / base_val - 1) * 100
            symbol = "↑" if improvement > 0 else "↓"
        else:
            improvement = (1 - opt_val / base_val) * 100
            symbol = "↓" if improvement > 0 else "↑"
        
        print(f"\n{name}")
        print(f"  Baseline:   {base_val:.2f}")
        print(f"  Optimized:  {opt_val:.2f}")
        print(f"  Change:     {symbol} {abs(improvement):.1f}%")
    
    print("\n" + "="*70 + "\n")
