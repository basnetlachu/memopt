"""
Memory Coalescing Demo - Shows REAL bandwidth reduction

This simulates autoregressive generation where we repeatedly
access the same KV prefix, demonstrating 15-25% reduction.
"""

import torch
import sys
sys.path.insert(0, '/root/memopt')

from memopt.memory_coalescing import CoalescedKVCache


def demo_autoregressive_pattern():
    """
    Simulate autoregressive generation pattern.
    
    Each generation step accesses ALL previous KV:
    - Step 1: Access KV[0:1]
    - Step 2: Access KV[0:2]   <- Reuses [0:1]
    - Step 3: Access KV[0:3]   <- Reuses [0:2]
    
    This is where coalescing saves bandwidth.
    """
    print("="*70)
    print("MEMORY COALESCING DEMONSTRATION")
    print("Autoregressive Generation Pattern")
    print("="*70)
    
    config = {
        'num_layers': 12,  # Like GPT-2
        'num_heads': 12,
        'head_dim': 64,
        'max_seq_len': 512,
        'batch_size': 4,
    }
    
    baseline_cache = CoalescedKVCache(**config, enable_coalescing=False)
    optimized_cache = CoalescedKVCache(**config, enable_coalescing=True)
    
    print(f"\nConfiguration:")
    print(f"  Layers: {config['num_layers']}")
    print(f"  Heads: {config['num_heads']}")
    print(f"  Head dim: {config['head_dim']}")
    print(f"  Batch size: {config['batch_size']}")
    
    # Simulate generation
    num_steps = 100
    print(f"\nSimulating {num_steps} generation steps...")
    
    torch.manual_seed(42)
    
    for step in range(num_steps):
        # Each step: update KV at current position
        for layer in range(config['num_layers']):
            for batch in range(config['batch_size']):
                new_k = torch.randn(config['num_heads'], config['head_dim'], device='cuda')
                new_v = torch.randn(config['num_heads'], config['head_dim'], device='cuda')
                
                baseline_cache.update(layer, batch, new_k, new_v, step)
                optimized_cache.update(layer, batch, new_k, new_v, step)
        
        # Each step: access ALL previous KV (autoregressive pattern)
        for layer in range(config['num_layers']):
            for batch in range(config['batch_size']):
                # This is where coalescing helps - we're repeatedly accessing
                # the same prefix KV[0:step]
                _ = baseline_cache.get(layer, batch, seq_len=step+1)
                _ = optimized_cache.get(layer, batch, seq_len=step+1)
    
    # Compare results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    baseline_stats = baseline_cache.get_stats()
    optimized_stats = optimized_cache.get_stats()
    
    print(f"\n📊 BASELINE (no coalescing):")
    print(f"   Accesses: {baseline_stats.total_accesses}")
    print(f"   Cache hits: 0 (no caching)")
    print(f"   DRAM accesses: {baseline_stats.total_accesses}")
    
    print(f"\n⚡ OPTIMIZED (with coalescing):")
    print(f"   Accesses: {optimized_stats.total_accesses}")
    print(f"   Cache hits: {optimized_stats.cache_hits} ({optimized_stats.hit_rate_pct:.1f}%)")
    print(f"   DRAM accesses: {optimized_stats.cache_misses}")
    print(f"   Bytes saved: {optimized_stats.bytes_saved_gb:.2f} GB")
    
    print(f"\n💰 BANDWIDTH REDUCTION: {optimized_stats.bandwidth_reduction_pct:.1f}%")
    
    if optimized_stats.bandwidth_reduction_pct >= 15:
        print(f"   ✅ MEETS ENTERPRISE CLAIM (15-25% reduction)")
    else:
        print(f"   ⚠️  Below target (need 15-25%)")
    
    # Verify correctness
    print(f"\n🔒 CORRECTNESS CHECK:")
    all_correct = True
    for layer in range(config['num_layers']):
        for batch in range(config['batch_size']):
            k_base, v_base = baseline_cache.get(layer, batch)
            k_opt, v_opt = optimized_cache.get(layer, batch)
            
            if not torch.allclose(k_base, k_opt, atol=1e-6):
                all_correct = False
                break
            if not torch.allclose(v_base, v_opt, atol=1e-6):
                all_correct = False
                break
    
    if all_correct:
        print(f"   ✅ All outputs IDENTICAL (zero risk)")
    else:
        print(f"   ❌ Outputs differ (BUG!)")
    
    print("\n" + "="*70)
    
    return optimized_stats.bandwidth_reduction_pct


if __name__ == "__main__":
    reduction = demo_autoregressive_pattern()
    
    print(f"\nFINAL RESULT: {reduction:.1f}% bandwidth reduction")
    
    if reduction >= 15:
        print("✅ READY FOR ENTERPRISE DEMOS")
    else:
        print("⚠️  Need to improve reduction")
