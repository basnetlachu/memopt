#!/usr/bin/env python3
"""Debug test for prefix sharing ref counting."""

import torch
from memopt.kv_cache import PagedKVCache

# Create cache
kv_cache = PagedKVCache(
    num_layers=2,
    num_heads=4,
    head_dim=16,
    block_size=8,
    max_blocks=32,
    device="cpu",
    quantize=False,
    enable_prefix_sharing=True
)

print("Initial state:")
print(f"  Free blocks: {len(kv_cache.free_blocks)}")
print(f"  Ref counts: {kv_cache.block_ref_counts}")

# Allocate first sequence
seq_id_1 = 0
blocks_1 = kv_cache.allocate_blocks(seq_id_1, 4)
print(f"\nAfter allocate_blocks(0, 4):")
print(f"  blocks_1 = {blocks_1}")
print(f"  block_tables[0] = {kv_cache.block_tables.get(0)}")
print(f"  Ref counts: {dict(sorted(kv_cache.block_ref_counts.items()))}")
print(f"  Free blocks: {len(kv_cache.free_blocks)}")

# Register prefix
prefix_tokens = list(range(32))
prefix_blocks = blocks_1[:2]
print(f"\nRegistering prefix with blocks {prefix_blocks}")
kv_cache.register_prefix(prefix_tokens, prefix_blocks)
print(f"  Prefix cache: {kv_cache.prefix_cache}")
print(f"  Ref counts (should be unchanged): {dict(sorted(kv_cache.block_ref_counts.items()))}")

# Allocate second sequence with sharing
seq_id_2 = 1
print(f"\nAllocating seq_id=1 with prefix sharing (need 4 blocks total)")
blocks_2, shared_count = kv_cache.allocate_with_prefix_sharing(
    seq_id_2, prefix_tokens, 4
)
print(f"  blocks_2 = {blocks_2}")
print(f"  shared_count = {shared_count}")
print(f"  block_tables[1] = {kv_cache.block_tables.get(1)}")
print(f"  Ref counts (0,1 should be 2): {dict(sorted(kv_cache.block_ref_counts.items()))}")
print(f"  Free blocks: {len(kv_cache.free_blocks)}")

# Check ref counts before freeing
print(f"\nBefore freeing seq_id_1:")
for i, block_id in enumerate(prefix_blocks):
    print(f"  Block {block_id}: ref_count = {kv_cache.block_ref_counts.get(block_id, 'NOT FOUND')}, in free_blocks = {block_id in kv_cache.free_blocks}")

# Free first sequence
print(f"\nFreeing seq_id_1...")
kv_cache.free_sequence(seq_id_1)

print(f"\nAfter freeing seq_id_1:")
print(f"  block_tables: {kv_cache.block_tables}")
print(f"  Ref counts: {dict(sorted(kv_cache.block_ref_counts.items()))}")
print(f"  Free blocks count: {len(kv_cache.free_blocks)}")

# Check shared blocks
print(f"\nChecking shared blocks:")
for block_id in prefix_blocks:
    in_free = block_id in kv_cache.free_blocks
    ref_count = kv_cache.block_ref_counts.get(block_id, "DELETED")
    print(f"  Block {block_id}: ref_count = {ref_count}, in free_blocks = {in_free}")

    if in_free:
        print(f"    ❌ ERROR: Block {block_id} should NOT be in free_blocks!")
    if ref_count != 1:
        print(f"    ❌ ERROR: Block {block_id} ref_count should be 1, got {ref_count}!")
