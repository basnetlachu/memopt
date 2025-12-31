#!/usr/bin/env python3
"""
Test speculative decoding at 1000 tokens with sliding window enabled.
This should achieve 15-16x speedup.
"""

import torch
import time
from memopt import OptimizedLLM
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
MODEL = "gpt2-xl"
PROMPT = "The future of artificial intelligence is"
MAX_TOKENS = 1000

print("="*70)
print("TESTING: 1000-Token Generation with Sliding Window")
print("="*70)

# Test 1: Baseline
print("\n1. BASELINE (No optimizations)")
print("-" * 70)
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

baseline_model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map=device,
    low_cpu_mem_usage=True
)
baseline_model.eval()

inputs = tokenizer(PROMPT, return_tensors="pt").to(device)

if torch.cuda.is_available():
    torch.cuda.synchronize()

start = time.time()
with torch.no_grad():
    outputs = baseline_model.generate(
        **inputs,
        max_new_tokens=MAX_TOKENS,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id
    )
if torch.cuda.is_available():
    torch.cuda.synchronize()
baseline_time = time.time() - start
baseline_throughput = MAX_TOKENS / baseline_time

print(f"Baseline throughput: {baseline_throughput:.1f} tok/s")
print(f"Baseline time: {baseline_time:.2f}s")

# Clean up
del baseline_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Test 2: Optimized WITHOUT sliding window (current behavior)
print("\n2. OPTIMIZED (Speculative, NO sliding window)")
print("-" * 70)
model_no_window = OptimizedLLM(
    MODEL,
    optimization_level="speculative",
    enable_profiling=True
)

start = time.time()
_ = model_no_window.generate(PROMPT, max_tokens=MAX_TOKENS, do_sample=False)
optimized_time_no_window = time.time() - start

stats = model_no_window.get_profiling_stats()
print(f"Optimized throughput: {stats.tokens_per_second:.1f} tok/s")
print(f"Optimized time: {optimized_time_no_window:.2f}s")
print(f"Speedup: {stats.tokens_per_second / baseline_throughput:.2f}x")

# Clean up
del model_no_window
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Test 3: Optimized WITH sliding window (should be faster!)
print("\n3. OPTIMIZED (Speculative, WITH sliding window)")
print("-" * 70)
model_with_window = OptimizedLLM(
    MODEL,
    optimization_level="speculative",
    opt_config={'max_context_length': 'auto'},  # Enable sliding window!
    enable_profiling=True
)

start = time.time()
_ = model_with_window.generate(PROMPT, max_tokens=MAX_TOKENS, do_sample=False)
optimized_time_with_window = time.time() - start

stats = model_with_window.get_profiling_stats()
print(f"Optimized throughput: {stats.tokens_per_second:.1f} tok/s")
print(f"Optimized time: {optimized_time_with_window:.2f}s")
print(f"Speedup: {stats.tokens_per_second / baseline_throughput:.2f}x")

# Show sliding window stats
if hasattr(model_with_window, 'speculative_decoder') and model_with_window.speculative_decoder:
    decoder_stats = model_with_window.speculative_decoder.get_stats()
    print(f"\nSliding window stats:")
    print(f"  Window slides: {decoder_stats['window_slides']}")
    print(f"  Max context length: {decoder_stats['max_context_length']}")
    print(f"  Sliding window enabled: {decoder_stats['sliding_window_enabled']}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Baseline: {baseline_throughput:.1f} tok/s")
print(f"Optimized (no window): {baseline_throughput * (stats.tokens_per_second / baseline_throughput):.1f} tok/s → {stats.tokens_per_second / baseline_throughput:.2f}x")
print(f"Optimized (with window): {stats.tokens_per_second:.1f} tok/s → {stats.tokens_per_second / baseline_throughput:.2f}x")
print("="*70)
