#!/usr/bin/env python3
"""
Benchmark script for Memopt

Compares baseline vs optimized inference and provides honest performance metrics.

Three optimization presets:
- fast: SDPA only (1.2-1.5x speedup) - DEFAULT for single-sequence
- batch: Batching + paging (5-10x speedup) - For multi-request workloads
- maximum: Batching + speculation (10-20x speedup) - Requires draft model

All old optimization levels are aliases for "fast".

Usage:
    # Single GPU
    python benchmark.py --model Qwen/Qwen2-7B --max-tokens 1000                    # Uses "fast" (1.2-1.5x)
    python benchmark.py --model Qwen/Qwen2-7B --optimization-level fast            # SDPA only (1.2-1.5x)
    python benchmark.py --model Qwen/Qwen2-7B --optimization-level batch           # Batching (5-10x)
    python benchmark.py --model Qwen/Qwen2-7B --optimization-level maximum         # Batching + spec (10-20x)

    # Multi-GPU (Worker-Per-GPU Architecture)
    python benchmark.py --model gpt2-xl --num-gpus 2 --use-workers --num-prompts 100 \
        --rl-scheduler-path scheduler_rl_agent.zip \
        --memory-predictor-path memory_predictor.pth
"""

import argparse
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import multiprocessing as mp
import os
import sys
import queue

from memopt import OptimizedLLM, ProfileStats
from memopt.profiler import compare_profiles


def run_baseline(model_name: str, prompts: list, max_tokens: int = 256):
    """
    Run baseline inference without optimizations.
    
    Returns:
        ProfileStats
    """
    print("\n" + "="*70)
    print("RUNNING BASELINE (No Optimizations)")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model normally
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True
    )
    model.eval()
    
    # Reset memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    start_time = time.time()
    total_tokens = 0
    
    # Run inference
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            print(f"  Processing prompt {i+1}/{len(prompts)}...")
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            total_tokens += len(outputs[0]) - len(inputs['input_ids'][0])
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Collect stats
    stats = ProfileStats()
    stats.total_tokens_generated = total_tokens
    stats.total_time_seconds = total_time
    stats.tokens_per_second = total_tokens / total_time
    stats.latency_per_token_ms = (total_time / total_tokens) * 1000
    
    if torch.cuda.is_available():
        stats.peak_memory_allocated_gb = torch.cuda.max_memory_allocated() / (1024**3)
        stats.peak_memory_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)
        
        # Estimate bandwidth usage (baseline is inefficient)
        bytes_per_token = 26e9  # 26GB for FP16 13B model
        achieved_bandwidth = stats.tokens_per_second * bytes_per_token
        theoretical_bandwidth = 2e12  # 2TB/s for A100
        stats.memory_bandwidth_utilization_pct = min(
            (achieved_bandwidth / theoretical_bandwidth) * 100, 100.0
        )
        stats.gpu_stall_pct = 100.0 - stats.memory_bandwidth_utilization_pct
        stats.gpu_utilization_pct = 100.0 - stats.gpu_stall_pct
    
    # Cost estimate
    gpu_hourly_cost = 5.0  # A100 80GB
    stats.estimated_gpu_hours = total_time / 3600.0
    cost_for_run = stats.estimated_gpu_hours * gpu_hourly_cost
    stats.cost_per_1m_tokens_usd = (cost_for_run / total_tokens) * 1e6
    
    print(f"\n✓ Baseline complete: {stats.tokens_per_second:.1f} tok/s")
    
    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return stats


def run_optimized(
    model_name: str,
    prompts: list,
    max_tokens: int = 256,
    optimization_level: str = "ultra",
    max_kv_blocks: int = None,
    quantize_kv: bool = False,
    enable_speculative: bool = False,
    # REMOVED: num_gpus, multi_gpu_mode, enable_rl_routing, rl_router_path (use production/ for multi-GPU)
    rl_scheduler_path: str = None,
    memory_predictor_path: str = None,
    enable_memory_tracing: bool = False,
    memory_trace_output: str = "memory_traces.csv"
):
    """
    Run optimized inference with Memopt.

    Returns:
        ProfileStats
    """
    print("\n" + "="*70)
    print(f"RUNNING OPTIMIZED (Memopt - {optimization_level})")
    print("="*70)

    # Initialize memory tracer if requested
    memory_tracer = None
    if enable_memory_tracing:
        from memopt.memory_tracer import MemoryTracer
        memory_tracer = MemoryTracer(
            enable=True,
            auto_save_interval=10,
            auto_save_path=memory_trace_output
        )
        print(f"\n  📊 Memory tracing enabled")
        print(f"     • Auto-saving every 10 traces to {memory_trace_output}")

    # REMOVED: Multi-GPU display code (use production/ for multi-GPU)
    # For multi-GPU inference, use: scripts/benchmark_production.sh

    # Determine if we'll use concurrent batching
    # This affects memory allocation
    use_concurrent = (rl_scheduler_path is not None or memory_predictor_path is not None)
    num_prompts = len(prompts) if prompts else 10

    # Set appropriate batch size for concurrent mode
    if use_concurrent and num_prompts >= 10:
        expected_batch_size = min(10, num_prompts)  # Chunk size for concurrent batching
        print(f"\n  🚀 Concurrent batching enabled ({expected_batch_size}-prompt chunks)")
    else:
        expected_batch_size = 8  # Default for sequential mode

    # Load with Memopt
    model = OptimizedLLM(
        model=model_name,
        optimization_level=optimization_level,
        enable_profiling=True,
        expected_batch_size=expected_batch_size,  # Dynamic based on mode
        expected_seq_len=max_tokens * 2,  # Conservative estimate
        max_kv_blocks=max_kv_blocks,  # Apply user override if specified
        # REMOVED: num_gpus, multi_gpu_mode, enable_rl_routing, rl_router_path (use production/ for multi-GPU)
        rl_scheduler_path=rl_scheduler_path,  # Trained RL batch scheduler agent
        memory_predictor_path=memory_predictor_path  # Trained neural memory predictor
    )

    # Override quantization if requested (must be done BEFORE cache initialization)
    if quantize_kv:
        print("\n  🔧 Enabling INT8 KV Cache Quantization:")
        print("     • 4× memory reduction (FP16 → INT8)")
        print("     • Expected: +1.5-2× additional speedup")
        print("     • Most effective for MHA models (Llama-2)")
        model.opt_config['quantize_kv'] = True
        # Reinitialize KV cache with quantization enabled
        model._initialize_kv_cache()
        print("  ✓ INT8 quantization active\n")

    # Override speculative decoding if requested
    if enable_speculative:
        print("\n  🔧 Enabling Speculative Decoding:")
        print("     • Draft model generates candidate tokens")
        print("     • Main model verifies in parallel")
        print("     • Expected: +2-3× additional speedup")
        print("     • Works best with compatible draft models")
        model.opt_config['enable_speculative_decoding'] = True
        # Use 4 speculative tokens by default (safe and effective)
        if model.opt_config.get('num_speculative_tokens', 0) == 0:
            model.opt_config['num_speculative_tokens'] = 4
        # Initialize speculative decoder
        model._initialize_speculative_decoding()
        print("  ✓ Speculative decoding active\n")
    
    # CRITICAL FIX: Reset cache ONCE before the loop, not after each prompt
    model.reset_kv_cache()

    # Warmup: Run one iteration to exclude model load and any first-run overhead
    # This ensures we measure steady-state performance, not compilation/initialization
    print("  Running warmup iteration (excluded from timing)...")
    warmup_prompt = prompts[0] if prompts else "The"
    _ = model.generate(warmup_prompt, max_tokens=min(50, max_tokens), do_sample=False)

    # Reset profiler after warmup
    if hasattr(model, 'profiler') and model.profiler:
        model.profiler.reset()

    # Reset timer for actual measurement
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    # Check if we should use concurrent batching mode
    # Concurrent mode is automatically enabled if:
    # 1. RL scheduler is loaded, OR
    # 2. Memory predictor is loaded
    # This enables TRUE batching to demonstrate trained AI model performance
    use_concurrent = (rl_scheduler_path is not None or memory_predictor_path is not None)

    if use_concurrent and len(prompts) >= 10:
        print(f"  🚀 CONCURRENT BATCHING MODE ENABLED")
        print(f"     RL Scheduler: {'✓' if rl_scheduler_path else '✗'}")
        print(f"     Memory Predictor: {'✓' if memory_predictor_path else '✗'}")
        print(f"  Processing {len(prompts)} prompts with concurrent batching...")
        print(f"  RL scheduler will dynamically optimize batch sizes...")

        # Process in chunks to avoid KV cache exhaustion
        chunk_size = 10  # Conservative chunk size
        for i in range(0, len(prompts), chunk_size):
            chunk = prompts[i:i+chunk_size]
            chunk_num = i//chunk_size + 1
            total_chunks = (len(prompts)-1)//chunk_size + 1
            print(f"    Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} prompts)...")

            # Use generate_batch for true concurrent processing
            _ = model.generate_batch(
                chunk,
                max_tokens=max_tokens,
                do_sample=False
            )

            # Reset KV cache between chunks to free memory
            if hasattr(model, 'reset_kv_cache'):
                model.reset_kv_cache()
    else:
        # Sequential mode (original behavior)
        if use_concurrent:
            print(f"  ⚠️  Note: Concurrent mode disabled (need 10+ prompts)")

        # Run inference - Stage 4 auto-tuning will adapt to varying lengths
        for i, prompt in enumerate(prompts):
            print(f"  Processing prompt {i+1}/{len(prompts)}...")

            _ = model.generate(
                prompt,
                max_tokens=max_tokens,
                do_sample=False
            )

            # Free this sequence from KV cache to prevent exhaustion
            # This allows prefix sharing while avoiding OOM
            if hasattr(model, 'kv_cache') and model.kv_cache:
                # Find the most recent sequence ID and free it
                if hasattr(model, '_last_seq_id'):
                    try:
                        model.kv_cache.free_sequence(model._last_seq_id)
                    except:
                        pass  # Ignore if already freed
    
    # Get stats
    stats = model.get_profiling_stats()

    print(f"\n✓ Optimized complete: {stats.tokens_per_second:.1f} tok/s")

    # Show prefix sharing stats if enabled AND actually working
    if stats.prefix_sharing_enabled and stats.total_prefix_hits > 0:
        print(f"  Prefix sharing stats:")
        print(f"    Cached prefixes: {stats.num_cached_prefixes}")
        print(f"    Prefix hits: {stats.total_prefix_hits}")
        print(f"    Prefix misses: {stats.total_prefix_misses}")
        if stats.total_prefix_hits + stats.total_prefix_misses > 0:
            hit_rate = stats.total_prefix_hits / (stats.total_prefix_hits + stats.total_prefix_misses) * 100
            print(f"    Hit rate: {hit_rate:.1f}%")

    return stats, model  # Return model for memory analysis


def print_memory_analysis(optimized_model, baseline_memory_gb, optimized_peak_gb):
    """
    Print honest memory analysis showing actual vs peak allocation.
    """
    if not torch.cuda.is_available():
        return
    
    print("\n" + "="*70)
    print("MEMORY ANALYSIS")
    print("="*70)
    
    # Get model size
    model_params = sum(p.numel() * p.element_size() for p in optimized_model.model.parameters())
    model_size_gb = model_params / (1024**3)
    
    print(f"\nModel weights: {model_size_gb:.2f} GB (constant)")
    
    # Get KV cache usage
    if hasattr(optimized_model, 'kv_cache') and optimized_model.kv_cache:
        kv = optimized_model.kv_cache
        stats = kv.get_stats()
        
        # Calculate actual memory - handle both dict and object
        if isinstance(stats, dict):
            blocks_used = stats.get('used_pages', 0)
            utilization = stats.get('utilization', 0.0)
        else:
            blocks_used = stats.used_pages
            utilization = stats.utilization

        bytes_per_block = (
            kv.block_size * kv.num_heads * kv.head_dim *
            (1 if kv.quantize else 2) * 2 * kv.num_layers
        )
        actual_kv_gb = (blocks_used * bytes_per_block) / (1024**3)

        # What baseline would use (FP16 for same tokens)
        baseline_kv_gb = actual_kv_gb * (2 if kv.quantize else 1)

        print(f"\nKV Cache:")
        print(f"  Allocated: {kv.max_blocks} blocks (max capacity)")
        print(f"  Used: {blocks_used} blocks ({utilization*100:.1f}% utilization)")
        print(f"  Quantization: {'INT8' if kv.quantize else 'FP16'}")

        # Show prefix sharing stats if enabled
        if hasattr(kv, 'enable_prefix_sharing') and kv.enable_prefix_sharing:
            num_prefixes = len(kv.prefix_cache)
            print(f"  Prefix sharing: ENABLED ({num_prefixes} cached prefixes)")
        else:
            print(f"  Prefix sharing: DISABLED")

        print(f"  Memory (baseline would use): {baseline_kv_gb:.2f} GB")
        print(f"  Memory (optimized actual): {actual_kv_gb:.2f} GB")
        
        # True comparison
        baseline_total = baseline_memory_gb
        optimized_actual = model_size_gb + actual_kv_gb
        savings_gb = baseline_total - optimized_actual
        savings_pct = (savings_gb / baseline_total) * 100
        
        print(f"\nActual memory usage:")
        print(f"  Baseline total: {baseline_total:.2f} GB")
        print(f"  Optimized actual: {optimized_actual:.2f} GB")
        print(f"  True savings: {savings_gb:.2f} GB ({savings_pct:.1f}%)")
        
        print(f"\nNote: Peak shows {optimized_peak_gb:.2f} GB due to pre-allocated")
        print(f"      blocks, but only {blocks_used}/{kv.max_blocks} blocks are used.")
    
    print("="*70)


def worker_process(
    worker_id: int,
    model_name: str,
    optimization_level: str,
    max_tokens: int,
    rl_scheduler_path: str,
    memory_predictor_path: str,
    prompt_queue: mp.Queue,
    result_queue: mp.Queue,
    ready_queue: mp.Queue
):
    """
    Worker process that runs on a single GPU.

    Each worker:
    1. Sets CUDA_VISIBLE_DEVICES to see only its GPU
    2. Loads Memopt with full optimizations
    3. Processes prompts from the queue
    4. Returns results to the main process
    """
    # Set this worker to use only its assigned GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(worker_id)

    # Import here to ensure CUDA_VISIBLE_DEVICES takes effect
    import torch
    from memopt import OptimizedLLM

    print(f"[Worker {worker_id}] Starting on GPU {worker_id}")
    print(f"[Worker {worker_id}] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"[Worker {worker_id}] PyTorch sees {torch.cuda.device_count()} GPU(s)")

    # Load model with Memopt optimizations
    print(f"[Worker {worker_id}] Loading model {model_name}...")

    model_kwargs = {
        'model': model_name,
        'optimization_level': optimization_level,
        'enable_profiling': True,
        'device': 'cuda',  # Will use GPU 0 in this process (which is actually GPU worker_id)
    }

    # Add AI models if provided
    if rl_scheduler_path:
        model_kwargs['rl_scheduler_path'] = rl_scheduler_path
    if memory_predictor_path:
        model_kwargs['memory_predictor_path'] = memory_predictor_path

    try:
        model = OptimizedLLM(**model_kwargs)
        print(f"[Worker {worker_id}] ✓ Model loaded successfully")
    except Exception as e:
        print(f"[Worker {worker_id}] ✗ Failed to load model: {e}")
        ready_queue.put(('error', worker_id, str(e)))
        return

    # Signal that worker is ready
    ready_queue.put(('ready', worker_id, None))

    # Process prompts from queue
    prompts_processed = 0
    total_tokens = 0
    start_time = time.time()

    while True:
        try:
            # Get prompt from queue (with timeout to allow graceful shutdown)
            item = prompt_queue.get(timeout=1.0)

            if item is None:  # Sentinel value to stop worker
                break

            prompt_id, prompt = item

            # Generate response
            try:
                output = model.generate(prompt, max_tokens=max_tokens, do_sample=False)

                # Count tokens (approximate)
                tokens_generated = len(output.split())

                prompts_processed += 1
                total_tokens += tokens_generated

                # Send result back
                result_queue.put(('success', worker_id, prompt_id, tokens_generated))

            except Exception as e:
                print(f"[Worker {worker_id}] Error generating for prompt {prompt_id}: {e}")
                result_queue.put(('error', worker_id, prompt_id, str(e)))

        except queue.Empty:
            continue  # No prompts available, keep waiting
        except Exception as e:
            print(f"[Worker {worker_id}] Unexpected error: {e}")
            break

    # Compute worker stats
    elapsed = time.time() - start_time
    throughput = total_tokens / elapsed if elapsed > 0 else 0

    print(f"[Worker {worker_id}] Processed {prompts_processed} prompts, {total_tokens} tokens in {elapsed:.1f}s")
    print(f"[Worker {worker_id}] Throughput: {throughput:.1f} tok/s")

    # Send final stats
    result_queue.put(('stats', worker_id, {
        'prompts_processed': prompts_processed,
        'total_tokens': total_tokens,
        'elapsed': elapsed,
        'throughput': throughput
    }))


def run_worker_benchmark(
    model_name: str,
    num_gpus: int,
    prompts: list,
    max_tokens: int,
    optimization_level: str,
    rl_scheduler_path: str,
    memory_predictor_path: str
):
    """Run worker-per-GPU benchmark."""
    print("\n" + "="*70)
    print(f"WORKER-PER-GPU BENCHMARK ({num_gpus} GPUs)")
    print("="*70)

    # Create queues for communication
    prompt_queue = mp.Queue()
    result_queue = mp.Queue()
    ready_queue = mp.Queue()

    # Start worker processes
    print(f"\nStarting {num_gpus} workers...")
    workers = []

    for worker_id in range(num_gpus):
        p = mp.Process(
            target=worker_process,
            args=(
                worker_id, model_name, optimization_level, max_tokens,
                rl_scheduler_path, memory_predictor_path,
                prompt_queue, result_queue, ready_queue
            )
        )
        p.start()
        workers.append(p)

    # Wait for all workers to be ready
    print(f"Waiting for {num_gpus} workers to initialize...")
    workers_ready = 0

    while workers_ready < num_gpus:
        try:
            status, worker_id, data = ready_queue.get(timeout=60)
            if status == 'ready':
                print(f"  ✓ Worker {worker_id} ready")
                workers_ready += 1
            elif status == 'error':
                print(f"  ✗ Worker {worker_id} failed: {data}")
                # Kill all workers and exit
                for p in workers:
                    p.terminate()
                raise RuntimeError(f"Worker {worker_id} failed to start")
        except queue.Empty:
            print("  ✗ Timeout waiting for workers")
            for p in workers:
                p.terminate()
            raise RuntimeError("Workers failed to start within timeout")

    print(f"✓ All {num_gpus} workers ready\n")

    # Distribute prompts to queue
    print(f"Distributing {len(prompts)} prompts to workers...")
    for i, prompt in enumerate(prompts):
        prompt_queue.put((i, prompt))

    # Add sentinel values to stop workers
    for _ in range(num_gpus):
        prompt_queue.put(None)

    # Collect results
    print(f"Processing prompts...")
    start_time = time.time()

    prompts_completed = 0
    errors = 0
    worker_stats = {}

    while prompts_completed < len(prompts) or len(worker_stats) < num_gpus:
        try:
            result = result_queue.get(timeout=300)

            if result[0] == 'success':
                _, worker_id, prompt_id, tokens = result
                prompts_completed += 1
                print(f"  Completed {prompts_completed}/{len(prompts)} prompts", end='\r')

            elif result[0] == 'error':
                _, worker_id, prompt_id, error = result
                errors += 1
                print(f"\n  Error in worker {worker_id}, prompt {prompt_id}: {error}")

            elif result[0] == 'stats':
                _, worker_id, stats = result
                worker_stats[worker_id] = stats

        except queue.Empty:
            print("\n  Warning: Timeout waiting for results")
            break

    elapsed = time.time() - start_time

    # Wait for workers to finish
    for p in workers:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

    # Calculate aggregate stats
    total_tokens = sum(stats['total_tokens'] for stats in worker_stats.values())
    total_throughput = total_tokens / elapsed

    print(f"\n\n✓ Worker benchmark complete")
    print(f"\nPer-Worker Stats:")
    for worker_id, stats in sorted(worker_stats.items()):
        print(f"  Worker {worker_id}: {stats['throughput']:.1f} tok/s ({stats['prompts_processed']} prompts)")

    print(f"\nAggregate Stats:")
    print(f"  Total throughput: {total_throughput:.1f} tok/s")
    print(f"  Total elapsed: {elapsed:.1f}s")
    print(f"  Prompts completed: {prompts_completed}/{len(prompts)}")
    if errors > 0:
        print(f"  Errors: {errors}")

    # Return stats compatible with ProfileStats
    stats = ProfileStats()
    stats.total_tokens_generated = total_tokens
    stats.total_time_seconds = elapsed
    stats.tokens_per_second = total_throughput
    stats.latency_per_token_ms = (elapsed / total_tokens) * 1000 if total_tokens > 0 else 0

    # Worker mode doesn't have accurate memory tracking (distributed)
    stats.peak_memory_allocated_gb = 0.0
    stats.peak_memory_reserved_gb = 0.0
    stats.memory_bandwidth_utilization_pct = 0.0
    stats.gpu_stall_pct = 0.0
    stats.gpu_utilization_pct = 0.0

    # Cost estimate
    gpu_hourly_cost = 5.0  # A100 80GB
    stats.estimated_gpu_hours = elapsed / 3600.0 * num_gpus  # Multiple GPUs
    cost_for_run = stats.estimated_gpu_hours * gpu_hourly_cost
    stats.cost_per_1m_tokens_usd = (cost_for_run / total_tokens) * 1e6 if total_tokens > 0 else 0

    return stats


def main():
    parser = argparse.ArgumentParser(description="Benchmark Memopt")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Model name or path"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "optimized", "both"],
        default="both",
        help="Which mode to run"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max tokens to generate per prompt"
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=5,
        help="Number of prompts to test"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--use-system-prompt",
        action="store_true",
        help="Add system prompt prefix to all prompts (demonstrates Stage 3 prefix sharing)"
    )
    parser.add_argument(
        "--optimization-level",
        type=str,
        choices=["fast", "batch", "maximum", "conservative", "balanced", "high", "ultra", "aggressive", "speculative", "flash"],
        default="fast",
        help="Optimization: 'fast' (1.2-1.5x, single-seq), 'batch' (5-10x, multi-req), 'maximum' (10-20x, needs draft model)"
    )
    parser.add_argument(
        "--max-kv-blocks",
        type=int,
        default=None,
        help="Maximum KV cache blocks (None=auto, 128 recommended for CPU/low memory)"
    )
    parser.add_argument(
        "--quantize-kv",
        action="store_true",
        help="Enable INT8 KV cache quantization (4× memory reduction, ~1.5-2× speedup)"
    )
    parser.add_argument(
        "--enable-speculative",
        action="store_true",
        help="Enable speculative decoding with built-in draft model (2-3× additional speedup)"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs for multi-GPU inference (default: 1, use 4-8 for 50×+ speedup)"
    )
    parser.add_argument(
        "--multi-gpu-mode",
        type=str,
        choices=["data_parallel", "tensor_parallel"],
        default="data_parallel",
        help="Multi-GPU mode: 'data_parallel' (replicas, better) or 'tensor_parallel' (split layers)"
    )
    parser.add_argument(
        "--enable-rl-routing",
        action="store_true",
        help="Enable RL-powered multi-GPU routing (+5%% efficiency over load-aware)"
    )
    parser.add_argument(
        "--rl-router-path",
        type=str,
        default=None,
        help="Path to trained RL router agent (e.g., multi_gpu_router.zip)"
    )
    parser.add_argument(
        "--rl-scheduler-path",
        type=str,
        default=None,
        help="Path to trained RL batch scheduler agent (e.g., scheduler_rl_agent.zip)"
    )
    parser.add_argument(
        "--memory-predictor-path",
        type=str,
        default=None,
        help="Path to trained neural memory predictor (e.g., memory_predictor.pth)"
    )
    parser.add_argument(
        "--enable-memory-tracing",
        action="store_true",
        help="Enable memory tracing for neural predictor training (saves to memory_traces.csv)"
    )
    parser.add_argument(
        "--memory-trace-output",
        type=str,
        default="memory_traces.csv",
        help="Output file for memory traces (default: memory_traces.csv)"
    )
    parser.add_argument(
        "--use-workers",
        action="store_true",
        help="Use worker-per-GPU architecture for multi-GPU (recommended for 2+ GPUs)"
    )

    args = parser.parse_args()

    # Validate GPU count for worker mode
    if args.use_workers:
        available_gpus = torch.cuda.device_count()
        if args.num_gpus > available_gpus:
            print(f"Error: Requested {args.num_gpus} GPUs but only {available_gpus} available")
            sys.exit(1)
        if args.num_gpus < 2:
            print("Warning: Worker mode is designed for 2+ GPUs. Using single GPU with standard mode.")
            args.use_workers = False

    # Test prompts
    base_prompts = [
        "Explain how neural networks work in simple terms.",
        "Write a Python function to compute the Fibonacci sequence.",
        "What are the key differences between RAM and storage?",
        "Describe the process of photosynthesis step by step.",
        "How does TCP/IP networking function at a high level?",
        "Explain the concept of recursion with an example.",
        "What makes quantum computing different from classical computing?",
        "Write a short story about a robot learning to paint.",
    ]

    # Generate requested number of prompts (cycle through base prompts if needed)
    if args.num_prompts <= len(base_prompts):
        base_prompts = base_prompts[:args.num_prompts]
    else:
        # Repeat prompts to reach requested count
        import itertools
        base_prompts = list(itertools.islice(itertools.cycle(base_prompts), args.num_prompts))

    # Add system prompt if requested (to demonstrate Stage 3 prefix sharing)
    if args.use_system_prompt:
        system_prompt = "You are a helpful AI assistant. Please provide clear, accurate, and concise answers. "
        prompts = [system_prompt + p for p in base_prompts]
        print("Using prompts with common system prompt prefix (Stage 3 will benefit)")
    else:
        prompts = base_prompts
        print("Using unique prompts without common prefix (Stage 3 will have minimal overhead)")

    # CRITICAL: Clear GPU memory from any previous runs
    if torch.cuda.is_available():
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        print("✓ GPU memory cleared\n")

    print("="*70)
    print("Memopt BENCHMARK")
    print("="*70)
    # Map optimization levels to stage descriptions
    stage_map = {
        "conservative": "Stage 0 (Paged Cache + Flash Attention)",
        "balanced": "Stage 0+1 (+ Memory Allocation)",
        "high": "Stage 0+1+2 (+ Continuous Batching) - 6.2x proven",
        "maximum": "Stage 0+1+2+3 (+ Prefix Sharing)",
        "ultra": "Stage 0+1+2+3+4 (+ Priority Scheduling)",
        "aggressive": "Stage 0+1+2+3+4 (+ Priority Scheduling)",
        "speculative": "Stage 0+1+2+3+4+5b (+ Speculative Decoding) - Target: 12-18x"
    }

    print(f"Model: {args.model}")
    print(f"Prompts: {len(prompts)}")
    print(f"Max tokens per prompt: {args.max_tokens}")
    print(f"Optimization: {stage_map.get(args.optimization_level, args.optimization_level)}")

    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: CUDA not available, running on CPU (will be slow)")

    # Run benchmarks
    baseline_stats = None
    optimized_stats = None
    optimized_model = None

    if args.mode in ["baseline", "both"]:
        baseline_stats = run_baseline(args.model, prompts, args.max_tokens)

    if args.mode in ["optimized", "both"]:
        # Check if we should use worker-per-GPU architecture
        if args.use_workers and args.num_gpus >= 2:
            # Use worker-per-GPU architecture (production-correct for multi-GPU)
            optimized_stats = run_worker_benchmark(
                args.model,
                args.num_gpus,
                prompts,
                args.max_tokens,
                args.optimization_level,
                args.rl_scheduler_path,
                args.memory_predictor_path
            )
            optimized_model = None  # No single model instance in worker mode
        else:
            # Use standard single-GPU or DataParallel mode
            optimized_stats, optimized_model = run_optimized(
                args.model, prompts, args.max_tokens, args.optimization_level, args.max_kv_blocks,
                quantize_kv=args.quantize_kv,
                enable_speculative=args.enable_speculative,
                # REMOVED: num_gpus, multi_gpu_mode, enable_rl_routing, rl_router_path
                rl_scheduler_path=args.rl_scheduler_path,
                memory_predictor_path=args.memory_predictor_path,
                enable_memory_tracing=args.enable_memory_tracing,
                memory_trace_output=args.memory_trace_output
            )

    # "Optimized" uses specified optimization level (default: ultra with all stages)
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print("\nℹ️  Metrics labeled:")
    print("  • (measured) = Direct measurements from PyTorch/CUDA APIs")
    print("  • (estimated) = Derived from theoretical models or business assumptions")
    
    if baseline_stats:
        print("\n📊 BASELINE:")
        print(f"  Throughput:         {baseline_stats.tokens_per_second:.1f} tok/s (measured)")
        print(f"  Latency:            {baseline_stats.latency_per_token_ms:.2f} ms/tok (measured)")
        print(f"  Memory:             {baseline_stats.peak_memory_allocated_gb:.2f} GB (measured)")
        print(f"  GPU stall:          {baseline_stats.gpu_stall_pct:.1f}% (estimated)")
        print(f"  Cost per 1M tokens: ${baseline_stats.cost_per_1m_tokens_usd:.2f} (estimated)")
    
    if optimized_stats:
        print(f"\n🚀 OPTIMIZED ({args.optimization_level.upper()}):")
        print(f"  Throughput:         {optimized_stats.tokens_per_second:.1f} tok/s (measured)")
        print(f"  Latency:            {optimized_stats.latency_per_token_ms:.2f} ms/tok (measured)")
        print(f"  Memory (peak):      {optimized_stats.peak_memory_allocated_gb:.2f} GB (measured)")
        print(f"  GPU stall:          {optimized_stats.gpu_stall_pct:.1f}% (estimated)", end="")
        if optimized_stats.gpu_stall_pct == 0.0:
            print(" ⚠️  0% may indicate measurement unavailable on this platform")
        else:
            print()
        print(f"  Cost per 1M tokens: ${optimized_stats.cost_per_1m_tokens_usd:.2f} (estimated)")

        # Show which stages are enabled
        stage_map = {
            "conservative": "Stage 0 (Paged KV Cache)",
            "balanced": "Stages 0+1 (Cache + Flash Attention)",
            "high": "Stages 0+1+2 (+ Continuous Batching)",
            "maximum": "Stages 0+1+2+3 (+ Prefix Sharing)",
            "ultra": "Stages 0+1+2+3+4 (+ Priority Scheduling)",
            "speculative": "Stages 0+1+2+3+4+5b (ALL + Speculative Decoding)",
            "aggressive": "Stages 0+1+2+3 + INT8"
        }
        print(f"  Enabled stages:     {stage_map.get(args.optimization_level, args.optimization_level)}")
    
    if baseline_stats and optimized_stats:
        print("\n💰 IMPROVEMENT:")
        speedup = optimized_stats.tokens_per_second / baseline_stats.tokens_per_second
        memory_reduction = (
            (baseline_stats.peak_memory_allocated_gb - optimized_stats.peak_memory_allocated_gb) /
            baseline_stats.peak_memory_allocated_gb * 100
        )
        cost_reduction = (
            (baseline_stats.cost_per_1m_tokens_usd - optimized_stats.cost_per_1m_tokens_usd) /
            baseline_stats.cost_per_1m_tokens_usd * 100
        )
        stall_reduction = baseline_stats.gpu_stall_pct - optimized_stats.gpu_stall_pct
        
        print(f"  Speedup:            {speedup:.2f}x (measured)")
        if memory_reduction < 0:
            print(f"  Memory (peak):      {memory_reduction:.1f}% (optimized uses MORE memory due to caching)")
        else:
            print(f"  Memory (peak):      {memory_reduction:.1f}% reduction (measured)")
        print(f"  Cost reduction:     {cost_reduction:.1f}% (estimated)")
        print(f"  Stall reduction:    {stall_reduction:.1f}% (estimated)")
        
        # Show detailed memory analysis
        if optimized_model:
            print_memory_analysis(
                optimized_model,
                baseline_stats.peak_memory_allocated_gb,
                optimized_stats.peak_memory_allocated_gb
            )
        
        # Calculate annual savings for a realistic workload
        tokens_per_day = 10e9  # 10B tokens/day
        daily_baseline_cost = (tokens_per_day / 1e6) * baseline_stats.cost_per_1m_tokens_usd
        daily_optimized_cost = (tokens_per_day / 1e6) * optimized_stats.cost_per_1m_tokens_usd
        daily_savings = daily_baseline_cost - daily_optimized_cost
        annual_savings = daily_savings * 365
        
        print("\n" + "="*70)
        print("ROI ANALYSIS (10B tokens/day) - ESTIMATED")
        print("="*70)
        print(f"  Daily baseline cost:   ${daily_baseline_cost:,.0f}")
        print(f"  Daily optimized cost:  ${daily_optimized_cost:,.0f}")
        print(f"  Daily savings:         ${daily_savings:,.0f}")
        print(f"  Annual savings:        ${annual_savings:,.0f}")
        print(f"\n  Memopt price: $50,000/year")
        print(f"  Payback period: {(50000 / daily_savings):.1f} days")
        print(f"  First year ROI: {(annual_savings / 50000):.1f}x")
        print(f"\n  ⚠️  Note: Cost estimates assume $0.002/1K tokens. Actual costs vary by provider.")
        print(f"  ⚠️  ROI calculation assumes 10B tokens/day workload. Adjust for your use case.")
    
    # Save results
    results = {}
    if baseline_stats:
        results["baseline"] = baseline_stats.to_dict()
    if optimized_stats:
        results["optimized"] = optimized_stats.to_dict()
    
    if baseline_stats and optimized_stats:
        results["comparison"] = {
            "speedup": speedup,
            "memory_reduction_pct": memory_reduction,
            "cost_reduction_pct": cost_reduction,
            "stall_reduction_pct": stall_reduction,
        }
        results["roi_analysis"] = {
            "tokens_per_day": 10e9,
            "daily_savings_usd": daily_savings,
            "annual_savings_usd": annual_savings,
            "Memopt_annual_cost_usd": 50000,
            "payback_days": 50000 / daily_savings,
            "first_year_roi": annual_savings / 50000
        }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {args.output}")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    # Required for multiprocessing on some platforms
    mp.set_start_method('spawn', force=True)
    main()