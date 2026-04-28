"""
Run GPT-2, Qwen2.5-7B, and Qwen2.5-32B through the memopt pillar stack.

For each model we:
  - Load via transformers
  - Run 20 prompts, 100 tokens each
  - Record per-token latency
  - Register prompt prefixes in GKDStore, count dedup hits on a repeated-
    prefix workload
  - Allocate KV blocks through VMM, measure allocations + prefetch stats
  - Record energy in OptimizationLedger
  - Measure GPU memory via pynvml before + during + after
"""
from __future__ import annotations
import gc
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, "/root/memopt")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memopt.vmm import VMM
from memopt.cluster.gkd_store import GKDStore
from memopt.observability.collector import MetricRegistry
from memopt.observability.ledger import OptimizationLedger

try:
    import pynvml
    pynvml.nvmlInit()
    NVML = True
except Exception:
    NVML = False


def gpu_mem_mb() -> float:
    if not NVML:
        return 0.0
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    return pynvml.nvmlDeviceGetMemoryInfo(h).used / 1024 / 1024


def gpu_power_w() -> float:
    if not NVML:
        return 0.0
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
    except Exception:
        return 0.0


PROMPTS_BASE = [
    "Explain the difference between paging and swapping in operating systems.",
    "Write a short essay on why GPU memory hierarchy matters for LLM inference.",
    "Describe what a KV cache is and why it matters for transformer inference.",
    "Give three reasons why RDMA transport benefits large language model serving.",
    "What is the roofline model in performance engineering?",
    "Compare HBM3 and HBM2e in terms of bandwidth and power draw.",
    "What does FlashAttention do that standard attention does not?",
    "Describe one advantage of content-addressed KV cache deduplication.",
    "Why does continuous batching help LLM serving throughput?",
    "Explain prefetching in the context of memory hierarchies.",
]


def run_model(model_name: str, n_repeat_prompts: int = 2,
              new_tokens: int = 64,
              dtype=torch.bfloat16,
              use_ledger: Optional[OptimizationLedger] = None) -> dict:
    print(f"\n{'=' * 70}\n  LOADING {model_name}\n{'=' * 70}")
    t_load = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mem_before = gpu_mem_mb()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="cuda",
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    model.eval()
    load_s = round(time.perf_counter() - t_load, 1)
    mem_after_load = gpu_mem_mb()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  loaded in {load_s}s, {total_params/1e9:.2f} B params, "
          f"GPU mem used: {mem_after_load:.0f} MB")

    # Pillar 2: GKD — register duplicated prompt prefixes, measure hit rate
    gkd = GKDStore(backend="local", node_id="h100-test")
    repeated_prompts = PROMPTS_BASE * n_repeat_prompts
    gkd_hits = 0
    gkd_misses = 0
    # First pass: register; second and later repeats should see hits.
    for i, p in enumerate(repeated_prompts):
        tokens = tok.encode(p, add_special_tokens=False)
        hit = gkd.lookup(tokens, len(tokens), tenant_id="big_model")
        if hit is not None:
            gkd_hits += 1
        else:
            gkd_misses += 1
            gkd.register(tokens, len(tokens), block_ref=f"{model_name}:{i}",
                         node_id="h100-test", tenant_id="big_model",
                         size_bytes=len(tokens) * 2)
    gkd_stats = gkd.stats()

    # Pillar 1: VMM — simulate KV block allocations per prompt
    vmm = VMM()
    kv_block_bytes = 2 * model.config.hidden_size * 128 * 2  # rough
    kv_blocks = 0
    for i in range(len(PROMPTS_BASE)):
        for b in range(4):
            vmm.allocate(f"run_{i}", b, kv_block_bytes, tenant_id="big_model")
            kv_blocks += 1

    # Inference loop — measure throughput + power
    latencies = []
    token_counts = []
    power_samples = []
    t_total_0 = time.perf_counter()
    with torch.no_grad():
        for i, p in enumerate(PROMPTS_BASE):
            inputs = tok(p, return_tensors="pt").to("cuda")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.generate(
                **inputs, max_new_tokens=new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id or tok.pad_token_id or 0,
            )
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            latencies.append(dt)
            generated = out.shape[1] - inputs.input_ids.shape[1]
            token_counts.append(generated)
            power_samples.append(gpu_power_w())
            if use_ledger is not None:
                use_ledger.record(
                    tokens=generated, tenant_id="big_model",
                    actual_j_per_token=0.0005,  # placeholder; real NVML needed
                )
    total_s = time.perf_counter() - t_total_0
    total_tokens = sum(token_counts)
    tok_per_s = round(total_tokens / total_s, 2) if total_s > 0 else 0.0

    mem_peak = max(mem_after_load, gpu_mem_mb())
    p_mean = round(sum(power_samples) / len(power_samples), 1) if power_samples else 0.0

    # Shutdown and free
    for i in range(len(PROMPTS_BASE)):
        try:
            vmm.free_sequence(f"run_{i}", tenant_id="big_model")
        except Exception:
            pass
    del model
    gc.collect()
    torch.cuda.empty_cache()
    mem_after_free = gpu_mem_mb()

    return {
        "model": model_name,
        "load_seconds": load_s,
        "param_count_b": round(total_params / 1e9, 3),
        "dtype": str(dtype),
        "gpu_mem_after_load_mb": round(mem_after_load, 0),
        "gpu_mem_peak_mb": round(mem_peak, 0),
        "gpu_mem_after_free_mb": round(mem_after_free, 0),
        "total_tokens_generated": total_tokens,
        "total_latency_s": round(total_s, 2),
        "tokens_per_sec": tok_per_s,
        "mean_gpu_power_w": p_mean,
        "latency_per_prompt_s": [round(x, 3) for x in latencies],
        "vmm_kv_blocks_allocated": kv_blocks,
        "gkd_hits_on_repeats": gkd_hits,
        "gkd_misses_on_first_pass": gkd_misses,
        "gkd_stats": gkd_stats,
    }


if __name__ == "__main__":
    # Ledger captures energy+cost across all three model runs
    ledger = OptimizationLedger(db_path="/root/memopt/big_model_ledger.db")
    results = {"summary": {}, "runs": []}
    # Progressively larger models
    models = [
        ("gpt2", torch.float32, 2),
        ("Qwen/Qwen2.5-7B-Instruct", torch.bfloat16, 2),
        ("Qwen/Qwen2.5-32B-Instruct", torch.bfloat16, 2),
    ]
    for model_name, dtype, n_repeat in models:
        try:
            r = run_model(model_name, n_repeat_prompts=n_repeat,
                          new_tokens=64, dtype=dtype, use_ledger=ledger)
            results["runs"].append(r)
        except Exception as e:
            results["runs"].append({
                "model": model_name,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })
        gc.collect()
        torch.cuda.empty_cache()

    try:
        ledger.flush()
    except Exception:
        pass
    results["ledger_totals"] = ledger.totals(tenant_id="big_model")

    with open("/root/memopt/big_model_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n\n=== FINAL ===")
    print(json.dumps(results, indent=2, default=str)[:5000])
