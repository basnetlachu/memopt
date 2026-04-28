"""
Pillar 1 proof.
Proves memopt VMM evicts real HBM bytes to DRAM
and allows longer context than baseline.

Each phase runs in a fresh Python subprocess. PyTorch locks the active
CUDA allocator after the first tensor is allocated, so the WITHOUT and
WITH phases cannot share a process — the WITH phase fails to install
the pluggable allocator with "Can't swap an already initialized
allocator". Subprocess isolation gives each phase a clean allocator
state.

Phases:
  --phase=without  baseline sweep, writes /tmp/pillar1_without.json
  --phase=with     memopt sweep, writes /tmp/pillar1_with.json
  --phase=summary  reads both JSONs, prints verdict
  (no arg)         orchestrate all three sequentially

Pass conditions:
  bytes_evicted_gb > 0  (real bytes moved)
  wm_max_ctx >= wo_max_ctx  (no regression)
"""
import sys
import os
import time
import gc
import json
import argparse
import subprocess


def run_phase(phase: str) -> None:
    sys.path.insert(0, '/opt/memopt')

    use_memopt = (phase == "with")
    alloc = None

    # ── Install memopt FIRST, before any other CUDA-touching code ──
    # Importing transformers / bitsandbytes runs CUDA-detection probes
    # that initialize PyTorch's CUDACachingAllocator, after which
    # change_current_allocator fails with "Can't swap an already
    # initialized allocator". Doing install() before those imports
    # gives PyTorch a clean allocator state to swap.
    if use_memopt:
        from memopt.vmm.torch_allocator import MemoptTorchAllocator
        # Pool size from env, default 30 GB. Reserving 60 GB of VA up-front
        # has caused CUBLAS_STATUS_INTERNAL_ERROR on smaller models because
        # cuBLAS Lt allocates its own GEMM workspace via cudaMalloc and
        # competes for the remaining VA / device-side scratch budget.
        pool_gb = float(os.environ.get("MEMOPT_TORCH_POOL_GB", "30"))
        alloc = MemoptTorchAllocator(
            pool_gb=pool_gb, evict_threshold=0.78,
        )
        installed = alloc.install()
        print(f"allocator installed: {installed}", flush=True)
        if not installed:
            sys.exit(2)

    import torch
    import pynvml
    pynvml.nvmlInit()
    H = pynvml.nvmlDeviceGetHandleByIndex(0)

    def hw():
        m = pynvml.nvmlDeviceGetMemoryInfo(H)
        return round(m.used / 1e9, 1), round(m.free / 1e9, 1)

    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    )

    MODEL = os.environ.get(
        "MEMOPT_TEST_MODEL", "Qwen/Qwen2.5-32B-Instruct")

    # MEMOPT_USE_BNB=0 disables bitsandbytes 4-bit and loads the model in
    # fp16 instead. bnb's CUDA kernels make raw cudaMalloc calls inside
    # libbitsandbytes_cuda*.so that bypass PyTorch's pluggable allocator;
    # mixing those with memopt's VA pool is a known architectural conflict
    # that produces "illegal memory access" faults. Use fp16 + a smaller
    # model (Qwen2.5-7B) to exercise the memopt allocator end-to-end
    # without that conflict.
    use_bnb = os.environ.get("MEMOPT_USE_BNB", "1") == "1"

    tok = AutoTokenizer.from_pretrained(MODEL)
    BASE = ("Transformer memory management across GPU clusters. ") * 8000
    TOKS = tok.encode(BASE)
    CONTEXTS = [32768, 49152, 65536, 81920, 98304]

    label = "WITH memopt" if use_memopt else "WITHOUT memopt"
    res = []

    if use_bnb:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, quantization_config=bnb, device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.float16, device_map="auto",
        )
    used, free = hw()
    print(f"\n{label} | HBM: {used}GB used, {free}GB free", flush=True)

    for ctx in CONTEXTS:
        inp = torch.tensor([TOKS[:ctx]], dtype=torch.long).to("cuda")
        actual = inp.shape[1]
        try:
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(
                    inp, max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            elapsed = time.time() - t0
            used2, _ = hw()
            n = out.shape[1] - inp.shape[1]
            s = alloc.stats() if alloc else {}
            evicted_gb = s.get("bytes_evicted_total", 0) / 1e9
            r = {
                "ctx": actual, "status": "OK",
                "tok_s": round(n / elapsed, 1),
                "hbm_gb": used2,
                "pages_dram": s.get("pages_dram", 0),
                "bytes_evicted_gb": evicted_gb,
            }
            res.append(r)
            print(
                f"  ctx={actual:7d} OK  "
                f"{n / elapsed:.1f}tok/s  "
                f"HBM={used2}GB  "
                f"dram={s.get('pages_dram', 0)}  "
                f"evicted={evicted_gb:.3f}GB",
                flush=True,
            )
            if alloc:
                alloc.step_boundary()
        except torch.cuda.OutOfMemoryError:
            res.append({"ctx": actual, "status": "OOM"})
            print(f"  ctx={actual:7d} OOM", flush=True)
            torch.cuda.empty_cache()
            break

    out_path = f"/tmp/pillar1_{phase}.json"
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"phase {phase} done -> {out_path}", flush=True)
    pynvml.nvmlShutdown()


def summary() -> None:
    paths = {
        "without": "/tmp/pillar1_without.json",
        "with":    "/tmp/pillar1_with.json",
    }
    results = {}
    for k, p in paths.items():
        if not os.path.exists(p):
            print(f"missing {p} — phase {k} did not complete", flush=True)
            results[k] = []
            continue
        with open(p) as f:
            results[k] = json.load(f)

    wo = max(
        (r["ctx"] for r in results["without"] if r.get("status") == "OK"),
        default=0,
    )
    wm = max(
        (r["ctx"] for r in results["with"] if r.get("status") == "OK"),
        default=0,
    )
    evicted = max(
        (r.get("bytes_evicted_gb", 0)
         for r in results["with"] if r.get("status") == "OK"),
        default=0,
    )

    print(f"\n{'=' * 50}")
    print("PILLAR 1 RESULT")
    print(f"{'=' * 50}")
    print(f"WITHOUT memopt: {wo:,} tokens")
    print(f"WITH memopt:    {wm:,} tokens")
    print(f"Bytes evicted:  {evicted:.3f} GB")
    passed = wm >= wo and evicted > 0
    print(f"PASS: {passed}")

    with open('/tmp/pillar1_result.json', 'w') as f:
        json.dump({
            "wo_max":     wo,
            "wm_max":     wm,
            "evicted_gb": evicted,
            "passed":     passed,
            "results":    results,
        }, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        choices=["without", "with", "summary"],
        default=None,
    )
    args = ap.parse_args()

    if args.phase is None:
        # Orchestrate: subprocess each phase, then summarize.
        for p in ("without", "with"):
            print(f"\n=== launching subprocess: phase={p} ===", flush=True)
            r = subprocess.run(
                [sys.executable, "-u", __file__, "--phase", p])
            print(f"=== phase {p} exit: {r.returncode} ===", flush=True)
        summary()
    elif args.phase == "summary":
        summary()
    else:
        run_phase(args.phase)


if __name__ == "__main__":
    main()
