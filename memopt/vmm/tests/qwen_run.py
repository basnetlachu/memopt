"""
Qwen 32B context-length comparison driver.

Runs ONE condition (baseline or with-memopt) and writes results JSON so
the two runs can be compared.

Invoked as: python3 qwen_run.py {baseline|memopt} <out_json>

Must be two processes because `torch.cuda.memory.change_current_allocator`
cannot be called after CUDA has already been used in-process.
"""
import json
import sys
import time
from pathlib import Path

if len(sys.argv) != 3 or sys.argv[1] not in ("baseline", "memopt"):
    print("usage: qwen_run.py {baseline|memopt} <out_json>", file=sys.stderr)
    sys.exit(2)
MODE, OUT_JSON = sys.argv[1], sys.argv[2]

sys.path.insert(0, "/opt/memopt")

# CRITICAL: install the allocator BEFORE any CUDA interaction.
if MODE == "memopt":
    from memopt.vmm.torch_allocator import install_memopt_allocator
    alloc = install_memopt_allocator(
        pool_gb=200.0,            # VA reservation (virtual, free). HBM is
        evict_threshold=0.80,     # bounded separately via cache cap.
        verbose=True,
    )
    print(f"MODE=memopt installed={alloc.is_installed()}", flush=True)
else:
    alloc = None
    print("MODE=baseline", flush=True)

import torch

# CUDAPluggableAllocator does not implement getDeviceStats. accelerate (via
# transformers) calls torch.cuda.memory_reserved/memory_allocated during
# device_map inference, which raises RuntimeError. Swallow and return 0 so
# the auto-device-map path doesn't crash. nvidia-smi / pynvml still report
# true HBM usage, so we don't lose observability.
if MODE == "memopt":
    _orig_reserved  = torch.cuda.memory_reserved
    _orig_allocated = torch.cuda.memory_allocated

    def _safe_reserved(*a, **kw):
        try: return _orig_reserved(*a, **kw)
        except RuntimeError: return 0

    def _safe_allocated(*a, **kw):
        try: return _orig_allocated(*a, **kw)
        except RuntimeError: return 0

    torch.cuda.memory_reserved  = _safe_reserved
    torch.cuda.memory_allocated = _safe_allocated

import pynvml
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

pynvml.nvmlInit()
nvh = pynvml.nvmlDeviceGetHandleByIndex(0)


def hw():
    m = pynvml.nvmlDeviceGetMemoryInfo(nvh)
    return {
        "hbm_gb":     round(m.used / 1e9, 2),
        "hbm_free_gb": round(m.free / 1e9, 2),
    }


# ── Model ────────────────────────────────────────────────────────────
MODEL = "Qwen/Qwen2.5-32B-Instruct"
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)
print(f"[{MODE}] loading tokenizer", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)

print(f"[{MODE}] loading model", flush=True)
t_load_0 = time.time()

# accelerate's device_map="auto" queries torch.cuda.memory_reserved, which
# CUDAPluggableAllocator does NOT implement. Pass max_memory explicitly so
# accelerate skips the probe.
from_pretrained_kwargs = dict(quantization_config=bnb, device_map="auto")
if MODE == "memopt":
    from_pretrained_kwargs["max_memory"] = {0: "45GiB", "cpu": "60GiB"}

model = AutoModelForCausalLM.from_pretrained(MODEL, **from_pretrained_kwargs)
t_load = time.time() - t_load_0
print(f"[{MODE}] model loaded in {t_load:.1f}s, HBM={hw()['hbm_gb']}GB",
      flush=True)

# Build a long tokenized corpus once.
BASE = ("Unified memory management across GPU clusters requires careful "
        "orchestration. ") * 20000  # ~220K tokens — enough for all ctx sizes
ALL_TOKS = tok.encode(BASE)
print(f"[{MODE}] corpus tokens: {len(ALL_TOKS)}", flush=True)

CONTEXTS = [16384, 32768, 49152, 65536, 81920, 98304]
results = []

for ctx in CONTEXTS:
    if ctx > len(ALL_TOKS):
        print(f"[{MODE}] ctx={ctx} exceeds corpus len, skipping", flush=True)
        break
    inp = torch.tensor([ALL_TOKS[:ctx]]).to("cuda")
    actual = inp.shape[1]
    row = {"ctx": actual}
    try:
        h0 = hw()
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                inp,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        if alloc is not None:
            s = alloc.step_boundary()
        else:
            s = {}
        elapsed = time.time() - t0
        h1 = hw()
        n_new = out.shape[1] - inp.shape[1]
        row.update({
            "status":  "OK",
            "tok_s":   round(n_new / elapsed, 1),
            "hbm_gb":  h1["hbm_gb"],
            "evictions":       s.get("eviction_count", 0),
            "promotions":      s.get("promotion_count", 0),
            "pages_hbm":       s.get("pages_hbm", 0),
            "pages_dram":      s.get("pages_dram", 0),
            "bytes_evicted_total": s.get("bytes_evicted_total", 0),
        })
        print(
            f"[{MODE}] ctx={actual:6d} OK  {row['tok_s']:.1f} tok/s  "
            f"HBM={row['hbm_gb']}GB  evict={row['evictions']}  "
            f"dram={row['pages_dram']}",
            flush=True,
        )
    except torch.cuda.OutOfMemoryError as e:
        row["status"] = "OOM"
        row["error"]  = str(e)[:200]
        print(f"[{MODE}] ctx={actual:6d} OOM", flush=True)
        torch.cuda.empty_cache()
        results.append(row)
        break
    except Exception as e:
        row["status"] = "ERR"
        row["error"]  = f"{type(e).__name__}: {str(e)[:200]}"
        print(f"[{MODE}] ctx={actual:6d} ERR: {row['error']}", flush=True)
        results.append(row)
        break
    results.append(row)
    del inp, out
    torch.cuda.empty_cache()

max_ctx = max(
    (r["ctx"] for r in results if r.get("status") == "OK"),
    default=0,
)
Path(OUT_JSON).write_text(
    json.dumps({"mode": MODE, "max_ctx": max_ctx,
                "load_sec": round(t_load, 1),
                "model": MODEL, "results": results}, indent=2)
)
print(f"[{MODE}] max_ctx_OK={max_ctx}  wrote {OUT_JSON}", flush=True)

pynvml.nvmlShutdown()
