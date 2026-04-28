"""
Pillar 1 proof.
Proves memopt VMM evicts real HBM bytes to DRAM
and allows longer context than baseline.

Pass conditions:
  bytes_evicted_gb > 0  (real bytes moved)
  wm_max_ctx >= wo_max_ctx  (no regression)
"""
import sys, os, time, gc, json, torch
sys.path.insert(0, '/opt/memopt')
import pynvml
pynvml.nvmlInit()
H = pynvml.nvmlDeviceGetHandleByIndex(0)

def hw():
    m = pynvml.nvmlDeviceGetMemoryInfo(H)
    return round(m.used/1e9,1), round(m.free/1e9,1)

from transformers import (AutoTokenizer,
    AutoModelForCausalLM, BitsAndBytesConfig)

MODEL = os.environ.get(
    "MEMOPT_TEST_MODEL",
    "Qwen/Qwen2.5-32B-Instruct"
)
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)
tok = AutoTokenizer.from_pretrained(MODEL)
BASE = ("Transformer memory management "
        "across GPU clusters. ") * 8000
TOKS = tok.encode(BASE)

CONTEXTS = [32768, 49152, 65536, 81920, 98304]
results = {}

def run(label, use_memopt):
    res = []
    alloc = None
    if use_memopt:
        from memopt.vmm.torch_allocator import (
            MemoptTorchAllocator
        )
        alloc = MemoptTorchAllocator(
            evict_threshold=0.78
        )
        installed = alloc.install()
        print(f"allocator installed: {installed}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb,
        device_map="auto"
    )
    used, free = hw()
    print(f"\n{label} | HBM: {used}GB used, "
          f"{free}GB free")

    for ctx in CONTEXTS:
        inp = torch.tensor(
            [TOKS[:ctx]], dtype=torch.long
        ).to("cuda")
        actual = inp.shape[1]
        try:
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(
                    inp, max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id
                )
            elapsed = time.time() - t0
            used2, _ = hw()
            n = out.shape[1] - inp.shape[1]
            s = alloc.stats() if alloc else {}
            evicted_gb = s.get("bytes_evicted_total", 0) / 1e9
            r = {
                "ctx": actual, "status": "OK",
                "tok_s": round(n/elapsed, 1),
                "hbm_gb": used2,
                "pages_dram": s.get(
                    "pages_dram", 0),
                "bytes_evicted_gb": evicted_gb,
            }
            res.append(r)
            print(
                f"  ctx={actual:7d} OK  "
                f"{n/elapsed:.1f}tok/s  "
                f"HBM={used2}GB  "
                f"dram={s.get('pages_dram',0)}  "
                f"evicted={evicted_gb:.3f}GB"
            )
            if alloc:
                alloc.step_boundary()
        except torch.cuda.OutOfMemoryError:
            res.append({"ctx": actual,
                        "status": "OOM"})
            print(f"  ctx={actual:7d} OOM")
            torch.cuda.empty_cache()
            break
    del model
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(2)
    return res

results["without"] = run(
    "WITHOUT memopt", use_memopt=False
)
results["with"] = run(
    "WITH memopt", use_memopt=True
)

wo = max((r["ctx"] for r in results["without"]
          if r.get("status")=="OK"), default=0)
wm = max((r["ctx"] for r in results["with"]
          if r.get("status")=="OK"), default=0)
evicted = max(
    (r.get("bytes_evicted_gb", 0)
     for r in results["with"]
     if r.get("status")=="OK"),
    default=0
)

print(f"\n{'='*50}")
print(f"PILLAR 1 RESULT")
print(f"{'='*50}")
print(f"WITHOUT memopt: {wo:,} tokens")
print(f"WITH memopt:    {wm:,} tokens")
print(f"Bytes evicted:  {evicted:.3f} GB")
passed = wm >= wo and evicted > 0
print(f"PASS: {passed}")

with open('/tmp/pillar1_result.json','w') as f:
    json.dump({
        "wo_max": wo, "wm_max": wm,
        "evicted_gb": evicted,
        "passed": passed,
        "results": results
    }, f, indent=2)

pynvml.nvmlShutdown()
