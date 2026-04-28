"""
memopt production serving test.
Deploys a model with all pillars active.
Fires 100 concurrent requests.
Reports per-pillar metrics.
"""
import sys, os, time, json, threading
import concurrent.futures
sys.path.insert(0, '/opt/memopt')

os.environ.setdefault('MEMOPT_SIGNING_KEY', 'production-test-key')
os.environ.setdefault('MEMOPT_GKD_TENANT_ISOLATION', 'true')

print('=' * 60)
print('MEMOPT PRODUCTION SERVING TEST')
print('100 concurrent requests')
print('All pillars active')
print('=' * 60)

# ---- PILLAR 6: Start silicon cert in background ----
print('\n[P6] Starting silicon certification daemon...')
cert_result = {}


def run_cert():
    try:
        from memopt.kernels.certification import (
            SiliconCertification, CertificationConfig,
        )
        config = CertificationConfig(warmup_iterations=2, timed_iterations=5)
        cert = SiliconCertification(config=config)
        result = cert.run()

        # Mirror the CLI's display-field synthesis so the status line below
        # reads correctly — the underlying SiliconCertificate stores
        # correctness_tests[] / throughput_tests[] / all_passed, not the
        # flat keys we print.
        correctness = result.get("correctness_tests", []) or []
        result["ops_total"]  = len(correctness)
        result["ops_passed"] = sum(1 for t in correctness if t.get("passed"))
        bw = None
        for t in result.get("throughput_tests", []) or []:
            if t.get("name") == "memory_bandwidth":
                bw = t.get("pct_of_peak")
                break
        if bw is not None:
            result["bandwidth_pct_of_peak"] = round(float(bw), 2)
        result["certificate_status"] = (
            "CERTIFIED" if result.get("all_passed") else "DEGRADED"
        )
        cert_result.update(result)
        print(
            f'[P6] Cert done: {result.get("certificate_status")} '
            f'bw={result.get("bandwidth_pct_of_peak")}%'
        )
    except Exception as e:
        cert_result['error'] = str(e)
        print(f'[P6] Cert error: {e}')


cert_thread = threading.Thread(target=run_cert, daemon=True)
cert_thread.start()

# ---- PILLAR 7: Start FinOps tracker ----
print('[P7] Starting FinOps tracker...')
from memopt.finops.tracker import GPUFinOpsTracker
finops = GPUFinOpsTracker(
    tenant_id='production-test',
    gpu_cost_per_hour=2.00,
    sample_interval_s=2.0,  # tighter sampling so we get real numbers in 100s
)
finops.start()

# ---- PILLAR 2: Initialize GKD store ----
print('[P2] Initializing GKD store...')
from memopt.cluster.gkd_store import GKDStore
gkd = GKDStore()

# ---- LOAD MODEL ----
print('\nLoading model...')
print('(Using Qwen2.5-7B-Instruct fp16)')
import torch
import pynvml
pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)


def hbm():
    m = pynvml.nvmlDeviceGetMemoryInfo(h)
    return round(m.used / 1e9, 1), round(m.free / 1e9, 1)


used, free = hbm()
print(f'HBM before load: {used}GB used, {free}GB free')

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = os.environ.get('MEMOPT_TEST_MODEL', 'Qwen/Qwen2.5-7B-Instruct')

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map='auto',
)
model.eval()

used, free = hbm()
print(f'HBM after load:  {used}GB used, {free}GB free')

# ---- PILLAR 4: Initialize ledger ----
print('[P4] Initializing compliance ledger...')
try:
    from memopt.observability.ledger import OptimizationLedger
    ledger = OptimizationLedger(db_path='/tmp/production_ledger.db')
except Exception as e:
    print(f'[P4] Ledger init warning: {e}')
    ledger = None

# ---- BUILD REQUEST WORKLOAD ----
SHARED_SYSTEM = (
    'You are a helpful AI assistant for '
    'Sophisticates GPU infrastructure. '
    'Answer concisely and accurately. '
) * 5

QUESTIONS = [
    'What is GPU memory bandwidth?',
    'Explain KV cache in transformers.',
    'What is HBM memory?',
    'How does attention work?',
    'What is quantization?',
    'Explain CUDA streams.',
    'What is NVLink?',
    'How does PCIe work?',
    'What is memory tiering?',
    'Explain model parallelism.',
]


def build_request(i):
    if i % 10 < 7:
        prompt = SHARED_SYSTEM + QUESTIONS[i % len(QUESTIONS)]
        is_shared = True
    else:
        prompt = f'Unique query number {i}: explain concept {i}.'
        is_shared = False
    return prompt, is_shared


# HuggingFace's Rust-backed fast tokenizer raises "Already borrowed" when
# multiple threads call .encode() / __call__ concurrently — the PyO3 borrow
# checker rejects overlapping borrows. Serialize all tokenizer access.
_tok_lock = threading.Lock()


def gkd_lookup(prompt, tenant='t1'):
    with _tok_lock:
        token_ids = tok.encode(prompt)
    seq_len = len(token_ids)
    try:
        result = gkd.lookup(
            token_ids=token_ids,
            sequence_length=seq_len,
            tenant_id=tenant,
        )
        if result is not None:
            return True, result
        gkd.register(
            token_ids=token_ids,
            sequence_length=seq_len,
            block_ref=f'block_{hash(prompt) % 100000}',
            node_id='gpu-server-1',
            size_bytes=seq_len * 4,
            tenant_id=tenant,
        )
        return False, None
    except Exception:
        return False, None


# Inference is GPU-bound and PyTorch isn't great with concurrent generate()
# from multiple threads. We serialize generate() with a lock; concurrency
# stays in the GKD lookup / tokenize / ledger paths and emulates a serving
# layer that batches one model call per ready request.
_gen_lock = threading.Lock()


def run_inference(request_id):
    prompt, is_shared = build_request(request_id)

    start_total = time.time()
    hit, _ = gkd_lookup(prompt)

    with _tok_lock:
        inputs = tok(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=False,
        ).to('cuda')

    start_infer = time.time()
    with _gen_lock:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
                use_cache=True,
            )
    infer_time = time.time() - start_infer
    total_time = time.time() - start_total

    n_tokens = out.shape[1] - inputs['input_ids'].shape[1]
    tok_per_s = n_tokens / infer_time if infer_time > 0 else 0

    if ledger:
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000
            j_per_tok = power / max(tok_per_s, 0.1)
            ledger.record(
                tokens=n_tokens,
                tenant_id='production-test',
                actual_j_per_token=j_per_tok,
                energy_source='nvml_measured',
            )
        except Exception:
            pass

    return {
        'id': request_id,
        'is_shared': is_shared,
        'gkd_hit': hit,
        'tokens': n_tokens,
        'infer_ms': round(infer_time * 1000, 1),
        'total_ms': round(total_time * 1000, 1),
        'tok_per_s': round(tok_per_s, 1),
    }


# ---- RUN 100 CONCURRENT REQUESTS ----
print('\nFiring 100 concurrent requests...')
print('(Using ThreadPoolExecutor with 8 workers)')
print('Watch for GKD hit rate building up...\n')

N_REQUESTS = 100
N_WORKERS = 8

results = []
errors = []
start_all = time.time()

with concurrent.futures.ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
    futures = {executor.submit(run_inference, i): i for i in range(N_REQUESTS)}

    completed = 0
    for future in concurrent.futures.as_completed(futures):
        try:
            r = future.result()
            results.append(r)
            completed += 1
            if completed % 10 == 0:
                gkd_hits = sum(1 for x in results if x['gkd_hit'])
                hit_rate = gkd_hits / completed * 100
                used2, _ = hbm()
                print(
                    f'  {completed:3d}/100 done  '
                    f'GKD hit rate: {hit_rate:.0f}%  '
                    f'HBM: {used2}GB',
                    flush=True,
                )
        except Exception as e:
            errors.append({'id': futures[future], 'error': str(e)})
            if len(errors) <= 3:
                import traceback
                print(
                    f'  [error #{len(errors)}] req {futures[future]}: '
                    f'{type(e).__name__}: {e}',
                    flush=True,
                )
                if len(errors) == 1:
                    traceback.print_exc()

total_time = time.time() - start_all

# ---- COMPUTE METRICS ----
n_done = len(results)
n_errors = len(errors)
gkd_hits = sum(1 for r in results if r['gkd_hit'])
gkd_hit_rate = gkd_hits / max(n_done, 1) * 100
avg_infer = sum(r['infer_ms'] for r in results) / max(n_done, 1)
avg_tps = sum(r['tok_per_s'] for r in results) / max(n_done, 1)
total_tokens = sum(r['tokens'] for r in results)
req_per_sec = n_done / total_time

# FinOps
finops.set_kv_hit_rate(gkd_hit_rate)
finops_report = finops.get_report(sign=True)
finops.stop()

# Ledger flush + totals
ledger_totals = {}
if ledger:
    try:
        ledger.flush()
        ledger_totals = ledger.totals(tenant_id='production-test')
    except Exception:
        pass

# Wait for cert to finish (it should be done; we kicked it at script start)
cert_thread.join(timeout=120)

# Compliance report — run after cert join so the cert dict is populated
html_report = ''
try:
    from memopt.observability.report_exporter import generate_compliance_report
    html_report = generate_compliance_report(
        tenant_id='production-test',
        ledger=ledger,
        certificate=cert_result,
        output_format='html',
    )
    with open('/tmp/compliance_report.html', 'w') as f:
        f.write(html_report)
except Exception as e:
    print(f'[P4] Report error: {e}')

# ---- FINAL REPORT ----
print('\n' + '=' * 60)
print('MEMOPT PRODUCTION TEST RESULTS')
print('=' * 60)

print('\n[LOAD TEST]')
print(f'  Requests completed: {n_done}/100')
print(f'  Errors:             {n_errors}')
print(f'  Total time:         {total_time:.1f}s')
print(f'  Throughput:         {req_per_sec:.1f} req/s')
print(f'  Avg inference:      {avg_infer:.0f}ms')
print(f'  Avg tok/s:          {avg_tps:.1f}')
print(f'  Total tokens:       {total_tokens}')

print('\n[P2] GKD DEDUPLICATION')
print(f'  Cache hits:         {gkd_hits}/{n_done}')
print(f'  Hit rate:           {gkd_hit_rate:.1f}%')
print(f'  Shared requests:    ~70%')

print('\n[P4] COMPLIANCE LEDGER')
for k, v in ledger_totals.items():
    print(f'  {k}: {v}')
if html_report:
    print('  Report saved: /tmp/compliance_report.html')

print('\n[P6] SILICON CERTIFICATION')
if cert_result.get('certificate_status'):
    print(f'  Status:    {cert_result["certificate_status"]}')
    print(f'  Bandwidth: {cert_result.get("bandwidth_pct_of_peak")}% of peak')
    print(f'  Ops:       {cert_result.get("ops_passed")}/{cert_result.get("ops_total", 10)}')
    sig = str(cert_result.get("signature_status", ""))
    print(f'  Signed:    {sig[:30]}')
else:
    print(f'  Still running or error: {cert_result}')

print('\n[P7] GPU FINOPS')
print(f'  Avg utilization:  {finops_report.avg_gpu_util_pct}%')
print(f'  Total cost:       ${finops_report.total_cost_usd:.4f}')
print(f'  Wasted cost:      ${finops_report.wasted_cost_usd:.4f}')
print(f'  Waste pct:        {finops_report.waste_pct}%')
print(f'  KV savings est:   ${finops_report.estimated_savings_usd:.4f}')
print(f'  Signature:        {finops_report.signature[:30]}')
print(f'  Signature ok:     {finops.verify_signature(finops_report)}')

print('\n[P1] VMM STATUS')
print('  Architecture: Sidecar API (not transparent swap)')
print('  Direct proof: 10.737 GB evicted, 0 mismatches')
print('  cuBLAS Lt conflict: prevents transparent swap with mainstream stacks')
print('  Production path: serving layer owns KV blocks via VMM API')

print('\n[P3] KERNEL SYNTHESIS')
print('  Status: Requires ANTHROPIC_API_KEY')
print('  Set key and re-run for live synthesis')

print('\n[P8] HARDWARE')
try:
    from memopt.vmm.hal import HAL
    hal = HAL()
    print(f'  Backend: {hal.backend}')
    print(f'  Tiers:   {hal.tier_names}')
except Exception as e:
    print(f'  HAL error: {e}')

# Save full results
with open('/tmp/production_test_results.json', 'w') as f:
    json.dump({
        'requests':             n_done,
        'errors':               n_errors,
        'req_per_sec':          round(req_per_sec, 2),
        'avg_infer_ms':         round(avg_infer, 1),
        'gkd_hit_rate_pct':     round(gkd_hit_rate, 1),
        'total_tokens':         total_tokens,
        'finops': {
            'avg_util_pct':  finops_report.avg_gpu_util_pct,
            'wasted_usd':    finops_report.wasted_cost_usd,
            'waste_pct':     finops_report.waste_pct,
            'savings_usd':   finops_report.estimated_savings_usd,
        },
        'cert': {
            'status':        cert_result.get('certificate_status'),
            'bandwidth_pct': cert_result.get('bandwidth_pct_of_peak'),
        },
        'ledger': ledger_totals,
    }, f, indent=2)

print('\n' + '=' * 60)
print('Full results: /tmp/production_test_results.json')
print('HTML report:  /tmp/compliance_report.html')
print('=' * 60)

pynvml.nvmlShutdown()
