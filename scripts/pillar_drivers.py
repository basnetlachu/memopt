"""
Pillar feature drivers for end-to-end verification on the H100.

Each pillar's real Python API is exercised with a non-trivial workload.
Results are collected into a single JSON document written to
/root/memopt/pillar_results.json.
"""
from __future__ import annotations
import json
import os
import sys
import time
import traceback
import socket
import threading
from pathlib import Path

sys.path.insert(0, "/root/memopt")

RESULTS: dict = {}


def pillar(name: str):
    def deco(fn):
        def wrap():
            print(f"\n=== {name} ===")
            t0 = time.perf_counter()
            try:
                out = fn()
                out["_status"] = "ok"
            except Exception as e:
                out = {"_status": "error", "error": str(e),
                       "traceback": traceback.format_exc()}
            out["_elapsed_s"] = round(time.perf_counter() - t0, 3)
            RESULTS[name] = out
            print(json.dumps(out, indent=2, default=str)[:1500])
            return out
        return wrap
    return deco


# ──────────────────────────────────────────────────────────────────────
# Pillar 1 — Infinite Context VMM
# ──────────────────────────────────────────────────────────────────────
@pillar("P1_VMM")
def p1_vmm():
    from memopt.vmm import VMM
    vmm = VMM()
    block_size = 128 * 1024  # 128 KB
    n_blocks = 200
    # Allocate across several sequences
    for s in range(4):
        for b in range(n_blocks):
            vmm.allocate(f"seq_{s}", b, block_size, tenant_id=f"t{s%2}")
    # Fetch with pattern that should trigger prefetch learning
    hits = 0
    for b in range(n_blocks):
        for s in range(4):
            try:
                vmm.fetch(f"seq_{s}", b, tenant_id=f"t{s%2}")
                hits += 1
            except Exception:
                pass
    stats = vmm.stats()
    # Free one sequence to test reclaim
    vmm.free_sequence("seq_0", tenant_id="t0")
    after_free = vmm.stats()
    # Page-table backend: C++ if _memopt_core loaded
    try:
        from memopt import _memopt_core  # noqa
        pt_backend = "cpp"
    except ImportError:
        pt_backend = "python"
    return {
        "block_size_bytes": block_size,
        "allocated_blocks": 4 * n_blocks,
        "fetches_succeeded": hits,
        "page_table_backend": pt_backend,
        "stats_after_alloc": stats,
        "stats_after_free": after_free,
    }


# ──────────────────────────────────────────────────────────────────────
# Pillar 2 — GKD
# ──────────────────────────────────────────────────────────────────────
@pillar("P2_GKD")
def p2_gkd():
    from memopt.cluster.gkd_store import GKDStore
    from memopt.cluster.prefix_index import find_lcp
    store = GKDStore(backend="local", node_id="h100-test")
    # Register 500 token sequences
    import random
    random.seed(42)
    base = list(range(2048))
    reg_count = 0
    for i in range(500):
        tokens = base[:1024] + [random.randint(0, 50000) for _ in range(64)]
        store.register(tokens, len(tokens), block_ref=f"blk_{i}",
                       node_id="h100-test", tenant_id="t0", size_bytes=8192)
        reg_count += 1
    # Exact-hit lookup
    exact_tokens = base[:1024] + [1, 2, 3]  # won't match unless we registered this
    exact = store.lookup(base[:1024] + [0, 0, 0, 0], 1028, tenant_id="t0")  # may or may not hit
    # Prefix match via LCP on two token lists
    lcp_len = find_lcp(list(range(128)), list(range(100)) + [999] * 28)
    # Register identical prompt twice to force a hit
    pdup = [7] * 256
    store.register(pdup, 256, block_ref="blk_dup", node_id="h100-test",
                   tenant_id="t0", size_bytes=1024)
    hit = store.lookup(pdup, 256, tenant_id="t0")
    stats = store.stats()
    # SIMD ISA?
    try:
        from memopt._memopt_simd import detected_isa
        isa = detected_isa()
    except Exception:
        isa = "python"
    return {
        "registrations": reg_count,
        "lcp_same_prefix_len": lcp_len,
        "exact_dup_hit": hit is not None,
        "hit_ref": getattr(hit, "block_ref", None),
        "simd_isa": isa,
        "stats": stats,
    }


# ──────────────────────────────────────────────────────────────────────
# Pillar 3 — Claude-powered JIT synthesis
# ──────────────────────────────────────────────────────────────────────
@pillar("P3_JIT")
def p3_jit():
    import torch
    from memopt.kernels.jit_generator import JITGenerator
    from memopt.kernels.kernel_cache import KernelCache
    from memopt.kernels.portability_layer import PortabilityLayer
    from memopt.kernels.bottleneck_detector import BottleneckEvent
    cache = KernelCache()
    port = PortabilityLayer()
    prof = port.profile()
    gen = JITGenerator(cache=cache, portability=port)
    # Fabricate a bottleneck event for a fused RoPE kernel
    ev = BottleneckEvent(
        op_name="apply_rope",
        input_shapes=[[1, 8, 256, 64], [256, 64], [256, 64]],
        dtype="float16",
        access_pattern="strided",
        stall_rate=0.72,
        hardware=f"cuda:{prof.arch_name}",
    )
    attempted = True
    result = None
    err = None
    t0 = time.perf_counter()
    try:
        result = gen.generate(ev.op_name, hardware_profile=prof)
    except Exception as e:
        err = str(e)
    synth_s = round(time.perf_counter() - t0, 2)
    # Also try flash_attention synthesis — exercises the real prompt path
    fa_result = None
    fa_err = None
    fa_s = 0.0
    if hasattr(gen, "synthesize_flash_attention"):
        import torch
        q = torch.randn(1, 8, 256, 64, dtype=torch.float16, device="cuda")
        k_ = torch.randn_like(q)
        v = torch.randn_like(q)
        tfa = time.perf_counter()
        try:
            fa_result = gen.synthesize_flash_attention(q, k_, v, causal=True)
        except Exception as e:
            fa_err = str(e)
        fa_s = round(time.perf_counter() - tfa, 2)
    return {
        "hardware_profile": {
            "arch_name": prof.arch_name,
            "backend": prof.backend,
            "device_name": prof.device_name,
            "compute_cap": prof.compute_cap,
            "max_block_size": prof.max_block_size,
        },
        "api_key_set": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "synthesis_attempted": attempted,
        "synthesis_seconds": synth_s,
        "result_type": type(result).__name__ if result is not None else None,
        "result_truthy": bool(result),
        "error": err,
        "flash_attention_synth_s": fa_s,
        "flash_attention_result_type": type(fa_result).__name__ if fa_result is not None else None,
        "flash_attention_error": fa_err,
    }


# ──────────────────────────────────────────────────────────────────────
# Pillar 4 — Collector + Ledger + Certificate + Arbitrage
# ──────────────────────────────────────────────────────────────────────
@pillar("P4_Observability")
def p4_obs():
    from memopt.observability.collector import MetricsCollector, MetricRegistry
    from memopt.observability.ledger import OptimizationLedger
    from memopt.observability.certificate import sign_entry, verify_certificate
    reg = MetricRegistry()
    reg.set("memopt_test_latency_ms", 12.5, labels={"op": "rope"})
    reg.set("memopt_test_stall_rate", 0.42)
    prom_text = reg.prometheus_text()
    ledger = OptimizationLedger(db_path="/tmp/ledger_test.db")
    # Record synthetic batches
    for i in range(10):
        ledger.record(
            tokens=1024,
            tenant_id="t0",
            actual_j_per_token=0.0005 + i * 1e-6,
        )
    try:
        ledger.flush()
    except Exception:
        pass
    totals = ledger.totals(tenant_id="t0")
    # Signature round-trip
    cert = sign_entry({
        "node_id": "h100-test", "tokens": 1024,
        "energy_saved_kwh": 0.001, "co2_saved_kg": 0.0002,
        "cost_saved_usd": 0.0001,
    })
    verified = verify_certificate(cert, os.environ.get("MEMOPT_SIGNING_KEY", ""))
    return {
        "prom_text_bytes": len(prom_text),
        "ledger_totals": totals,
        "cert_status": cert.get("signature_status") if isinstance(cert, dict) else str(type(cert).__name__),
        "cert_verified": bool(verified),
    }


# ──────────────────────────────────────────────────────────────────────
# Pillar 5 — Cross-node block protocol (loopback)
# ──────────────────────────────────────────────────────────────────────
@pillar("P5_GUM")
def p5_gum():
    from memopt.cluster.block_directory import LocalBlockDirectory, BlockEntry
    from memopt.cluster.remote_block import RemoteBlockServer, RemoteBlockClient
    import tempfile
    # Two "nodes" on the same box, different ports
    dir_a = LocalBlockDirectory(node_id="node-a")
    dir_b = LocalBlockDirectory(node_id="node-b")
    # Write a block file and register it on node-a
    tmp = tempfile.NamedTemporaryFile(delete=False)
    payload = b"Hello, GUM!" * 1024  # 11 KB
    tmp.write(payload)
    tmp.close()
    entry = BlockEntry(
        content_hash="sha256_test_01",
        node_id="node-a",
        tier="nvme",
        path=tmp.name,
        size_bytes=len(payload),
    )
    dir_a.register(entry)

    def read_block(path: str):
        try:
            with open(path, "rb") as f:
                return f.read()
        except Exception:
            return None

    server = RemoteBlockServer(
        node_id="node-a",
        block_directory=dir_a,
        read_block_fn=read_block,
        port=19516,
    )
    server.start()
    time.sleep(0.3)
    client = RemoteBlockClient(node_id="node-b", timeout_s=2.0)
    fetched = None
    err = None
    t0 = time.perf_counter()
    try:
        fetched = client.fetch_block(
            content_hash="sha256_test_01",
            remote_host="127.0.0.1",
            remote_port=19516,
        )
    except Exception as e:
        err = str(e)
    lat_ms = round((time.perf_counter() - t0) * 1000, 2)
    server.stop()
    ok = fetched is not None and len(fetched) == len(payload)
    try:
        Path(tmp.name).unlink()
    except Exception:
        pass
    return {
        "transferred_bytes": len(fetched) if fetched else 0,
        "expected_bytes": len(payload),
        "fetch_ok": ok,
        "fetch_latency_ms": lat_ms,
        "error": err,
    }


# ──────────────────────────────────────────────────────────────────────
# Pillar 6 — Silicon certification
# ──────────────────────────────────────────────────────────────────────
@pillar("P6_Certification")
def p6_cert():
    from memopt.kernels.certification import run_certification
    from memopt.kernels.drift_detector import DriftDetector
    cert = run_certification()
    # The certification returns a dataclass — extract numbers
    summary = {
        "device_name": getattr(cert, "device_name", None),
        "compute_cap": getattr(cert, "compute_cap", None),
        "all_passed": getattr(cert, "all_passed", None),
        "signature_status": getattr(cert, "signature_status", None),
        "n_correctness": len(getattr(cert, "correctness_tests", []) or []),
        "n_throughput": len(getattr(cert, "throughput_tests", []) or []),
        "correctness": [{"name": t.name, "dtype": t.dtype, "passed": t.passed,
                         "max_err": t.max_err}
                        for t in getattr(cert, "correctness_tests", []) or []],
        "throughput": [{"name": t.name, "achieved_gb_s": t.achieved_gb_s,
                        "theoretical_gb_s": t.theoretical_gb_s,
                        "pct_of_peak": t.pct_of_peak}
                       for t in getattr(cert, "throughput_tests", []) or []],
    }
    # Drift detector round-trip
    dd = DriftDetector(node_id="h100-test")
    for t in summary["throughput"]:
        if t["name"] == "memory_bandwidth":
            dd.record(t["pct_of_peak"])
    summary["drift_stats"] = dd.stats()
    return summary


if __name__ == "__main__":
    p1_vmm()
    p2_gkd()
    p3_jit()
    p4_obs()
    p5_gum()
    p6_cert()
    out_path = "/root/memopt/pillar_results.json"
    with open(out_path, "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print(f"\n\n=== All results written to {out_path} ===")
    # Short verdict summary
    print("\nSUMMARY:")
    for k, v in RESULTS.items():
        status = v.get("_status", "?")
        print(f"  {k:20} {status:6}  ({v.get('_elapsed_s', '?')}s)")
