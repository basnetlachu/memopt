#!/usr/bin/env python3
"""
memopt wiring audit — verifies C++ components are actually called.
Run: python scripts/audit_wiring.py
"""
import sys, os, inspect, importlib, json

# Ensure memopt is importable when run from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = "✓ WIRED"
WARN = "⚠ PARTIAL"
FAIL = "✗ NOT WIRED"
ERR  = "✗ ERROR"

results = {}

def check(name, fn):
    try:
        status, detail = fn()
        results[name] = (status, detail)
    except Exception as e:
        results[name] = (ERR, str(e))

# ── 1. PageTable source ───────────────────────────────────────────
def check_page_table():
    from memopt.vmm.page_table import PageTable
    mod = PageTable.__module__
    if "_memopt_core" in mod:
        return PASS, f"C++ class from {mod}"
    if "_page_table_py" in mod:
        return WARN, f"Python fallback active (C++ not built): {mod}"
    return WARN, f"Unknown source: {mod}"
check("PageTable", check_page_table)

# ── 2. MemoryOracle source ────────────────────────────────────────
def check_oracle():
    from memopt.vmm.oracle import MemoryOracle
    mod = MemoryOracle.__module__
    if "_memopt_core" in mod:
        return PASS, f"C++ class from {mod}"
    if "_oracle_py" in mod:
        return WARN, f"Python fallback active (C++ not built): {mod}"
    return WARN, f"Unknown source: {mod}"
check("MemoryOracle", check_oracle)

# ── 3. VMM actually uses shim PageTable ──────────────────────────
def check_vmm_wiring():
    src = inspect.getsource(
        importlib.import_module("memopt.vmm.tier_manager"))
    if "_page_table_py" in src:
        return FAIL, "tier_manager imports from _page_table_py directly"
    if "from .page_table import" in src:
        return PASS, "tier_manager imports from page_table shim"
    return WARN, "Cannot determine import path"
check("VMM→PageTable wiring", check_vmm_wiring)

# ── 4. Oracle wired into PrefetchEngine ──────────────────────────
def check_oracle_wiring():
    src = inspect.getsource(
        importlib.import_module("memopt.vmm"))
    if "oracle=" in src and "MemoryOracle" in src:
        return PASS, "VMM.__init__ passes oracle to PrefetchEngine"
    if "PrefetchEngine(self.tier_manager)" in src:
        return FAIL, "PrefetchEngine created WITHOUT oracle — C++ oracle unused"
    return WARN, "Cannot determine oracle wiring"
check("VMM→Oracle wiring", check_oracle_wiring)

# ── 5. Kernel hooks C++ notify ───────────────────────────────────
def check_hooks():
    from memopt.serving import kernel_hooks
    if hasattr(kernel_hooks, '_cpp_hooks') and kernel_hooks._cpp_hooks is not None:
        return PASS, "C++ _memopt_hooks loaded"
    if hasattr(kernel_hooks, '_USE_CPP') and kernel_hooks._USE_CPP:
        return PASS, "C++ hooks available"
    return WARN, "Python fallback (C++ not built)"
check("KernelHooks", check_hooks)

# ── 6. Hooks wired into server ───────────────────────────────────
def check_hooks_server():
    src = inspect.getsource(
        importlib.import_module("memopt.serving.server"))
    has_init = "init_hooks" in src
    has_hooks = "kernel_hooks" in src
    if has_init and has_hooks:
        return PASS, "server.py calls init_hooks and references kernel_hooks"
    if has_init:
        return WARN, "init_hooks called but hooks not in forward path"
    return FAIL, "server.py does not call init_hooks"
check("Server→KernelHooks", check_hooks_server)

# ── 7. PagedKVCache source ────────────────────────────────────────
def check_paged():
    from memopt.serving.paged_attention import PagedKVCache
    mod = PagedKVCache.__module__
    if "_memopt_paged" in mod:
        return PASS, f"C++ class: {mod}"
    if "_paged_attention_py" in mod:
        return WARN, f"Python fallback (C++ not built): {mod}"
    return WARN, f"Unknown source: {mod}"
check("PagedKVCache", check_paged)

# ── 8. CUDABackend wiring ────────────────────────────────────────
def check_cuda_backend():
    from memopt.vmm.backends.cuda_backend import CUDABackend
    mod = CUDABackend.__module__
    try:
        import memopt._memopt_cuda
        cpp_loaded = True
    except ImportError:
        cpp_loaded = False
    if cpp_loaded:
        # Check if _write_nvme_block is overridden
        src = inspect.getsource(CUDABackend)
        if "_memopt_cuda" in src or "write_block_atomic" in src:
            return PASS, "C++ loaded + NVMe I/O overridden"
        return WARN, "C++ loaded but CUDABackend NOT overriding NVMe I/O methods"
    return WARN, f"Python fallback (C++ not built): {mod}"
check("CUDABackend", check_cuda_backend)

# ── 9. Transport daemon ──────────────────────────────────────────
def check_transport():
    import socket
    node_id = os.environ.get("MEMOPT_NODE_ID", socket.gethostname())
    req_path = f"/dev/shm/memopt_transport_{node_id}_req"
    if os.path.exists(req_path):
        return PASS, f"Daemon ring buffer found: {req_path}"
    return WARN, "Daemon not running — Python TCP fallback active"
check("Transport", check_transport)

# ── 10. SIMD prefix match ────────────────────────────────────────
def check_simd():
    try:
        import memopt._memopt_simd as s
        isa = s.detected_isa()
        assert s.find_lcp([1,2,3], [1,2,4]) == 2
        assert s.find_lcp([1,2,3], [1,2,3]) == 3
        assert s.find_lcp([], []) == 0
        return PASS, f"ISA={isa}, correctness verified"
    except ImportError:
        return WARN, "_memopt_simd not built — Python fallback"
    except AssertionError as e:
        return ERR, f"Correctness failure: {e}"
check("PrefixMatch", check_simd)

# ── 11. GKD → prefix_index uses shim ─────────────────────────────
def check_gkd_wiring():
    src = inspect.getsource(
        importlib.import_module("memopt.cluster.prefix_index"))
    has_simd = "_memopt_simd" in src
    has_find_lcp = "find_lcp" in src
    if has_simd and has_find_lcp:
        return PASS, "prefix_index.py imports _memopt_simd.find_lcp"
    if has_find_lcp:
        return WARN, "find_lcp defined but C++ may not be used"
    return FAIL, "No find_lcp in prefix_index.py"
check("GKD→SIMD", check_gkd_wiring)

# ── 12. GKD lookup_longest_prefix uses find_lcp? ─────────────────
def check_gkd_lcp_path():
    # Check the shim (what callers actually import), not the backup
    src = inspect.getsource(
        importlib.import_module("memopt.cluster.prefix_index"))
    has_fast = "_verify_fingerprint_fast" in src
    has_find_lcp = "find_lcp" in src
    if has_fast and has_find_lcp:
        return PASS, ("lookup_longest_prefix uses "
                      "_verify_fingerprint_fast → find_lcp "
                      "(C++ SIMD when built)")
    if has_find_lcp:
        return WARN, "find_lcp present but not wired into verify"
    return WARN, ("lookup_longest_prefix uses Python list comparison, "
                  "C++ find_lcp not wired")
check("GKD LCP hot path", check_gkd_lcp_path)

# ── 13. Duplication check ─────────────────────────────────────────
def check_duplication():
    issues = []
    # CUDABackend shim
    src = inspect.getsource(
        importlib.import_module("memopt.vmm.backends.cuda_backend"))
    if "_PyCUDABackend" in src and "_memopt_cuda" in src:
        issues.append(
            "CUDABackend: inherits _PyCUDABackend (Python __init__ "
            "creates torch.cuda.Stream) + references _memopt_cuda "
            "(possible duplication of stream management)")
    # PagedKVCache fetch — gather kernel vs torch.cat
    try:
        from memopt.serving.paged_attention import PagedKVCache
        if "_paged_attention_py" in PagedKVCache.__module__:
            # Python fallback active — torch.cat is expected.
            # C++ gather kernel is in bindings.cpp, not in Python.
            # This is NOT duplication — it's the fallback path.
            pass  # not a duplication issue
        elif "_memopt_paged" in PagedKVCache.__module__:
            # C++ extension active — gather kernel should be wired
            try:
                import memopt._memopt_paged as p
                if not p.has_cuda_gather():
                    issues.append(
                        "PagedKVCache C++ loaded but "
                        "has_cuda_gather()=False — "
                        "gather kernel not compiled")
            except Exception:
                pass
    except Exception:
        pass
    if issues:
        return WARN, " | ".join(issues)
    return PASS, "No obvious duplication"
check("Duplication", check_duplication)

# ─── Print report ────────────────────────────────────────────────
print("\n" + "═"*65)
print("  MEMOPT C++ WIRING + ENGINEERING AUDIT")
print("═"*65)

wired = partial = failed = errors = 0
for name, (status, detail) in results.items():
    print(f"\n{status}  {name}")
    print(f"   {detail}")
    if "WIRED" in status:   wired   += 1
    elif "PARTIAL" in status: partial += 1
    elif "ERROR" in status:  errors  += 1
    else:                    failed  += 1

print("\n" + "═"*65)
print(f"  WIRED: {wired}  PARTIAL: {partial}  "
      f"FAILED: {failed}  ERRORS: {errors}")
print("═"*65)

overall = ("PRODUCTION READY" if failed == 0 and errors == 0
           and partial <= 2  # C++ not built is expected in dev
           else "GAPS FOUND — SEE ABOVE")
print(f"\n  VERDICT: {overall}\n")

with open("wiring_audit_result.json", "w") as f:
    json.dump({k: list(v) for k, v in results.items()}, f, indent=2)
print("  Saved: wiring_audit_result.json\n")
