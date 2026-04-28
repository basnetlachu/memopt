"""Merge pillar_results.json + big_model_results.json into one final JSON."""
import json
from pathlib import Path

final = {
    "environment": {
        "gpu": "NVIDIA H100 80GB HBM3",
        "driver": "580.126.09",
        "cuda": "12.8",
        "cpu_simd": "avx512 (bw, cd, dq, f)",
        "ram_gb": 181,
        "python": "3.10.12",
        "torch": "2.5.1+cu121",
    },
    "cpp_build": {
        "targets_built": [
            "_memopt_core.so", "_memopt_cuda.so", "_memopt_hooks.so",
            "_memopt_paged.so", "_memopt_rocm.so", "_memopt_simd.so",
            "memopt-transport",
            "test_block_pool", "test_cuda_backend", "test_dispatch_table",
            "test_oracle", "test_page_table", "test_prefix_match",
            "test_transport",
        ],
        "ctest_pass": 5,
        "ctest_fail": 2,
        "failing_cases": ["RingBuffer.WrapAround", "QPExchange.FileBackendRoundTrip"],
    },
    "pytest": {
        "passed": 791, "failed": 29, "skipped": 17, "seconds": 103,
    },
}

try:
    final["pillars"] = json.load(open("/root/memopt/pillar_results.json"))
except Exception as e:
    final["pillars"] = {"error": str(e)}

try:
    final["big_models"] = json.load(open("/root/memopt/big_model_results.json"))
except Exception as e:
    final["big_models"] = {"error": str(e)}

Path("/root/memopt/FINAL_REPORT.json").write_text(
    json.dumps(final, indent=2, default=str))
print("FINAL_REPORT.json written")
