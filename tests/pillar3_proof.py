"""
Pillar 3 proof — Self-Synthesizing Kernels.
Tests what can be verified without a GPU.

GPU-required tests are marked and skipped locally.
"""
import sys, os, json
sys.path.insert(0, '.')

print("="*50)
print("PILLAR 3 — SELF-SYNTHESIZING KERNELS")
print("="*50)

results = {}

# Test 1: _validate_kernel_source exists and handles all cases
print("\n[Test 1] Validator")
try:
    from memopt.kernels.jit_generator import JITGenerator
    gen = JITGenerator.__new__(JITGenerator)

    cases = [
        ("clean",
         "import triton\nimport triton.language as tl\n\n"
         "@triton.jit\n"
         "def run_kernel(x_ptr, n: tl.constexpr):\n    pass\n",
         True),
        ("fenced",
         "```python\nimport triton\n@triton.jit\n"
         "def run_kernel(x): pass\n```",
         True),
        ("no_triton",
         "import torch\ndef foo(): pass",
         False),
        ("no_jit",
         "import triton\ndef foo(): pass",
         False),
        ("syntax_error",
         "import triton\n@triton.jit\ndef run_kernel(x\n    pass",
         False),
    ]

    passed = 0
    for name, src, expected in cases:
        valid, result = gen._validate_kernel_source(src)
        ok = valid == expected
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            print(f"    expected={expected} got={valid} reason={result}")
        if ok:
            passed += 1

    results["validator"] = {
        "passed": passed == len(cases),
        "score": f"{passed}/{len(cases)}",
    }
    print(f"  Validator: {passed}/{len(cases)}")
except Exception as e:
    results["validator"] = {"passed": False, "error": str(e)}
    print(f"  FAIL: {e}")


# Test 2: MAX_TOKENS is 6000
print("\n[Test 2] MAX_TOKENS = 6000")
try:
    import ast
    with open("memopt/kernels/jit_generator.py") as f:
        src = f.read()
    tree = ast.parse(src)

    found_6000 = "6000" in src
    found_2048_as_token_limit = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            if node.value == 2048:
                found_2048_as_token_limit = True

    passed2 = found_6000 and not found_2048_as_token_limit
    results["max_tokens"] = {
        "passed":          passed2,
        "found_6000":      found_6000,
        "still_has_2048":  found_2048_as_token_limit,
    }
    print(f"  6000 present: {found_6000}")
    print(f"  2048 still present: {found_2048_as_token_limit}")
    print(f"  Test 2: {'PASS' if passed2 else 'FAIL'}")
except Exception as e:
    results["max_tokens"] = {"passed": False, "error": str(e)}
    print(f"  FAIL: {e}")


# Test 3: KernelCache saves and retrieves
print("\n[Test 3] KernelCache persistence")
try:
    import tempfile
    from memopt.kernels.kernel_cache import KernelCache

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = KernelCache(cache_dir=tmpdir)

        fake_source = (
            "import triton\n"
            "import triton.language as tl\n\n"
            "@triton.jit\n"
            "def run_kernel(x_ptr, n: tl.constexpr):\n"
            "    pass\n"
        )

        cache.put(
            op_name="test_op",
            input_shapes=[(32, 128)],
            dtype="float16",
            source=fake_source,
            hardware="test_gpu",
        )

        result = cache.get(
            op_name="test_op",
            input_shapes=[(32, 128)],
            dtype="float16",
            hardware="test_gpu",
        )

        passed3 = result is not None
        results["cache_persist"] = {
            "passed":    passed3,
            "retrieved": result is not None,
        }
        print(f"  Saved and retrieved: {passed3}")
        print(f"  Test 3: {'PASS' if passed3 else 'FAIL'}")
except Exception as e:
    results["cache_persist"] = {"passed": False, "error": str(e)}
    print(f"  FAIL: {e}")


# Test 4: Kernel library catalog
print("\n[Test 4] Kernel library catalog")
try:
    from memopt.kernels.kernel_cache import KernelCache
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = KernelCache(cache_dir=tmpdir)

        for i in range(3):
            cache.put(
                op_name=f"op_{i}",
                input_shapes=[(32, 64)],
                dtype="float16",
                source="import triton\n@triton.jit\n"
                       "def run_kernel(x): pass\n",
                hardware="a100",
            )

        catalog = cache.catalog()
        passed4 = len(catalog) == 3
        results["catalog"] = {"passed": passed4, "count": len(catalog)}
        print(f"  Catalog count: {len(catalog)}")
        print(f"  Test 4: {'PASS' if passed4 else 'FAIL'}")
except Exception as e:
    results["catalog"] = {"passed": False, "error": str(e)}
    print(f"  FAIL: {e}")


# Test 5: GPU-required — skip locally
print("\n[Test 5] Live synthesis (GPU required)")
print("  SKIP — requires CUDA + ANTHROPIC_API_KEY")
print("  Will run on GPU server")
results["live_synthesis"] = {"passed": None, "note": "GPU required"}


# Summary
local_tests = {k: v for k, v in results.items()
               if v.get("passed") is not None}
passed_count = sum(1 for v in local_tests.values() if v.get("passed"))
total = len(local_tests)

print(f"\n{'='*50}")
print("PILLAR 3 RESULT (local)")
print(f"{'='*50}")
print(f"Local tests: {passed_count}/{total} pass")
print("GPU test: deferred to server")
print(f"PASS: {passed_count == total}")

with open('/tmp/pillar3_result.json', 'w') as f:
    json.dump({
        "local_passed": passed_count,
        "local_total":  total,
        "passed":       passed_count == total,
        "results":      results,
    }, f, indent=2)
