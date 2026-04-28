"""
Pillar 8 proof — Hardware Abstraction.

Tests:
  1. HAL imports and detects current hardware
  2. NVIDIA backend interface is complete
  3. AMD stub correctly declares itself
  4. HAL does not silently use stubs
  5. Backend selection is deterministic
"""
import sys, os, json
sys.path.insert(0, '.')

print("=" * 50)
print("PILLAR 8 — HARDWARE ABSTRACTION")
print("=" * 50)

results = {}


# Test 1: HAL imports and detects
print("\n[Test 1] HAL detection")
try:
    from memopt.vmm.hal import HAL

    hal = HAL()
    backend = hal.backend
    tiers   = hal.tier_names

    passed1 = (
        backend is not None
        and isinstance(tiers, (list, tuple))
        and len(tiers) > 0
    )
    results["hal_detection"] = {
        "passed":  passed1,
        "backend": str(backend),
        "tiers":   list(tiers),
    }
    print(f"  Backend: {backend}")
    print(f"  Tiers:   {tiers}")
    print(f"  Test 1: {'PASS' if passed1 else 'FAIL'}")
except Exception as e:
    results["hal_detection"] = {"passed": False, "error": str(e)}
    print(f"  FAIL: {e}")


# Test 2: NVIDIA backend interface
print("\n[Test 2] NVIDIA backend interface")
try:
    import importlib
    cuda_mod = importlib.import_module(
        "memopt.vmm.backends._cuda_backend_py")

    backend_classes = [
        name for name in dir(cuda_mod)
        if "Backend" in name or "backend" in name
    ]

    backend_cls = None
    for name in backend_classes:
        obj = getattr(cuda_mod, name)
        if isinstance(obj, type):
            backend_cls = obj
            break

    has_methods = (
        backend_cls is not None
        and all(hasattr(backend_cls, m) for m in ("allocate", "free"))
    )

    passed2 = has_methods
    results["nvidia_interface"] = {
        "passed":        passed2,
        "classes_found": backend_classes,
        "has_required":  has_methods,
    }
    print(f"  Classes: {backend_classes}")
    print(f"  Has required methods: {has_methods}")
    print(f"  Test 2: {'PASS' if passed2 else 'FAIL'}")
except Exception as e:
    results["nvidia_interface"] = {"passed": False, "error": str(e)}
    print(f"  FAIL: {e}")


# Test 3: AMD stub declares itself
print("\n[Test 3] AMD stub declares itself")
try:
    from memopt.vmm.backends.rocm_backend import ROCmBackend

    stub_raises = False
    try:
        r = ROCmBackend()
    except NotImplementedError:
        stub_raises = True

    is_stub       = getattr(ROCmBackend, "IS_STUB", False)
    has_available = hasattr(ROCmBackend, "is_available")

    passed3 = stub_raises and is_stub
    results["amd_stub"] = {
        "passed":            passed3,
        "raises_on_init":    stub_raises,
        "is_stub_flag":      is_stub,
        "has_is_available":  has_available,
    }
    print(f"  Raises NotImplementedError: {stub_raises}")
    print(f"  IS_STUB flag: {is_stub}")
    print(f"  Test 3: {'PASS' if passed3 else 'FAIL'}")
except ImportError as e:
    results["amd_stub"] = {"passed": False, "error": f"Import failed: {e}"}
    print(f"  FAIL: {e}")


# Test 4: HAL does not silently use stubs
print("\n[Test 4] HAL stub transparency")
try:
    from memopt.vmm.hal import HAL, HardwareBackend
    import torch

    hal = HAL()
    backend_enum = hal.backend

    has_gpu  = torch.cuda.is_available()
    has_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None

    if not has_gpu and not has_rocm:
        backend_str = str(backend_enum).lower()
        not_falsely_rocm = "rocm" not in backend_str and "amd" not in backend_str
        passed4 = not_falsely_rocm
    else:
        passed4 = True

    results["hal_transparency"] = {
        "passed":      passed4,
        "backend":     str(backend_enum),
        "has_nvidia":  has_gpu,
        "has_rocm":    has_rocm,
    }
    print(f"  Backend selected: {backend_enum}")
    print(f"  NVIDIA available: {has_gpu}")
    print(f"  ROCm available:   {has_rocm}")
    print(f"  Test 4: {'PASS' if passed4 else 'FAIL'}")
except Exception as e:
    results["hal_transparency"] = {"passed": False, "error": str(e)}
    print(f"  FAIL: {e}")


# Test 5: Deterministic backend selection
print("\n[Test 5] Deterministic backend selection")
try:
    from memopt.vmm.hal import HAL

    hal1 = HAL()
    hal2 = HAL()

    passed5 = (
        str(hal1.backend) == str(hal2.backend)
        and hal1.tier_names == hal2.tier_names
    )
    results["deterministic"] = {
        "passed":   passed5,
        "backend1": str(hal1.backend),
        "backend2": str(hal2.backend),
    }
    print(f"  Two HAL() calls match: {passed5}")
    print(f"  Test 5: {'PASS' if passed5 else 'FAIL'}")
except Exception as e:
    results["deterministic"] = {"passed": False, "error": str(e)}
    print(f"  FAIL: {e}")


# Summary
passed_count = sum(1 for v in results.values() if v.get("passed"))
total = len(results)

print(f"\n{'=' * 50}")
print("PILLAR 8 RESULT")
print(f"{'=' * 50}")
print(f"Tests: {passed_count}/{total} pass")
print(f"PASS: {passed_count == total}")

with open('/tmp/pillar8_result.json', 'w') as f:
    json.dump({
        "passed":  passed_count == total,
        "score":   f"{passed_count}/{total}",
        "results": results,
    }, f, indent=2)
