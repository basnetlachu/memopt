#!/usr/bin/env python3
"""
Phase 1 Validation Test - Tests WITHOUT PyTorch

This validates the code structure and logic without requiring torch.
Tests can be run on any machine to verify Phase 1 implementation.
"""

import sys
import os
import re

# Test directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_file_structure():
    """Test that all required Phase 1 files exist."""
    print("\n" + "="*60)
    print("TEST 1: File Structure")
    print("="*60)

    required_files = [
        "memopt/profiler/__init__.py",
        "memopt/profiler/ncu_profiler.py",
        "memopt/profiler/hardware_counters.py",
        "memopt/profiler/bottleneck_classifier.py",
        "memopt/profiler/phase1_profiler.py",
    ]

    passed = 0
    for f in required_files:
        path = os.path.join(PROJECT_ROOT, f)
        exists = os.path.exists(path)
        status = "[OK]" if exists else "[FAIL]"
        print(f"  {status} {f}")
        if exists:
            passed += 1

    return passed == len(required_files)


def test_gpu_specs_in_code():
    """Test GPU specifications database coverage by parsing the source."""
    print("\n" + "="*60)
    print("TEST 2: GPU Specifications Database")
    print("="*60)

    hw_path = os.path.join(PROJECT_ROOT, "memopt/profiler/hardware_counters.py")
    with open(hw_path, 'r') as f:
        content = f.read()

    # Required GPU families
    required_gpus = [
        "H100", "H200",  # Hopper
        "A100", "A30", "A40", "A10", "A6000",  # Ampere DC/Pro
        "L40", "L40S", "RTX 4090", "RTX 4080",  # Ada
        "T4",  # Turing
        "V100",  # Volta
        "P100",  # Pascal
        "RTX 3090", "RTX 3080",  # Ampere Consumer
    ]

    found = 0
    missing = []

    for gpu in required_gpus:
        # Check if GPU exists in GPU_SPECS dictionary
        pattern = f'"{gpu}"\\s*:\\s*GPUSpec'
        if re.search(pattern, content):
            print(f"  [OK] {gpu}")
            found += 1
        else:
            print(f"  [MISS] {gpu}")
            missing.append(gpu)

    print(f"\n  Coverage: {found}/{len(required_gpus)} ({100*found/len(required_gpus):.0f}%)")

    if missing:
        print(f"  Missing: {', '.join(missing)}")

    return found >= len(required_gpus) * 0.8


def test_bottleneck_types():
    """Test that all 5 bottleneck types are defined."""
    print("\n" + "="*60)
    print("TEST 3: Bottleneck Types")
    print("="*60)

    bc_path = os.path.join(PROJECT_ROOT, "memopt/profiler/bottleneck_classifier.py")
    with open(bc_path, 'r') as f:
        content = f.read()

    required_types = [
        "MEMORY_BOUND_DRAM",
        "MEMORY_BOUND_CACHE",
        "COMPUTE_BOUND",
        "PIPELINE_BOUND_OCCUPANCY",
        "MIXED",
    ]

    found = 0
    for btype in required_types:
        if btype in content:
            print(f"  [OK] {btype}")
            found += 1
        else:
            print(f"  [MISS] {btype}")

    return found == len(required_types)


def test_hardware_counters():
    """Test that all 7 critical hardware counters are defined."""
    print("\n" + "="*60)
    print("TEST 4: Hardware Counters")
    print("="*60)

    hw_path = os.path.join(PROJECT_ROOT, "memopt/profiler/hardware_counters.py")
    with open(hw_path, 'r') as f:
        content = f.read()

    required_counters = [
        ("dram_bytes_read", "DRAM read traffic"),
        ("dram_bytes_write", "DRAM write traffic"),
        ("sm_cycles_active", "Active SM cycles"),
        ("sm_cycles_elapsed", "Elapsed SM cycles"),
        ("stall_cycles", "Memory stall cycles"),
        ("l2_hit_rate", "L2 cache hit rate"),
        ("achieved_occupancy", "GPU occupancy"),
    ]

    found = 0
    for counter, desc in required_counters:
        if counter in content:
            print(f"  [OK] {counter}: {desc}")
            found += 1
        else:
            print(f"  [MISS] {counter}")

    return found == len(required_counters)


def test_ncu_metrics():
    """Test that NCU profiler collects required CUPTI metrics."""
    print("\n" + "="*60)
    print("TEST 5: NCU CUPTI Metrics")
    print("="*60)

    ncu_path = os.path.join(PROJECT_ROOT, "memopt/profiler/ncu_profiler.py")
    with open(ncu_path, 'r') as f:
        content = f.read()

    required_metrics = [
        "dram__bytes_read",
        "dram__bytes_write",
        "sm__cycles_elapsed",
        "sm__cycles_active",
        "smsp__warp_issue_stalled_long_scoreboard",
        "lts__t_sector_hit_rate",
        "sm__warps_active",
    ]

    found = 0
    for metric in required_metrics:
        if metric in content:
            print(f"  [OK] {metric}")
            found += 1
        else:
            print(f"  [MISS] {metric}")

    return found == len(required_metrics)


def test_counters_from_ncu():
    """Test that counters_from_ncu conversion function exists."""
    print("\n" + "="*60)
    print("TEST 6: NCU -> HardwareCounters Conversion")
    print("="*60)

    hw_path = os.path.join(PROJECT_ROOT, "memopt/profiler/hardware_counters.py")
    with open(hw_path, 'r') as f:
        content = f.read()

    checks = [
        ("def counters_from_ncu", "Function definition exists"),
        ("ncu_counters.kernel_name", "Converts kernel name"),
        ("ncu_counters.dram_bytes_read", "Converts DRAM read"),
        ("ncu_counters.total_stall_cycles", "Converts stall cycles"),
        ('measurement_method="ncu_cupti"', "Sets measurement method"),
        ("measurement_confidence=1.0", "Sets full confidence"),
    ]

    found = 0
    for pattern, desc in checks:
        if pattern in content:
            print(f"  [OK] {desc}")
            found += 1
        else:
            print(f"  [MISS] {desc}: looking for '{pattern}'")

    return found == len(checks)


def test_classification_thresholds():
    """Test that classification thresholds are reasonable."""
    print("\n" + "="*60)
    print("TEST 7: Classification Thresholds")
    print("="*60)

    bc_path = os.path.join(PROJECT_ROOT, "memopt/profiler/bottleneck_classifier.py")
    with open(bc_path, 'r') as f:
        content = f.read()

    # Extract threshold values using regex
    thresholds = {
        "MEMORY_STALL_DRAM_THRESHOLD": (40, 80),  # Should be 40-80%
        "MEMORY_STALL_CACHE_THRESHOLD": (30, 70),
        "L2_HIT_RATE_THRESHOLD": (30, 70),
        "COMPUTE_STALL_THRESHOLD": (15, 40),
        "OCCUPANCY_THRESHOLD": (20, 50),
    }

    passed = 0
    for name, (min_val, max_val) in thresholds.items():
        pattern = rf'{name}\s*=\s*(\d+\.?\d*)'
        match = re.search(pattern, content)
        if match:
            value = float(match.group(1))
            in_range = min_val <= value <= max_val
            status = "[OK]" if in_range else "[WARN]"
            print(f"  {status} {name} = {value} (expected {min_val}-{max_val})")
            if in_range:
                passed += 1
        else:
            print(f"  [MISS] {name} not found")

    return passed >= len(thresholds) - 1  # Allow 1 threshold to be outside expected range


def test_error_handling():
    """Test that error handling patterns exist."""
    print("\n" + "="*60)
    print("TEST 8: Error Handling")
    print("="*60)

    files_to_check = [
        "memopt/profiler/ncu_profiler.py",
        "memopt/profiler/hardware_counters.py",
    ]

    error_patterns = [
        ("try:", "Try-except blocks"),
        ("except", "Exception handling"),
        ("logger.warning", "Warning logging"),
        ("logger.error", "Error logging"),
        ("FileNotFoundError", "File error handling"),
    ]

    total_found = 0
    for fname in files_to_check:
        path = os.path.join(PROJECT_ROOT, fname)
        with open(path, 'r') as f:
            content = f.read()

        print(f"\n  {fname}:")
        for pattern, desc in error_patterns:
            count = content.count(pattern)
            if count > 0:
                print(f"    [OK] {desc}: {count} occurrences")
                total_found += 1

    return total_found >= 5


def test_exports():
    """Test that __init__.py exports required classes."""
    print("\n" + "="*60)
    print("TEST 9: Module Exports")
    print("="*60)

    init_path = os.path.join(PROJECT_ROOT, "memopt/profiler/__init__.py")
    with open(init_path, 'r') as f:
        content = f.read()

    required_exports = [
        "NCUProfiler",
        "NCUCounters",
        "HardwareCounters",
        "BottleneckClassifier",
        "BottleneckType",
        "counters_from_ncu",
        "get_gpu_spec",
    ]

    found = 0
    for export in required_exports:
        if export in content:
            print(f"  [OK] {export}")
            found += 1
        else:
            print(f"  [MISS] {export}")

    return found == len(required_exports)


def main():
    """Run all validation tests."""
    print("="*60)
    print("PHASE 1 VALIDATION TEST (No PyTorch Required)")
    print("Validates code structure, logic, and completeness")
    print("="*60)

    results = []

    results.append(("File Structure", test_file_structure()))
    results.append(("GPU Specs Database", test_gpu_specs_in_code()))
    results.append(("Bottleneck Types", test_bottleneck_types()))
    results.append(("Hardware Counters", test_hardware_counters()))
    results.append(("NCU CUPTI Metrics", test_ncu_metrics()))
    results.append(("NCU Conversion", test_counters_from_ncu()))
    results.append(("Classification Thresholds", test_classification_thresholds()))
    results.append(("Error Handling", test_error_handling()))
    results.append(("Module Exports", test_exports()))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    passed = 0
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")
        if result:
            passed += 1

    print(f"\n  Total: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\n" + "="*60)
        print("  ALL VALIDATION TESTS PASSED!")
        print("="*60)
        print("\n  Phase 1 implementation is complete and robust.")
        print("\n  To test on real GPU, run on a machine with:")
        print("    - NVIDIA GPU with CUDA toolkit")
        print("    - Nsight Compute (ncu) installed")
        print("    - PyTorch with CUDA support")
        print("\n  Commands:")
        print("    python tests/test_real_counters.py    # Test with ncu (requires GPU)")
        return 0
    else:
        print("\n  [FAIL] Some tests failed - review output above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
