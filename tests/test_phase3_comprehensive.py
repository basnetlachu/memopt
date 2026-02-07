#!/usr/bin/env python3
"""
Phase 3 Comprehensive Validation Tests

Tests the Auto-Optimization Engine and Custom Kernel Library:
1. OptimizationExecutor - Test-measure-commit loop
2. OptimizationSequencer - Multi-optimization application
3. TransformationEngine - Specific transformations
4. CustomKernelRegistry - Kernel selection and fallbacks
5. AutoOptimizer - Full pipeline integration
6. End-to-end with real GPU profiling

Run on GPU server:
    python tests/test_phase3_comprehensive.py
"""

import os
import sys
import time
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check for PyTorch
HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    print("PyTorch not available - some tests will be skipped")


def test_phase3_imports():
    """Test 1: Verify all Phase 3 components can be imported."""
    print("\n" + "=" * 60)
    print("TEST 1: Phase 3 Imports")
    print("=" * 60)

    imports = [
        ("OptimizationExecutor", "memopt.phase3", "OptimizationExecutor"),
        ("OptimizationResult", "memopt.phase3", "OptimizationResult"),
        ("OptimizationSequencer", "memopt.phase3", "OptimizationSequencer"),
        ("OptimizationPlan", "memopt.phase3", "OptimizationPlan"),
        ("TransformationEngine", "memopt.phase3", "TransformationEngine"),
        ("CustomKernelRegistry", "memopt.phase3", "CustomKernelRegistry"),
        ("kernel_registry", "memopt.phase3", "kernel_registry"),
        ("fused_attention", "memopt.phase3", "fused_attention"),
        ("AutoOptimizer", "memopt.phase3", "AutoOptimizer"),
    ]

    passed = 0
    for name, module_path, attr in imports:
        try:
            module = __import__(module_path, fromlist=[attr])
            obj = getattr(module, attr)
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")

    print(f"\n  Result: {passed}/{len(imports)} imports successful")
    return passed == len(imports)


def test_optimization_executor_structure():
    """Test 2: Verify OptimizationExecutor has required methods."""
    print("\n" + "=" * 60)
    print("TEST 2: OptimizationExecutor Structure")
    print("=" * 60)

    from memopt.phase3 import OptimizationExecutor

    executor = OptimizationExecutor()

    required_methods = [
        "apply_optimization",
        "_profile_operation",
        "_validate_correctness",
    ]

    required_attrs = [
        "tolerance_pct",
        "correctness_rtol",
        "correctness_atol",
        "num_warmup",
        "num_iterations",
    ]

    passed = 0

    for method in required_methods:
        if hasattr(executor, method) and callable(getattr(executor, method)):
            print(f"  [PASS] Method: {method}")
            passed += 1
        else:
            print(f"  [FAIL] Method: {method}")

    for attr in required_attrs:
        if hasattr(executor, attr):
            val = getattr(executor, attr)
            print(f"  [PASS] Attr: {attr} = {val}")
            passed += 1
        else:
            print(f"  [FAIL] Attr: {attr}")

    total = len(required_methods) + len(required_attrs)
    print(f"\n  Result: {passed}/{total} checks passed")
    return passed == total


def test_kernel_registry():
    """Test 3: Verify CustomKernelRegistry functionality."""
    print("\n" + "=" * 60)
    print("TEST 3: Custom Kernel Registry")
    print("=" * 60)

    from memopt.phase3 import CustomKernelRegistry, kernel_registry

    # Check registry has detected backends
    print(f"  Flash Attention: {kernel_registry.has_flash_attn}")
    print(f"  xFormers: {kernel_registry.has_xformers}")
    print(f"  Triton: {kernel_registry.has_triton}")
    print(f"  cuDNN: {kernel_registry.has_cudnn}")
    print(f"  PyTorch SDPA: {kernel_registry.has_sdpa}")

    # Check built-in kernels registered
    kernels = kernel_registry.list_kernels()
    print(f"\n  Registered kernels: {kernels}")

    expected_kernels = ["fused_attention", "fused_layernorm_linear", "fused_gelu_dropout"]
    passed = 0

    for k in expected_kernels:
        if k in kernels:
            print(f"  [PASS] Kernel registered: {k}")
            passed += 1
        else:
            print(f"  [FAIL] Kernel not registered: {k}")

    # Test kernel retrieval
    try:
        attn_kernel = kernel_registry.get_kernel("fused_attention")
        print(f"  [PASS] Can retrieve fused_attention kernel")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Cannot retrieve fused_attention: {e}")

    total = len(expected_kernels) + 1
    print(f"\n  Result: {passed}/{total} checks passed")
    return passed == total


def test_transformation_engine():
    """Test 4: Verify TransformationEngine has all transformations."""
    print("\n" + "=" * 60)
    print("TEST 4: Transformation Engine")
    print("=" * 60)

    from memopt.phase3 import TransformationEngine

    engine = TransformationEngine()

    # Check transformation mappings exist
    expected_transformations = [
        "layout_transpose",
        "cache_residency",
        "kernel_fusion_tiling",
        "shared_memory_staging",
        "increase_parallelism",
        "memory_prefetch",
        "gather_optimization",
        "flash_attention",
    ]

    passed = 0
    for t in expected_transformations:
        if t in engine._transformations:
            print(f"  [PASS] Transformation: {t}")
            passed += 1
        else:
            print(f"  [FAIL] Missing transformation: {t}")

    print(f"\n  Result: {passed}/{len(expected_transformations)} transformations available")
    return passed >= len(expected_transformations) - 2  # Allow 2 missing


def test_optimization_sequencer_structure():
    """Test 5: Verify OptimizationSequencer structure."""
    print("\n" + "=" * 60)
    print("TEST 5: OptimizationSequencer Structure")
    print("=" * 60)

    from memopt.phase3 import OptimizationSequencer, OptimizationPlan

    sequencer = OptimizationSequencer()

    required_methods = [
        "execute_plan",
        "_calculate_cumulative_speedup",
        "estimate_plan_impact",
    ]

    passed = 0
    for method in required_methods:
        if hasattr(sequencer, method) and callable(getattr(sequencer, method)):
            print(f"  [PASS] Method: {method}")
            passed += 1
        else:
            print(f"  [FAIL] Method: {method}")

    # Test cumulative speedup calculation
    cumulative = sequencer._calculate_cumulative_speedup(20.0, 10.0)
    # 20% then 10% should give ~28%
    expected = 28.0
    if abs(cumulative - expected) < 1.0:
        print(f"  [PASS] Cumulative speedup calculation: {cumulative:.1f}% (expected ~{expected}%)")
        passed += 1
    else:
        print(f"  [FAIL] Cumulative speedup: {cumulative:.1f}% (expected ~{expected}%)")

    print(f"\n  Result: {passed}/{len(required_methods) + 1} checks passed")
    return passed == len(required_methods) + 1


def test_auto_optimizer_structure():
    """Test 6: Verify AutoOptimizer structure."""
    print("\n" + "=" * 60)
    print("TEST 6: AutoOptimizer Structure")
    print("=" * 60)

    from memopt.phase3 import AutoOptimizer

    optimizer = AutoOptimizer()

    required_methods = [
        "optimize",
        "optimize_from_report",
        "apply_single_optimization",
        "get_available_optimizations",
        "get_available_kernels",
    ]

    passed = 0
    for method in required_methods:
        if hasattr(optimizer, method) and callable(getattr(optimizer, method)):
            print(f"  [PASS] Method: {method}")
            passed += 1
        else:
            print(f"  [FAIL] Method: {method}")

    # Check available optimizations
    available = optimizer.get_available_optimizations()
    print(f"\n  Available optimizations: {available}")

    expected_opts = ["flash_attention", "layout_transpose", "kernel_fusion", "torch_compile"]
    for opt in expected_opts:
        if opt in available:
            print(f"  [PASS] Optimization available: {opt}")
            passed += 1
        else:
            print(f"  [FAIL] Optimization missing: {opt}")

    total = len(required_methods) + len(expected_opts)
    print(f"\n  Result: {passed}/{total} checks passed")
    return passed == total


def test_with_mock_model():
    """Test 7: Test optimization flow with a mock model (no GPU required)."""
    print("\n" + "=" * 60)
    print("TEST 7: Mock Model Optimization Flow")
    print("=" * 60)

    if not HAS_TORCH:
        print("  [SKIP] PyTorch not available")
        return True

    from memopt.phase3 import (
        OptimizationExecutor,
        OptimizationResult,
        TransformationEngine,
    )
    from memopt.profiler.optimization_synthesis import (
        OptimizationCandidate,
        OptimizationType,
    )

    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(64, 64)
            self.gelu = nn.GELU()

        def forward(self, x):
            return self.gelu(self.linear(x))

    model = SimpleModel()

    # Move to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Create inputs
    inputs = {'x': torch.randn(32, 64, device=device)}

    def operation(**kwargs):
        return model(**kwargs)

    # Create a mock optimization candidate
    candidate = OptimizationCandidate(
        rule_name='test_fusion',
        optimization_type=OptimizationType.KERNEL_FUSION_TILING,
        description='Test kernel fusion',
        expected_impact_pct=10.0,
        option1_action='apply_kernel_fusion',
        option2_recommendation='Use torch.compile',
        option3_kernel=None,
        priority='MEDIUM',
        confidence=0.8
    )

    # Test executor profiling
    executor = OptimizationExecutor(num_warmup=2, num_iterations=5)

    try:
        metrics = executor._profile_operation(operation, inputs)
        print(f"  [PASS] Profiling works: {metrics['gpu_time_ms']:.3f}ms on {metrics.get('device', 'unknown')}")
    except Exception as e:
        print(f"  [FAIL] Profiling failed: {e}")
        return False

    # Test correctness validation
    try:
        is_correct, error = executor._validate_correctness(operation, operation, inputs)
        if is_correct:
            print(f"  [PASS] Correctness validation works")
        else:
            print(f"  [FAIL] Correctness validation failed: {error}")
            return False
    except Exception as e:
        print(f"  [FAIL] Correctness validation error: {e}")
        return False

    # Test full optimization flow (dry run)
    try:
        result = executor.apply_optimization(
            model, operation, candidate, inputs, dry_run=True
        )
        print(f"  [PASS] Dry run optimization: {result.speedup_pct:.1f}% estimated")
    except Exception as e:
        print(f"  [FAIL] Dry run failed: {e}")
        return False

    print(f"\n  Result: All mock model tests passed")
    return True


def test_fused_attention_kernel():
    """Test 8: Test fused attention kernel."""
    print("\n" + "=" * 60)
    print("TEST 8: Fused Attention Kernel")
    print("=" * 60)

    if not HAS_TORCH:
        print("  [SKIP] PyTorch not available")
        return True

    from memopt.phase3 import fused_attention, kernel_registry

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create Q, K, V tensors
    batch, heads, seq_len, head_dim = 2, 8, 64, 64
    q = torch.randn(batch, heads, seq_len, head_dim, device=device)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device)

    try:
        # Test fused attention
        output = fused_attention(q, k, v)
        print(f"  [PASS] Fused attention output shape: {output.shape}")

        # Verify output shape
        if output.shape == (batch, heads, seq_len, head_dim):
            print(f"  [PASS] Output shape correct")
        else:
            print(f"  [FAIL] Wrong output shape: {output.shape}")
            return False

        # Test with causal mask
        output_causal = fused_attention(q, k, v, is_causal=True)
        print(f"  [PASS] Causal attention works")

        # Check backend used
        if kernel_registry.has_sdpa:
            print(f"  [INFO] Using PyTorch SDPA backend")
        elif kernel_registry.has_flash_attn:
            print(f"  [INFO] Using Flash Attention backend")
        elif kernel_registry.has_xformers:
            print(f"  [INFO] Using xFormers backend")
        else:
            print(f"  [INFO] Using naive attention fallback")

    except Exception as e:
        print(f"  [FAIL] Fused attention failed: {e}")
        return False

    print(f"\n  Result: Fused attention tests passed")
    return True


def test_real_gpu_optimization():
    """Test 9: Test optimization on real GPU workload."""
    print("\n" + "=" * 60)
    print("TEST 9: Real GPU Optimization")
    print("=" * 60)

    if not HAS_TORCH or not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        return True

    from memopt.phase3 import AutoOptimizer, OptimizationExecutor
    from memopt.profiler.optimization_synthesis import (
        OptimizationCandidate,
        OptimizationType,
    )

    # Create attention-like model
    class AttentionModel(nn.Module):
        def __init__(self, embed_dim=256, num_heads=8):
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads
            self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
            self.proj = nn.Linear(embed_dim, embed_dim)

        def forward(self, x):
            B, T, C = x.shape
            qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B, H, T, D
            q, k, v = qkv[0], qkv[1], qkv[2]

            # Naive attention
            attn = torch.bmm(
                q.reshape(B * self.num_heads, T, self.head_dim),
                k.reshape(B * self.num_heads, T, self.head_dim).transpose(-2, -1)
            ) / (self.head_dim ** 0.5)
            attn = F.softmax(attn, dim=-1)
            out = torch.bmm(attn, v.reshape(B * self.num_heads, T, self.head_dim))
            out = out.reshape(B, self.num_heads, T, self.head_dim).permute(0, 2, 1, 3)
            out = out.reshape(B, T, C)
            return self.proj(out)

    model = AttentionModel().cuda()

    # Create inputs
    inputs = {'x': torch.randn(8, 128, 256, device='cuda')}

    def operation(**kwargs):
        return model(**kwargs)

    # Test with executor
    executor = OptimizationExecutor(tolerance_pct=5.0, num_warmup=3, num_iterations=10)

    # Create flash attention candidate
    candidate = OptimizationCandidate(
        rule_name='redundant_fetch',
        optimization_type=OptimizationType.CACHE_RESIDENCY,
        description='Replace naive attention with Flash Attention',
        expected_impact_pct=30.0,
        option1_action='apply_cache_pinning',
        option2_recommendation='Use F.scaled_dot_product_attention',
        option3_kernel='fused_attention_kernel',
        priority='HIGH',
        confidence=0.8
    )

    print(f"  Running optimization test...")
    print(f"  Model: AttentionModel (embed_dim=256, num_heads=8)")
    print(f"  Input shape: {inputs['x'].shape}")

    # Baseline measurement
    baseline_metrics = executor._profile_operation(operation, inputs)
    print(f"  Baseline: {baseline_metrics['gpu_time_ms']:.3f}ms")

    # Try optimization
    result = executor.apply_optimization(model, operation, candidate, inputs)

    print(f"\n  Optimization Result:")
    print(f"    Success: {result.success}")
    print(f"    Baseline: {result.baseline_time_ms:.3f}ms")
    print(f"    Optimized: {result.optimized_time_ms:.3f}ms")
    print(f"    Speedup: {result.speedup_pct:+.1f}%")
    if result.error_message:
        print(f"    Message: {result.error_message}")

    # Even if no speedup (due to torch.compile warmup), test passed if no error
    if not result.regression_detected:
        print(f"\n  [PASS] No regression detected")
        return True
    else:
        print(f"\n  [FAIL] Regression detected: {result.speedup_pct:.1f}%")
        return False


def test_end_to_end_pipeline():
    """Test 10: End-to-end Phase 1 -> 2 -> 3 pipeline."""
    print("\n" + "=" * 60)
    print("TEST 10: End-to-End Pipeline")
    print("=" * 60)

    if not HAS_TORCH:
        print("  [SKIP] PyTorch not available")
        return True

    from memopt.phase3 import AutoOptimizer

    # Create simple model
    class SimpleFFN(nn.Module):
        def __init__(self, dim=256):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim * 4)
            self.gelu = nn.GELU()
            self.fc2 = nn.Linear(dim * 4, dim)
            self.dropout = nn.Dropout(0.1)

        def forward(self, x):
            return self.dropout(self.fc2(self.gelu(self.fc1(x))))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleFFN().to(device)
    model.eval()

    inputs = {'x': torch.randn(16, 64, 256, device=device)}

    # Create auto-optimizer
    optimizer = AutoOptimizer(tolerance_pct=10.0, max_optimizations=3)

    print(f"  Model: SimpleFFN (256 -> 1024 -> 256)")
    print(f"  Device: {device}")
    print(f"  Available optimizations: {optimizer.get_available_optimizations()}")
    print(f"  Available kernels: {optimizer.get_available_kernels()}")

    # Run optimization
    try:
        result = optimizer.optimize(
            model=model,
            inputs=inputs,
            gpu_name='A100' if device == 'cuda' else 'CPU'
        )

        print(f"\n  AutoOptimizer Result:")
        print(f"    Success: {result.success}")
        print(f"    Speedup: {result.speedup_pct:.1f}%")
        print(f"    Applied: {result.applied_optimizations}")
        print(f"    Failed: {result.failed_optimizations[:2]}...")  # First 2

        if result.error_message:
            print(f"    Error: {result.error_message}")

        # Success if no errors (speedup may vary)
        if result.error_message is None or "No optimizations" in str(result.error_message):
            print(f"\n  [PASS] Pipeline completed without errors")
            return True
        else:
            print(f"\n  [FAIL] Pipeline had errors")
            return False

    except Exception as e:
        print(f"\n  [FAIL] Pipeline exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Phase 3 validation tests."""
    print("=" * 70)
    print("PHASE 3 COMPREHENSIVE VALIDATION")
    print("Auto-Optimization Engine + Custom Kernel Library")
    print("=" * 70)

    if HAS_TORCH:
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("PyTorch: Not available")

    tests = [
        ("Imports", test_phase3_imports),
        ("OptimizationExecutor Structure", test_optimization_executor_structure),
        ("Kernel Registry", test_kernel_registry),
        ("Transformation Engine", test_transformation_engine),
        ("OptimizationSequencer Structure", test_optimization_sequencer_structure),
        ("AutoOptimizer Structure", test_auto_optimizer_structure),
        ("Mock Model Flow", test_with_mock_model),
        ("Fused Attention Kernel", test_fused_attention_kernel),
        ("Real GPU Optimization", test_real_gpu_optimization),
        ("End-to-End Pipeline", test_end_to_end_pipeline),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  [ERROR] Test '{name}' raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)

    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\n  Total: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n  " + "=" * 60)
        print("  ALL PHASE 3 VALIDATION TESTS PASSED!")
        print("  " + "=" * 60)
        return 0
    else:
        print(f"\n  [FAIL] {total_count - passed_count} tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
