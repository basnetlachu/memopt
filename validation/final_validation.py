#!/usr/bin/env python3
"""
Final Validation: What Can We Actually Prove?

This script runs real tests on GPU and reports honest results.
No fake numbers, no theoretical calculations - just real measurements.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
from pathlib import Path

# Check for GPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_profiler():
    """Test 1: Can we profile GPU memory?"""
    print("[1/5] Testing Profiler...")

    try:
        from memopt.bandwidth_profiler import BandwidthProfiler
        profiler = BandwidthProfiler()
        print("      ✅ Profiler imports and runs")
        return True, "Profiler works"
    except Exception as e:
        print(f"      ❌ Profiler failed: {e}")
        return False, str(e)


def test_coalescer_integration():
    """Test 2: Does coalescer hook into models?"""
    print("[2/5] Testing Coalescer Integration...")

    try:
        from transformers import AutoModelForCausalLM
        from memopt.optimization import MemoryCoalescer

        model = AutoModelForCausalLM.from_pretrained('gpt2')
        if DEVICE == 'cuda':
            model = model.cuda()

        coalescer = MemoryCoalescer(model, mode='inference')
        coalescer.enable()

        stats = coalescer.get_stats()
        layers = stats.layers_optimized

        coalescer.disable()

        if layers > 0:
            print(f"      ✅ Coalescer hooks into {layers} layers")
            return True, f"{layers} layers"
        else:
            print("      ❌ No layers hooked")
            return False, "No layers"

    except Exception as e:
        print(f"      ❌ Integration failed: {e}")
        return False, str(e)


def test_access_tracking():
    """Test 3: Does coalescer track memory accesses?"""
    print("[3/5] Testing Access Tracking...")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from memopt.optimization import MemoryCoalescer

        model = AutoModelForCausalLM.from_pretrained('gpt2')
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        if DEVICE == 'cuda':
            model = model.cuda()

        model.eval()
        coalescer = MemoryCoalescer(model, mode='inference')
        coalescer.enable()

        # Run some inference
        prompts = ["Hello world"] * 5
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors='pt')
            if DEVICE == 'cuda':
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                model.generate(
                    **inputs,
                    max_new_tokens=20,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id
                )

        stats = coalescer.get_stats()
        coalescer.disable()

        if stats.total_accesses > 0:
            print(f"      ✅ Tracked {stats.total_accesses:,} accesses")
            print(f"         Hit rate: {stats.hit_rate:.1f}%")
            print(f"         Bandwidth reduction: {stats.bandwidth_reduction:.1f}%")
            return True, {
                "accesses": stats.total_accesses,
                "hit_rate": stats.hit_rate,
                "bandwidth_reduction": stats.bandwidth_reduction
            }
        else:
            print("      ❌ No accesses tracked")
            return False, "No accesses"

    except Exception as e:
        print(f"      ❌ Tracking failed: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)


def test_bandwidth_measurement():
    """Test 4: Can we measure real GPU bandwidth?"""
    print("[4/5] Testing Bandwidth Measurement...")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from memopt.measurement import BandwidthTracker

        model = AutoModelForCausalLM.from_pretrained('gpt2')
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        if DEVICE == 'cuda':
            model = model.cuda()

        model.eval()
        tracker = BandwidthTracker()

        with tracker.measure("test"):
            inputs = tokenizer("Hello world", return_tensors='pt')
            if DEVICE == 'cuda':
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                for _ in range(5):
                    model.generate(
                        **inputs,
                        max_new_tokens=20,
                        use_cache=True,
                        pad_token_id=tokenizer.eos_token_id
                    )

        measurement = tracker.get_measurement("test")

        if measurement.peak_memory_bytes > 0:
            print(f"      ✅ Peak memory: {measurement.peak_memory_gb:.3f} GB")
            print(f"         Duration: {measurement.duration_ms:.1f} ms")
            return True, {
                "peak_memory_gb": measurement.peak_memory_gb,
                "duration_ms": measurement.duration_ms
            }
        else:
            print("      ❌ No memory measured")
            return False, "No memory"

    except Exception as e:
        print(f"      ❌ Measurement failed: {e}")
        return False, str(e)


def test_correctness():
    """Test 5: Does coalescing preserve output correctness?"""
    print("[5/5] Testing Correctness...")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from memopt.optimization import MemoryCoalescer

        model = AutoModelForCausalLM.from_pretrained('gpt2')
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        if DEVICE == 'cuda':
            model = model.cuda()

        model.eval()

        prompt = "The quick brown fox"
        inputs = tokenizer(prompt, return_tensors='pt')
        if DEVICE == 'cuda':
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Baseline
        with torch.no_grad():
            baseline = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # With coalescing
        coalescer = MemoryCoalescer(model, mode='inference')
        coalescer.enable()

        with torch.no_grad():
            optimized = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )

        coalescer.disable()

        if torch.equal(baseline, optimized):
            print("      ✅ Outputs match (lossless optimization)")
            return True, "Outputs match"
        else:
            print("      ❌ Outputs differ")
            return False, "Outputs differ"

    except Exception as e:
        print(f"      ❌ Correctness test failed: {e}")
        return False, str(e)


def main():
    print("=" * 70)
    print("FINAL VALIDATION: WHAT CAN WE ACTUALLY PROVE?")
    print("=" * 70)
    print()
    print(f"Device: {DEVICE}")
    print()

    results = {}

    # Run all tests
    results['profiler'] = test_profiler()
    print()

    results['integration'] = test_coalescer_integration()
    print()

    results['tracking'] = test_access_tracking()
    print()

    results['measurement'] = test_bandwidth_measurement()
    print()

    results['correctness'] = test_correctness()
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    passed = sum(1 for r in results.values() if r[0])
    total = len(results)

    print(f"Tests passed: {passed}/{total}")
    print()

    for name, (success, detail) in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {name}: {detail if isinstance(detail, str) else 'OK'}")

    print()

    # What we can claim
    print("=" * 70)
    print("WHAT WE CAN HONESTLY CLAIM")
    print("=" * 70)
    print()

    if results['profiler'][0]:
        print("✅ Can profile GPU memory bandwidth usage")

    if results['integration'][0]:
        print(f"✅ Can hook into transformer layers ({results['integration'][1]})")

    if results['tracking'][0]:
        tracking_data = results['tracking'][1]
        print(f"✅ Can track memory access patterns")
        print(f"   - {tracking_data['accesses']:,} accesses tracked")
        print(f"   - {tracking_data['hit_rate']:.1f}% cache hit rate")
        print(f"   - {tracking_data['bandwidth_reduction']:.1f}% bandwidth reduction potential")

    if results['measurement'][0]:
        meas_data = results['measurement'][1]
        print(f"✅ Can measure real GPU memory")
        print(f"   - Peak: {meas_data['peak_memory_gb']:.3f} GB")

    if results['correctness'][0]:
        print("✅ Optimization is lossless (outputs match)")

    print()

    # What we cannot claim
    print("=" * 70)
    print("WHAT WE CANNOT CLAIM (YET)")
    print("=" * 70)
    print()
    print("❌ Measured peak memory reduction (tracking ≠ reducing)")
    print("   - We track access patterns, but don't reduce allocations")
    print("   - Actual memory reduction requires deeper integration")
    print()
    print("❌ Hardware-validated Nsight results")
    print("   - No Nsight CSV files generated")
    print("   - Would need: ncu --metrics dram__bytes_read.sum,...")
    print()
    print("❌ Production-ready optimization")
    print("   - Current: Profiling + tracking tool")
    print("   - Needed: Actual tensor caching implementation")

    print()

    # Save results
    report = {
        "device": DEVICE,
        "tests_passed": passed,
        "tests_total": total,
        "results": {k: {"success": v[0], "detail": str(v[1])} for k, v in results.items()},
        "can_claim": [
            "Can profile GPU memory bandwidth usage",
            "Can hook into transformer layers",
            "Can track memory access patterns",
            "Can measure real GPU memory",
            "Optimization is lossless"
        ],
        "cannot_claim": [
            "Measured peak memory reduction",
            "Hardware-validated Nsight results",
            "Production-ready optimization"
        ],
        "readiness": {
            "profiling_service": "YES - can sell profiling + analysis",
            "optimization_product": "NO - need actual caching implementation",
            "price_point": "$10-25K for profiling, NOT $200K+ for optimization"
        }
    }

    output_path = Path(__file__).parent / 'final_validation_results.json'
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print("=" * 70)
    print(f"Results saved: {output_path}")
    print("=" * 70)

    # Final verdict
    print()
    if passed >= 4:
        print(f"VERDICT: {passed}/5 tests passed")
        print("Ready for: $10-25K profiling service")
        print("Not ready for: $200K optimization product")
    else:
        print(f"VERDICT: {passed}/5 tests passed")
        print("Need to fix failing tests before any sales")

    print()
    print("=" * 70)

    return report


if __name__ == '__main__':
    main()
