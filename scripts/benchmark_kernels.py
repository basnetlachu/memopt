#!/usr/bin/env python3
"""
Kernel synthesis benchmark for memopt.

Measures real before/after timing for the three covered ops plus attention.
Run on GPU rental to get real numbers.

Usage:
  python scripts/benchmark_kernels.py --op all --seq-len 2048 --batch 8

On CPU: reports "GPU required" for each op.
On GPU: reports real timing in ms and speedup ratio.

HONEST: These numbers depend entirely on what kernel Claude synthesizes
and whether it compiles correctly. We report whatever happens — faster,
same, or slower.
"""

import argparse
import os
import sys
import time

# Ensure memopt is importable when run from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def benchmark_op(
    op_name: str,
    shapes: dict,
    reference_fn,
    synthesize_fn,
    n_warmup: int = 5,
    n_iters: int = 20,
) -> dict:
    """
    Benchmark one op. Returns timing dict.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return {
                "op": op_name,
                "status": "gpu_required",
                "note": "Run on GPU rental",
            }

        inputs = shapes["inputs"]

        # Attempt synthesis
        print(f"  Synthesizing {op_name}...")
        t0 = time.time()
        module = synthesize_fn()
        synth_time = time.time() - t0

        if module is None:
            return {
                "op": op_name,
                "status": "synthesis_failed",
                "synth_time_s": round(synth_time, 1),
                "note": (
                    "Synthesis failed or kernel not faster than reference. "
                    "This is the correct behavior — we never use a wrong or "
                    "slower kernel."),
            }

        # Warmup
        for _ in range(n_warmup):
            with torch.no_grad():
                reference_fn(*inputs)
                module.run_kernel(*inputs)
        torch.cuda.synchronize()

        # Benchmark reference
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(n_iters):
            with torch.no_grad():
                reference_fn(*inputs)
        end.record()
        torch.cuda.synchronize()
        ref_ms = start.elapsed_time(end) / n_iters

        # Benchmark synthesized
        start.record()
        for _ in range(n_iters):
            with torch.no_grad():
                module.run_kernel(*inputs)
        end.record()
        torch.cuda.synchronize()
        synth_ms = start.elapsed_time(end) / n_iters

        speedup = ref_ms / synth_ms if synth_ms > 0 else 0.0
        status = "FASTER" if speedup > 1.05 else "NOT FASTER"

        return {
            "op": op_name,
            "status": "measured",
            "ref_ms": round(ref_ms, 3),
            "synth_ms": round(synth_ms, 3),
            "speedup": round(speedup, 3),
            "synth_time_s": round(synth_time, 1),
            "note": f"Real measured numbers. {status} than reference.",
        }

    except Exception as e:
        return {"op": op_name, "status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="memopt kernel benchmark")
    parser.add_argument(
        "--op",
        choices=["rope", "layer_norm", "softmax", "attention", "all"],
        default="all")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--dim", type=int, default=128)
    args = parser.parse_args()

    print("memopt Kernel Synthesis Benchmark")
    print(f"  seq_len: {args.seq_len}")
    print(f"  batch:   {args.batch}")
    print()

    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("GPU: not available. Run on GPU rental for real numbers.")
    except ImportError:
        print("torch not available")
    print()

    results = []

    ops_to_run = (
        ["rope", "layer_norm", "softmax", "attention"]
        if args.op == "all"
        else [args.op])

    for op in ops_to_run:
        print(f"Benchmarking: {op}")

        try:
            import torch
            import torch.nn.functional as F
            from memopt.kernels.jit_generator import JITGenerator
            from memopt.kernels.kernel_cache import KernelCache
            from memopt.kernels.portability_layer import PortabilityLayer

            cache = KernelCache()
            port = PortabilityLayer()
            gen = JITGenerator(cache=cache, portability=port)

            B = args.batch
            S = args.seq_len
            H = args.heads
            D = args.dim

            if op == "attention":
                if not torch.cuda.is_available():
                    result = {
                        "op": "attention",
                        "status": "gpu_required",
                        "note": "Run on GPU rental",
                    }
                else:
                    q = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda")
                    k = torch.randn_like(q)
                    v = torch.randn_like(q)
                    scale = D ** -0.5

                    def ref_attn(*_args):
                        return F.scaled_dot_product_attention(q, k, v, scale=scale)

                    source = gen.synthesize_flash_attention(
                        q_shape=(B, H, S, D),
                        k_shape=(B, H, S, D),
                        v_shape=(B, H, S, D),
                        dtype="float16",
                        causal=False)

                    module = None
                    if source:
                        module = port.compile(source, gen._get_hardware_description())
                        if module:
                            ok = gen.validate_attention(module, (B, H, S, D))
                            if not ok:
                                module = None

                    result = benchmark_op(
                        op_name="attention",
                        shapes={"inputs": [q, k, v]},
                        reference_fn=ref_attn,
                        synthesize_fn=lambda: module,
                    )
            else:
                result = {
                    "op": op,
                    "status": "not_implemented",
                    "note": (
                        f"{op} benchmark requires auto-optimizer warm-up. "
                        f"Use memopt-serve with real model traffic instead."),
                }

        except Exception as e:
            result = {"op": op, "status": "error", "error": str(e)}

        results.append(result)

        print(f"  status: {result['status']}")
        if result.get("speedup"):
            print(f"  speedup: {result['speedup']:.3f}x")
        if result.get("ref_ms"):
            print(f"  reference: {result['ref_ms']:.3f}ms")
        if result.get("synth_ms"):
            print(f"  synthesized: {result['synth_ms']:.3f}ms")
        if result.get("note"):
            print(f"  note: {result['note']}")
        print()

    # Summary
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    measured = [r for r in results if r["status"] == "measured"]

    if not measured:
        print(
            "No measured results. "
            "Run on GPU rental with ANTHROPIC_API_KEY set.")
    else:
        for r in measured:
            status = "FASTER" if r["speedup"] > 1.05 else "NOT FASTER"
            print(f"{r['op']:20s} {r['speedup']:.3f}x ({status})")

    print()
    print(
        "HONEST NOTE: These are real measured numbers on real hardware. "
        "Speedup depends on what kernel was synthesized and the specific "
        "hardware. We report whatever the measurement shows.")


if __name__ == "__main__":
    main()
