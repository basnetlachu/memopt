#!/usr/bin/env python3
"""
memopt CLI - Command Line Interface

Usage:
    memopt optimize --model path/to/model.pt --input-shape 8,512,1024
    memopt profile --model path/to/model.pt
    memopt info
    memopt sessions
"""

import argparse
import sys
import json


def cmd_info(args):
    """Show memopt and GPU info."""
    from memopt.profiler import api

    print("=" * 50)
    print("MEMOPT INFO")
    print("=" * 50)
    print(f"Version: {api.__version__}")
    print(f"API Contract: {api.__api_version__}")
    print()

    gpu_info = api.get_gpu_info()
    print("GPU:")
    for key, value in gpu_info.items():
        print(f"  {key}: {value}")


def cmd_sessions(args):
    """List saved optimization sessions."""
    from memopt.profiler import api

    sessions = api.list_sessions(args.model if hasattr(args, 'model') else None)

    if not sessions:
        print("No saved sessions found.")
        return

    print("=" * 70)
    print(f"{'Model':<20} {'Session':<15} {'Speedup':<10} {'Committed':<10}")
    print("=" * 70)

    for s in sessions:
        print(f"{s.get('model_name', 'unknown'):<20} "
              f"{s.get('session_id', ''):<15} "
              f"{s.get('total_speedup', 1.0):.3f}x     "
              f"{s.get('committed_count', 0):<10}")


def cmd_optimize(args):
    """Optimize a saved model."""
    import torch
    from memopt.profiler import api

    print(f"Loading model from: {args.model}")

    # Parse input shape
    shape = [int(x) for x in args.input_shape.split(",")]
    print(f"Input shape: {shape}")

    # Load model
    try:
        model = torch.load(args.model)
    except Exception as e:
        # Try loading as state dict
        print(f"Trying to load as state dict: {e}")
        model = torch.jit.load(args.model)

    if torch.cuda.is_available():
        model = model.cuda()
        sample = torch.randn(*shape).cuda()
    else:
        sample = torch.randn(*shape)

    # Optimize
    print("\nOptimizing...")
    model, session = api.optimize(model, sample, verbose=True)

    print(f"\nResults saved to: ~/.memopt/sessions/")


def cmd_profile(args):
    """Profile a model without optimization."""
    import torch
    from memopt.profiler import api

    print(f"Loading model from: {args.model}")

    shape = [int(x) for x in args.input_shape.split(",")]

    model = torch.load(args.model)
    if torch.cuda.is_available():
        model = model.cuda()

    snapshot = api.profile(
        model,
        lambda: torch.randn(*shape).cuda() if torch.cuda.is_available() else torch.randn(*shape),
        num_iterations=args.iterations
    )

    print("\n" + "=" * 50)
    print("PROFILE RESULTS")
    print("=" * 50)
    print(f"Total GPU Time: {snapshot.total_gpu_time_ms:.2f} ms")
    print(f"Memory-Bound: {snapshot.memory_bound_pct:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        prog="memopt",
        description="GPU Memory Optimization Platform"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # info command
    info_parser = subparsers.add_parser("info", help="Show memopt and GPU info")
    info_parser.set_defaults(func=cmd_info)

    # sessions command
    sessions_parser = subparsers.add_parser("sessions", help="List saved sessions")
    sessions_parser.add_argument("--model", help="Filter by model name")
    sessions_parser.set_defaults(func=cmd_sessions)

    # optimize command
    opt_parser = subparsers.add_parser("optimize", help="Optimize a model")
    opt_parser.add_argument("--model", required=True, help="Path to saved model (.pt)")
    opt_parser.add_argument("--input-shape", required=True, help="Input shape (e.g., 8,512,1024)")
    opt_parser.set_defaults(func=cmd_optimize)

    # profile command
    prof_parser = subparsers.add_parser("profile", help="Profile a model")
    prof_parser.add_argument("--model", required=True, help="Path to saved model (.pt)")
    prof_parser.add_argument("--input-shape", required=True, help="Input shape (e.g., 8,512,1024)")
    prof_parser.add_argument("--iterations", type=int, default=5, help="Profile iterations")
    prof_parser.set_defaults(func=cmd_profile)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
