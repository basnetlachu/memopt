"""
MemOpt CLI Tool

Command-line interface for bandwidth profiling and optimization.
"""

import argparse
import sys
import torch

from .optimization_engine import run_optimization_demo
from .hardware_validator import HardwareValidator


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MemOpt - GPU Memory Bandwidth Profiling & Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run optimization demo
  memopt-profile --demo lazy-kv

  # Run with hardware validation
  memopt-profile --demo lazy-kv --validate-hardware

  # Check GPU info
  memopt-profile --gpu-info

  # Show validation setup instructions
  memopt-profile --validation-setup

For more information: https://github.com/yourusername/memopt
        """,
    )

    parser.add_argument(
        "--demo",
        choices=["lazy-kv", "quantization", "all"],
        help="Run optimization demo",
    )

    parser.add_argument(
        "--gpu-info",
        action="store_true",
        help="Show GPU information",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use (cuda or cpu)",
    )

    parser.add_argument(
        "--validate-hardware",
        action="store_true",
        help="Validate bandwidth measurements using Nsight Compute hardware counters",
    )

    parser.add_argument(
        "--validation-setup",
        action="store_true",
        help="Show instructions for installing hardware validation tools",
    )

    args = parser.parse_args()

    # Show validation setup instructions
    if args.validation_setup:
        validator = HardwareValidator()
        print("\n" + "="*70)
        print("HARDWARE VALIDATION SETUP")
        print("="*70)
        print(validator.get_installation_instructions())
        print("\n" + "="*70 + "\n")
        return 0

    # Check hardware validation tools
    if args.validate_hardware:
        validator = HardwareValidator()
        if not validator.ncu_available:
            print("\n⚠️  WARNING: Nsight Compute (ncu) not found!")
            print("   Hardware validation requires Nsight Compute.")
            print("   Run: memopt-profile --validation-setup for installation instructions\n")
            return 1

    # Show GPU info
    if args.gpu_info or not any([args.demo, args.validate_hardware]):
        print("\n" + "="*70)
        print("GPU INFORMATION")
        print("="*70)

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  Memory: {props.total_memory / (1024**3):.1f} GB")
                print(f"  Compute Capability: {props.major}.{props.minor}")
                print(f"  Multi-processors: {props.multi_processor_count}")
        else:
            print("\n⚠️  No CUDA GPUs available")
            print("   MemOpt requires NVIDIA GPU for bandwidth profiling")

        print("\n" + "="*70 + "\n")

        if not args.demo:
            return 0

    # Run demo
    if args.demo:
        print("\n" + "="*70)
        print(f"MEMOPT - {args.demo.upper()} DEMO")
        if args.validate_hardware:
            print("WITH HARDWARE VALIDATION")
        print("="*70 + "\n")

        if args.validate_hardware:
            print("⚠️  Note: Hardware validation adds significant overhead")
            print("   Nsight Compute will profile all GPU kernels\n")

        if args.demo in ["lazy-kv", "all"]:
            run_optimization_demo(
                device=args.device,
                validate_hardware=args.validate_hardware
            )

        if args.demo == "all":
            print("\n✅ All demos complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
