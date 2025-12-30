"""
MemOpt Command Line Interface

Provides diagnostic and management commands for MemOpt.
"""

import os
import sys
import argparse
from typing import Optional
import warnings


def check_license() -> bool:
    """Check license validity and print status."""
    print("=" * 60)
    print("License Status")
    print("=" * 60)

    try:
        from .license import LicenseManager

        # Check dev mode
        dev_mode = os.getenv("MEMOPT_DEV_MODE") == "1"
        if dev_mode:
            print("Mode: DEV MODE (bypassing license check)")
            print("Status: OK (development)")
            print()
            return True

        # Check license path
        license_path = os.getenv("MEMOPT_LICENSE_PATH", "/etc/memopt/license.json")
        print(f"License path: {license_path}")

        if not os.path.exists(license_path):
            print(f"Status: ERROR - License file not found")
            print(f"  Set MEMOPT_LICENSE_PATH or create {license_path}")
            print()
            return False

        # Validate license
        manager = LicenseManager(dev_mode=False)
        license_obj = manager.load_and_validate()

        print(f"Customer ID: {license_obj.customer_id}")
        print(f"Tier: {license_obj.tier}")
        print(f"Expires: {license_obj.expires_at}")
        print(f"Max GPUs: {license_obj.max_gpus}")
        print(f"Features: {', '.join(license_obj.features)}")
        print(f"Telemetry: {'Required' if license_obj.telemetry_required else 'Optional'}")
        print(f"Status: VALID")
        print()
        return True

    except ImportError as e:
        print(f"Status: ERROR - Cannot import license module: {e}")
        print()
        return False
    except Exception as e:
        print(f"Status: ERROR - {e}")
        print()
        return False


def check_vllm() -> bool:
    """Check vLLM installation and version."""
    print("=" * 60)
    print("vLLM Status")
    print("=" * 60)

    try:
        import vllm
        print(f"Installed: Yes")
        print(f"Version: {vllm.__version__}")
        print(f"Location: {vllm.__file__}")
        print()
        return True
    except ImportError:
        print(f"Installed: No")
        print(f"  Install vLLM: pip install vllm")
        print()
        return False


def check_gpu() -> bool:
    """Check GPU availability and CUDA status."""
    print("=" * 60)
    print("GPU Status")
    print("=" * 60)

    try:
        import torch

        if not torch.cuda.is_available():
            print("CUDA: Not available")
            print("  MemOpt requires CUDA for GPU acceleration")
            print()
            return False

        print(f"CUDA: Available")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    Memory: {props.total_memory / (1024**3):.2f} GB")
            print(f"    Compute: {props.major}.{props.minor}")

        print()
        return True

    except ImportError:
        print("PyTorch: Not installed")
        print("  Install PyTorch: pip install torch")
        print()
        return False
    except Exception as e:
        print(f"Error: {e}")
        print()
        return False


def check_plugin_status() -> bool:
    """Check MemOpt vLLM plugin status."""
    print("=" * 60)
    print("MemOpt Plugin Status")
    print("=" * 60)

    # Check environment variables
    enabled = os.getenv("MEMOPT_ENABLED") == "1"
    strict = os.getenv("MEMOPT_STRICT") == "1"
    safe_mode = os.getenv("MEMOPT_SAFE_MODE") == "1"

    print(f"MEMOPT_ENABLED: {enabled}")
    print(f"MEMOPT_STRICT: {strict}")
    print(f"MEMOPT_SAFE_MODE: {safe_mode}")

    if not enabled:
        print(f"Status: DISABLED")
        print(f"  Set MEMOPT_ENABLED=1 to enable")
        print()
        return False

    # Try to get plugin status
    try:
        from .vllm_plugin import get_status
        status = get_status()

        if not status.get('enabled', False):
            print(f"Status: NOT INITIALIZED")
            print(f"  Reason: {status.get('reason', 'Unknown')}")
            print()
            return False

        print(f"Status: ACTIVE")
        print(f"vLLM Version: {status.get('vllm_version', 'Unknown')}")

        patches = status.get('patches_applied', {})
        if patches:
            print(f"Patches Applied:")
            for component, applied in patches.items():
                status_str = "OK" if applied else "FAILED"
                print(f"  - {component}: {status_str}")

        errors = status.get('patch_errors', {})
        if errors:
            print(f"Patch Errors:")
            for component, error in errors.items():
                print(f"  - {component}: {error}")

        print()
        return True

    except ImportError:
        print(f"Status: ERROR - Cannot import vllm_plugin")
        print()
        return False
    except Exception as e:
        print(f"Status: ERROR - {e}")
        print()
        return False


def cmd_doctor(args):
    """Run MemOpt diagnostics."""
    print()
    print("MemOpt System Diagnostics")
    print()

    # Run all checks
    license_ok = check_license()
    vllm_ok = check_vllm()
    gpu_ok = check_gpu()
    plugin_ok = check_plugin_status()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    all_ok = license_ok and vllm_ok and gpu_ok and plugin_ok

    if all_ok:
        print("Status: ALL SYSTEMS OPERATIONAL")
        print()
        print("MemOpt is ready to use!")
        print("Run vLLM with MEMOPT_ENABLED=1 to activate optimizations.")
        return 0
    else:
        print("Status: ISSUES DETECTED")
        print()
        print("Please fix the errors above before using MemOpt.")
        return 1


def cmd_version(args):
    """Show MemOpt version."""
    from . import __version__
    print(f"MemOpt version {__version__}")
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="memopt",
        description="MemOpt - GPU Memory Optimization for LLM Inference"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Doctor command
    parser_doctor = subparsers.add_parser(
        "doctor",
        help="Run system diagnostics"
    )
    parser_doctor.set_defaults(func=cmd_doctor)

    # Version command
    parser_version = subparsers.add_parser(
        "version",
        help="Show version information"
    )
    parser_version.set_defaults(func=cmd_version)

    # Parse and execute
    args = parser.parse_args()

    if not hasattr(args, 'func'):
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
