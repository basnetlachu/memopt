#!/usr/bin/env python3
"""
Quick fix for PyTorch API compatibility issue.

Run this on the GPU server to fix the cuda_time_total AttributeError.

Usage:
    python3 apply_pytorch_fix.py
"""

import os
import sys
from pathlib import Path

def apply_fix():
    """Apply PyTorch compatibility fix to bandwidth_profiler.py"""

    # Find the file
    script_dir = Path(__file__).parent
    target_file = script_dir / "memopt" / "bandwidth_profiler.py"

    if not target_file.exists():
        print(f"❌ Error: {target_file} not found")
        print(f"   Make sure you're running this from the memopt directory")
        return False

    print(f"📝 Applying PyTorch compatibility fix to {target_file}")

    # Read the file
    with open(target_file, 'r') as f:
        content = f.read()

    # Check if already fixed
    if 'device_time_total' in content:
        print("✅ Fix already applied! File is up to date.")
        return True

    # Backup original
    backup_file = target_file.with_suffix('.py.backup')
    with open(backup_file, 'w') as f:
        f.write(content)
    print(f"💾 Backup created: {backup_file}")

    # Old code to replace
    old_code = """        for event in events:
            # Sum CUDA time
            if event.device_type == torch.profiler.DeviceType.CUDA:
                total_cuda_time_us += event.cuda_time_total

                # Estimate memory traffic from tensor sizes
                # This is approximate - PyTorch doesn't expose exact HBM traffic
                if hasattr(event, 'cuda_memory_usage') and event.cuda_memory_usage > 0:
                    total_memory_bytes += event.cuda_memory_usage"""

    # New code with fix
    new_code = """        for event in events:
            # Sum CUDA time
            if event.device_type == torch.profiler.DeviceType.CUDA:
                # Handle both old and new PyTorch API
                if hasattr(event, 'cuda_time_total'):
                    total_cuda_time_us += event.cuda_time_total
                elif hasattr(event, 'device_time_total'):
                    total_cuda_time_us += event.device_time_total
                else:
                    # Fallback to self_cuda_time_total if available
                    total_cuda_time_us += getattr(event, 'self_cuda_time_total', 0)

                # Estimate memory traffic from tensor sizes
                # This is approximate - PyTorch doesn't expose exact HBM traffic
                if hasattr(event, 'cuda_memory_usage') and event.cuda_memory_usage > 0:
                    total_memory_bytes += event.cuda_memory_usage"""

    # Apply fix
    if old_code in content:
        content = content.replace(old_code, new_code)

        # Write fixed content
        with open(target_file, 'w') as f:
            f.write(content)

        print("✅ Fix applied successfully!")
        print("\nVerify the fix:")
        print("  python3 test_installation.py")
        print("  python3 -m memopt.cli --demo lazy-kv")
        return True
    else:
        print("⚠️  Warning: Could not find exact code to replace")
        print("   The file may have already been modified or is a different version")
        print("   Please apply the fix manually using PYTORCH_COMPATIBILITY_FIX.md")
        return False

if __name__ == "__main__":
    print("="*70)
    print("PYTORCH COMPATIBILITY FIX")
    print("="*70)
    print()

    success = apply_fix()

    print()
    print("="*70)

    if success:
        sys.exit(0)
    else:
        sys.exit(1)
