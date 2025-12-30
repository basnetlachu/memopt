#!/usr/bin/env python3
"""
MemOpt Pre-Flight Checklist

Validates that MemOpt is ready for customer deployment.

Usage:
    python tools/preflight_check.py
"""

import os
import sys
import json
import subprocess
from pathlib import Path


def check(description, passed):
    """Print check result"""
    symbol = "✅" if passed else "❌"
    status = "PASS" if passed else "FAIL"
    print(f"{symbol} {status}: {description}")
    return passed


def run_command(cmd):
    """Run command and return success status"""
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
        return True
    except:
        return False


def main():
    print("=" * 80)
    print("MemOpt Pre-Flight Checklist")
    print("=" * 80)
    print()

    all_passed = True

    # Check 1: Repository structure
    print("📁 Repository Structure")
    print("-" * 80)

    required_files = [
        "memopt/__init__.py",
        "memopt/license.py",
        "memopt/vllm_plugin.py",
        "memopt/cli.py",
        "setup.py",
        "README.md",
    ]

    for file_path in required_files:
        exists = Path(file_path).exists()
        all_passed &= check(f"File exists: {file_path}", exists)

    print()

    # Check 2: License system configuration
    print("🔐 License System")
    print("-" * 80)

    try:
        with open("memopt/license.py") as f:
            license_code = f.read()

        # Check if placeholder key is still there
        has_placeholder = "a1b2c3d4e5f6a1b2c3d4" in license_code
        all_passed &= check(
            "Public key updated (not placeholder)",
            not has_placeholder
        )

        if has_placeholder:
            print("   ⚠️  Run: python tools/generate_keypair.py")
    except:
        all_passed &= check("Can read memopt/license.py", False)

    print()

    # Check 3: License signing tool
    print("🔏 License Signing")
    print("-" * 80)

    try:
        with open("tools/sign_license.py") as f:
            signing_code = f.read()

        has_placeholder = "REPLACE_WITH_YOUR_PRIVATE_KEY" in signing_code
        all_passed &= check(
            "Private key configured in tools/sign_license.py",
            not has_placeholder
        )

        if has_placeholder:
            print("   ⚠️  Update PRIVATE_KEY_HEX in tools/sign_license.py")
    except:
        all_passed &= check("Can read tools/sign_license.py", False)

    print()

    # Check 4: Dependencies
    print("📦 Dependencies")
    print("-" * 80)

    try:
        import nacl
        all_passed &= check("PyNaCl installed", True)
    except ImportError:
        all_passed &= check("PyNaCl installed", False)
        print("   ⚠️  Run: pip install pynacl")

    try:
        import torch
        all_passed &= check("PyTorch installed", True)
    except ImportError:
        all_passed &= check("PyTorch installed", False)
        print("   ⚠️  Run: pip install torch")

    try:
        import transformers
        all_passed &= check("Transformers installed", True)
    except ImportError:
        all_passed &= check("Transformers installed", False)
        print("   ⚠️  Run: pip install transformers")

    print()

    # Check 5: Package build
    print("🔨 Package Build")
    print("-" * 80)

    can_build = run_command("python -m build --help > /dev/null 2>&1")
    all_passed &= check("Build tools available", can_build)

    if not can_build:
        print("   ⚠️  Run: pip install build")

    print()

    # Check 6: Tests
    print("🧪 Tests")
    print("-" * 80)

    test_files_exist = (
        Path("tests/test_license.py").exists() and
        Path("tests/test_vllm_plugin.py").exists()
    )
    all_passed &= check("Test files exist", test_files_exist)

    print()

    # Check 7: Documentation
    print("📚 Documentation")
    print("-" * 80)

    doc_files = [
        "docs/ONPREM_VLLM_PLUGIN.md",
        "docs/PERFORMANCE_SAFETY.md",
    ]

    for doc_file in doc_files:
        exists = Path(doc_file).exists()
        all_passed &= check(f"Documentation: {doc_file}", exists)

    print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()

    if all_passed:
        print("✅ ALL CHECKS PASSED - Ready for customer deployment!")
        print()
        print("Next steps:")
        print("1. Build package: python -m build")
        print("2. Test installation: pip install dist/*.whl")
        print("3. Create test license: python tools/sign_license.py --test")
        print("4. Validate license: MEMOPT_ENABLED=1 memopt doctor")
        print("5. Upload to PyPI: python -m twine upload dist/*")
        print()
        return 0
    else:
        print("❌ ISSUES DETECTED - Fix errors above before deployment")
        print()
        print("Quick fixes:")
        print("1. Generate keys: python tools/generate_keypair.py")
        print("2. Update memopt/license.py with public key")
        print("3. Update tools/sign_license.py with private key")
        print("4. Install dependencies: pip install pynacl torch transformers build")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
