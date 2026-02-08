#!/usr/bin/env python3
"""
Build Script for Compiled memopt Distribution

This script creates a deployable package with compiled .so files (no source code).
Your customers will receive binary extensions that cannot be easily reverse-engineered.

Usage:
    python build_compiled.py

Output:
    dist/memopt-1.0.0-compiled/
        memopt/
            __init__.cpython-3X.so
            profiler/
                __init__.cpython-3X.so
                phase1_profiler.cpython-3X.so
                ...
            phase3/
                auto_optimizer.cpython-3X.so
                ...
        install.sh           # Installation script
        requirements.txt     # Dependencies
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import List

# Configuration
PROJECT_ROOT = Path(__file__).parent
MEMOPT_DIR = PROJECT_ROOT / 'memopt'
BUILD_DIR = PROJECT_ROOT / 'build'
DIST_DIR = PROJECT_ROOT / 'dist'
COMPILED_DIR = DIST_DIR / 'memopt-1.0.0-compiled'

# Directories/files to exclude from distribution
EXCLUDE_PATTERNS = [
    '__pycache__',
    '*.pyc',
    '*.pyo',
    '.git',
    '.venv',
    'tests',
    'examples',
    'dashboard',
    '*_test',
    'build',
    '*.egg-info',
]


def clean_build():
    """Clean previous build artifacts."""
    print("Cleaning previous builds...")

    for path in [BUILD_DIR, COMPILED_DIR]:
        if path.exists():
            shutil.rmtree(path)

    # Clean .c files and .so files from source
    for pattern in ['*.c', '*.so', '*.pyd']:
        for f in MEMOPT_DIR.rglob(pattern):
            f.unlink()


def check_cython():
    """Verify Cython is installed."""
    try:
        import Cython
        print(f"Cython version: {Cython.__version__}")
        return True
    except ImportError:
        print("ERROR: Cython not installed!")
        print("Install with: pip install cython>=3.0")
        return False


def compile_extensions():
    """Compile Python to C extensions."""
    print("\nCompiling Python to C extensions...")

    result = subprocess.run(
        [sys.executable, 'setup.py', 'build_ext', '--inplace'],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("Compilation failed!")
        print(result.stderr)
        return False

    print("Compilation successful!")
    return True


def create_distribution():
    """Create the compiled distribution package."""
    print("\nCreating distribution package...")

    COMPILED_DIR.mkdir(parents=True, exist_ok=True)
    compiled_memopt = COMPILED_DIR / 'memopt'

    # Copy compiled .so files (not .py files)
    for so_file in MEMOPT_DIR.rglob('*.cpython-*.so'):
        # Get relative path
        rel_path = so_file.relative_to(MEMOPT_DIR)
        dest_path = compiled_memopt / rel_path

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(so_file, dest_path)
        print(f"  Copied: {rel_path}")

    # Also copy .pyd files for Windows compatibility
    for pyd_file in MEMOPT_DIR.rglob('*.pyd'):
        rel_path = pyd_file.relative_to(MEMOPT_DIR)
        dest_path = compiled_memopt / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pyd_file, dest_path)

    return True


def create_stub_inits():
    """Create minimal __init__.py stubs for package discovery."""
    print("\nCreating package stubs...")

    compiled_memopt = COMPILED_DIR / 'memopt'

    # Find all directories with .so files
    dirs_with_so = set()
    for so_file in compiled_memopt.rglob('*.so'):
        dirs_with_so.add(so_file.parent)

    # Create __init__.py stubs
    for dir_path in dirs_with_so:
        init_path = dir_path / '__init__.py'
        if not init_path.exists():
            # Check if there's a compiled __init__
            init_so = list(dir_path.glob('__init__.cpython-*.so'))
            if init_so:
                # __init__ is compiled, create import redirect
                init_path.write_text(
                    f'# Compiled module - imports from .so extension\n'
                    f'from {init_so[0].stem.replace(".cpython", "")} import *\n'
                )
            else:
                # Create empty __init__
                init_path.write_text('# Package marker\n')


def create_install_script():
    """Create installation script for customers."""
    print("\nCreating installation script...")

    install_script = COMPILED_DIR / 'install.sh'
    install_script.write_text('''#!/bin/bash
# memopt Installation Script
# Run this script on your GPU server

set -e

echo "Installing memopt dependencies..."
pip install torch>=2.0.0 numpy>=1.24.0

echo "Installing memopt..."
pip install -e .

echo ""
echo "memopt installed successfully!"
echo ""
echo "Quick test:"
echo "  python -c 'from memopt.phase3 import AutoOptimizer; print(\"OK\")'"
echo ""
echo "Usage example:"
echo "  from memopt.phase3 import AutoOptimizer"
echo "  optimizer = AutoOptimizer()"
echo "  result = optimizer.optimize(model, inputs)"
''')
    install_script.chmod(0o755)


def create_requirements():
    """Create requirements.txt for customers."""
    print("Creating requirements.txt...")

    req_file = COMPILED_DIR / 'requirements.txt'
    req_file.write_text('''# memopt Dependencies
torch>=2.0.0
numpy>=1.24.0

# Optional: For LLM optimization
# transformers>=4.30.0
# accelerate>=0.20.0
# flash-attn>=2.0.0
''')


def create_setup_py():
    """Create setup.py for pip install."""
    print("Creating setup.py for pip install...")

    setup_content = '''
from setuptools import setup, find_packages

setup(
    name='memopt',
    version='1.0.0',
    description='GPU Memory Optimization Platform',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.24.0',
    ],
    package_data={
        'memopt': ['*.so', '**/*.so', '*.pyd', '**/*.pyd'],
    },
    include_package_data=True,
    zip_safe=False,
)
'''
    setup_file = COMPILED_DIR / 'setup.py'
    setup_file.write_text(setup_content.strip())


def verify_no_source_code():
    """Verify no Python source code is in the distribution."""
    print("\nVerifying source code protection...")

    compiled_memopt = COMPILED_DIR / 'memopt'
    py_files = list(compiled_memopt.rglob('*.py'))

    # Filter out __init__.py stubs (these are OK)
    source_files = [f for f in py_files if f.name != '__init__.py']

    if source_files:
        print("WARNING: Found Python source files in distribution:")
        for f in source_files:
            print(f"  - {f}")
        return False

    # Check that we have .so files
    so_files = list(compiled_memopt.rglob('*.so'))
    if not so_files:
        print("ERROR: No compiled .so files found!")
        return False

    print(f"Distribution contains {len(so_files)} compiled modules")
    print("No source code exposed")
    return True


def print_summary():
    """Print build summary."""
    compiled_memopt = COMPILED_DIR / 'memopt'
    so_files = list(compiled_memopt.rglob('*.so'))

    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {COMPILED_DIR}")
    print(f"Compiled modules: {len(so_files)}")
    print("\nContents:")

    for item in sorted(COMPILED_DIR.iterdir()):
        if item.is_file():
            print(f"  {item.name}")
        else:
            count = len(list(item.rglob('*.so')))
            print(f"  {item.name}/ ({count} compiled modules)")

    print("\n" + "-" * 60)
    print("DEPLOYMENT INSTRUCTIONS")
    print("-" * 60)
    print(f"""
1. Copy the compiled package to your GPU server:
   scp -r {COMPILED_DIR} user@gpu-server:/opt/

2. On the GPU server, install:
   cd /opt/memopt-1.0.0-compiled
   ./install.sh

3. Test the installation:
   python -c "from memopt.phase3 import AutoOptimizer; print('OK')"

4. Use in your code:
   from memopt.phase3 import AutoOptimizer
   optimizer = AutoOptimizer()
   result = optimizer.optimize(model, inputs)
   print(f"Speedup: {{result.speedup_pct:.1f}}%")
""")


def main():
    print("=" * 60)
    print("memopt Compiled Distribution Builder")
    print("=" * 60)

    # Step 1: Check Cython
    if not check_cython():
        sys.exit(1)

    # Step 2: Clean previous builds
    clean_build()

    # Step 3: Compile to .so files
    if not compile_extensions():
        sys.exit(1)

    # Step 4: Create distribution
    if not create_distribution():
        sys.exit(1)

    # Step 5: Create package files
    create_stub_inits()
    create_install_script()
    create_requirements()
    create_setup_py()

    # Step 6: Verify no source code
    if not verify_no_source_code():
        print("\nWARNING: Build completed but source code protection incomplete")

    # Step 7: Print summary
    print_summary()


if __name__ == '__main__':
    main()
