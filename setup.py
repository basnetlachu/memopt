"""
Cython Build Setup for memopt

This setup.py compiles all Python modules to C extensions (.so files on Linux,
.pyd files on Windows) to protect source code during deployment.

Usage:
    # Build compiled package (no source code)
    python setup.py build_ext --inplace

    # Build wheel with compiled extensions only
    python build_compiled.py
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.extension import Extension

# Check if Cython is available
try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False
    cythonize = None
    build_ext = None

# Directories to exclude from compilation
EXCLUDE_DIRS = {
    '__pycache__', '.git', '.venv', 'venv', 'env',
    'build', 'dist', '*.egg-info', 'tests', 'examples',
    'dashboard', 'gpt2xl_test', 'qwen_test'
}

# Files to exclude from compilation (keep as .py)
EXCLUDE_FILES = {
    'setup.py', 'build_compiled.py', 'conftest.py'
}


def get_python_files(base_path: str = 'memopt') -> list:
    """Get all Python files to compile."""
    py_files = []
    base = Path(base_path)

    for py_file in base.rglob('*.py'):
        # Skip excluded directories
        if any(excluded in py_file.parts for excluded in EXCLUDE_DIRS):
            continue

        # Skip excluded files
        if py_file.name in EXCLUDE_FILES:
            continue

        # Skip __pycache__
        if '__pycache__' in str(py_file):
            continue

        py_files.append(str(py_file))

    return py_files


def make_extensions(py_files: list) -> list:
    """Create Extension objects for each Python file."""
    extensions = []

    for py_file in py_files:
        # Convert path to module name
        # memopt/profiler/phase1_profiler.py -> memopt.profiler.phase1_profiler
        module_name = py_file.replace('/', '.').replace('\\', '.').replace('.py', '')

        ext = Extension(
            name=module_name,
            sources=[py_file],
            extra_compile_args=['-O3'],  # Maximum optimization
        )
        extensions.append(ext)

    return extensions


if USE_CYTHON:
    py_files = get_python_files()
    extensions = make_extensions(py_files)
    ext_modules = cythonize(
        extensions,
        compiler_directives={
            'language_level': '3',
            'boundscheck': False,
            'wraparound': False,
            'embedsignature': True,
        },
        nthreads=os.cpu_count() or 4,
    )
    cmdclass = {'build_ext': build_ext}
else:
    ext_modules = []
    cmdclass = {}
    print("WARNING: Cython not found. Building pure Python package.")
    print("Install Cython for compiled distribution: pip install cython")


setup(
    name='memopt',
    version='1.0.0',
    description='GPU Memory Optimization Platform - Compiled Distribution',
    author='MemOpt Team',
    author_email='hello@memopt.ai',
    packages=find_packages(exclude=['tests', 'tests.*', 'examples', 'dashboard', 'build']),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.24.0',
    ],
    extras_require={
        'full': [
            'transformers>=4.30.0',
            'accelerate>=0.20.0',
            'flash-attn>=2.0.0',
        ],
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
        'Programming Language :: Cython',
        'Operating System :: POSIX :: Linux',
    ],
    zip_safe=False,
)
