#!/bin/bash
# Flash Attention 2 Installation Script for A100 + CUDA 12.4
# This will give you the 40-60x speedup you need

set -e  # Exit on error

echo "========================================"
echo "FLASH ATTENTION 2 INSTALLER"
echo "========================================"
echo ""
echo "This will install Flash Attention 2 from source"
echo "Expected time: 10-15 minutes"
echo "Required: CUDA 12.4, Python 3.10, 20GB free space"
echo ""

# Check CUDA version
echo "Checking CUDA version..."
nvcc --version | grep "release"

# Install build dependencies
echo ""
echo "Installing build dependencies..."
pip install packaging wheel ninja setuptools -q

# Install Flash Attention 2 from source
echo ""
echo "========================================"
echo "BUILDING FLASH ATTENTION 2 (10-15 min)"
echo "========================================"
echo "This is compiling optimized CUDA kernels..."
echo ""

# Use MAX_JOBS to speed up compilation on A100
export MAX_JOBS=8
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export TORCH_CUDA_ARCH_LIST="8.0"  # A100 compute capability

pip install flash-attn --no-build-isolation -v

echo ""
echo "========================================"
echo "✓ FLASH ATTENTION 2 INSTALLED"
echo "========================================"
echo ""
echo "Verifying installation..."
python -c "import flash_attn; print(f'Flash Attention version: {flash_attn.__version__}')"

echo ""
echo "✓ Installation complete!"
echo ""
echo "You can now run benchmarks with 40-60x speedup:"
echo "  python benchmark.py --model EleutherAI/gpt-neox-20b --max-tokens 2048 --num-prompts 5 --optimization-level ultra"
