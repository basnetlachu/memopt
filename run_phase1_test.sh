#!/bin/bash
# Phase 1 GPU Test Runner for memopt
# Run this script on the GPU server: bash run_phase1_test.sh

set -e

echo "========================================"
echo "MEMOPT Phase 1 GPU Test"
echo "========================================"

# Install the wheel
echo "Installing memopt..."
pip3 install --break-system-packages --upgrade memopt-0.4.0-py3-none-any.whl

# Run the test
echo ""
echo "Running Phase 1 GPU tests..."
echo ""
python3 test_phase1_gpu.py

echo ""
echo "========================================"
echo "Test complete!"
echo "========================================"
