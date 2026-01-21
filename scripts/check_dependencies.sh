#!/bin/bash
# Check if all dependencies for production serving are installed

echo "======================================="
echo "Checking Dependencies"
echo "======================================="

EXIT_CODE=0

# Check Python packages
echo "Checking Python packages..."

REQUIRED_PACKAGES="fastapi uvicorn redis httpx pydantic numpy"

for pkg in $REQUIRED_PACKAGES; do
    if python3 -c "import $pkg" 2>/dev/null; then
        echo "  ✓ $pkg"
    else
        echo "  ✗ $pkg (missing)"
        EXIT_CODE=1
    fi
done

# Check Redis
echo ""
echo "Checking Redis..."
if redis-cli ping > /dev/null 2>&1; then
    echo "  ✓ Redis running"
else
    echo "  ✗ Redis not running"
    echo "    Start with: docker run -d -p 6379:6379 redis:latest"
    EXIT_CODE=1
fi

# Check nvidia-smi
echo ""
echo "Checking GPU tools..."
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    echo "  ✓ nvidia-smi found ($NUM_GPUS GPUs available)"
else
    echo "  ✗ nvidia-smi not found"
    EXIT_CODE=1
fi

# Check jq (for JSON parsing)
echo ""
echo "Checking utilities..."
if command -v jq &> /dev/null; then
    echo "  ✓ jq"
else
    echo "  ✗ jq (optional, for better output)"
    echo "    Install with: sudo apt-get install jq"
fi

if command -v bc &> /dev/null; then
    echo "  ✓ bc"
else
    echo "  ✗ bc (needed for calculations)"
    echo "    Install with: sudo apt-get install bc"
    EXIT_CODE=1
fi

echo ""
echo "======================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ All dependencies satisfied"
    echo ""
    echo "You can now run:"
    echo "  ./scripts/benchmark_production.sh 2 100 50"
else
    echo "✗ Missing dependencies"
    echo ""
    echo "Install missing packages:"
    echo "  pip install fastapi uvicorn redis httpx pydantic numpy"
fi
echo "======================================="

exit $EXIT_CODE
