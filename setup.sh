#!/bin/bash
# Memopt Quick Setup Script
# Builds and runs Memopt in Docker

set -e

echo "========================================"
echo "Memopt Docker Quick Setup"
echo "========================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found. Please install Docker first.${NC}"
    exit 1
fi

# Check NVIDIA Docker
if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo -e "${RED}❌ NVIDIA Docker runtime not working.${NC}"
    echo "Please install nvidia-docker2:"
    echo "  sudo apt-get install nvidia-docker2"
    echo "  sudo systemctl restart docker"
    exit 1
fi

echo -e "${GREEN}✓ Docker and NVIDIA runtime detected${NC}"

# Create directories
mkdir -p models outputs examples

# Build image
echo ""
echo "Building Docker image..."
docker build -t Memopt:latest .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Image built successfully${NC}"
else
    echo -e "${RED}❌ Build failed${NC}"
    exit 1
fi

# Run quick test
echo ""
echo "Running quick test..."
docker run --rm --gpus all Memopt:latest \
    python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Prompt for benchmark
echo ""
echo -e "${YELLOW}Run benchmark? (y/n)${NC}"
read -r response

if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo ""
    echo "Running benchmark (this may take a few minutes)..."
    docker run --gpus all \
        -v $(pwd)/outputs:/app/outputs \
        Memopt:latest \
        python3 benchmark.py --model gpt2 --mode both --num-prompts 3
    
    echo ""
    echo -e "${GREEN}✓ Benchmark complete!${NC}"
    echo "Results saved to: outputs/benchmark_results.json"
fi

# Summary
echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Quick commands:"
echo ""
echo "  # Run benchmark"
echo "  docker run --gpus all -v \$(pwd)/outputs:/app/outputs Memopt:latest \\"
echo "    python3 benchmark.py --model gpt2-xl --mode both"
echo ""
echo "  # Start API server"
echo "  docker-compose up -d"
echo ""
echo "  # Interactive shell"
echo "  docker run --gpus all -it Memopt:latest bash"
echo ""
echo "For more info, see DOCKER_README.md"
echo ""