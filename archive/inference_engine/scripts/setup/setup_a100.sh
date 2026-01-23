#!/bin/bash
# Setup script for A100 - run this on the GPU

set -e

echo "================================================================"
echo "MEMOPT A100 SETUP - Getting 60x Speedup"
echo "================================================================"
echo ""

# Step 1: Check GPU
echo "Step 1: Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Step 2: Check Python and CUDA
echo "Step 2: Checking Python and CUDA..."
python3 --version
echo "CUDA: $(python3 -c 'import torch; print(torch.version.cuda)')"
echo ""

# Step 3: Install/Update MemOpt
echo "Step 3: Installing MemOpt..."
cd /root/memopt
pip3 install -e . --quiet
echo "✓ MemOpt installed"
echo ""

# Step 4: Install Flash Attention (critical for 60x speedup)
echo "Step 4: Installing Flash Attention (takes 5-10 minutes)..."
echo "This is CRITICAL for 40-60x speedup"
echo ""

if python3 -c "import flash_attn" 2>/dev/null; then
    echo "✓ Flash Attention already installed"
else
    echo "Installing flash-attn..."
    pip3 install flash-attn --no-build-isolation
    if [ $? -eq 0 ]; then
        echo "✓ Flash Attention installed successfully"
    else
        echo "⚠ Flash Attention install failed"
        echo "Will use GPT-NeoX for 15-20x speedup instead"
    fi
fi
echo ""

# Step 5: Quick test
echo "================================================================"
echo "Step 5: Quick Test"
echo "================================================================"
echo ""

if python3 -c "import flash_attn" 2>/dev/null; then
    echo "Testing Llama-2-7b with Flash Attention (40-60x expected)..."
    echo ""
    python3 benchmark.py \
        --model meta-llama/Llama-2-7b-hf \
        --max-tokens 1000 \
        --num-prompts 2 \
        --optimization-level ultra \
        --max-kv-blocks 3000
else
    echo "Testing GPT-NeoX-20b without Flash Attention (15-20x expected)..."
    echo ""
    python3 benchmark.py \
        --model EleutherAI/gpt-neox-20b \
        --max-tokens 1000 \
        --num-prompts 2 \
        --optimization-level ultra \
        --max-kv-blocks 2000
fi

echo ""
echo "================================================================"
echo "SETUP COMPLETE!"
echo "================================================================"
echo ""
echo "For full benchmark with 10,000 tokens:"
if python3 -c "import flash_attn" 2>/dev/null; then
    echo "  python3 benchmark.py --model meta-llama/Llama-2-7b-hf \\"
    echo "    --max-tokens 10000 --num-prompts 2 \\"
    echo "    --optimization-level ultra --max-kv-blocks 3000"
else
    echo "  python3 benchmark.py --model EleutherAI/gpt-neox-20b \\"
    echo "    --max-tokens 5000 --num-prompts 2 \\"
    echo "    --optimization-level ultra --max-kv-blocks 2000"
fi
echo ""
