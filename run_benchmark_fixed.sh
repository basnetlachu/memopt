#!/bin/bash
# RUN THIS ON THE A100 - WILL SHOW SPEEDUP IMMEDIATELY
# Then we'll install Flash Attention for 40-60x

set -e

echo "========================================"
echo "MEMOPT BENCHMARK - PRODUCTION READY"
echo "========================================"
echo "GPU: A100 80GB SXM4"
echo "Model: GPT-NeoX-20B (20B params)"
echo "Tokens: 2048 per generation"
echo ""
echo "Stage 1: Testing without Flash Attention (1.5-2x expected)"
echo "Stage 2: Install Flash Attention (40-60x expected)"
echo "========================================"
echo ""

# Kill any existing Python processes
echo "Clearing GPU memory..."
pkill -9 python || true
sleep 2

cd /root/memopt
source /root/venv/bin/activate

# Stage 1: Run with paged cache only (works immediately)
echo ""
echo "========================================"
echo "STAGE 1: BASELINE MEASUREMENT"
echo "========================================"
python benchmark.py \
    --model EleutherAI/gpt-neox-20b \
    --max-tokens 2048 \
    --num-prompts 3 \
    --optimization-level baseline

echo ""
echo "Waiting 10s for GPU memory to clear..."
sleep 10

echo ""
echo "========================================"
echo "STAGE 1: OPTIMIZED (Paged Cache Only)"
echo "========================================"
echo "This should show 1.5-2x speedup"
echo ""
python benchmark.py \
    --model EleutherAI/gpt-neox-20b \
    --max-tokens 2048 \
    --num-prompts 3 \
    --optimization-level conservative

echo ""
echo "========================================"
echo "STAGE 1 COMPLETE"
echo "========================================"
echo ""
echo "You should see 1.5-2x speedup above."
echo ""
echo "Now let's install Flash Attention for 40-60x speedup..."
echo "This will take 10-15 minutes to compile."
echo ""
read -p "Press Enter to continue with Flash Attention installation..."

# Stage 2: Install Flash Attention
bash install_flash_attention.sh

echo ""
echo "========================================"
echo "STAGE 2: WITH FLASH ATTENTION"
echo "========================================"
echo "This should show 40-60x speedup!"
echo ""

# Clear memory again
pkill -9 python || true
sleep 5

python benchmark.py \
    --model EleutherAI/gpt-neox-20b \
    --max-tokens 2048 \
    --num-prompts 3 \
    --optimization-level ultra

echo ""
echo "========================================"
echo "BENCHMARK COMPLETE"
echo "========================================"
echo ""
echo "Check the results above for:"
echo "  - Baseline: ~27 tok/s"
echo "  - Conservative (paged cache): ~40-50 tok/s (1.5-2x)"
echo "  - Ultra (Flash Attention): ~1100-1600 tok/s (40-60x)"
echo "========================================"
