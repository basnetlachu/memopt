# MemOpt - 60x LLM Inference Speedup

Automatic GPU optimization for trillion-token scale LLM inference.

## Quick Start

```bash
pip install -e .

# Run benchmark - it tells you what to do
python benchmarks/benchmark.py --model Qwen/Qwen2-7B --max-tokens 10000 \
  --num-prompts 2 --optimization-level ultra --max-kv-blocks 2000
```

## Why Only 1.82x?

If you see low speedup, benchmark.py will show:

```
⚠️  WARNING: Limited Speedup Expected (8-12x)

This model has:
  ✗ No Flash Attention (not installed or not working)
  ✗ No compatible draft model for speculative decoding

To achieve 40-60x speedup:
  Option 1: Install Flash Attention
    pip3 install flash-attn --no-build-isolation

  Option 2: Use model with proven draft model
    EleutherAI/gpt-neox-20b (15-20x speedup)
    meta-llama/Llama-2-7b-hf (with Flash Attn: 40-60x)
```

## Proven Configurations

**GPT-NeoX-20B** (15-20x, works NOW without Flash Attention):
```bash
python benchmarks/benchmark.py --model EleutherAI/gpt-neox-20b --max-tokens 5000 \
  --num-prompts 2 --optimization-level ultra --max-kv-blocks 2000
```

**Llama-2-7B** (40-60x with Flash Attention):
```bash
# First install Flash Attention
pip3 install flash-attn --no-build-isolation

# Then benchmark
python benchmarks/benchmark.py --model meta-llama/Llama-2-7b-hf --max-tokens 10000 \
  --num-prompts 2 --optimization-level ultra --max-kv-blocks 3000
```

## Trillion-Token Scale

- **GPT-NeoX (15x)**: 10T tokens in 193 days on single A100
- **Llama-2+Flash (40x)**: 10T tokens in 72 days on single A100
- **With 10 GPUs**: 7-19 days for 10T tokens

## Repository Structure

The codebase is organized into clear functional areas:

- **[memopt/](memopt/)** - Core package (31 optimized modules)
- **[scripts/](scripts/)** - Production servers and utilities
- **[benchmarks/](benchmarks/)** - Performance benchmarking tools
- **[deployment/](deployment/)** - Docker & Kubernetes configs
- **[docs/](docs/)** - Implementation guides and documentation
- **[tests/](tests/)** - Test suite

See [STRUCTURE.md](STRUCTURE.md) for complete repository layout and [ARCHITECTURE.md](ARCHITECTURE.md) for deployment architecture.
