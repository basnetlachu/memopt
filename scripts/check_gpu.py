#!/usr/bin/env python3
"""
GPU Diagnostic Script
Check if PyTorch can actually use your NVIDIA GPU
"""

import torch
import sys

print("="*70)
print("GPU DIAGNOSTIC")
print("="*70)

# Check PyTorch version
print(f"\nPyTorch version: {torch.__version__}")
print(f"Python version: {sys.version}")

# Check CUDA
print(f"\n1. CUDA availability:")
print(f"   torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"   torch.version.cuda: {torch.version.cuda}")

if not torch.cuda.is_available():
    print("\n❌ CUDA NOT AVAILABLE")
    print("\nPossible reasons:")
    print("1. PyTorch CPU-only version installed")
    print("2. CUDA drivers not installed")
    print("3. CUDA version mismatch")
    print("\nTo fix:")
    print("Visit: https://pytorch.org/get-started/locally/")
    print("Reinstall PyTorch with CUDA support:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    sys.exit(1)

# Check GPU details
print(f"\n2. GPU devices:")
print(f"   Device count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"   Device {i}: {torch.cuda.get_device_name(i)}")
    props = torch.cuda.get_device_properties(i)
    print(f"      Total memory: {props.total_memory / 1024**3:.1f} GB")
    print(f"      CUDA capability: {props.major}.{props.minor}")

# Test actual GPU usage
print(f"\n3. Testing GPU tensor creation:")
try:
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    z = torch.matmul(x, y)
    torch.cuda.synchronize()
    print(f"   ✅ GPU tensor operations work!")
    print(f"   Device: {z.device}")
except Exception as e:
    print(f"   ❌ GPU operations failed: {e}")
    sys.exit(1)

# Test model loading
print(f"\n4. Testing model loading on GPU:")
try:
    from transformers import AutoModelForCausalLM
    print(f"   Loading gpt2 (small model) on GPU...")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    print(f"   ✅ Model loaded on GPU successfully!")
    print(f"   Model device: {next(model.parameters()).device}")

    # Check memory usage
    torch.cuda.synchronize()
    mem_allocated = torch.cuda.memory_allocated() / 1024**3
    print(f"   GPU memory used: {mem_allocated:.2f} GB")

except Exception as e:
    print(f"   ❌ Model loading failed: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("✅ ALL GPU CHECKS PASSED!")
print("Your GPU is working correctly with PyTorch.")
print("="*70)
