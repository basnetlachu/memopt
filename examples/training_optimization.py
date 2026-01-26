#!/usr/bin/env python3
"""
Training Optimization Example

Demonstrates memory optimization for LLM fine-tuning.
Measures real GPU memory reduction during training.

Usage:
    python examples/training_optimization.py

Requirements:
    - PyTorch with CUDA
    - transformers
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from memopt.optimization import MemoryCoalescer, CoalescingConfig
from memopt.measurement import BandwidthTracker


class SimpleTextDataset(Dataset):
    """Simple text dataset for demonstration."""

    def __init__(self, tokenizer, num_samples: int = 100, max_length: int = 128):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_length = max_length

        # Sample texts for training
        self.texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming the world.",
            "Artificial intelligence enables new capabilities.",
            "Deep neural networks learn complex patterns.",
            "Natural language processing understands text.",
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        text = self.texts[idx % len(self.texts)]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': encoding['input_ids'].squeeze(0)
        }


def train_step(model, batch, optimizer):
    """Single training step."""
    outputs = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        labels=batch['labels']
    )
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()


def run_training_baseline(model, dataloader, optimizer, num_steps: int = 10):
    """Run training without optimization."""
    model.train()
    losses = []

    for step, batch in enumerate(dataloader):
        if step >= num_steps:
            break

        if torch.cuda.is_available():
            batch = {k: v.cuda() for k, v in batch.items()}

        loss = train_step(model, batch, optimizer)
        losses.append(loss)

    return losses


def run_training_optimized(model, dataloader, optimizer, coalescer, num_steps: int = 10):
    """Run training with memory coalescing."""
    model.train()
    losses = []

    for step, batch in enumerate(dataloader):
        if step >= num_steps:
            break

        if torch.cuda.is_available():
            batch = {k: v.cuda() for k, v in batch.items()}

        loss = train_step(model, batch, optimizer)
        losses.append(loss)

    return losses


def main():
    print("=" * 70)
    print("TRAINING OPTIMIZATION DEMO")
    print("=" * 70)
    print()

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    if device == 'cpu':
        print("⚠️  Warning: Running on CPU. For real memory measurement, use GPU.")
        print()

    # Load model
    print("Loading GPT-2...")
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        model = model.cuda()

    print(f"Model loaded: GPT-2 ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")
    print()

    # Create dataset and dataloader
    dataset = SimpleTextDataset(tokenizer, num_samples=100, max_length=128)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    num_steps = 10

    print(f"Training config:")
    print(f"  Batch size: 4")
    print(f"  Sequence length: 128")
    print(f"  Training steps: {num_steps}")
    print()

    # Initialize tracking
    tracker = BandwidthTracker()

    # Baseline training
    print("[1/2] Measuring BASELINE training...")
    model_baseline = AutoModelForCausalLM.from_pretrained('gpt2')
    if torch.cuda.is_available():
        model_baseline = model_baseline.cuda()

    optimizer_baseline = torch.optim.AdamW(model_baseline.parameters(), lr=1e-5)

    with tracker.measure("baseline"):
        baseline_losses = run_training_baseline(
            model_baseline, dataloader, optimizer_baseline, num_steps
        )

    baseline = tracker.get_measurement("baseline")
    print(f"      Peak memory: {baseline.peak_memory_gb:.3f} GB")
    print(f"      Duration: {baseline.duration_ms:.1f} ms")
    print(f"      Final loss: {baseline_losses[-1]:.4f}")
    print()

    # Clean up baseline model
    del model_baseline, optimizer_baseline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Optimized training
    print("[2/2] Measuring OPTIMIZED training...")
    model_optimized = AutoModelForCausalLM.from_pretrained('gpt2')
    if torch.cuda.is_available():
        model_optimized = model_optimized.cuda()

    optimizer_optimized = torch.optim.AdamW(model_optimized.parameters(), lr=1e-5)

    # Configure coalescer for training
    config = CoalescingConfig(
        cache_size_mb=128,
        optimize_attention=True,
        optimize_mlp=True,
        optimize_checkpointing=True,
        enable_profiling=True
    )
    coalescer = MemoryCoalescer(model_optimized, mode='training', config=config)
    coalescer.enable()

    with tracker.measure("optimized"):
        optimized_losses = run_training_optimized(
            model_optimized, dataloader, optimizer_optimized, coalescer, num_steps
        )

    optimized = tracker.get_measurement("optimized")
    print(f"      Peak memory: {optimized.peak_memory_gb:.3f} GB")
    print(f"      Duration: {optimized.duration_ms:.1f} ms")
    print(f"      Final loss: {optimized_losses[-1]:.4f}")
    print()

    # Get coalescing stats
    coalescer_stats = coalescer.get_stats()
    coalescer.disable()

    # Generate comparison report
    report = tracker.compare("baseline", "optimized")

    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    print("Memory Measurements:")
    print(f"  Baseline peak:   {report.baseline.peak_memory_gb:.3f} GB")
    print(f"  Optimized peak:  {report.optimized.peak_memory_gb:.3f} GB")
    print(f"  Reduction:       {report.memory_reduction_pct:.1f}%")
    print()

    print("Timing:")
    print(f"  Baseline:   {report.baseline.duration_ms:.1f} ms")
    print(f"  Optimized:  {report.optimized.duration_ms:.1f} ms")
    print(f"  Speedup:    {report.speedup_ratio:.2f}x")
    print()

    print("Training Loss:")
    print(f"  Baseline final:   {baseline_losses[-1]:.4f}")
    print(f"  Optimized final:  {optimized_losses[-1]:.4f}")
    loss_diff = abs(baseline_losses[-1] - optimized_losses[-1])
    print(f"  Difference:       {loss_diff:.6f} (should be ~0)")
    print()

    print("Coalescing Statistics:")
    print(f"  Layers optimized:   {coalescer_stats.layers_optimized}")
    print(f"  Forward passes:     {coalescer_stats.forward_passes}")
    print(f"  Backward passes:    {coalescer_stats.backward_passes}")
    print(f"  Total accesses:     {coalescer_stats.total_accesses:,}")
    print(f"  Cache hit rate:     {coalescer_stats.hit_rate:.1f}%")
    print()

    print("=" * 70)

    # Summary
    if report.memory_reduction_pct >= 15:
        print(f"✅ SUCCESS: {report.memory_reduction_pct:.1f}% memory reduction")
        print("   Allows larger batch sizes or sequence lengths")
    elif report.memory_reduction_pct >= 5:
        print(f"⚠️  PARTIAL: {report.memory_reduction_pct:.1f}% memory reduction")
        print("   Some improvement in memory efficiency")
    else:
        print(f"❌ MINIMAL: {report.memory_reduction_pct:.1f}% memory reduction")
        print("   Training workload may not benefit from coalescing")

    print("=" * 70)

    return report


if __name__ == '__main__':
    main()
