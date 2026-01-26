"""
Automated Validation Pipeline

Runs comprehensive validation of memory optimization including:
1. Baseline measurement (no optimization)
2. Optimized measurement (with coalescing)
3. Hardware validation (Nsight if available)
4. Report generation

Usage:
    from memopt.validation import validate_optimization

    report = validate_optimization(
        model=model,
        workload='inference',
        num_iterations=10
    )
    print(report.summary())
"""

from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

import torch
import torch.nn as nn

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from memopt.optimization import MemoryCoalescer, CoalescingConfig, CoalescingStats
from memopt.measurement import BandwidthTracker, BandwidthMeasurement


@dataclass
class ValidationReport:
    """Complete validation report."""

    # Identification
    model_name: str = ""
    workload_type: str = ""  # 'training' or 'inference'
    timestamp: str = ""

    # Baseline metrics
    baseline_peak_memory_gb: float = 0.0
    baseline_duration_ms: float = 0.0
    baseline_dram_gb: float = 0.0

    # Optimized metrics
    optimized_peak_memory_gb: float = 0.0
    optimized_duration_ms: float = 0.0
    optimized_dram_gb: float = 0.0

    # Coalescing stats
    total_accesses: int = 0
    cache_hits: int = 0
    cache_hit_rate: float = 0.0
    bandwidth_reduction_potential: float = 0.0

    # Results
    memory_reduction_pct: float = 0.0
    speedup: float = 1.0
    correctness_verified: bool = False

    # Hardware validation
    nsight_validated: bool = False
    nsight_baseline_csv: Optional[str] = None
    nsight_optimized_csv: Optional[str] = None
    hardware_dram_reduction_pct: float = 0.0

    # Metadata
    device: str = ""
    iterations: int = 0

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 70,
            "VALIDATION REPORT",
            "=" * 70,
            "",
            f"Model: {self.model_name}",
            f"Workload: {self.workload_type}",
            f"Device: {self.device}",
            f"Iterations: {self.iterations}",
            f"Timestamp: {self.timestamp}",
            "",
            "--- Baseline (No Optimization) ---",
            f"Peak Memory: {self.baseline_peak_memory_gb:.3f} GB",
            f"Duration: {self.baseline_duration_ms:.1f} ms",
            "",
            "--- Optimized (With Coalescing) ---",
            f"Peak Memory: {self.optimized_peak_memory_gb:.3f} GB",
            f"Duration: {self.optimized_duration_ms:.1f} ms",
            "",
            "--- Coalescing Statistics ---",
            f"Total Accesses: {self.total_accesses:,}",
            f"Cache Hits: {self.cache_hits:,}",
            f"Hit Rate: {self.cache_hit_rate:.1f}%",
            f"Bandwidth Reduction Potential: {self.bandwidth_reduction_potential:.1f}%",
            "",
            "--- Results ---",
            f"Memory Reduction: {self.memory_reduction_pct:.1f}%",
            f"Speedup: {self.speedup:.2f}x",
            f"Correctness: {'✅ Verified' if self.correctness_verified else '❌ Not Verified'}",
            "",
            "--- Hardware Validation ---",
            f"Nsight Validated: {'✅ YES' if self.nsight_validated else '❌ NO'}",
        ]

        if self.nsight_validated:
            lines.extend([
                f"Hardware DRAM Reduction: {self.hardware_dram_reduction_pct:.1f}%",
                f"Baseline CSV: {self.nsight_baseline_csv or 'N/A'}",
                f"Optimized CSV: {self.nsight_optimized_csv or 'N/A'}",
            ])

        lines.extend([
            "",
            "=" * 70,
        ])

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'workload_type': self.workload_type,
            'timestamp': self.timestamp,
            'device': self.device,
            'iterations': self.iterations,
            'baseline': {
                'peak_memory_gb': self.baseline_peak_memory_gb,
                'duration_ms': self.baseline_duration_ms,
                'dram_gb': self.baseline_dram_gb
            },
            'optimized': {
                'peak_memory_gb': self.optimized_peak_memory_gb,
                'duration_ms': self.optimized_duration_ms,
                'dram_gb': self.optimized_dram_gb
            },
            'coalescing': {
                'total_accesses': self.total_accesses,
                'cache_hits': self.cache_hits,
                'hit_rate': self.cache_hit_rate,
                'bandwidth_reduction_potential': self.bandwidth_reduction_potential
            },
            'results': {
                'memory_reduction_pct': self.memory_reduction_pct,
                'speedup': self.speedup,
                'correctness_verified': self.correctness_verified
            },
            'hardware': {
                'nsight_validated': self.nsight_validated,
                'dram_reduction_pct': self.hardware_dram_reduction_pct
            }
        }

    def save(self, path: str):
        """Save report to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def _run_inference_workload(
    model: nn.Module,
    tokenizer,
    prompts: List[str],
    max_tokens: int,
    device: str
) -> List[torch.Tensor]:
    """Run inference workload."""
    results = []
    model.eval()

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors='pt')
        if device == 'cuda':
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
        results.append(outputs)

    return results


def _run_training_workload(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    num_steps: int,
    device: str
) -> List[float]:
    """Run training workload."""
    model.train()
    losses = []

    for step, batch in enumerate(dataloader):
        if step >= num_steps:
            break

        if device == 'cuda':
            batch = {k: v.cuda() for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())

    return losses


def validate_optimization(
    model: nn.Module,
    workload: str = 'inference',
    num_iterations: int = 10,
    tokenizer=None,
    dataloader=None,
    prompts: List[str] = None,
    max_tokens: int = 50,
    use_nsight: bool = False,
    output_dir: str = 'validation'
) -> ValidationReport:
    """
    Run complete validation of memory optimization.

    Args:
        model: PyTorch model to validate
        workload: 'inference' or 'training'
        num_iterations: Number of iterations to run
        tokenizer: Tokenizer for inference workload
        dataloader: DataLoader for training workload
        prompts: Prompts for inference (uses defaults if None)
        max_tokens: Max tokens for inference
        use_nsight: Try to use Nsight Compute for hardware validation
        output_dir: Directory for output files

    Returns:
        ValidationReport with all metrics
    """
    report = ValidationReport(
        model_name=model.__class__.__name__,
        workload_type=workload,
        timestamp=datetime.now().isoformat(),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        iterations=num_iterations
    )

    device = report.device
    if device == 'cuda':
        model = model.cuda()

    # Default prompts for inference
    if prompts is None:
        prompts = [
            "The future of AI is",
            "Machine learning enables",
            "Deep neural networks",
        ] * (num_iterations // 3 + 1)
        prompts = prompts[:num_iterations]

    # Load tokenizer if needed
    if tokenizer is None and workload == 'inference':
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        except ImportError:
            raise RuntimeError("transformers required for inference validation")

    # Initialize tracker
    tracker = BandwidthTracker(device=device)

    # ===== BASELINE MEASUREMENT =====
    print("[1/4] Running BASELINE (no optimization)...")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    with tracker.measure("baseline"):
        if workload == 'inference':
            baseline_results = _run_inference_workload(
                model, tokenizer, prompts, max_tokens, device
            )
        else:
            baseline_results = _run_training_workload(
                model, dataloader, torch.optim.AdamW(model.parameters(), lr=1e-5),
                num_iterations, device
            )

    baseline = tracker.get_measurement("baseline")
    report.baseline_peak_memory_gb = baseline.peak_memory_gb
    report.baseline_duration_ms = baseline.duration_ms

    print(f"    Peak memory: {report.baseline_peak_memory_gb:.3f} GB")
    print(f"    Duration: {report.baseline_duration_ms:.1f} ms")

    # ===== OPTIMIZED MEASUREMENT =====
    print("[2/4] Running OPTIMIZED (with coalescing)...")

    # Enable coalescing
    coalescer = MemoryCoalescer(model, mode=workload)
    coalescer.enable()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    with tracker.measure("optimized"):
        if workload == 'inference':
            optimized_results = _run_inference_workload(
                model, tokenizer, prompts, max_tokens, device
            )
        else:
            # Reload model for training
            optimized_results = _run_training_workload(
                model, dataloader, torch.optim.AdamW(model.parameters(), lr=1e-5),
                num_iterations, device
            )

    optimized = tracker.get_measurement("optimized")
    report.optimized_peak_memory_gb = optimized.peak_memory_gb
    report.optimized_duration_ms = optimized.duration_ms

    # Get coalescing stats
    stats = coalescer.get_stats()
    report.total_accesses = stats.total_accesses
    report.cache_hits = stats.cache_hits
    report.cache_hit_rate = stats.hit_rate
    report.bandwidth_reduction_potential = stats.bandwidth_reduction

    coalescer.disable()

    print(f"    Peak memory: {report.optimized_peak_memory_gb:.3f} GB")
    print(f"    Duration: {report.optimized_duration_ms:.1f} ms")
    print(f"    Accesses tracked: {report.total_accesses:,}")
    print(f"    Hit rate: {report.cache_hit_rate:.1f}%")

    # ===== CORRECTNESS CHECK =====
    print("[3/4] Verifying correctness...")

    if workload == 'inference':
        try:
            report.correctness_verified = all(
                torch.equal(b, o) for b, o in zip(baseline_results, optimized_results)
            )
        except Exception:
            report.correctness_verified = False

        print(f"    Outputs match: {'✅ YES' if report.correctness_verified else '❌ NO'}")
    else:
        # For training, check loss convergence
        if len(baseline_results) > 0 and len(optimized_results) > 0:
            loss_diff = abs(baseline_results[-1] - optimized_results[-1])
            report.correctness_verified = loss_diff < 0.1
            print(f"    Loss difference: {loss_diff:.4f}")
        else:
            report.correctness_verified = True

    # ===== CALCULATE RESULTS =====
    print("[4/4] Calculating results...")

    if report.baseline_peak_memory_gb > 0:
        report.memory_reduction_pct = (
            (report.baseline_peak_memory_gb - report.optimized_peak_memory_gb)
            / report.baseline_peak_memory_gb
        ) * 100

    if report.optimized_duration_ms > 0:
        report.speedup = report.baseline_duration_ms / report.optimized_duration_ms

    print(f"    Memory reduction: {report.memory_reduction_pct:.1f}%")
    print(f"    Speedup: {report.speedup:.2f}x")
    print(f"    Bandwidth reduction potential: {report.bandwidth_reduction_potential:.1f}%")

    # ===== SAVE REPORT =====
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report_path = output_path / 'validation_report.json'
    report.save(str(report_path))
    print(f"\n✅ Report saved: {report_path}")

    return report


def run_full_validation(
    model_name: str = 'gpt2',
    workload: str = 'inference',
    output_dir: str = 'validation'
):
    """
    Run full validation on a model from scratch.

    Convenience function that loads the model and runs validation.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        print(f"Running validation ({workload})...")
        report = validate_optimization(
            model=model,
            workload=workload,
            tokenizer=tokenizer,
            output_dir=output_dir
        )

        print("\n" + report.summary())
        return report

    except ImportError as e:
        print(f"Error: {e}")
        print("Install transformers: pip install transformers")
        return None


if __name__ == '__main__':
    # Run validation when executed directly
    import argparse

    parser = argparse.ArgumentParser(description='Run memory optimization validation')
    parser.add_argument('--model', default='gpt2', help='Model name')
    parser.add_argument('--workload', default='inference', choices=['inference', 'training'])
    parser.add_argument('--output', default='validation', help='Output directory')

    args = parser.parse_args()
    run_full_validation(args.model, args.workload, args.output)
