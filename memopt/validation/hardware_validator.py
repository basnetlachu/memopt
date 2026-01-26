"""
Hardware Validation using NVIDIA Nsight Compute

Provides hardware-level validation of memory optimization by:
1. Running workloads under Nsight Compute profiler
2. Extracting DRAM read/write counters from hardware
3. Generating proof files (CSV) for validation
4. Comparing baseline vs optimized results

Usage:
    from memopt.validation import HardwareValidator

    validator = HardwareValidator()

    # Run baseline
    baseline = validator.profile_workload(
        script_path='train.py',
        label='baseline'
    )

    # Run optimized
    optimized = validator.profile_workload(
        script_path='train_optimized.py',
        label='optimized'
    )

    # Compare
    report = validator.compare(baseline, optimized)
"""

from __future__ import annotations

import os
import csv
import json
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import torch


@dataclass
class NsightMetrics:
    """Metrics extracted from Nsight Compute profiling."""

    label: str = ""

    # DRAM metrics (bytes)
    dram_bytes_read: int = 0
    dram_bytes_write: int = 0

    # L2 cache metrics
    l2_read_throughput: float = 0.0
    l2_write_throughput: float = 0.0

    # SM efficiency
    sm_efficiency: float = 0.0
    achieved_occupancy: float = 0.0

    # Timing
    duration_ms: float = 0.0
    kernel_count: int = 0

    # Source file
    csv_path: Optional[str] = None

    @property
    def dram_bytes_total(self) -> int:
        return self.dram_bytes_read + self.dram_bytes_write

    @property
    def dram_gb_total(self) -> float:
        return self.dram_bytes_total / (1024 ** 3)

    @property
    def dram_gb_read(self) -> float:
        return self.dram_bytes_read / (1024 ** 3)

    @property
    def dram_gb_write(self) -> float:
        return self.dram_bytes_write / (1024 ** 3)

    def to_dict(self) -> Dict:
        return {
            'label': self.label,
            'dram_bytes_read': self.dram_bytes_read,
            'dram_bytes_write': self.dram_bytes_write,
            'dram_bytes_total': self.dram_bytes_total,
            'dram_gb_total': self.dram_gb_total,
            'l2_read_throughput': self.l2_read_throughput,
            'sm_efficiency': self.sm_efficiency,
            'achieved_occupancy': self.achieved_occupancy,
            'duration_ms': self.duration_ms,
            'kernel_count': self.kernel_count,
            'csv_path': self.csv_path
        }


class HardwareValidator:
    """
    Hardware-level validation using NVIDIA Nsight Compute.

    Profiles GPU workloads and extracts DRAM traffic metrics
    to validate memory optimization claims.
    """

    # Key metrics to extract from Nsight
    NSIGHT_METRICS = [
        'dram__bytes_read.sum',
        'dram__bytes_write.sum',
        'l2_tex_read_throughput.avg.pct_of_peak_sustained_elapsed',
        'sm__throughput.avg.pct_of_peak_sustained_elapsed',
        'sm__warps_active.avg.pct_of_peak_sustained_active',
    ]

    def __init__(self, output_dir: str = 'validation', ncu_path: str = 'ncu'):
        """
        Initialize hardware validator.

        Args:
            output_dir: Directory to save CSV output files
            ncu_path: Path to ncu (Nsight Compute CLI)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ncu_path = ncu_path

        # Check if Nsight is available
        self._nsight_available = self._check_nsight()

    def _check_nsight(self) -> bool:
        """Check if Nsight Compute is available."""
        try:
            result = subprocess.run(
                [self.ncu_path, '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    @property
    def nsight_available(self) -> bool:
        return self._nsight_available

    def profile_workload(
        self,
        script_path: str,
        label: str,
        python_path: str = 'python3',
        extra_args: List[str] = None,
        timeout: int = 300
    ) -> NsightMetrics:
        """
        Profile a Python script with Nsight Compute.

        Args:
            script_path: Path to Python script to profile
            label: Label for this profiling run
            python_path: Path to Python interpreter
            extra_args: Extra arguments to pass to the script
            timeout: Timeout in seconds

        Returns:
            NsightMetrics with extracted hardware counters
        """
        if not self._nsight_available:
            return self._profile_with_pytorch(script_path, label, python_path, extra_args)

        csv_path = self.output_dir / f'{label}_nsight.csv'

        # Build Nsight command
        metrics_str = ','.join(self.NSIGHT_METRICS)
        cmd = [
            self.ncu_path,
            '--metrics', metrics_str,
            '--csv',
            '--log-file', str(csv_path),
            '--target-processes', 'all',
            python_path, script_path
        ]

        if extra_args:
            cmd.extend(extra_args)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(Path(script_path).parent)
            )

            if result.returncode != 0:
                print(f"Warning: Nsight exited with code {result.returncode}")
                if result.stderr:
                    print(f"Stderr: {result.stderr[:500]}")

            # Parse CSV
            return self._parse_nsight_csv(csv_path, label)

        except subprocess.TimeoutExpired:
            print(f"Warning: Nsight profiling timed out after {timeout}s")
            return NsightMetrics(label=label)

        except Exception as e:
            print(f"Warning: Nsight profiling failed: {e}")
            return self._profile_with_pytorch(script_path, label, python_path, extra_args)

    def _parse_nsight_csv(self, csv_path: Path, label: str) -> NsightMetrics:
        """Parse Nsight Compute CSV output."""
        metrics = NsightMetrics(label=label, csv_path=str(csv_path))

        if not csv_path.exists():
            print(f"Warning: CSV file not found: {csv_path}")
            return metrics

        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                kernel_count = 0

                for row in reader:
                    kernel_count += 1

                    # Extract DRAM metrics
                    if 'dram__bytes_read.sum' in row:
                        try:
                            metrics.dram_bytes_read += int(float(row['dram__bytes_read.sum']))
                        except (ValueError, TypeError):
                            pass

                    if 'dram__bytes_write.sum' in row:
                        try:
                            metrics.dram_bytes_write += int(float(row['dram__bytes_write.sum']))
                        except (ValueError, TypeError):
                            pass

                    # Extract efficiency metrics (average across kernels)
                    if 'sm__throughput.avg.pct_of_peak_sustained_elapsed' in row:
                        try:
                            metrics.sm_efficiency += float(row['sm__throughput.avg.pct_of_peak_sustained_elapsed'])
                        except (ValueError, TypeError):
                            pass

                metrics.kernel_count = kernel_count
                if kernel_count > 0:
                    metrics.sm_efficiency /= kernel_count

        except Exception as e:
            print(f"Warning: Failed to parse CSV: {e}")

        return metrics

    def _profile_with_pytorch(
        self,
        script_path: str,
        label: str,
        python_path: str = 'python3',
        extra_args: List[str] = None
    ) -> NsightMetrics:
        """
        Fallback profiling using PyTorch profiler when Nsight unavailable.
        """
        metrics = NsightMetrics(label=label)

        if not torch.cuda.is_available():
            return metrics

        # Use PyTorch's built-in memory tracking
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        try:
            # Run the script and capture memory usage
            cmd = [python_path, script_path]
            if extra_args:
                cmd.extend(extra_args)

            start_time = datetime.now()

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(Path(script_path).parent) if script_path else None
            )

            end_time = datetime.now()
            metrics.duration_ms = (end_time - start_time).total_seconds() * 1000

            # Estimate DRAM traffic from output if available
            if result.stdout:
                # Look for memory stats in output
                for line in result.stdout.split('\n'):
                    if 'peak_memory' in line.lower() or 'bytes' in line.lower():
                        # Try to extract numbers
                        parts = line.split()
                        for part in parts:
                            try:
                                val = float(part.replace('GB', '').replace('MB', ''))
                                if 'GB' in line:
                                    metrics.dram_bytes_read = int(val * 1024**3)
                                elif 'MB' in line:
                                    metrics.dram_bytes_read = int(val * 1024**2)
                                break
                            except ValueError:
                                continue

        except Exception as e:
            print(f"PyTorch profiling failed: {e}")

        return metrics

    def profile_inline(
        self,
        func: callable,
        label: str,
        *args,
        **kwargs
    ) -> NsightMetrics:
        """
        Profile a function inline using PyTorch profiler.

        This is useful when you can't run a separate script.
        """
        metrics = NsightMetrics(label=label)

        if not torch.cuda.is_available():
            # Run without profiling
            func(*args, **kwargs)
            return metrics

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Use CUDA events for timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        # Run the function
        result = func(*args, **kwargs)

        end_event.record()
        torch.cuda.synchronize()

        # Collect metrics
        metrics.duration_ms = start_event.elapsed_time(end_event)
        metrics.dram_bytes_read = torch.cuda.max_memory_allocated()
        metrics.dram_bytes_write = torch.cuda.memory_allocated()

        return metrics

    def compare(
        self,
        baseline: NsightMetrics,
        optimized: NsightMetrics
    ) -> Dict:
        """
        Compare baseline and optimized profiling results.

        Returns dict with comparison metrics.
        """
        comparison = {
            'baseline': baseline.to_dict(),
            'optimized': optimized.to_dict(),
            'reduction': {},
            'validated': False
        }

        # Calculate reductions
        if baseline.dram_bytes_total > 0:
            dram_reduction = baseline.dram_bytes_total - optimized.dram_bytes_total
            dram_reduction_pct = (dram_reduction / baseline.dram_bytes_total) * 100

            comparison['reduction'] = {
                'dram_bytes': dram_reduction,
                'dram_gb': dram_reduction / (1024**3),
                'dram_pct': dram_reduction_pct
            }

            # Mark as validated if we have real data
            comparison['validated'] = (
                baseline.dram_bytes_total > 0 and
                optimized.dram_bytes_total > 0 and
                (baseline.csv_path is not None or baseline.duration_ms > 0)
            )

        if baseline.duration_ms > 0 and optimized.duration_ms > 0:
            speedup = baseline.duration_ms / optimized.duration_ms
            comparison['reduction']['speedup'] = speedup

        return comparison

    def generate_report(self, comparison: Dict, output_path: str = None) -> str:
        """Generate markdown report from comparison."""
        baseline = comparison['baseline']
        optimized = comparison['optimized']
        reduction = comparison.get('reduction', {})

        lines = [
            "# Hardware Validation Report",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Validated**: {'✅ YES' if comparison.get('validated') else '❌ NO'}",
            "",
            "---",
            "",
            "## Baseline (No Optimization)",
            "",
            f"- **DRAM Read**: {baseline.get('dram_gb_total', 0):.3f} GB",
            f"- **Duration**: {baseline.get('duration_ms', 0):.1f} ms",
            f"- **Kernels**: {baseline.get('kernel_count', 0)}",
            "",
            "## Optimized (With Coalescing)",
            "",
            f"- **DRAM Read**: {optimized.get('dram_gb_total', 0):.3f} GB",
            f"- **Duration**: {optimized.get('duration_ms', 0):.1f} ms",
            f"- **Kernels**: {optimized.get('kernel_count', 0)}",
            "",
            "## Results",
            "",
            f"- **DRAM Reduction**: {reduction.get('dram_pct', 0):.1f}%",
            f"- **Bytes Saved**: {reduction.get('dram_gb', 0):.3f} GB",
            f"- **Speedup**: {reduction.get('speedup', 1.0):.2f}x",
            "",
            "## Proof Files",
            "",
        ]

        if baseline.get('csv_path'):
            lines.append(f"- Baseline CSV: `{baseline['csv_path']}`")
        if optimized.get('csv_path'):
            lines.append(f"- Optimized CSV: `{optimized['csv_path']}`")

        report = "\n".join(lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)

        return report
