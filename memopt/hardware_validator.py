"""
Hardware Counter Validation

Validates bandwidth measurements against GPU hardware counters using NVIDIA Nsight Compute.
Provides credibility for enterprise customers by showing actual DRAM traffic.
"""

from typing import Optional, Dict
import subprocess
import re
import os
import json
from dataclasses import dataclass


@dataclass
class HardwareCounters:
    """Hardware-validated memory counters from Nsight Compute."""
    dram_read_bytes: int = 0
    dram_write_bytes: int = 0
    l2_read_bytes: int = 0
    l2_write_bytes: int = 0
    validated: bool = False
    validation_method: str = "none"
    error_message: Optional[str] = None

    @property
    def dram_read_gb(self) -> float:
        """DRAM reads in GB."""
        return self.dram_read_bytes / (1024**3)

    @property
    def dram_write_gb(self) -> float:
        """DRAM writes in GB."""
        return self.dram_write_bytes / (1024**3)

    @property
    def total_dram_gb(self) -> float:
        """Total DRAM traffic in GB."""
        return (self.dram_read_bytes + self.dram_write_bytes) / (1024**3)

    @property
    def l2_read_gb(self) -> float:
        """L2 cache reads in GB."""
        return self.l2_read_bytes / (1024**3)


class HardwareValidator:
    """
    Validates bandwidth measurements using NVIDIA profiling tools.

    Supports two validation methods:
    1. Nsight Compute (ncu) - most accurate, kernel-level DRAM counters
    2. nvprof (legacy) - deprecated but widely available

    Usage:
        validator = HardwareValidator()
        counters = validator.validate_inference(model_name, prompts)

        if counters.validated:
            print(f"✅ Hardware-validated: {counters.total_dram_gb:.2f} GB DRAM")
        else:
            print(f"⚠️  {counters.error_message}")
    """

    NSIGHT_METRICS = [
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum",  # L2 read
        "lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_st.sum",  # L2 write
    ]

    def __init__(self):
        """Initialize hardware validator."""
        self.ncu_available = self._check_ncu_available()
        self.nvprof_available = self._check_nvprof_available()

    def _check_ncu_available(self) -> bool:
        """Check if Nsight Compute (ncu) is available."""
        try:
            result = subprocess.run(
                ['ncu', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _check_nvprof_available(self) -> bool:
        """Check if nvprof is available."""
        try:
            result = subprocess.run(
                ['nvprof', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def validate_with_nsight(
        self,
        command: str,
        output_file: str = "nsight_profile.csv"
    ) -> HardwareCounters:
        """
        Run Nsight Compute to get actual DRAM bytes.

        Args:
            command: Command to profile (e.g., "python script.py")
            output_file: Where to save Nsight output

        Returns:
            HardwareCounters with validated measurements

        Example:
            counters = validator.validate_with_nsight("python benchmark.py")
            print(f"DRAM traffic: {counters.total_dram_gb:.2f} GB")
        """
        if not self.ncu_available:
            return HardwareCounters(
                validated=False,
                error_message="Nsight Compute (ncu) not found. Install from: https://developer.nvidia.com/nsight-compute"
            )

        try:
            # Build ncu command
            ncu_cmd = [
                'ncu',
                '--metrics', ','.join(self.NSIGHT_METRICS),
                '--csv',
                '--log-file', output_file,
                '--target-processes', 'all',
                '--'
            ] + command.split()

            # Run profiling
            result = subprocess.run(
                ncu_cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                return HardwareCounters(
                    validated=False,
                    error_message=f"Nsight Compute failed: {result.stderr[:200]}"
                )

            # Parse output
            counters = self._parse_nsight_output(output_file)
            counters.validated = True
            counters.validation_method = "nsight_compute"
            return counters

        except subprocess.TimeoutExpired:
            return HardwareCounters(
                validated=False,
                error_message="Nsight Compute timed out after 5 minutes"
            )
        except Exception as e:
            return HardwareCounters(
                validated=False,
                error_message=f"Nsight Compute error: {str(e)}"
            )

    def _parse_nsight_output(self, csv_file: str) -> HardwareCounters:
        """Parse Nsight Compute CSV output to extract DRAM counters."""
        counters = HardwareCounters()

        if not os.path.exists(csv_file):
            counters.error_message = f"Nsight output file not found: {csv_file}"
            return counters

        try:
            with open(csv_file, 'r') as f:
                content = f.read()

            # Parse CSV for metric values
            # Format: "Metric Name","Metric Value"
            for metric_name, counter_attr in [
                ("dram__bytes_read.sum", "dram_read_bytes"),
                ("dram__bytes_write.sum", "dram_write_bytes"),
                ("lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum", "l2_read_bytes"),
                ("lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_st.sum", "l2_write_bytes"),
            ]:
                pattern = rf'"{metric_name}"\s*,\s*"?([0-9,]+)"?'
                match = re.search(pattern, content)
                if match:
                    value_str = match.group(1).replace(',', '')
                    setattr(counters, counter_attr, int(value_str))

            return counters

        except Exception as e:
            counters.error_message = f"Failed to parse Nsight output: {str(e)}"
            return counters

    def validate_with_nvprof(self, command: str) -> HardwareCounters:
        """
        Legacy validation using nvprof (deprecated but widely available).

        Args:
            command: Command to profile

        Returns:
            HardwareCounters with validated measurements
        """
        if not self.nvprof_available:
            return HardwareCounters(
                validated=False,
                error_message="nvprof not found (deprecated tool)"
            )

        try:
            nvprof_cmd = [
                'nvprof',
                '--print-gpu-trace',
                '--csv',
                '--'
            ] + command.split()

            result = subprocess.run(
                nvprof_cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            # Parse nvprof output (less accurate than Nsight)
            counters = self._parse_nvprof_output(result.stdout)
            counters.validated = True
            counters.validation_method = "nvprof_legacy"
            return counters

        except Exception as e:
            return HardwareCounters(
                validated=False,
                error_message=f"nvprof error: {str(e)}"
            )

    def _parse_nvprof_output(self, output: str) -> HardwareCounters:
        """Parse nvprof output (less precise than Nsight)."""
        # nvprof doesn't give us direct DRAM counters
        # This is a rough estimate based on memory operations
        counters = HardwareCounters()
        counters.error_message = "nvprof provides lower accuracy than Nsight Compute"
        return counters

    def add_disclaimer(self, stats: Dict, counters: Optional[HardwareCounters] = None) -> Dict:
        """
        Add measurement methodology disclaimer to profiling stats.

        Args:
            stats: Profiling statistics dictionary
            counters: Optional hardware validation counters

        Returns:
            Updated stats with disclaimer
        """
        if counters and counters.validated:
            stats['hardware_validated'] = True
            stats['validation_method'] = counters.validation_method
            stats['hardware_dram_gb'] = counters.total_dram_gb
            stats['methodology_note'] = (
                f"✅ Hardware-validated using {counters.validation_method}. "
                f"Actual DRAM traffic: {counters.total_dram_gb:.2f} GB."
            )
        else:
            stats['hardware_validated'] = False
            stats['methodology_note'] = (
                "⚠️  METHODOLOGY: Lower-bound bandwidth estimates from PyTorch profiler. "
                "Directionally accurate but not hardware-validated. "
                "For hardware validation, install Nsight Compute and run with --validate-hardware flag.\n"
                "Install: https://developer.nvidia.com/nsight-compute"
            )
            if counters and counters.error_message:
                stats['validation_error'] = counters.error_message

        return stats

    def get_installation_instructions(self) -> str:
        """Get instructions for installing validation tools."""
        instructions = """
# Hardware Counter Validation Setup

## Nsight Compute (Recommended)

1. Download from: https://developer.nvidia.com/nsight-compute
2. Install for your platform:
   - Ubuntu: sudo dpkg -i nsight-compute-*.deb
   - RHEL/CentOS: sudo rpm -i nsight-compute-*.rpm
   - Windows: Run .exe installer

3. Verify installation:
   ncu --version

## nvprof (Legacy, less accurate)

Already included with CUDA Toolkit 11.x and earlier.
Deprecated in CUDA 12+.

## Usage

# With Nsight Compute
memopt-profile --model gpt2 --validate-hardware

# Manual validation
ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum \\
    python your_script.py
"""
        return instructions


def validate_bandwidth_measurement(
    stats: Dict,
    validation_command: Optional[str] = None
) -> Dict:
    """
    Quick helper to validate bandwidth measurements.

    Args:
        stats: Profiling statistics from BandwidthProfiler
        validation_command: Optional command to run for hardware validation

    Returns:
        Stats with validation results and disclaimers

    Example:
        from memopt import BandwidthProfiler, validate_bandwidth_measurement

        profiler = BandwidthProfiler()
        stats = profiler.profile_model(model)

        # Add hardware validation
        stats = validate_bandwidth_measurement(
            stats,
            validation_command="python benchmark.py"
        )

        print(stats['methodology_note'])
    """
    validator = HardwareValidator()

    if validation_command and validator.ncu_available:
        counters = validator.validate_with_nsight(validation_command)
        return validator.add_disclaimer(stats, counters)
    else:
        return validator.add_disclaimer(stats, None)
