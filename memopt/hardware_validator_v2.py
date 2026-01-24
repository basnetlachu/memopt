"""
Hardware Validation using NVIDIA Nsight Compute

This provides ACTUAL DRAM traffic measurements from GPU hardware counters.
Critical for enterprise credibility - validates PyTorch estimates against real HW.
"""

import subprocess
import os
import re
import tempfile
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class HardwareValidationResult:
    """Result from Nsight Compute hardware validation."""

    validated: bool  # True if hardware counters were successfully read
    dram_read_gb: float = 0.0  # DRAM bytes read (from HW counters)
    dram_write_gb: float = 0.0  # DRAM bytes written (from HW counters)
    total_dram_gb: float = 0.0  # Total DRAM traffic
    l2_read_gb: float = 0.0  # L2 cache reads (optional)
    l2_write_gb: float = 0.0  # L2 cache writes (optional)
    note: str = ""  # Error message or additional info

    def __repr__(self):
        if self.validated:
            return f"HardwareValidation(✅ DRAM: {self.total_dram_gb:.2f} GB, Read: {self.dram_read_gb:.2f} GB, Write: {self.dram_write_gb:.2f} GB)"
        else:
            return f"HardwareValidation(❌ {self.note})"


class HardwareValidator:
    """
    Validates bandwidth measurements using NVIDIA Nsight Compute.

    This is the enterprise credibility layer - proves measurements are real.
    """

    def __init__(self, ncu_path: str = "/usr/local/cuda-12.4/bin/ncu"):
        """
        Initialize hardware validator.

        Args:
            ncu_path: Path to ncu binary (Nsight Compute)
        """
        self.ncu_path = ncu_path

        # Check if ncu exists
        if not os.path.exists(ncu_path):
            # Try to find it
            possible_paths = [
                "/usr/local/cuda/bin/ncu",
                "/usr/local/cuda-12.4/bin/ncu",
                "/usr/local/cuda-12.3/bin/ncu",
                "/usr/local/cuda-11.8/bin/ncu",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    self.ncu_path = path
                    break

    def is_available(self) -> bool:
        """Check if Nsight Compute is available."""
        return os.path.exists(self.ncu_path)

    def validate_script(
        self,
        script_content: str,
        timeout: int = 300
    ) -> HardwareValidationResult:
        """
        Run a Python script under Nsight Compute profiling.

        Args:
            script_content: Python code to profile
            timeout: Timeout in seconds

        Returns:
            HardwareValidationResult with DRAM counters
        """
        if not self.is_available():
            return HardwareValidationResult(
                validated=False,
                note=f"Nsight Compute not found at {self.ncu_path}"
            )

        # Create temporary script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            script_path = f.name

        # Create temporary output file
        output_file = tempfile.mktemp(suffix='.csv')

        try:
            # Run ncu with DRAM metrics
            cmd = [
                self.ncu_path,
                '--metrics', 'dram__bytes_read.sum,dram__bytes_write.sum',
                '--csv',
                '--log-file', output_file,
                '--target-processes', 'all',
                'python3', script_path
            ]

            # Run with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            # Parse output
            return self._parse_ncu_output(output_file)

        except subprocess.TimeoutExpired:
            return HardwareValidationResult(
                validated=False,
                note=f"Nsight Compute timed out after {timeout}s"
            )
        except Exception as e:
            return HardwareValidationResult(
                validated=False,
                note=f"Nsight Compute failed: {str(e)}"
            )
        finally:
            # Cleanup
            if os.path.exists(script_path):
                os.remove(script_path)

    def _parse_ncu_output(self, output_file: str) -> HardwareValidationResult:
        """Parse Nsight Compute CSV output."""
        if not os.path.exists(output_file):
            return HardwareValidationResult(
                validated=False,
                note="Nsight output file not found"
            )

        try:
            with open(output_file, 'r') as f:
                content = f.read()

            dram_read_total = 0
            dram_write_total = 0

            # Parse metrics
            read_matches = re.findall(r'dram__bytes_read\.sum[,\s]+(\d+)', content)
            write_matches = re.findall(r'dram__bytes_write\.sum[,\s]+(\d+)', content)

            for match in read_matches:
                dram_read_total += int(match)

            for match in write_matches:
                dram_write_total += int(match)

            if dram_read_total == 0 and dram_write_total == 0:
                return HardwareValidationResult(
                    validated=False,
                    note="Could not parse DRAM metrics from Nsight output"
                )

            return HardwareValidationResult(
                validated=True,
                dram_read_gb=dram_read_total / 1e9,
                dram_write_gb=dram_write_total / 1e9,
                total_dram_gb=(dram_read_total + dram_write_total) / 1e9,
                note="Hardware-validated with Nsight Compute"
            )

        except Exception as e:
            return HardwareValidationResult(
                validated=False,
                note=f"Failed to parse Nsight output: {str(e)}"
            )

    def quick_test(self) -> HardwareValidationResult:
        """Quick test to verify Nsight Compute works."""
        test_script = """
import torch

a = torch.randn(1024, 1024, device='cuda')
b = torch.randn(1024, 1024, device='cuda')
c = torch.matmul(a, b)
torch.cuda.synchronize()
"""

        print(f"Running Nsight Compute test...")
        print(f"  (This may take 30-60 seconds)")

        result = self.validate_script(test_script, timeout=120)

        if result.validated:
            print(f"  ✅ Nsight Compute works!")
            print(f"     DRAM read: {result.dram_read_gb:.2f} GB")
            print(f"     DRAM write: {result.dram_write_gb:.2f} GB")
        else:
            print(f"  ❌ Failed: {result.note}")

        return result


if __name__ == "__main__":
    validator = HardwareValidator()
    print(f"Nsight available: {validator.is_available()}")
    if validator.is_available():
        validator.quick_test()
