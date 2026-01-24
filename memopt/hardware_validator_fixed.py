"""Hardware Validation using NVIDIA Nsight Compute"""

import subprocess
import os
import csv
import tempfile
from dataclasses import dataclass


@dataclass
class HardwareValidationResult:
    validated: bool
    dram_read_gb: float = 0.0
    dram_write_gb: float = 0.0
    total_dram_gb: float = 0.0
    note: str = ""


class HardwareValidator:
    def __init__(self, ncu_path: str = "/usr/local/cuda-12.4/bin/ncu"):
        self.ncu_path = ncu_path
        if not os.path.exists(ncu_path):
            possible_paths = [
                "/usr/local/cuda/bin/ncu",
                "/usr/local/cuda-12.4/bin/ncu",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    self.ncu_path = path
                    break

    def is_available(self) -> bool:
        return os.path.exists(self.ncu_path)

    def validate_script(self, script_content: str, timeout: int = 300) -> HardwareValidationResult:
        if not self.is_available():
            return HardwareValidationResult(False, note=f"ncu not found at {self.ncu_path}")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            script_path = f.name

        output_file = tempfile.mktemp(suffix='.csv')

        try:
            cmd = [
                self.ncu_path,
                '--metrics', 'dram__bytes_read.sum,dram__bytes_write.sum',
                '--csv',
                '--log-file', output_file,
                '--target-processes', 'all',
                'python3', script_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return self._parse_csv(output_file)

        except subprocess.TimeoutExpired:
            return HardwareValidationResult(False, note=f"Timeout after {timeout}s")
        except Exception as e:
            return HardwareValidationResult(False, note=str(e))
        finally:
            if os.path.exists(script_path):
                os.remove(script_path)

    def _parse_csv(self, csv_file: str) -> HardwareValidationResult:
        if not os.path.exists(csv_file):
            return HardwareValidationResult(False, note="CSV file not found")

        try:
            dram_read = 0
            dram_write = 0

            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    metric = row.get('Metric Name', '')
                    value = row.get('Metric Value', '0')
                    
                    if 'dram__bytes_read' in metric:
                        dram_read += int(value)
                    elif 'dram__bytes_write' in metric:
                        dram_write += int(value)

            if dram_read == 0 and dram_write == 0:
                return HardwareValidationResult(False, note="No DRAM metrics found")

            return HardwareValidationResult(
                validated=True,
                dram_read_gb=dram_read / 1e9,
                dram_write_gb=dram_write / 1e9,
                total_dram_gb=(dram_read + dram_write) / 1e9,
                note="Hardware-validated"
            )

        except Exception as e:
            return HardwareValidationResult(False, note=f"Parse error: {str(e)}")

    def quick_test(self):
        test_script = """
import torch
a = torch.randn(1024, 1024, device='cuda')
b = torch.matmul(a, a)
torch.cuda.synchronize()
"""
        print("Testing Nsight Compute...")
        result = self.validate_script(test_script, timeout=120)
        
        if result.validated:
            print(f"  ✅ Works! DRAM read: {result.dram_read_gb:.3f} GB, write: {result.dram_write_gb:.3f} GB")
        else:
            print(f"  ❌ Failed: {result.note}")
        
        return result


if __name__ == "__main__":
    v = HardwareValidator()
    if v.is_available():
        v.quick_test()
    else:
        print("Nsight not available")
