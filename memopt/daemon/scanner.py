"""
GPU process scanner for memopt.
Uses pynvml (nvidia-smi programmatic API) to discover
all processes using GPUs on this server.
No external dependencies beyond pynvml (already in requirements).
Never crashes — returns empty list if no GPUs or pynvml fails.
"""
import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict

log = logging.getLogger(__name__)


@dataclass
class GPUProcess:
    pid: int
    gpu_ids: List[int]           # can span multiple GPUs
    gpu_memory_mb: int           # total across all GPUs
    gpu_utilization_pct: float   # average across GPUs
    cmdline: str                 # full command line
    script_name: str             # just the .py filename
    framework: str               # pytorch / tensorflow / jax / unknown
    mode: str                    # inference / training / unknown
    model_family: str            # llama / bert / resnet / unknown
    model_size_b: Optional[float]  # parameter count in billions if detectable
    working_dir: str


class GPUScanner:
    """
    Scans all GPUs on this server and returns running ML processes.

    Usage:
        scanner = GPUScanner()
        processes = scanner.scan()
        for proc in processes:
            print(proc.pid, proc.model_family, proc.mode)
    """

    def scan(self) -> List[GPUProcess]:
        """
        Return all GPU processes currently running.
        Never raises — returns [] on any failure.
        """
        try:
            import pynvml
            pynvml.nvmlInit()
            try:
                raw = self._get_raw_processes()
                merged = self._merge_by_pid(raw)
                enriched = [self._enrich(p) for p in merged]
                return enriched
            finally:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
        except Exception as e:
            log.warning(f"GPU scan failed: {e}")
            return []

    def _get_raw_processes(self) -> List[dict]:
        """Get raw process list from pynvml across all GPUs."""
        import pynvml
        raw = []
        device_count = pynvml.nvmlDeviceGetCount()

        for gpu_id in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            except pynvml.NVMLError:
                continue

            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
            except pynvml.NVMLError:
                gpu_util = 0

            for proc in procs:
                raw.append({
                    "pid": proc.pid,
                    "gpu_id": gpu_id,
                    "gpu_memory_mb": proc.usedGpuMemory // (1024 * 1024),
                    "gpu_utilization_pct": gpu_util,
                })

        return raw

    def _merge_by_pid(self, raw: List[dict]) -> List[dict]:
        """Merge entries for same PID across multiple GPUs."""
        by_pid: Dict[int, dict] = {}
        for entry in raw:
            pid = entry["pid"]
            if pid not in by_pid:
                by_pid[pid] = {
                    "pid": pid,
                    "gpu_ids": [],
                    "gpu_memory_mb": 0,
                    "gpu_utilization_pct": 0,
                    "util_count": 0,
                }
            by_pid[pid]["gpu_ids"].append(entry["gpu_id"])
            by_pid[pid]["gpu_memory_mb"] += entry["gpu_memory_mb"]
            by_pid[pid]["gpu_utilization_pct"] += entry["gpu_utilization_pct"]
            by_pid[pid]["util_count"] += 1

        # Average utilization
        for p in by_pid.values():
            if p["util_count"] > 0:
                p["gpu_utilization_pct"] /= p["util_count"]

        return list(by_pid.values())

    def _enrich(self, raw: dict) -> GPUProcess:
        """Add cmdline, framework, mode, model_family from /proc."""
        pid = raw["pid"]
        cmdline, script_name, working_dir = self._read_proc(pid)

        framework = self._detect_framework(cmdline)
        mode = self._detect_mode(cmdline)
        model_family, model_size_b = self._detect_model(
            cmdline, raw["gpu_memory_mb"]
        )

        return GPUProcess(
            pid=pid,
            gpu_ids=raw["gpu_ids"],
            gpu_memory_mb=raw["gpu_memory_mb"],
            gpu_utilization_pct=raw["gpu_utilization_pct"],
            cmdline=cmdline,
            script_name=script_name,
            framework=framework,
            mode=mode,
            model_family=model_family,
            model_size_b=model_size_b,
            working_dir=working_dir,
        )

    def _read_proc(self, pid: int):
        """Read /proc/<pid>/cmdline and /proc/<pid>/cwd."""
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                raw = f.read().replace(b"\x00", b" ").decode("utf-8", errors="replace")
            cmdline = raw.strip()
        except Exception:
            cmdline = ""

        # Extract script name
        parts = cmdline.split()
        script_name = ""
        for part in parts:
            if part.endswith(".py"):
                script_name = os.path.basename(part)
                break

        # Working directory
        try:
            working_dir = os.readlink(f"/proc/{pid}/cwd")
        except Exception:
            working_dir = ""

        return cmdline, script_name, working_dir

    def _detect_framework(self, cmdline: str) -> str:
        """Detect ML framework from command line."""
        cmdline_lower = cmdline.lower()
        if "tensorflow" in cmdline_lower or "tf." in cmdline_lower:
            return "tensorflow"
        if "jax" in cmdline_lower:
            return "jax"
        # Default to pytorch — most common, and pynvml finds CUDA processes
        return "pytorch"

    def _detect_mode(self, cmdline: str) -> str:
        """Detect inference vs training from command line."""
        cmdline_lower = cmdline.lower()
        training_signals = [
            "train", "finetune", "fine_tune", "fine-tune",
            "fit", "epoch", "backward", "optimizer",
        ]
        inference_signals = [
            "serve", "infer", "predict", "inference",
            "generate", "api", "server", "deploy",
            "fastapi", "flask", "uvicorn", "gunicorn",
        ]
        if any(s in cmdline_lower for s in training_signals):
            return "training"
        if any(s in cmdline_lower for s in inference_signals):
            return "inference"
        return "inference"  # default assumption

    def _detect_model(
        self, cmdline: str, gpu_memory_mb: int
    ):
        """
        Detect model family and estimate size from cmdline + VRAM.
        Returns (family, size_in_billions or None).
        """
        cmdline_lower = cmdline.lower()

        # Model family detection — order matters (specific before generic)
        families = [
            (["llama-3", "llama3"], "llama3"),
            (["llama-2", "llama2"], "llama2"),
            (["llama"], "llama"),
            (["mistral"], "mistral"),
            (["mixtral"], "mixtral"),
            (["falcon"], "falcon"),
            (["gemma"], "gemma"),
            (["qwen"], "qwen"),
            (["phi-3", "phi3"], "phi3"),
            (["bert", "roberta", "albert"], "bert"),
            (["gpt-2", "gpt2"], "gpt2"),
            (["gpt-j", "gptj"], "gptj"),
            (["gpt-neox", "gptneox"], "gptneox"),
            (["resnet"], "resnet"),
            (["vit", "vision-transformer"], "vit"),
            (["whisper"], "whisper"),
            (["stable-diffusion", "sdxl", "diffusion"], "diffusion"),
            (["clip"], "clip"),
        ]

        detected_family = "unknown"
        for signals, family in families:
            if any(s in cmdline_lower for s in signals):
                detected_family = family
                break

        # Estimate size from VRAM usage
        # FP16: ~2 bytes/param. 7B = ~14GB, 13B = ~26GB, 70B = ~140GB
        # Add ~20% overhead for activations/KV cache
        estimated_size_b = None
        if gpu_memory_mb > 500:  # ignore tiny processes
            params_b = (gpu_memory_mb / 1024) / 2.4  # rough FP16 estimate
            estimated_size_b = round(params_b, 1)

        # Override size from explicit cmdline hints
        size_hints = {
            "7b": 7.0, "7-b": 7.0,
            "13b": 13.0, "13-b": 13.0,
            "30b": 30.0, "34b": 34.0,
            "70b": 70.0, "72b": 72.0,
            "8b": 8.0, "8-b": 8.0,
        }
        for hint, size in size_hints.items():
            if hint in cmdline_lower:
                estimated_size_b = size
                break

        return detected_family, estimated_size_b
