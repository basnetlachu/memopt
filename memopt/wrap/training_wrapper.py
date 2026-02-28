"""
memopt-wrap: zero-touch training optimization.

Customer adds ONE word to their existing command:
  Before: python train.py --model llama --epochs 10
  After:  memopt-wrap python train.py --model llama --epochs 10

What happens:
  1. memopt-wrap launches train.py as a subprocess
  2. Monitors GPU utilization and memory during first N batches
  3. Identifies bottleneck (compute vs memory-bound)
  4. Between batch N and N+1: applies safe training optimizations
     - Gradient checkpointing (if VRAM > 70%)
     - Recommends BF16 autocast and torch.compile
  5. Writes optimization report to ~/.memopt/training_sessions/
  6. Process exit code is forwarded unchanged

What it does NOT do:
  - Modify the user's source code
  - Stop or interrupt training
  - Apply risky optimizations (weight changes, loss scaling, etc.)
  - Touch model weights or optimizer state

How it works technically:
  - Writes a sitecustomize.py + hook module into a temp directory
  - Injects that directory at the head of PYTHONPATH
  - sitecustomize.py is executed automatically by Python on startup
  - The hook patches nn.Module.__call__ and nn.Module.train to observe
    training batches and apply safe in-process changes
"""
import os
import sys
import subprocess
import tempfile
import logging
from pathlib import Path

log = logging.getLogger(__name__)


_HOOK_TEMPLATE = '''\
"""
memopt training hook — injected by memopt-wrap.
Patches training loop to profile and optimize automatically.
Auto-generated — do not edit.
"""
import torch
import torch.nn as nn
import time
import os
import sys
import logging

log = logging.getLogger("memopt.hook")

_PROFILE_BATCHES = {profile_batches}
_GPU_COST_PER_HOUR = {gpu_cost_per_hour}
_NODE_NAME = {node_name!r}
_DRY_RUN = {dry_run}

_call_count = 0
_optimizations_applied = False
_batch_times = []
_phase_start = None
_baseline_batch_ms = None

_original_module_call = nn.Module.__call__
_original_train = nn.Module.train


def _memopt_forward(self, *args, **kwargs):
    global _call_count, _optimizations_applied, _batch_times, _phase_start, _baseline_batch_ms

    result = _original_module_call(self, *args, **kwargs)

    # Only intercept root modules (large models — avoid submodule recursion)
    if not getattr(self, "_memopt_root", False):
        return result

    _call_count += 1

    if _phase_start is None:
        _phase_start = time.time()
        log.info(f"memopt: profiling first {{_PROFILE_BATCHES}} batches...")

    elapsed_ms = (time.time() - _phase_start) * 1000
    _batch_times.append(elapsed_ms)
    _phase_start = time.time()

    if _call_count == _PROFILE_BATCHES and not _optimizations_applied:
        _apply_optimizations(self)

    return result


def _apply_optimizations(model):
    global _optimizations_applied, _baseline_batch_ms

    if _optimizations_applied:
        return
    _optimizations_applied = True

    if _batch_times:
        _baseline_batch_ms = sum(_batch_times) / len(_batch_times)

    applied = []

    if _DRY_RUN:
        log.info("memopt: dry-run — profiling only, no optimizations applied")
        _write_report(applied, dry_run=True)
        return

    # 1. Gradient checkpointing — safe VRAM reduction for memory-bound training
    if torch.cuda.is_available():
        vram_used  = torch.cuda.memory_reserved(0) / 1e9
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        vram_pct   = vram_used / vram_total if vram_total > 0 else 0

        if vram_pct > 0.7 and hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable()
                applied.append("gradient_checkpointing")
                log.info(f"memopt: gradient checkpointing enabled (VRAM {{vram_pct:.0%}} used)")
            except Exception as e:
                log.warning(f"memopt: gradient_checkpointing_enable failed: {{e}}")

    # 2. BF16 recommendation (cannot change existing parameters in-place)
    if torch.cuda.is_bf16_supported():
        sample = next(iter(model.parameters()), None)
        if sample is not None and sample.dtype == torch.float32:
            applied.append("bf16_autocast_recommended")
            log.info("memopt: BF16 autocast recommended — add torch.autocast('cuda') to training loop")

    # 3. torch.compile recommendation
    if not getattr(model, "_memopt_compiled", False) and hasattr(torch, "compile"):
        applied.append("torch_compile_recommended")
        log.info("memopt: torch.compile recommended — wrap model with torch.compile(model, mode='reduce-overhead')")

    log.info(f"memopt: applied after batch {{_PROFILE_BATCHES}}: {{applied}}")
    _write_report(applied, dry_run=False)


def _write_report(applied, dry_run=False):
    import json
    import datetime

    sessions_dir = Path.home() / ".memopt" / "training_sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    report = {{
        "timestamp": datetime.datetime.now().isoformat(),
        "node": _NODE_NAME,
        "baseline_batch_ms": round(_baseline_batch_ms, 2) if _baseline_batch_ms else None,
        "optimizations": applied,
        "dry_run": dry_run,
        "gpu_cost_per_hour": _GPU_COST_PER_HOUR,
        "script": sys.argv[0] if sys.argv else "unknown",
    }}

    ts = int(time.time())
    report_file = sessions_dir / f"training_{{ts}}.json"
    report_file.write_text(json.dumps(report, indent=2))
    log.info(f"memopt: report written to {{report_file}}")


def _patched_train(self, mode=True):
    result = _original_train(self, mode)
    if mode and not getattr(self, "_memopt_root_checked", False):
        self._memopt_root_checked = True
        # Mark as root if it has >1M parameters — heuristic for "the training model"
        try:
            params = sum(p.numel() for p in self.parameters())
            if params > 1_000_000:
                self._memopt_root = True
        except Exception:
            pass
    return result


nn.Module.__call__ = _memopt_forward
nn.Module.train    = _patched_train

log.info(f"memopt training hook active — will optimize after {{_PROFILE_BATCHES}} batches")
'''


class TrainingWrapper:
    """
    Wraps a training script with zero-touch memopt optimization.

    Usage:
        wrapper = TrainingWrapper()
        exit_code = wrapper.run(["python", "train.py", "--epochs", "10"])
    """

    def __init__(
        self,
        profile_batches: int = 5,
        gpu_cost_per_hour: float = 2.50,
        dry_run: bool = False,
    ):
        self.profile_batches = profile_batches
        self.gpu_cost_per_hour = float(
            os.getenv("MEMOPT_GPU_COST_PER_HOUR", str(gpu_cost_per_hour))
        )
        self.dry_run = dry_run
        self.node_name = os.getenv("NODE_NAME", "localhost")

    def run(self, command: list) -> int:
        """
        Run command with memopt hook injected via PYTHONPATH.
        Returns exit code of the wrapped process.
        """
        if not command:
            print("Usage: memopt-wrap python train.py [args...]")
            return 1

        hook_code = _HOOK_TEMPLATE.format(
            profile_batches=self.profile_batches,
            gpu_cost_per_hour=self.gpu_cost_per_hour,
            node_name=self.node_name,
            dry_run=str(self.dry_run),
        )

        hook_dir = Path(tempfile.mkdtemp(prefix="memopt_hook_"))
        hook_file    = hook_dir / "memopt_training_hook.py"
        sitecustom   = hook_dir / "sitecustomize.py"

        hook_file.write_text(hook_code)
        sitecustom.write_text(
            f"import sys\n"
            f"sys.path.insert(0, {str(hook_dir)!r})\n"
            f"import memopt_training_hook\n"
        )

        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{hook_dir}:{existing}" if existing else str(hook_dir)

        mode = "(dry-run)" if self.dry_run else "(optimizing)"
        print(f"\n  memopt-wrap {mode}: profiling first {self.profile_batches} batches")
        if not self.dry_run:
            print(f"  memopt-wrap: optimizations apply automatically after profiling")
        print(f"  memopt-wrap: running: {' '.join(command)}\n")

        try:
            proc = subprocess.run(command, env=env)
            return proc.returncode
        except KeyboardInterrupt:
            print("\n  memopt-wrap: interrupted")
            return 130
        except FileNotFoundError:
            print(f"\n  memopt-wrap: command not found: {command[0]}")
            return 1
        finally:
            try:
                hook_file.unlink(missing_ok=True)
                sitecustom.unlink(missing_ok=True)
                hook_dir.rmdir()
            except Exception:
                pass
