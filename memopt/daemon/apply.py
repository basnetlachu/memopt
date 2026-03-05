"""
Apply engine for memopt — generates an optimized wrapper and restarts a process.

Flow:
  1. Read the target process cmdline / cwd from /proc
  2. Generate a Python wrapper script that prepends memopt optimizations
  3. Show the diff and ask y/N
  4. SIGTERM the original process
  5. Start the wrapper in a subprocess
  6. Monitor for 60 s — rollback (kill wrapper, restart original) if it crashes
  7. Return ApplyResult

Never raises — errors are captured in ApplyResult.error.
"""
import os
import re
import sys
import signal
import subprocess
import tempfile
import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from .process_inspector import ProcessProfile, OPT_LABELS

log = logging.getLogger(__name__)

# How long (seconds) to watch the new process before declaring success
_MONITOR_SECONDS = 60


@dataclass
class ApplyResult:
    pid_original: int
    pid_new: Optional[int]
    success: bool
    rolled_back: bool
    optimizations_applied: List[str]
    wrapper_path: Optional[str]
    error: Optional[str] = None


class ApplyEngine:
    """
    Applies the top recommendations from a ProcessProfile to a live process.

    Usage:
        engine = ApplyEngine()
        result = engine.apply(profile, dry_run=False)
    """

    def apply(
        self,
        profile: ProcessProfile,
        dry_run: bool = False,
        mode: str = "auto",
    ) -> ApplyResult:
        """
        Apply optimizations to the process described by profile.

        Args:
            profile:  ProcessProfile from ProcessInspector.profile()
            dry_run:  If True, generate wrapper but do not execute anything.
            mode:     "auto" | "batch" | "vllm"
                      "vllm"  — generate vLLM server migration script
                      "batch" — generate wrapper tuned for optimal batch size
                      "auto"  — default existing behaviour

        Returns:
            ApplyResult — always returns, never raises.
        """
        try:
            if mode == "vllm":
                return self._apply_vllm(profile)
            return self._apply_inner(profile, dry_run, mode=mode)
        except Exception as e:
            log.error(f"ApplyEngine.apply failed for pid {profile.pid}: {e}", exc_info=True)
            return ApplyResult(
                pid_original=profile.pid,
                pid_new=None,
                success=False,
                rolled_back=False,
                optimizations_applied=[],
                wrapper_path=None,
                error=str(e),
            )

    # ── Private methods ──────────────────────────────────────────────────────

    def _apply_inner(
        self,
        profile: ProcessProfile,
        dry_run: bool,
        mode: str = "auto",
    ) -> ApplyResult:
        # 1. Read original process info
        cmdline, cwd, python_exe = self._read_proc_info(profile.pid)
        if not cmdline:
            return ApplyResult(
                pid_original=profile.pid,
                pid_new=None,
                success=False,
                rolled_back=False,
                optimizations_applied=[],
                wrapper_path=None,
                error=f"Cannot read /proc/{profile.pid}/cmdline — process may have exited.",
            )

        # 2. Generate wrapper script
        wrapper_code, applied_opts = self._generate_wrapper(
            cmdline=cmdline,
            cwd=cwd,
            python_exe=python_exe,
            profile=profile,
            mode=mode,
        )

        # 3. Write wrapper to a temp file
        wrapper_path = self._write_wrapper(wrapper_code, profile.pid)

        # 4. Print diff + prompt
        print()
        print(f"  PID {profile.pid}  →  wrapper: {wrapper_path}")
        print(f"  Optimizations to apply: {', '.join(OPT_LABELS.get(k, k) for k in applied_opts)}")
        print()
        print("  === WRAPPER PREVIEW ===")
        for line in wrapper_code.splitlines()[:30]:
            print(f"  {line}")
        if len(wrapper_code.splitlines()) > 30:
            print(f"  ... ({len(wrapper_code.splitlines()) - 30} more lines)")
        print()

        if dry_run:
            print("  [dry-run] No changes made.")
            return ApplyResult(
                pid_original=profile.pid,
                pid_new=None,
                success=True,
                rolled_back=False,
                optimizations_applied=applied_opts,
                wrapper_path=wrapper_path,
            )

        # 5. Ask for confirmation
        try:
            answer = input("  Apply? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"

        if answer != "y":
            print("  Aborted.")
            return ApplyResult(
                pid_original=profile.pid,
                pid_new=None,
                success=False,
                rolled_back=False,
                optimizations_applied=[],
                wrapper_path=wrapper_path,
                error="User declined.",
            )

        # 6. SIGTERM original process
        print(f"  Sending SIGTERM to PID {profile.pid}...")
        try:
            os.kill(profile.pid, signal.SIGTERM)
        except ProcessLookupError:
            log.warning(f"PID {profile.pid} already gone before SIGTERM")
        except PermissionError as e:
            return ApplyResult(
                pid_original=profile.pid,
                pid_new=None,
                success=False,
                rolled_back=False,
                optimizations_applied=[],
                wrapper_path=wrapper_path,
                error=f"Permission denied sending SIGTERM: {e}",
            )

        # Give process time to clean up
        time.sleep(2)

        # 7. Start wrapper
        env = os.environ.copy()
        env["MEMOPT_WRAPPER"] = "1"
        proc = subprocess.Popen(
            [python_exe, wrapper_path],
            cwd=cwd or None,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        print(f"  Started wrapper PID {proc.pid}")

        # 8. Monitor for _MONITOR_SECONDS
        pid_new, success, rolled_back = self._monitor(
            proc=proc,
            original_cmdline=cmdline,
            cwd=cwd,
            python_exe=python_exe,
        )

        if success:
            self._post_event(profile, applied_opts)

        return ApplyResult(
            pid_original=profile.pid,
            pid_new=pid_new,
            success=success,
            rolled_back=rolled_back,
            optimizations_applied=applied_opts if success else [],
            wrapper_path=wrapper_path,
            error=None if success else "Process crashed; rolled back.",
        )

    def _monitor(
        self,
        proc: subprocess.Popen,
        original_cmdline: str,
        cwd: str,
        python_exe: str,
    ):
        """Watch new process for _MONITOR_SECONDS. Roll back on crash."""
        deadline = time.time() + _MONITOR_SECONDS
        print(f"  Monitoring for {_MONITOR_SECONDS}s...")

        while time.time() < deadline:
            ret = proc.poll()
            if ret is not None:
                # Process exited — crash
                stderr_out = b""
                try:
                    stderr_out = proc.stderr.read(2000)
                except Exception:
                    pass
                log.error(
                    f"Wrapper PID {proc.pid} exited with code {ret}. "
                    f"stderr: {stderr_out.decode('utf-8', errors='replace')}"
                )
                # Rollback: restart original
                print(f"  Wrapper crashed (exit={ret}). Rolling back...")
                self._restart_original(original_cmdline, cwd, python_exe)
                return proc.pid, False, True
            time.sleep(2)

        print(f"  Process stable after {_MONITOR_SECONDS}s. Optimizations committed.")
        return proc.pid, True, False

    def _restart_original(self, cmdline: str, cwd: str, python_exe: str) -> None:
        """Restart the original command after a failed wrapper."""
        try:
            parts = cmdline.split()
            subprocess.Popen(
                parts,
                cwd=cwd or None,
                env=os.environ.copy(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            log.info(f"Restarted original process: {cmdline[:80]}")
        except Exception as e:
            log.error(f"Failed to restart original: {e}")

    def _read_proc_info(self, pid: int):
        """Read cmdline, cwd, python_exe from /proc/<pid>."""
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                raw = f.read().replace(b"\x00", b" ").decode("utf-8", errors="replace")
            cmdline = raw.strip()
        except Exception:
            return "", "", sys.executable

        try:
            cwd = os.readlink(f"/proc/{pid}/cwd")
        except Exception:
            cwd = ""

        # Determine python executable from cmdline
        parts = cmdline.split()
        python_exe = parts[0] if parts and "python" in parts[0].lower() else sys.executable

        return cmdline, cwd, python_exe

    def _generate_wrapper(
        self,
        cmdline: str,
        cwd: str,
        python_exe: str,
        profile: ProcessProfile,
        mode: str = "auto",
    ):
        """
        Generate a Python wrapper script that applies optimizations then
        exec()s the original command.

        Returns (wrapper_code: str, applied_opts: List[str]).
        """
        parts = cmdline.split()
        # Remove interpreter from args; wrapper will re-invoke via exec
        if parts and "python" in parts[0].lower():
            args = parts[1:]
        else:
            args = parts

        if mode == "batch":
            # Batch mode: add continuous_batch + kv_cache to top of recommendations
            recs = list(profile.recommendations)
            for opt in ("kv_cache", "continuous_batch"):
                if opt not in recs:
                    recs.insert(0, opt)
            applied_opts = recs[:3]
        else:
            applied_opts = list(profile.recommendations[:3])  # top 3 opts

        # Build preamble lines
        preamble_lines = [
            "# === memopt auto-generated wrapper ===",
            f"# Original PID: {profile.pid}",
            f"# Model family: {profile.model_family}",
            f"# Bottleneck: {profile.bottleneck}",
            f"# Applied: {', '.join(applied_opts)}",
            "",
            "import os, sys",
            "",
        ]

        env_lines = []

        if "flash_attention" in applied_opts:
            env_lines += [
                "# Enable Flash Attention 2",
                "os.environ.setdefault('MEMOPT_FLASH_ATTN', '1')",
            ]

        if "bf16" in applied_opts:
            env_lines += [
                "# Enable BF16 precision",
                "os.environ.setdefault('MEMOPT_BF16', '1')",
            ]

        if "int8" in applied_opts:
            env_lines += [
                "# Enable INT8 quantization hook",
                "os.environ.setdefault('MEMOPT_INT8', '1')",
            ]

        if "torch_compile" in applied_opts:
            env_lines += [
                "# Enable torch.compile",
                "os.environ.setdefault('MEMOPT_COMPILE', '1')",
            ]

        if "channels_last" in applied_opts:
            env_lines += [
                "# Enable channels-last memory format",
                "os.environ.setdefault('MEMOPT_CHANNELS_LAST', '1')",
            ]

        if "kv_cache" in applied_opts:
            env_lines += [
                "# Enable KV-cache reuse",
                "os.environ.setdefault('MEMOPT_KV_CACHE', '1')",
            ]

        if "continuous_batch" in applied_opts:
            env_lines += [
                "# Enable continuous batching",
                "os.environ.setdefault('MEMOPT_CONTINUOUS_BATCH', '1')",
            ]

        exec_lines = [
            "",
            "# Hand off to original script",
            f"sys.argv = {args!r}",
            f"os.chdir({cwd!r})" if cwd else "",
            "exec(open(sys.argv[0]).read())" if args else "# no script to exec",
        ]
        exec_lines = [l for l in exec_lines if l is not None]

        code = "\n".join(preamble_lines + env_lines + exec_lines) + "\n"
        return code, applied_opts

    def _detect_model_from_cmdline(self, pid: int) -> str:
        """
        Extract model name/path from a process cmdline.
        Tries --model <path>, HF org/name patterns, then falls back to empty string.
        """
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                raw = f.read().replace(b"\x00", b" ").decode("utf-8", errors="replace")
            cmdline = raw.strip()
        except Exception:
            return ""

        # --model /path/to/model  or  --model_name_or_path /path
        m = re.search(r'--model(?:_name_or_path)?[= ](\S+)', cmdline)
        if m:
            return m.group(1)

        # HF hub slug: org/model-name
        m = re.search(
            r'((?:mistralai|meta-llama|openlm-research|tiiuae|google|microsoft|'
            r'Qwen|deepseek-ai|01-ai|NousResearch)/[\w\-\.]+)',
            cmdline,
        )
        if m:
            return m.group(1)

        return ""

    def _apply_vllm(self, profile: ProcessProfile) -> ApplyResult:
        """
        Generate a ready-to-run vLLM server script for continuous-batching migration.
        Prints the script path and launch instructions; does NOT restart the process.
        Posts an event to the control plane on success.
        """
        model_path = self._detect_model_from_cmdline(profile.pid) or profile.model_family or "your-model"
        script_path = f"/tmp/memopt_vllm_{profile.pid}.py"

        script = (
            "#!/usr/bin/env python3\n"
            "# === memopt vLLM migration script (auto-generated) ===\n"
            f"# Original PID : {profile.pid}\n"
            f"# Model family : {profile.model_family}\n"
            f"# Model path   : {model_path}\n"
            "#\n"
            "# Run this script to replace the existing inference process with a\n"
            "# vLLM OpenAI-compatible server using continuous batching.\n"
            "# Throughput gain: ~5-6x vs single-request inference (float16, no quantization).\n"
            "#\n"
            "import subprocess, sys\n"
            "\n"
            "cmd = [\n"
            "    sys.executable, \"-m\", \"vllm.entrypoints.openai.api_server\",\n"
            f"    \"--model\", \"{model_path}\",\n"
            "    \"--dtype\", \"float16\",\n"
            "    \"--gpu-memory-utilization\", \"0.85\",\n"
            "    \"--max-num-seqs\", \"32\",\n"
            "    \"--enable-prefix-caching\",\n"
            "    \"--port\", \"8001\",\n"
            "]\n"
            "\n"
            "print(f\"Starting vLLM server: {' '.join(cmd)}\")\n"
            "subprocess.run(cmd)\n"
        )

        with open(script_path, "w") as f:
            f.write(script)
        os.chmod(script_path, 0o755)

        print()
        print(f"  vLLM migration script  →  {script_path}")
        print(f"  Model : {model_path}")
        print()
        print("  To migrate (stop current process first):")
        print(f"    python {script_path}")
        print()
        print("  Or run vLLM directly:")
        print(f"    python -m vllm.entrypoints.openai.api_server \\")
        print(f"      --model {model_path} \\")
        print(f"      --dtype float16 \\")
        print(f"      --gpu-memory-utilization 0.85 \\")
        print(f"      --max-num-seqs 32 \\")
        print(f"      --enable-prefix-caching \\")
        print(f"      --port 8001")
        print()
        print("  OpenAI-compatible endpoint: http://localhost:8001/v1/completions")

        self._post_event(profile, ["vllm_continuous_batching"])

        return ApplyResult(
            pid_original=profile.pid,
            pid_new=None,
            success=True,
            rolled_back=False,
            optimizations_applied=["vllm_continuous_batching"],
            wrapper_path=script_path,
        )

    def _post_event(self, profile: "ProcessProfile", applied_opts: List[str]) -> None:
        """Fire-and-forget POST to control plane after a successful apply."""
        try:
            import httpx
            control_plane = os.getenv("MEMOPT_CONTROL_PLANE", "http://localhost:8080")
            api_key = os.getenv("MEMOPT_API_KEY", "")
            httpx.post(
                f"{control_plane}/api/v1/events",
                headers={"X-Memopt-API-Key": api_key},
                json={
                    "node": os.getenv("NODE_NAME", "unknown"),
                    "pid": profile.pid,
                    "model": profile.model_family or "unknown",
                    "optimizations": applied_opts or [],
                    "status": "applied",
                    "speedup": getattr(profile, "speedup_ratio", None),
                    "saved_per_hour": getattr(profile, "saved_per_hour", None),
                },
                timeout=5.0,
            )
        except Exception:
            pass  # never block the apply flow

    def _write_wrapper(self, code: str, pid: int) -> str:
        """Write wrapper code to a temp file and return its path."""
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=f"_memopt_wrapper_{pid}.py",
            delete=False,
            prefix="/tmp/",
        )
        tmp.write(code)
        tmp.flush()
        tmp.close()
        os.chmod(tmp.name, 0o755)
        return tmp.name
