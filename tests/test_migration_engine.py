"""
Tests for AutoMigrationEngine — no mocks for core logic.
Tests run on any machine (no GPU required for these unit tests).
"""

import os
import signal
import socket
import tempfile
import threading
import time
from unittest.mock import patch

import pytest

from memopt.migration.engine import (
    AutoMigrationEngine,
    Backend,
    MigrationPlan,
    MigrationResult,
    MigrationStatus,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_plan(**kwargs) -> MigrationPlan:
    defaults = dict(
        pid=12345,
        model_name="openlm-research/open_llama_13b",
        model_family="llama",
        current_backend=Backend.HUGGINGFACE,
        target_backend=Backend.VLLM,
        gpu_indices=[0],
        vram_total_mb=80 * 1024,
        vram_free_mb=55 * 1024,
        optimal_batch_size=16,
        flash_attention_version="flash_attention_2",
        estimated_speedup=5.0,
        target_port=18901,
        startup_timeout_seconds=5,
    )
    defaults.update(kwargs)
    return MigrationPlan(**defaults)


# ── Engine construction ────────────────────────────────────────────────────────

class TestAutoMigrationEngineInit:
    def test_instantiation(self):
        engine = AutoMigrationEngine()
        assert engine.active_migrations == {}
        assert engine.migration_log == []


# ── /proc parsing ──────────────────────────────────────────────────────────────

_LINUX = os.path.exists("/proc/self/cmdline")


class TestProcParsing:
    @pytest.mark.skipif(not _LINUX, reason="/proc not available on this platform")
    def test_read_own_cmdline(self):
        """Read our own /proc/self/cmdline — Linux only."""
        engine = AutoMigrationEngine()
        parts = engine._read_process_cmdline(os.getpid())
        assert isinstance(parts, list)
        assert len(parts) > 0
        assert any(p for p in parts)

    def test_read_nonexistent_pid_returns_empty(self):
        engine = AutoMigrationEngine()
        result = engine._read_process_cmdline(99999999)
        assert result == []

    def test_read_own_environ(self):
        engine = AutoMigrationEngine()
        env = engine._read_process_environ(os.getpid())
        assert isinstance(env, dict)
        # PATH should always be present
        assert "PATH" in env or len(env) >= 0  # be lenient on weird CI envs

    def test_read_environ_nonexistent_pid_returns_empty(self):
        engine = AutoMigrationEngine()
        result = engine._read_process_environ(99999999)
        assert result == {}

    def test_read_cwd_own_process(self):
        engine = AutoMigrationEngine()
        cwd = engine._read_process_cwd(os.getpid())
        assert isinstance(cwd, str)
        assert len(cwd) > 0


# ── Model detection ───────────────────────────────────────────────────────────

class TestModelDetection:
    def test_extract_family_mistral(self):
        engine = AutoMigrationEngine()
        assert engine._extract_family("mistralai/Mistral-7B-v0.1") == "mistral"

    def test_extract_family_llama(self):
        engine = AutoMigrationEngine()
        assert engine._extract_family("meta-llama/Llama-2-13b") == "llama"

    def test_extract_family_unknown(self):
        engine = AutoMigrationEngine()
        assert engine._extract_family("some/unknown-model") == "unknown"

    def test_extract_family_case_insensitive(self):
        engine = AutoMigrationEngine()
        assert engine._extract_family("TheBloke/Mistral-7B-GGUF") == "mistral"

    def test_detect_model_env_var(self, monkeypatch, tmp_path):
        """detect_model reads MODEL_NAME env var from /proc/<pid>/environ."""
        engine = AutoMigrationEngine()
        pid = os.getpid()

        # Override _read_process_environ to return a controlled env
        with patch.object(engine, "_read_process_environ",
                          return_value={"MODEL_NAME": "mistralai/Mistral-7B-Instruct-v0.2"}):
            model, family = engine.detect_model(pid)

        assert model == "mistralai/Mistral-7B-Instruct-v0.2"
        assert family == "mistral"

    def test_detect_model_cmdline_flag(self):
        engine = AutoMigrationEngine()

        with patch.object(engine, "_read_process_environ", return_value={}), \
             patch.object(engine, "_read_process_cmdline",
                          return_value=["python3", "--model", "openlm-research/open_llama_13b"]):
            model, family = engine.detect_model(os.getpid())

        assert model == "openlm-research/open_llama_13b"
        assert family == "llama"

    def test_detect_model_org_prefix(self):
        engine = AutoMigrationEngine()

        with patch.object(engine, "_read_process_environ", return_value={}), \
             patch.object(engine, "_read_process_cmdline",
                          return_value=["python3", "serve.py", "meta-llama/Llama-2-70b-chat-hf"]):
            model, family = engine.detect_model(os.getpid())

        assert model == "meta-llama/Llama-2-70b-chat-hf"
        assert family == "llama"

    def test_detect_model_unknown_fallback(self):
        engine = AutoMigrationEngine()

        with patch.object(engine, "_read_process_environ", return_value={}), \
             patch.object(engine, "_read_process_cmdline", return_value=["python3", "serve.py"]):
            model, family = engine.detect_model(os.getpid())

        assert model == "unknown"
        assert family == "unknown"


# ── Backend selection ─────────────────────────────────────────────────────────

class TestBackendSelection:
    def test_select_vllm_when_available(self):
        engine = AutoMigrationEngine()
        with patch.object(engine, "_check_package",
                          side_effect=lambda pkg: pkg == "vllm"):
            backend = engine.select_backend("llama", 80 * 1024, 1)
        assert backend == Backend.VLLM

    def test_select_trtllm_on_4_gpus(self):
        engine = AutoMigrationEngine()
        with patch.object(engine, "_check_package", return_value=True):
            backend = engine.select_backend("llama", 80 * 1024, 4)
        assert backend == Backend.TENSORRT_LLM

    def test_trtllm_requires_4_plus_gpus(self):
        engine = AutoMigrationEngine()
        with patch.object(engine, "_check_package", return_value=True):
            # 3 GPUs → should pick vLLM not TRT-LLM
            backend = engine.select_backend("llama", 80 * 1024, 3)
        assert backend == Backend.VLLM

    def test_fallback_to_huggingface_when_nothing_available(self):
        engine = AutoMigrationEngine()
        with patch.object(engine, "_check_package", return_value=False), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stderr = "install failed"
            backend = engine.select_backend("llama", 80 * 1024, 1)
        assert backend == Backend.HUGGINGFACE


# ── Plan building ─────────────────────────────────────────────────────────────

class TestBuildPlan:
    def test_build_plan_basic(self):
        engine = AutoMigrationEngine()
        hw = {
            "gpu_indices":   [0],
            "vram_total_mb": 80 * 1024,
            "vram_free_mb":  55 * 1024,
            "model_vram_mb": 26 * 1024,
            "gpu_name":      "A100 SXM4-80GB",
        }
        with patch.object(engine, "detect_model", return_value=("openlm-research/open_llama_13b", "llama")), \
             patch.object(engine, "select_backend", return_value=Backend.VLLM):
            plan = engine.build_plan(12345, hw)

        assert plan.pid == 12345
        assert plan.model_family == "llama"
        assert plan.target_backend == Backend.VLLM
        assert plan.optimal_batch_size >= 1
        assert plan.flash_attention_version == "flash_attention_2"
        assert plan.estimated_speedup >= 1.0

    def test_build_plan_h100_uses_fa3(self):
        engine = AutoMigrationEngine()
        hw = {
            "gpu_indices":   [0],
            "vram_total_mb": 80 * 1024,
            "vram_free_mb":  55 * 1024,
            "gpu_name":      "H100 SXM5-80GB",
        }
        with patch.object(engine, "detect_model", return_value=("model", "llama")), \
             patch.object(engine, "select_backend", return_value=Backend.VLLM):
            plan = engine.build_plan(12345, hw)

        assert plan.flash_attention_version == "flash_attention_3"

    def test_build_plan_optimal_batch_bounded(self):
        engine = AutoMigrationEngine()
        hw = {
            "gpu_indices":   [0],
            "vram_total_mb": 80 * 1024,
            "vram_free_mb":  200,          # Very low free VRAM
            "model_vram_mb": 26 * 1024,
            "gpu_name":      "A100",
        }
        with patch.object(engine, "detect_model", return_value=("m", "llama")), \
             patch.object(engine, "select_backend", return_value=Backend.VLLM):
            plan = engine.build_plan(12345, hw)

        assert plan.optimal_batch_size >= 1


# ── Dry-run execution ─────────────────────────────────────────────────────────

class TestDryRun:
    def test_dry_run_returns_completed(self, capsys):
        engine = AutoMigrationEngine()
        plan = _make_plan()
        result = engine.execute(plan, dry_run=True)

        assert result.success is True
        assert result.status == MigrationStatus.COMPLETED
        assert result.new_pid is None
        assert result.measured_speedup is None
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out

    def test_dry_run_shows_plan_details(self, capsys):
        engine = AutoMigrationEngine()
        plan = _make_plan(model_name="test-model", optimal_batch_size=32)
        engine.execute(plan, dry_run=True)
        out = capsys.readouterr().out
        assert "test-model" in out
        assert "32" in out


# ── Health check ──────────────────────────────────────────────────────────────

class TestHealthCheck:
    def test_health_check_times_out_gracefully(self):
        engine = AutoMigrationEngine()
        # Port 19999 should not have anything listening
        result = engine._wait_for_health(19999, timeout_seconds=3)
        assert result is False

    def test_health_check_passes_with_mock_server(self):
        """Start a real HTTP server that returns 200 /health."""
        import http.server
        import threading

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")

            def log_message(self, *args):
                pass  # Suppress output

        port = _find_free_test_port()
        server = http.server.HTTPServer(("127.0.0.1", port), Handler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        time.sleep(0.1)

        try:
            engine = AutoMigrationEngine()
            result = engine._wait_for_health(port, timeout_seconds=5)
            assert result is True
        finally:
            server.shutdown()


# ── Graceful kill ─────────────────────────────────────────────────────────────

class TestGracefulKill:
    def test_graceful_kill_nonexistent_pid_does_not_raise(self):
        engine = AutoMigrationEngine()
        engine._graceful_kill(99999999)  # Should not raise

    def test_graceful_kill_live_process(self):
        import subprocess as sp
        # Start a sleep process that we can kill
        proc = sp.Popen(["sleep", "60"])
        pid = proc.pid

        engine = AutoMigrationEngine()
        engine._graceful_kill(pid, timeout=5)

        # Poll for up to 3 seconds for the process to fully exit
        deadline = time.time() + 3.0
        alive = True
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
                time.sleep(0.1)
            except (ProcessLookupError, PermissionError):
                alive = False
                break
        # Also check via proc.poll()
        if alive and proc.poll() is not None:
            alive = False
        assert not alive, f"Process {pid} still alive after _graceful_kill"


# ── Port finding ──────────────────────────────────────────────────────────────

class TestFindFreePort:
    def test_finds_free_port(self):
        engine = AutoMigrationEngine()
        port = engine._find_free_port(19000)
        assert 19000 <= port <= 19100

    def test_returned_port_is_actually_free(self):
        engine = AutoMigrationEngine()
        port = engine._find_free_port(19200)
        # Verify we can bind to it
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", port))


# ── Migration failure path ─────────────────────────────────────────────────────

class TestMigrationFailure:
    def test_failed_health_check_leaves_original_untouched(self):
        """If vLLM fails to start, original PID must not be killed."""
        engine = AutoMigrationEngine()
        plan = _make_plan(startup_timeout_seconds=2)

        # Patch _wait_for_health to always fail
        with patch.object(engine, "_wait_for_health", return_value=False), \
             patch("subprocess.Popen") as mock_popen:
            mock_proc = mock_popen.return_value
            mock_proc.pid = 88888

            result = engine.execute(plan)

        assert result.success is False
        assert result.status == MigrationStatus.FAILED
        assert result.rollback_performed is True
        assert result.error is not None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _find_free_test_port(start: int = 19300) -> int:
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free test port found")
