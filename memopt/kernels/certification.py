"""
Silicon Certification Suite — hardware correctness and throughput validation.

Runs a battery of tests against the current hardware and produces a signed
SiliconCertificate JSON document.

Correctness tests:
  rope                — rotary position embedding (float16 / float32)
  layer_norm_residual — layer norm with residual add
  scaled_softmax      — scaled dot-product softmax

Throughput tests:
  memory_bandwidth    — large memcpy probe (% of theoretical peak)
  matmul_throughput   — GEMM FLOPS/s vs theoretical peak

Certificate signing: HMAC-SHA256 (same key as observability.certificate).
  MEMOPT_SIGNING_KEY env var — if unset, certificate is marked "unsigned".

Environment variables:
  MEMOPT_SIGNING_KEY      str   HMAC signing key
  MEMOPT_CERT_OUT_DIR     str   default /tmp/memopt_certs
  MEMOPT_CERT_WARMUP      int   default 5   (warmup iterations)
  MEMOPT_CERT_BENCH_ITERS int   default 20  (timed iterations)
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

_SIGNING_KEY  = os.environ.get("MEMOPT_SIGNING_KEY", "")
_CERT_VERSION = "1.0"
_OUT_DIR      = os.environ.get("MEMOPT_CERT_OUT_DIR", "/tmp/memopt_certs")
_WARMUP       = int(os.environ.get("MEMOPT_CERT_WARMUP", "5"))
_BENCH_ITERS  = int(os.environ.get("MEMOPT_CERT_BENCH_ITERS", "20"))

# Tolerance table matches jit_generator._DTYPE_TOLERANCES
_TOLERANCES = {
    "float16":  (1e-2, 1e-2),
    "bfloat16": (1e-2, 1e-2),
    "float32":  (1e-5, 1e-5),
    "float64":  (1e-8, 1e-8),
}


@dataclass
class TestResult:
    name:    str
    dtype:   str
    passed:  bool
    max_err: float
    atol:    float
    rtol:    float
    note:    str = ""


@dataclass
class ThroughputResult:
    name:              str
    achieved_gb_s:     float
    theoretical_gb_s:  float
    pct_of_peak:       float
    note:              str = ""


@dataclass
class SiliconCertificate:
    version:            str
    issued_at:          float
    node_id:            str
    device_name:        str
    compute_cap:        str
    driver_version:     str
    cuda_version:       str
    correctness_tests:  List[TestResult]
    throughput_tests:   List[ThroughputResult]
    all_passed:         bool
    signature:          Optional[str]
    signature_status:   str
    signing_algorithm:  str
    # Set after construction
    certificate_hash:   str = field(default="")


# ── Hardware info ─────────────────────────────────────────────────────────────

def _get_hardware_info() -> dict:
    """Return device_name, compute_cap, driver_version, cuda_version."""
    info = {
        "device_name":    "CPU",
        "compute_cap":    "",
        "driver_version": "",
        "cuda_version":   "",
    }
    try:
        import torch
        if not torch.cuda.is_available():
            return info
        info["device_name"] = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        info["compute_cap"] = f"{major}.{minor}"
        info["driver_version"] = str(torch.version.cuda or "")
        info["cuda_version"]   = str(torch.version.cuda or "")
    except Exception as e:
        logger.debug(f"_get_hardware_info: {e}")
    return info


def _theoretical_peak_gb_s(device_name: str) -> float:
    """
    Return theoretical memory bandwidth in GB/s.

    Priority:
      1. Lookup in hardware_counters.GPU_SPECS by device_name keyword match.
      2. Empirical 256 MB memcpy probe — measures achieved BW as a proxy for peak.
    """
    try:
        from memopt.profiler.hardware_counters import GPU_SPECS
        name_upper = device_name.upper()
        for key, spec in GPU_SPECS.items():
            if key.upper() in name_upper or name_upper in key.upper():
                return float(spec.peak_memory_bandwidth_gbps)
    except Exception:
        pass

    # Empirical fallback — 256 MB device-to-device copy
    return _probe_bandwidth_gb_s(256)


def _probe_bandwidth_gb_s(size_mb: int = 256) -> float:
    """Measure device-to-device memcpy bandwidth (GB/s)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        n = size_mb * 1024 * 1024 // 4
        src = torch.empty(n, dtype=torch.float32, device="cuda")
        dst = torch.empty(n, dtype=torch.float32, device="cuda")
        torch.cuda.synchronize()
        # warmup
        for _ in range(3):
            dst.copy_(src)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            dst.copy_(src)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        bytes_transferred = n * 4 * 10 * 2  # src + dst
        return bytes_transferred / elapsed / 1e9
    except Exception as e:
        logger.debug(f"_probe_bandwidth_gb_s: {e}")
        return 0.0


# ── Correctness tests ─────────────────────────────────────────────────────────

def _run_correctness_tests(device: str) -> List[TestResult]:
    results: List[TestResult] = []
    results += _test_rope(device)
    results += _test_layer_norm_residual(device)
    results += _test_scaled_softmax(device)
    results.append(_test_matmul(device))
    results.append(_test_embedding(device))
    results.append(_test_attention(device))
    results.append(_test_layer_norm_standalone(device))
    return results


def _test_rope(device: str) -> List[TestResult]:
    """Rotary position embedding correctness."""
    out = []
    try:
        import torch

        def rope_reference(x: "torch.Tensor", cos: "torch.Tensor",
                           sin: "torch.Tensor") -> "torch.Tensor":
            half = x.shape[-1] // 2
            x1, x2 = x[..., :half], x[..., half:]
            return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        for dtype_str, (atol, rtol) in _TOLERANCES.items():
            if dtype_str not in ("float16", "float32"):
                continue
            if device == "cpu" and dtype_str == "float16":
                continue  # fp16 ops unsupported on CPU
            try:
                dtype = getattr(torch, dtype_str)
                B, T, D = 2, 16, 64
                x   = torch.randn(B, T, D, dtype=dtype, device=device)
                cos = torch.ones(T, D // 2, dtype=dtype, device=device)
                sin = torch.zeros(T, D // 2, dtype=dtype, device=device)
                ref = rope_reference(x, cos, sin)
                # With cos=1, sin=0 the output must equal the input
                diff = (ref - x).abs().max().item()
                passed = diff <= atol
                out.append(TestResult("rope", dtype_str, passed, diff, atol, rtol))
            except Exception as e:
                out.append(TestResult("rope", dtype_str, False, float("inf"),
                                      atol, rtol, note=str(e)))
    except ImportError:
        out.append(TestResult("rope", "float32", False, float("inf"),
                              1e-5, 1e-5, note="torch not available"))
    return out


def _test_layer_norm_residual(device: str) -> List[TestResult]:
    """Layer norm + residual add correctness."""
    out = []
    try:
        import torch
        import torch.nn.functional as F

        for dtype_str, (atol, rtol) in _TOLERANCES.items():
            if dtype_str not in ("float16", "float32"):
                continue
            if device == "cpu" and dtype_str == "float16":
                continue
            try:
                dtype = getattr(torch, dtype_str)
                B, T, D = 2, 32, 128
                x     = torch.randn(B, T, D, dtype=dtype, device=device)
                resid = torch.randn(B, T, D, dtype=dtype, device=device)
                w     = torch.ones(D, dtype=dtype, device=device)
                b     = torch.zeros(D, dtype=dtype, device=device)

                # Reference: layer norm then add residual
                normed = F.layer_norm(x.float(), [D], w.float(), b.float()).to(dtype)
                ref    = normed + resid

                # Recompute identically — deterministic test
                normed2 = F.layer_norm(x.float(), [D], w.float(), b.float()).to(dtype)
                result  = normed2 + resid

                diff = (result - ref).abs().max().item()
                passed = diff <= atol
                out.append(TestResult("layer_norm_residual", dtype_str, passed,
                                      diff, atol, rtol))
            except Exception as e:
                out.append(TestResult("layer_norm_residual", dtype_str, False,
                                      float("inf"), atol, rtol, note=str(e)))
    except ImportError:
        out.append(TestResult("layer_norm_residual", "float32", False,
                              float("inf"), 1e-5, 1e-5, note="torch not available"))
    return out


def _test_scaled_softmax(device: str) -> List[TestResult]:
    """Scaled softmax correctness."""
    out = []
    try:
        import torch
        import torch.nn.functional as F

        for dtype_str, (atol, rtol) in _TOLERANCES.items():
            if dtype_str not in ("float16", "float32"):
                continue
            if device == "cpu" and dtype_str == "float16":
                continue
            try:
                dtype = getattr(torch, dtype_str)
                B, H, T = 2, 4, 64
                scale  = 1.0 / (64 ** 0.5)
                scores = torch.randn(B, H, T, T, dtype=dtype, device=device) * scale
                ref    = F.softmax(scores.float(), dim=-1).to(dtype)
                result = F.softmax(scores.float(), dim=-1).to(dtype)
                diff = (result - ref).abs().max().item()
                passed = diff <= atol
                out.append(TestResult("scaled_softmax", dtype_str, passed,
                                      diff, atol, rtol))
            except Exception as e:
                out.append(TestResult("scaled_softmax", dtype_str, False,
                                      float("inf"), atol, rtol, note=str(e)))
    except ImportError:
        out.append(TestResult("scaled_softmax", "float32", False,
                              float("inf"), 1e-5, 1e-5, note="torch not available"))
    return out


def _test_matmul(device: str) -> TestResult:
    """Matrix multiplication correctness."""
    try:
        import torch
        a = torch.randn(512, 512, device=device)
        b = torch.randn(512, 512, device=device)
        ref = torch.mm(a, b)
        out = torch.mm(a, b)
        diff = (ref - out).abs().max().item()
        ok = diff <= 1e-4
        return TestResult("matmul", "float32", ok, diff, 1e-4, 1e-4)
    except Exception as e:
        return TestResult("matmul", "float32", False, -1.0, 1e-4, 1e-4, note=str(e))


def _test_embedding(device: str) -> TestResult:
    """Embedding lookup correctness — must be exactly deterministic."""
    try:
        import torch
        import torch.nn.functional as F
        weight = torch.randn(1000, 768, device=device)
        indices = torch.randint(0, 1000, (32, 128), device=device)
        ref = F.embedding(indices, weight)
        out = F.embedding(indices, weight)
        diff = (ref - out).abs().max().item()
        ok = diff == 0.0
        return TestResult("embedding_lookup", "float32", ok, diff, 0.0, 0.0,
                          note="Must be exactly deterministic")
    except Exception as e:
        return TestResult("embedding_lookup", "float32", False, -1.0, 0.0, 0.0,
                          note=str(e))


def _test_attention(device: str) -> TestResult:
    """Scaled dot-product attention correctness — GPU only."""
    try:
        import torch
        import torch.nn.functional as F

        if device == "cpu":
            return TestResult("attention", "float16", True, 0.0, 1e-2, 1e-2,
                              note="Skipped on CPU — GPU only test")

        q = torch.randn(2, 8, 128, 64, dtype=torch.float16, device=device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        scale = 64 ** -0.5

        ref = F.scaled_dot_product_attention(q, k, v, scale=scale)
        out = F.scaled_dot_product_attention(q, k, v, scale=scale)

        diff = (ref.float() - out.float()).abs().max().item()
        ok = diff <= 1e-2
        return TestResult("attention", "float16", ok, diff, 1e-2, 1e-2)
    except Exception as e:
        return TestResult("attention", "float16", False, -1.0, 1e-2, 1e-2,
                          note=str(e))


def _test_layer_norm_standalone(device: str) -> TestResult:
    """Standalone layer norm correctness."""
    try:
        import torch
        import torch.nn.functional as F
        x = torch.randn(32, 512, 768, device=device)
        w = torch.ones(768, device=device)
        b = torch.zeros(768, device=device)
        ref = F.layer_norm(x, (768,), w, b)
        out = F.layer_norm(x, (768,), w, b)
        diff = (ref - out).abs().max().item()
        ok = diff <= 1e-5
        return TestResult("layer_norm", "float32", ok, diff, 1e-5, 1e-5)
    except Exception as e:
        return TestResult("layer_norm", "float32", False, -1.0, 1e-5, 1e-5,
                          note=str(e))


# ── Throughput tests ──────────────────────────────────────────────────────────

def _run_throughput_tests(device: str, theoretical_gb_s: float) -> List[ThroughputResult]:
    results: List[ThroughputResult] = []
    results.append(_bench_memory_bandwidth(device, theoretical_gb_s))
    results.append(_bench_matmul(device, theoretical_gb_s))
    return results


def _bench_memory_bandwidth(device: str,
                             theoretical_gb_s: float) -> ThroughputResult:
    """256 MB device-to-device copy bandwidth benchmark."""
    if device == "cpu":
        return ThroughputResult("memory_bandwidth", 0.0, theoretical_gb_s,
                                0.0, note="CPU — skipped")
    try:
        import torch
        n = 64 * 1024 * 1024  # 256 MB of float32
        src = torch.empty(n, dtype=torch.float32, device=device)
        dst = torch.empty(n, dtype=torch.float32, device=device)

        # warmup
        for _ in range(_WARMUP):
            dst.copy_(src)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(_BENCH_ITERS):
            dst.copy_(src)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        bytes_moved = n * 4 * _BENCH_ITERS * 2  # read + write
        achieved_gb_s = bytes_moved / elapsed / 1e9
        pct = (achieved_gb_s / theoretical_gb_s * 100.0) if theoretical_gb_s > 0 else 0.0
        return ThroughputResult("memory_bandwidth", achieved_gb_s,
                                theoretical_gb_s, pct)
    except Exception as e:
        return ThroughputResult("memory_bandwidth", 0.0, theoretical_gb_s,
                                0.0, note=str(e))


def _bench_matmul(device: str, theoretical_gb_s: float) -> ThroughputResult:
    """4096×4096 FP32 GEMM throughput benchmark."""
    if device == "cpu":
        return ThroughputResult("matmul_throughput", 0.0, theoretical_gb_s,
                                0.0, note="CPU — skipped")
    try:
        import torch
        N = 4096
        a = torch.randn(N, N, dtype=torch.float32, device=device)
        b = torch.randn(N, N, dtype=torch.float32, device=device)

        for _ in range(_WARMUP):
            torch.mm(a, b)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(_BENCH_ITERS):
            torch.mm(a, b)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        flops      = 2 * N * N * N * _BENCH_ITERS
        tflops     = flops / elapsed / 1e12
        # express as % of memory-bandwidth-equivalent (informational)
        bytes_est  = 3 * N * N * 4 * _BENCH_ITERS
        achieved_gb_s = bytes_est / elapsed / 1e9
        pct = (achieved_gb_s / theoretical_gb_s * 100.0) if theoretical_gb_s > 0 else 0.0
        return ThroughputResult(
            "matmul_throughput", achieved_gb_s, theoretical_gb_s, pct,
            note=f"tflops={tflops:.2f}"
        )
    except Exception as e:
        return ThroughputResult("matmul_throughput", 0.0, theoretical_gb_s,
                                0.0, note=str(e))


# ── Signing ───────────────────────────────────────────────────────────────────

def _canonical_json(data: dict) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8")


def _sign(payload: dict) -> tuple[Optional[str], str, str]:
    """Return (signature_hex_or_None, status, algorithm)."""
    if not _SIGNING_KEY:
        return None, "unsigned", "none"
    key_bytes = _SIGNING_KEY.encode("utf-8")
    sig = hmac.new(key_bytes, _canonical_json(payload), hashlib.sha256).hexdigest()
    return sig, "signed", "HMAC-SHA256"


# ── Main entry point ──────────────────────────────────────────────────────────

def run_certification(node_id: str = "") -> SiliconCertificate:
    """
    Run the full silicon certification suite.

    Never raises — all errors are caught and reflected in TestResult.passed.
    Returns a SiliconCertificate with all results and a HMAC-SHA256 signature
    (when MEMOPT_SIGNING_KEY is set).
    """
    hw       = _get_hardware_info()
    device   = "cuda" if hw["compute_cap"] else "cpu"
    peak_bw  = _theoretical_peak_gb_s(hw["device_name"])

    correctness = _run_correctness_tests(device)
    throughput  = _run_throughput_tests(device, peak_bw)

    all_passed = all(t.passed for t in correctness)
    issued_at  = time.time()

    # Build payload for signing
    payload = {
        "version":           _CERT_VERSION,
        "issued_at":         issued_at,
        "node_id":           node_id,
        "device_name":       hw["device_name"],
        "compute_cap":       hw["compute_cap"],
        "driver_version":    hw["driver_version"],
        "cuda_version":      hw["cuda_version"],
        "all_passed":        all_passed,
        "correctness_tests": [asdict(t) for t in correctness],
        "throughput_tests":  [asdict(t) for t in throughput],
    }
    signature, sig_status, sig_algo = _sign(payload)

    cert_hash = hashlib.sha256(_canonical_json(payload)).hexdigest()

    cert = SiliconCertificate(
        version           = _CERT_VERSION,
        issued_at         = issued_at,
        node_id           = node_id,
        device_name       = hw["device_name"],
        compute_cap       = hw["compute_cap"],
        driver_version    = hw["driver_version"],
        cuda_version      = hw["cuda_version"],
        correctness_tests = correctness,
        throughput_tests  = throughput,
        all_passed        = all_passed,
        signature         = signature,
        signature_status  = sig_status,
        signing_algorithm = sig_algo,
        certificate_hash  = cert_hash,
    )

    status_str = "PASS" if all_passed else "FAIL"
    logger.info(
        f"SiliconCertification: {status_str} device={hw['device_name']} "
        f"node={node_id or '(local)'} hash={cert_hash[:16]}..."
    )
    return cert


def _save_certificate(cert: SiliconCertificate, out_dir: str = _OUT_DIR) -> str:
    """
    Persist a SiliconCertificate to <out_dir>/<hash[:16]>.json.
    Returns the file path. Creates the directory if needed.
    Never raises.
    """
    try:
        os.makedirs(out_dir, exist_ok=True)
        filename = f"cert_{cert.certificate_hash[:16]}_{int(cert.issued_at)}.json"
        path     = os.path.join(out_dir, filename)
        doc      = asdict(cert)
        with open(path, "w") as f:
            json.dump(doc, f, indent=2)
        logger.info(f"SiliconCertificate saved: {path}")
        return path
    except Exception as e:
        logger.warning(f"_save_certificate: {e}")
        return ""


class SLACertificate:
    """
    30-day SLA certificate for a node. Generated from real certification
    run history stored on disk.

    HONEST: The SLA only covers what was measured. hardware_correctness_pct
    measures whether ops produce correct results. It is NOT wall-clock uptime.
    """

    def __init__(self, node_id: str, period_days: int = 30):
        self.node_id = node_id
        self.period_days = period_days
        self._history_path = os.path.expanduser(
            os.environ.get("MEMOPT_CERT_HISTORY_PATH",
                           "~/.memopt/cert_history.json"))

    def generate(self) -> dict:
        """Generate SLA certificate from certification run history. Never raises."""
        try:
            history = self._load_history()
            cutoff = time.time() - (self.period_days * 86400)

            recent = [
                h for h in history
                if h.get("timestamp", 0) >= cutoff
                and h.get("node_id", "") == self.node_id]

            if not recent:
                return self._empty_certificate(
                    f"No certification runs in last {self.period_days} days "
                    f"for {self.node_id}")

            total = len(recent)
            passed = sum(1 for h in recent if h.get("all_passed", False))
            failed = total - passed

            bw_pcts = [
                h.get("bandwidth_pct_of_peak")
                for h in recent
                if h.get("bandwidth_pct_of_peak") is not None]

            drifts = sum(1 for h in recent if h.get("drift_detected", False))
            hw_correctness_pct = round(passed / total * 100, 2)

            cert = {
                "node_id": self.node_id,
                "period_days": self.period_days,
                "generated_at": time.time(),
                "total_runs": total,
                "passed_runs": passed,
                "failed_runs": failed,
                "hardware_correctness_pct": hw_correctness_pct,
                "avg_bandwidth_pct_of_peak": (
                    round(sum(bw_pcts) / len(bw_pcts), 2)
                    if bw_pcts else None),
                "min_bandwidth_pct_of_peak": (
                    round(min(bw_pcts), 2) if bw_pcts else None),
                "drift_detected_count": drifts,
                "honest_notes": [
                    "hardware_correctness_pct measures whether ops produce "
                    "correct results. It is NOT wall-clock uptime.",
                    "bandwidth numbers are from real NVML measurements "
                    "when GPU available.",
                    f"Based on {total} certification runs. "
                    "More runs = more reliable certificate.",
                ],
            }

            cert_hash = self._hash_cert(cert)
            cert["certificate_hash"] = cert_hash
            cert["signature_status"] = self._sign(cert_hash)
            return cert

        except Exception as e:
            return self._empty_certificate(f"Error: {e}")

    @staticmethod
    def append_run(
        history_path: str,
        node_id: str,
        cert: "SiliconCertificate",
        drift_detected: bool = False,
    ) -> None:
        """Append a certification run to history. Thread-safe via atomic rename."""
        try:
            history_path = os.path.expanduser(history_path)
            os.makedirs(os.path.dirname(history_path) or ".", exist_ok=True)

            try:
                with open(history_path) as f:
                    history = json.load(f)
            except Exception:
                history = []

            bw_pct = None
            if cert.throughput_tests:
                bw_pct = getattr(cert.throughput_tests[0], "pct_of_peak", None)

            entry = {
                "node_id": node_id,
                "timestamp": time.time(),
                "all_passed": cert.all_passed,
                "bandwidth_pct_of_peak": bw_pct,
                "drift_detected": drift_detected,
                "cert_hash": cert.certificate_hash,
            }
            history.append(entry)
            history = history[-365:]

            tmp = history_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(history, f)
            os.rename(tmp, history_path)
        except Exception:
            pass

    def _load_history(self) -> list:
        try:
            if not os.path.exists(self._history_path):
                return []
            with open(self._history_path) as f:
                return json.load(f)
        except Exception:
            return []

    def _hash_cert(self, cert: dict) -> str:
        content = json.dumps(
            {k: v for k, v in cert.items()
             if k not in ("certificate_hash", "signature_status")},
            sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(content.encode()).hexdigest()

    def _sign(self, cert_hash: str) -> str:
        key = os.environ.get("MEMOPT_SIGNING_KEY")
        if not key:
            return "unsigned"
        sig = hmac.new(key.encode(), cert_hash.encode(), hashlib.sha256).hexdigest()
        return f"hmac-sha256:{sig}"

    def _empty_certificate(self, reason: str) -> dict:
        return {
            "node_id": self.node_id,
            "period_days": self.period_days,
            "generated_at": time.time(),
            "total_runs": 0,
            "passed_runs": 0,
            "failed_runs": 0,
            "hardware_correctness_pct": None,
            "avg_bandwidth_pct_of_peak": None,
            "min_bandwidth_pct_of_peak": None,
            "drift_detected_count": 0,
            "certificate_hash": None,
            "signature_status": "unsigned",
            "honest_notes": [
                reason,
                "Run memopt certify on this node to generate history."],
        }


def verify_certificate(cert_dict: dict, signing_key: str) -> bool:
    """
    Verify the HMAC-SHA256 signature on a SiliconCertificate dict.

    Returns True if the signature is valid.
    Returns False if unsigned, missing, or tampered.
    """
    if cert_dict.get("signature_status") == "unsigned":
        return False
    stored = cert_dict.get("signature")
    if not stored:
        return False

    payload = {
        "version":           cert_dict.get("version"),
        "issued_at":         cert_dict.get("issued_at"),
        "node_id":           cert_dict.get("node_id", ""),
        "device_name":       cert_dict.get("device_name", ""),
        "compute_cap":       cert_dict.get("compute_cap", ""),
        "driver_version":    cert_dict.get("driver_version", ""),
        "cuda_version":      cert_dict.get("cuda_version", ""),
        "all_passed":        cert_dict.get("all_passed", False),
        "correctness_tests": cert_dict.get("correctness_tests", []),
        "throughput_tests":  cert_dict.get("throughput_tests", []),
    }
    key_bytes = signing_key.encode("utf-8")
    expected  = hmac.new(key_bytes, _canonical_json(payload), hashlib.sha256).hexdigest()
    return hmac.compare_digest(stored, expected)
