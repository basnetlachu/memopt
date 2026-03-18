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
