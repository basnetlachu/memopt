"""
memopt-certify — standalone silicon certification CLI.

Usage:
  memopt-certify [--gpu 0] [--output cert.json]
                 [--baseline] [--compare]
                 [--format json|text]

Exit codes:
  0 — CERTIFIED
  1 — DEGRADED or certification failed
  2 — No GPU available
"""
import argparse
import json
import os
import sys


def get_cert_history_path() -> str:
    return os.path.expanduser("~/.memopt/baseline_cert.json")


def run_certification(gpu_idx: int = 0) -> dict:
    """
    Run silicon certification on the specified GPU.
    Returns the certificate dict.
    Raises RuntimeError if no GPU available.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU available")
    except ImportError:
        raise RuntimeError("PyTorch not installed")

    try:
        from memopt.kernels.certification import (
            SiliconCertification, CertificationConfig,
        )
    except ImportError as e:
        raise RuntimeError(f"memopt not installed: {e}")

    config = CertificationConfig(
        warmup_iterations=3,
        timed_iterations=10,
        dtypes=["fp16", "fp32"],
    )
    cert = SiliconCertification(config=config)
    result = cert.run()

    # Add power reading if pynvml available
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
        power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000
        result["power_watts_nvml"] = round(power, 1)
        pynvml.nvmlShutdown()
    except Exception:
        result["power_watts_nvml"] = None

    # ── Derive display-friendly fields from the real cert schema ────
    # SiliconCertificate persists correctness_tests[] + throughput_tests[]
    # + all_passed + issued_at. The CLI text formatter / exit-code logic
    # want flat keys, so synthesize them here. These are display-only —
    # they are NOT part of the signed payload, so adding them does not
    # invalidate the existing signature.
    correctness = result.get("correctness_tests", []) or []
    result["ops_total"]  = len(correctness)
    result["ops_passed"] = sum(1 for t in correctness if t.get("passed"))

    bw_pct = None
    for t in result.get("throughput_tests", []) or []:
        if t.get("name") == "memory_bandwidth":
            bw_pct = t.get("pct_of_peak")
            break
    if bw_pct is not None:
        result["bandwidth_pct_of_peak"] = round(float(bw_pct), 2)

    issued = result.get("issued_at")
    if issued is not None:
        from datetime import datetime, timezone
        result["timestamp"] = datetime.fromtimestamp(
            float(issued), tz=timezone.utc).isoformat()

    result["certificate_status"] = (
        "CERTIFIED" if result.get("all_passed") else "DEGRADED"
    )

    return result


def save_baseline(cert: dict) -> str:
    """Save cert as baseline for future comparisons."""
    path = get_cert_history_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cert, f, indent=2)
    return path


def load_baseline() -> dict:
    """Load saved baseline cert."""
    path = get_cert_history_path()
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def compare_with_baseline(current: dict, baseline: dict) -> dict:
    """Compare current cert with baseline."""
    if not baseline:
        return {"status": "no_baseline"}

    curr_bw = current.get("bandwidth_pct_of_peak", 0)
    base_bw = baseline.get("bandwidth_pct_of_peak", 0)

    drift_pct = 0.0 if base_bw == 0 else (base_bw - curr_bw) / base_bw * 100

    return {
        "baseline_bandwidth_pct": base_bw,
        "current_bandwidth_pct":  curr_bw,
        "drift_pct":              round(drift_pct, 2),
        "degraded":               drift_pct > 5.0,
        "baseline_date":          baseline.get("timestamp", "unknown"),
    }


def format_text_output(cert: dict, comparison: dict = None) -> str:
    """Human-readable cert output."""
    lines = []
    lines.append("=" * 55)
    lines.append("memopt Silicon Certification")
    lines.append("=" * 55)

    gpu = cert.get("gpu", cert.get("device_name", "unknown"))
    lines.append(f"GPU:          {gpu}")
    lines.append(f"Timestamp:    {cert.get('timestamp', 'N/A')}")

    ops_passed = cert.get("ops_passed", "?")
    ops_total  = cert.get("ops_total", 10)
    lines.append(f"Ops passed:   {ops_passed}/{ops_total}")

    bw = cert.get("bandwidth_pct_of_peak", "?")
    lines.append(f"Bandwidth:    {bw}% of peak")

    power = cert.get("power_watts_nvml")
    if power:
        lines.append(f"Power:        {power}W (NVML measured)")

    status = cert.get("certificate_status", "UNKNOWN")
    lines.append(f"Status:       {status}")

    sig = cert.get("signature_status", cert.get("signature", "unsigned"))
    if isinstance(sig, str) and len(sig) > 30:
        sig = sig[:30] + "..."
    lines.append(f"Signature:    {sig}")

    if comparison:
        lines.append("")
        lines.append("Drift from baseline:")
        lines.append(f"  Bandwidth:  {comparison.get('drift_pct', 0):.1f}%")
        lines.append(f"  Degraded:   {comparison.get('degraded', False)}")

    lines.append("=" * 55)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        prog="memopt-certify",
        description=(
            "memopt Silicon Certification — verifies GPU hardware "
            "correctness and produces a signed certificate."
        ),
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU index (default: 0)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save cert JSON to this path",
    )
    parser.add_argument(
        "--baseline", action="store_true",
        help="Save this cert as baseline for future comparisons",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare with saved baseline",
    )
    parser.add_argument(
        "--format", choices=["json", "text"], default="text",
        help="Output format (default: text)",
    )

    args = parser.parse_args()

    try:
        cert = run_certification(gpu_idx=args.gpu)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    comparison = None
    if args.compare:
        comparison = compare_with_baseline(cert, load_baseline())

    if args.baseline:
        path = save_baseline(cert)
        print(f"Baseline saved: {path}", file=sys.stderr)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(cert, f, indent=2)
        print(f"Certificate saved: {args.output}", file=sys.stderr)

    if args.format == "json":
        print(json.dumps(cert, indent=2))
    else:
        print(format_text_output(cert, comparison))

    status = cert.get("certificate_status", "UNKNOWN")
    sys.exit(0 if status == "CERTIFIED" else 1)


if __name__ == "__main__":
    main()
