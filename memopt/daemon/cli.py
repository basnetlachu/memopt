"""
memopt daemon CLI — scan and apply subcommands.

Can be invoked as:
  python -m memopt.daemon.cli scan [--json] [--watch] [--interval N] [--sample-seconds N]
  python -m memopt.daemon.cli apply --pid PID [--dry-run]

Or via the main memopt CLI:
  memopt scan [options]
  memopt apply --pid PID [--dry-run]
"""
import argparse
import sys
import time
import logging

log = logging.getLogger(__name__)


def cmd_scan(args) -> int:
    """
    Scan all GPUs, profile processes, print actionable report.
    Returns exit code (0 = OK, 1 = error, 2 = no GPU processes).
    """
    from .scanner import GPUScanner
    from .process_inspector import ProcessInspector
    from .report import ScanReporter

    scanner  = GPUScanner()
    inspector = ProcessInspector()
    reporter  = ScanReporter()

    sample_seconds = getattr(args, "sample_seconds", 5)
    as_json        = getattr(args, "json", False)
    color          = not as_json and sys.stdout.isatty()

    while True:
        processes = scanner.scan()

        if not processes:
            if as_json:
                import json
                print(json.dumps({"processes": []}))
            else:
                print("No GPU processes found.")
            code = 2
        else:
            profiles = []
            for proc in processes:
                try:
                    profile = inspector.profile(
                        pid=proc.pid,
                        gpu_ids=proc.gpu_ids,
                        gpu_memory_mb=proc.gpu_memory_mb,
                        model_family=proc.model_family,
                        mode=proc.mode,
                        sample_seconds=sample_seconds,
                    )
                    profiles.append(profile)
                except Exception as e:
                    log.warning(f"Failed to profile PID {proc.pid}: {e}")

            if as_json:
                print(reporter.to_json(profiles))
            else:
                reporter.print_report(profiles, color=color)

            code = 0

        watch = getattr(args, "watch", False)
        if not watch:
            return code

        interval = getattr(args, "interval", 30)
        if not as_json:
            print(f"\n  [watch] refreshing in {interval}s  (Ctrl+C to stop)\n")
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            return 0


def cmd_apply(args) -> int:
    """
    Apply optimizations to a running GPU process.
    Returns exit code (0 = success, 1 = error/rollback).
    """
    from .scanner import GPUScanner
    from .process_inspector import ProcessInspector
    from .apply import ApplyEngine

    pid = args.pid
    dry_run = getattr(args, "dry_run", False)

    # Scan to find the process
    scanner = GPUScanner()
    processes = scanner.scan()
    target = next((p for p in processes if p.pid == pid), None)

    if target is None:
        print(f"Error: PID {pid} is not a known GPU process.")
        print("Run 'memopt scan' to see active GPU processes.")
        return 1

    # Profile it
    inspector = ProcessInspector()
    sample_seconds = getattr(args, "sample_seconds", 5)
    profile = inspector.profile(
        pid=target.pid,
        gpu_ids=target.gpu_ids,
        gpu_memory_mb=target.gpu_memory_mb,
        model_family=target.model_family,
        mode=target.mode,
        sample_seconds=sample_seconds,
    )

    mode = getattr(args, "mode", "auto")

    # Script-generating modes do not require prior recommendations
    if mode not in ("turbo", "draft", "turbo+draft") and not profile.recommendations:
        print(f"PID {pid}: No recommendations found (bottleneck={profile.bottleneck}).")
        return 0

    # Apply
    engine = ApplyEngine()
    result = engine.apply(profile, dry_run=dry_run, mode=mode)

    if result.success:
        opts = ", ".join(result.optimizations_applied)
        print(f"\n  Success  PID {result.pid_new}  applied=[{opts}]")
        return 0
    elif result.rolled_back:
        print(f"\n  Rolled back. Original process restarted.")
        return 1
    else:
        print(f"\n  Not applied: {result.error}")
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memopt.daemon.cli",
        description="memopt scan/apply — zero-touch GPU optimization",
    )
    sub = parser.add_subparsers(dest="command")

    # ── scan ──────────────────────────────────────────────────────────────────
    scan_p = sub.add_parser("scan", help="Scan GPU processes and show bottleneck report")
    scan_p.add_argument(
        "--json", action="store_true",
        help="Output as JSON instead of human-readable text",
    )
    scan_p.add_argument(
        "--watch", "-w", action="store_true",
        help="Continuously refresh (like watch)",
    )
    scan_p.add_argument(
        "--interval", "-i", type=int, default=30,
        help="Refresh interval in seconds when --watch is set (default: 30)",
    )
    scan_p.add_argument(
        "--sample-seconds", type=int, default=5, dest="sample_seconds",
        help="Seconds to sample GPU utilization per process (default: 5)",
    )
    scan_p.set_defaults(func=cmd_scan)

    # ── apply ─────────────────────────────────────────────────────────────────
    apply_p = sub.add_parser("apply", help="Apply optimizations to a running GPU process")
    apply_p.add_argument(
        "--pid", type=int, required=True,
        help="PID of the GPU process to optimize",
    )
    apply_p.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Generate wrapper but do not restart the process",
    )
    apply_p.add_argument(
        "--sample-seconds", type=int, default=5, dest="sample_seconds",
        help="Seconds to sample utilization before choosing optimizations (default: 5)",
    )
    apply_p.add_argument(
        "--mode",
        choices=["auto", "batch", "turbo", "draft", "turbo+draft"],
        default="auto",
        help=(
            "Optimization mode (float16 only, zero quality loss): "
            "auto = flash+compile (default); "
            "batch = optimal batch size wrapper; "
            "turbo = Turbo Engine continuous batching server (~5-6x); "
            "draft = Draft Acceleration (~1.8x latency); "
            "turbo+draft = both combined (~6-8x)"
        ),
    )
    apply_p.set_defaults(func=cmd_apply)

    return parser


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
