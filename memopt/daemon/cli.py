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
        print("\n  Rolled back. Original process restarted.")
        return 1
    else:
        print(f"\n  Not applied: {result.error}")
        return 1


def cmd_migrate(args) -> int:
    """
    Migrate a running GPU process to a faster backend (e.g., Turbo Engine).
    Returns exit code (0 = success/no-op, 1 = error).
    """
    from .scanner import GPUScanner
    try:
        from memopt.migration.engine import AutoMigrationEngine
    except ImportError:
        print(
            "The 'migrate' feature is not available in this build "
            "(memopt.migration module is not installed).",
        )
        return 1
    from memopt.profiler.roofline import RooflineProfiler

    pid = args.pid
    dry_run = getattr(args, "dry_run", False)

    scanner = GPUScanner()
    processes = scanner.scan()
    target = next((p for p in processes if p.pid == pid), None)

    if target is None:
        print(f"Error: PID {pid} is not a known GPU process.")
        print("Run 'memopt scan' to see active GPU processes.")
        return 1

    profiler = RooflineProfiler()
    gpu_id = target.gpu_ids[0] if target.gpu_ids else 0
    hw = profiler.profile_gpu(gpu_id)
    hw_dict = {
        "gpu_indices":   target.gpu_ids,
        "vram_total_mb": hw.vram_total_mb,
        "vram_free_mb":  hw.vram_free_mb,
        "model_vram_mb": target.gpu_memory_mb or 0,
        "gpu_name":      hw.gpu_name,
    }

    engine = AutoMigrationEngine()
    plan = engine.build_plan(pid, hw_dict)

    print(f"\n  PID {pid}  →  target: {plan.target_backend}")
    print(f"  Model   : {plan.model_name}")
    print(f"  Expected: {plan.estimated_speedup:.1f}x speedup")

    if plan.estimated_speedup < 1.5:
        print("  No beneficial migration available.")
        return 0

    if dry_run:
        print("  [dry-run] No changes made.")
        return 0

    result = engine.execute(plan)
    if result.success:
        print(
            f"\n  Migrated  PID {result.new_pid}  "
            f"backend={result.backend}  measured={result.measured_speedup:.2f}x"
        )
        return 0
    else:
        print(f"\n  Migration failed: {result.error}")
        return 1


def cmd_preflight(args) -> int:
    """
    Analyze model graph before running.
    Shows optimization opportunities and estimated speedup.
    No forward pass required.
    """
    from memopt.profiler.graph_analyzer import StaticGraphAnalyzer

    model_path = args.model
    gpu_index  = getattr(args, "gpu", 0)

    print(f"\n  Analyzing {model_path}...")
    print("  Loading model structure (no weights required)...")

    try:
        from transformers import AutoConfig, AutoModelForCausalLM
        import torch

        config = AutoConfig.from_pretrained(model_path)
        print("  Config loaded. Tracing graph...")

        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config)

        from memopt.profiler.roofline import RooflineProfiler
        profiler = RooflineProfiler()
        hw = profiler.profile_gpu(gpu_index)

        analyzer = StaticGraphAnalyzer()
        result = analyzer.analyze(
            model=model,
            gpu_name=hw.gpu_name,
            ridge_point=hw.ridge_point_flops_per_byte,
            dtype="float16",
        )

        print(analyzer.format_report(result))

    except ImportError:
        print("  transformers not installed. Install: pip install transformers")
        return 1
    except Exception as e:
        log.error("Preflight failed", exc_info=True)
        print(f"  Analysis failed: {e}")
        return 1

    return 0


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

    # ── migrate ───────────────────────────────────────────────────────────────
    migrate_p = sub.add_parser(
        "migrate",
        help="Migrate a running GPU process to a faster backend (Turbo Engine, TensorRT-LLM)",
    )
    migrate_p.add_argument(
        "--pid", type=int, required=True,
        help="PID of the GPU process to migrate",
    )
    migrate_p.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Plan migration but do not execute",
    )
    migrate_p.set_defaults(func=cmd_migrate)

    # ── preflight ─────────────────────────────────────────────────────────────
    preflight_p = sub.add_parser(
        "preflight",
        help="Analyze model graph before running — shows optimization opportunities and estimated speedup",
    )
    preflight_p.add_argument(
        "--model", required=True,
        help="HuggingFace model name or local path (e.g. mistralai/Mistral-7B-Instruct-v0.2)",
    )
    preflight_p.add_argument(
        "--gpu", type=int, default=0,
        help="GPU index to target (default: 0)",
    )
    preflight_p.set_defaults(func=cmd_preflight)

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
