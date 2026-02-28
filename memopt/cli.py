#!/usr/bin/env python3
"""
memopt CLI - Command Line Interface

Usage:
    memopt optimize --model path/to/model.pt --input-shape 8,512,1024
    memopt profile --model path/to/model.pt
    memopt analyze --model path/to/model.pt --input-shape 8,512,1024 --format html
    memopt agent  --model path/to/model.pt --input-shape 1,2048 --target 2.0
    memopt info
    memopt sessions
    memopt daemon start|stop|status|logs
"""

import argparse
import os
import sys
import json
from pathlib import Path


def cmd_info(args):
    """Show memopt and GPU info."""
    from memopt.profiler import api

    print("=" * 50)
    print("MEMOPT INFO")
    print("=" * 50)
    print(f"Version: {api.__version__}")
    print(f"API Contract: {api.__api_version__}")
    print()

    gpu_info = api.get_gpu_info()
    print("GPU:")
    for key, value in gpu_info.items():
        print(f"  {key}: {value}")


def cmd_sessions(args):
    """List saved optimization sessions."""
    from memopt.profiler import api

    sessions = api.list_sessions(args.model if hasattr(args, 'model') else None)

    if not sessions:
        print("No saved sessions found.")
        return

    print("=" * 70)
    print(f"{'Model':<20} {'Session':<15} {'Speedup':<10} {'Committed':<10}")
    print("=" * 70)

    for s in sessions:
        print(f"{s.get('model_name', 'unknown'):<20} "
              f"{s.get('session_id', ''):<15} "
              f"{s.get('total_speedup', 1.0):.3f}x     "
              f"{s.get('committed_count', 0):<10}")


def cmd_optimize(args):
    """Optimize a saved model."""
    import torch
    from memopt.profiler import api

    print(f"Loading model from: {args.model}")

    # Parse input shape
    shape = [int(x) for x in args.input_shape.split(",")]
    print(f"Input shape: {shape}")

    # Load model
    try:
        model = torch.load(args.model)
    except Exception as e:
        # Try loading as state dict
        print(f"Trying to load as state dict: {e}")
        model = torch.jit.load(args.model)

    if torch.cuda.is_available():
        model = model.cuda()
        sample = torch.randn(*shape).cuda()
    else:
        sample = torch.randn(*shape)

    # Optimize
    print("\nOptimizing...")
    model, session = api.optimize(model, sample, verbose=True)

    print(f"\nResults saved to: ~/.memopt/sessions/")


def cmd_profile(args):
    """Profile a model without optimization."""
    import torch
    from memopt.profiler import api

    print(f"Loading model from: {args.model}")

    shape = [int(x) for x in args.input_shape.split(",")]

    model = torch.load(args.model)
    if torch.cuda.is_available():
        model = model.cuda()

    snapshot = api.profile(
        model,
        lambda: torch.randn(*shape).cuda() if torch.cuda.is_available() else torch.randn(*shape),
        num_iterations=args.iterations
    )

    print("\n" + "=" * 50)
    print("PROFILE RESULTS")
    print("=" * 50)
    print(f"Total GPU Time: {snapshot.total_gpu_time_ms:.2f} ms")
    print(f"Memory-Bound: {snapshot.memory_bound_pct:.1f}%")


def cmd_analyze(args):
    """Analyze a model and generate optimization report with ROI."""
    import torch
    from memopt.workflows import AnalyzeWorkflow

    print(f"Loading model from: {args.model}")

    # Parse input shape
    shape = [int(x) for x in args.input_shape.split(",")]
    print(f"Input shape: {shape}")

    # Load model
    try:
        model = torch.load(args.model, weights_only=False)
    except Exception as e:
        # Try loading as JIT model
        print(f"Trying to load as TorchScript: {e}")
        model = torch.jit.load(args.model)

    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        sample = torch.randn(*shape).cuda()
    else:
        sample = torch.randn(*shape)

    # Get model name from file path
    model_name = Path(args.model).stem

    # Create workflow with fleet parameters
    workflow = AnalyzeWorkflow(
        gpu_count=args.gpu_count,
        gpu_cost_per_hour=args.gpu_cost,
        verbose=True
    )

    # Run analysis
    result = workflow.analyze(
        model=model,
        sample_input=sample,
        model_name=model_name,
        output_format=args.format,
        output_path=args.output,
        num_iterations=args.iterations
    )

    if not result.success:
        print(f"\nAnalysis failed: {result.error}")
        sys.exit(1)

    # Print output based on format
    if args.format == 'text' and result.output_text:
        print(result.output_text)
    elif args.format == 'json' and result.output_json:
        print(result.output_json)
    elif args.format == 'html' and result.output_html:
        if args.output:
            print(f"\nHTML report saved to: {args.output}")
        else:
            # Save to default location
            output_path = Path(f"./{model_name}_report.html")
            output_path.write_text(result.output_html)
            print(f"\nHTML report saved to: {output_path}")

    # Print ROI summary
    if result.roi_report:
        print(result.roi_report)


def cmd_agent(args):
    """Run autonomous multi-round optimization agent."""
    import torch
    from memopt.agent import MemoptAgent

    print(f"Loading model from: {args.model}")
    shape = [int(x) for x in args.input_shape.split(",")]

    try:
        model = torch.load(args.model, weights_only=False)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # Build sample input — reuse same strategy as server._build_sample_input()
    sample_input = _agent_build_sample(model, shape, device)

    agent = MemoptAgent(
        target_speedup=args.target,
        max_rounds=args.max_rounds,
    )

    print(f"\nStarting agent | target={args.target:.2f}x | max_rounds={args.max_rounds}")
    report = agent.run(model, sample_input)

    # Pretty output matching the spec
    print("\n" + "=" * 64)
    for r in report.rounds:
        committed_str = ", ".join(r.candidates_committed)  or "-"
        rolled_str    = ", ".join(r.candidates_rolled_back) or "-"
        stop_str      = f"\n  STOP: {r.stop_reason}" if r.stop_reason else ""
        print(
            f"Round {r.round_number}: {r.bottleneck_type} (conf={r.confidence:.2f}) "
            f"→ committed=[{committed_str}]  rolled=[{rolled_str}]  "
            f"cumulative={r.cumulative_speedup:.2f}x{stop_str}"
        )
    print("=" * 64)
    gap = abs(report.target_speedup - report.final_speedup)
    print(
        f"\nFinal:  {report.final_speedup:.2f}x  |  "
        f"Target: {report.target_speedup:.2f}x  |  "
        f"Gap: {gap:.2f}x"
    )
    print(f"Ceiling: {report.honest_ceiling}")

    if report.optimized_model is not None:
        out_path = args.output or (Path(args.model).stem + "_optimized.pt")
        torch.save(report.optimized_model, out_path)
        print(f"Optimized model saved to: {out_path}")
    else:
        print("No optimization committed — original model unchanged.")


def _agent_build_sample(model, shape, device):
    """
    Build sample input dict for the agent CLI.
    Mirrors server._build_sample_input() strategy without importing FastAPI.
    """
    import torch

    # Strategy 1: transformer input_ids (2D shape)
    if len(shape) == 2:
        for vocab in (30522, 50257, 32000):
            try:
                ids  = torch.randint(0, vocab, shape, device=device)
                mask = torch.ones(shape, dtype=torch.long, device=device)
                sample = {"input_ids": ids, "attention_mask": mask}
                with torch.no_grad():
                    model(**sample)
                return sample
            except Exception:
                pass

    # Strategy 2: float tensor with common kwarg names
    t = torch.randn(*shape, device=device)
    for kw in ("x", "input", "inputs", "hidden_states"):
        try:
            sample = {kw: t}
            with torch.no_grad():
                model(**sample)
            return sample
        except Exception:
            pass

    # Strategy 3: last-resort — try "input" as kwarg name and warn
    print(
        f"Warning: could not probe model with standard kwarg names for shape {shape}. "
        "Falling back to {'input': tensor}. If this fails, pass inputs manually."
    )
    return {"input": t}


def cmd_scan(args):
    """Scan GPU processes and print bottleneck report."""
    from memopt.daemon.cli import cmd_scan as _scan
    sys.exit(_scan(args))


def cmd_apply(args):
    """Apply optimizations to a running GPU process."""
    from memopt.daemon.cli import cmd_apply as _apply
    sys.exit(_apply(args))


def cmd_daemon(args):
    """Daemon management commands."""
    from memopt.daemon import MemoptDaemon, DaemonConfig

    # Load config if specified
    config = None
    if hasattr(args, 'config') and args.config:
        config = DaemonConfig.from_yaml(args.config)
    else:
        # Check default locations
        default_config = Path(os.path.expanduser("~/.memopt/daemon_config.yaml"))
        if default_config.exists():
            config = DaemonConfig.from_yaml(str(default_config))
        else:
            config = DaemonConfig()

    action = args.action

    if action == "start":
        daemon = MemoptDaemon(config)

        if daemon.is_running():
            print("Daemon is already running")
            return

        foreground = getattr(args, 'foreground', False)

        if foreground:
            print("Starting memopt daemon in foreground (Ctrl+C to stop)...")
            daemon.start(foreground=True)
        else:
            print("Starting memopt daemon...")
            daemon.start(foreground=False)
            print(f"Daemon started (PID: {os.getpid()})")
            print(f"Log file: {config.log_file}")
            print("Use 'memopt daemon status' to check status")
            print("Use 'memopt daemon stop' to stop")

            # Keep main thread alive
            import time
            try:
                while daemon._running:
                    time.sleep(1)
            except KeyboardInterrupt:
                daemon.stop()

    elif action == "stop":
        pid = MemoptDaemon.get_running_pid(config)
        if pid is None:
            print("Daemon is not running")
            return

        print(f"Stopping daemon (PID: {pid})...")
        if MemoptDaemon.stop_running(config):
            print("Daemon stopped")
        else:
            print("Failed to stop daemon")

    elif action == "status":
        pid = MemoptDaemon.get_running_pid(config)

        if pid is None:
            print("Daemon is not running")
            return

        print(f"Daemon is running (PID: {pid})")

        # Try to get detailed status from state file
        state_file = Path(os.path.expanduser(config.state_file))
        if state_file.exists():
            try:
                state = json.loads(state_file.read_text())
                print(f"Started: {state.get('start_time', 'unknown')}")
                print(f"Last update: {state.get('last_update', 'unknown')}")
                print(f"Profiled processes: {len(state.get('profiled_pids', {}))}")
            except Exception:
                pass

        # Show current GPU state
        try:
            from memopt.daemon import ProcessMonitor
            monitor = ProcessMonitor()
            states = monitor.get_gpu_states()

            print("\nGPU Status:")
            for state in states:
                print(f"  GPU {state.index}: {state.name}")
                print(f"    Memory: {state.memory_used_mb:.0f}/{state.memory_total_mb:.0f} MB")
                print(f"    Utilization: {state.utilization_pct:.0f}%")
                print(f"    Processes: {len(state.processes)}")
            monitor.shutdown()
        except Exception as e:
            print(f"Could not get GPU status: {e}")

    elif action == "logs":
        log_file = Path(os.path.expanduser(config.log_file))

        if not log_file.exists():
            print("No log file found")
            return

        lines = getattr(args, 'lines', 50)
        follow = getattr(args, 'follow', False)

        if follow:
            # Tail -f behavior
            import subprocess
            try:
                subprocess.run(["tail", "-f", str(log_file)])
            except KeyboardInterrupt:
                pass
        else:
            # Show last N lines
            with open(log_file) as f:
                all_lines = f.readlines()
                for line in all_lines[-lines:]:
                    print(line, end="")

    else:
        print(f"Unknown action: {action}")
        print("Use: memopt daemon start|stop|status|logs")


def main():
    parser = argparse.ArgumentParser(
        prog="memopt",
        description="GPU Memory Optimization Platform"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # info command
    info_parser = subparsers.add_parser("info", help="Show memopt and GPU info")
    info_parser.set_defaults(func=cmd_info)

    # sessions command
    sessions_parser = subparsers.add_parser("sessions", help="List saved sessions")
    sessions_parser.add_argument("--model", help="Filter by model name")
    sessions_parser.set_defaults(func=cmd_sessions)

    # optimize command
    opt_parser = subparsers.add_parser("optimize", help="Optimize a model")
    opt_parser.add_argument("--model", required=True, help="Path to saved model (.pt)")
    opt_parser.add_argument("--input-shape", required=True, help="Input shape (e.g., 8,512,1024)")
    opt_parser.set_defaults(func=cmd_optimize)

    # profile command
    prof_parser = subparsers.add_parser("profile", help="Profile a model")
    prof_parser.add_argument("--model", required=True, help="Path to saved model (.pt)")
    prof_parser.add_argument("--input-shape", required=True, help="Input shape (e.g., 8,512,1024)")
    prof_parser.add_argument("--iterations", type=int, default=5, help="Profile iterations")
    prof_parser.set_defaults(func=cmd_profile)

    # analyze command (full analysis with ROI)
    analyze_parser = subparsers.add_parser("analyze", help="Full analysis with ROI calculation")
    analyze_parser.add_argument("--model", required=True, help="Path to saved model (.pt)")
    analyze_parser.add_argument("--input-shape", required=True, help="Input shape (e.g., 8,512,1024)")
    analyze_parser.add_argument("--format", choices=["text", "json", "html", "all"],
                                default="text", help="Output format (default: text)")
    analyze_parser.add_argument("--output", "-o", help="Output file path")
    analyze_parser.add_argument("--gpu-count", type=int, default=1,
                                help="Number of GPUs in fleet for ROI calculation")
    analyze_parser.add_argument("--gpu-cost", type=float, default=3.00,
                                help="GPU cost per hour in dollars (default: $3.00)")
    analyze_parser.add_argument("--iterations", type=int, default=5,
                                help="Profile iterations (default: 5)")
    analyze_parser.set_defaults(func=cmd_analyze)

    # agent command
    agent_parser = subparsers.add_parser(
        "agent", help="Autonomous multi-round optimization agent"
    )
    agent_parser.add_argument("--model", required=True, help="Path to saved model (.pt)")
    agent_parser.add_argument(
        "--input-shape", required=True, help="Input shape (e.g., 1,2048 or 8,3,224,224)"
    )
    agent_parser.add_argument(
        "--target", type=float, default=2.0, help="Target speedup multiplier (default: 2.0)"
    )
    agent_parser.add_argument(
        "--max-rounds", type=int, default=10, help="Maximum optimization rounds (default: 10)"
    )
    agent_parser.add_argument(
        "--output", "-o", help="Output path for optimized model (default: <model>_optimized.pt)"
    )
    agent_parser.set_defaults(func=cmd_agent)

    # scan command
    scan_parser = subparsers.add_parser("scan", help="Scan GPU processes and show bottleneck report")
    scan_parser.add_argument("--json", action="store_true", help="Output as JSON")
    scan_parser.add_argument("--watch", "-w", action="store_true", help="Continuously refresh")
    scan_parser.add_argument("--interval", "-i", type=int, default=30, help="Refresh interval in seconds (default: 30)")
    scan_parser.add_argument("--sample-seconds", type=int, default=5, dest="sample_seconds",
                             help="Seconds to sample GPU utilization per process (default: 5)")
    scan_parser.set_defaults(func=cmd_scan)

    # apply command
    apply_parser = subparsers.add_parser("apply", help="Apply optimizations to a running GPU process")
    apply_parser.add_argument("--pid", type=int, required=True, help="PID of the GPU process to optimize")
    apply_parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                              help="Generate wrapper but do not restart the process")
    apply_parser.add_argument("--sample-seconds", type=int, default=5, dest="sample_seconds",
                              help="Seconds to sample utilization (default: 5)")
    apply_parser.set_defaults(func=cmd_apply)

    # daemon command
    daemon_parser = subparsers.add_parser("daemon", help="Daemon management")
    daemon_parser.add_argument("action", choices=["start", "stop", "status", "logs"],
                               help="Daemon action")
    daemon_parser.add_argument("--config", help="Path to config file")
    daemon_parser.add_argument("--foreground", "-f", action="store_true",
                               help="Run in foreground (start only)")
    daemon_parser.add_argument("--lines", "-n", type=int, default=50,
                               help="Number of log lines to show (logs only)")
    daemon_parser.add_argument("--follow", "-F", action="store_true",
                               help="Follow log output (logs only)")
    daemon_parser.set_defaults(func=cmd_daemon)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
