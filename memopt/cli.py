#!/usr/bin/env python3
"""
memopt CLI - Command Line Interface

Usage:
    memopt optimize --model path/to/model.pt --input-shape 8,512,1024
    memopt profile --model path/to/model.pt
    memopt analyze --model path/to/model.pt --input-shape 8,512,1024 --format html
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
