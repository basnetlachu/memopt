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

    print("\nResults saved to: ~/.memopt/sessions/")


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
    print(
        "error: The 'agent' subcommand has been removed. "
        "Use 'memopt optimize' or 'memopt serve' instead.",
        file=sys.stderr,
    )
    sys.exit(2)


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


def cmd_certificates(args):
    """List or inspect optimization certificates."""
    try:
        from memopt.certificates import CertificateStore
    except ImportError:
        print(
            "The 'certificates' feature is not available in this build "
            "(memopt.certificates module is not installed).",
        )
        return 1
    import time as _time

    store = CertificateStore()

    if args.cert_command == "list":
        certs = store.list_certs(
            limit=args.limit,
            model_name=getattr(args, "model_name", None),
        )
        if not certs:
            print("No certificates found.")
            return
        print(f"{'ID':10}  {'Issued':20}  {'Model':20}  {'Type':20}  {'Speedup':>8}  {'Valid':5}")
        print("-" * 90)
        for c in certs:
            ts = _time.strftime("%Y-%m-%d %H:%M:%S", _time.localtime(c.issued_at))
            valid = "OK" if store.verify(c) else "FAIL"
            print(f"{c.certificate_id[:8]:10}  {ts:20}  {c.model_name[:20]:20}  "
                  f"{c.optimization_type[:20]:20}  {c.speedup_pct:>7.1f}%  {valid:5}")

    elif args.cert_command == "show":
        cert = store.get(args.id)
        if cert is None:
            print(f"Certificate not found: {args.id}")
            return
        import json
        d = cert.as_dict()
        d["signature_valid"] = store.verify(cert)
        print(json.dumps(d, indent=2))

    else:
        print("Usage: memopt certificates list|show")


def cmd_serve(args):
    """Start the memopt serving server (OpenAI-compatible HTTP interface)."""
    from memopt.serving.server import main as _serve_main
    import sys
    # Rebuild sys.argv so argparse inside server.main() sees the right flags
    argv = ["memopt-serve", "--model", args.model]
    if args.tokenizer:
        argv += ["--tokenizer", args.tokenizer]
    argv += ["--port", str(args.port), "--host", args.host]
    if args.device:
        argv += ["--device", args.device]
    argv += ["--max-batch", str(args.max_batch), "--max-seq", str(args.max_seq)]
    sys.argv = argv
    _serve_main()


def cmd_control_plane(args):
    """Control plane management."""
    from memopt.control_plane.cli import cmd_start
    cmd_start(args)


def cmd_cluster(args):
    """Cluster status/nodes/events commands."""
    from memopt.control_plane import cli as cp_cli
    if args.cluster_command == "status":
        cp_cli.cmd_status(args)
    elif args.cluster_command == "nodes":
        cp_cli.cmd_nodes(args)
    elif args.cluster_command == "events":
        cp_cli.cmd_events(args)
    else:
        print("Usage: memopt cluster status|nodes|events")


def cmd_scan(args):
    """Scan GPU processes and print bottleneck report."""
    from memopt.daemon.cli import cmd_scan as _scan
    sys.exit(_scan(args))


def cmd_apply(args):
    """Apply optimizations to a running GPU process."""
    from memopt.daemon.cli import cmd_apply as _apply
    sys.exit(_apply(args))


def cmd_certify(args):
    """Run silicon certification suite and print results."""
    from memopt.kernels.certification import run_certification, _save_certificate

    node_id  = getattr(args, "node_id", "") or ""
    out_dir  = getattr(args, "output_dir", None) or "/tmp/memopt_certs"
    no_save  = getattr(args, "no_save", False)

    print(f"Running Silicon Certification Suite (node={node_id or '(local)'}) ...")
    cert = run_certification(node_id=node_id)

    status = "PASS" if cert.all_passed else "FAIL"
    print(f"\nResult: {status}")
    print(f"Device: {cert.device_name}  CC={cert.compute_cap or 'N/A'}")
    print(f"Cert hash: {cert.certificate_hash[:32]}...")

    print("\nCorrectness tests:")
    for t in cert.correctness_tests:
        icon = "PASS" if t.passed else "FAIL"
        print(f"  [{icon}] {t.name:30s} dtype={t.dtype:10s} max_err={t.max_err:.2e}"
              + (f"  [{t.note}]" if t.note else ""))

    print("\nThroughput tests:")
    for t in cert.throughput_tests:
        print(f"  {t.name:30s} {t.achieved_gb_s:8.1f} GB/s  "
              f"({t.pct_of_peak:.1f}% of {t.theoretical_gb_s:.0f} GB/s peak)"
              + (f"  [{t.note}]" if t.note else ""))

    print(f"\nSignature: {cert.signature_status}")

    if not no_save:
        path = _save_certificate(cert, out_dir=out_dir)
        if path:
            print(f"Certificate saved: {path}")


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


def cmd_operator(args):
    """Start or query the Kubernetes operator."""
    import os
    import signal
    import threading

    op_cmd = getattr(args, "operator_command", None)
    if not op_cmd:
        print(
            "Usage: memopt operator start "
            "[--namespace NS] [--interval S] [--dry-run]")
        return

    from memopt.operator.controller import (
        MemoptOperator, K8S_AVAILABLE)

    if op_cmd == "start":
        if not K8S_AVAILABLE and \
                not getattr(args, "dry_run", False):
            print(
                "ERROR: kubernetes package not installed. "
                "Run: pip install kubernetes "
                "(or use --dry-run for validation).")
            sys.exit(1)

        namespace = getattr(args, "namespace", "") or \
            os.getenv("MEMOPT_NAMESPACE", "memopt")
        interval = float(
            getattr(args, "interval", 30.0) or 30.0)
        dry_run = bool(getattr(args, "dry_run", False))

        operator = MemoptOperator(
            namespace=namespace,
            reconcile_interval_s=interval,
            dry_run=dry_run)

        stop_event = threading.Event()

        def handle_signal(sig, frame):
            print("\nStopping operator...")
            stop_event.set()

        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

        if dry_run:
            print(
                f"Operator starting (dry-run): "
                f"namespace={namespace} interval={interval}s")
            # In dry-run the start() path skips Kubernetes connect;
            # run a few reconcile passes directly.
            try:
                operator._reconcile_all()
                print("Dry-run reconcile complete. "
                      "Press Ctrl+C to stop.")
                stop_event.wait()
            finally:
                print("Operator stopped.")
            return

        success = operator.start()
        if not success:
            print(
                "ERROR: operator failed to start "
                "(is the cluster reachable?)")
            sys.exit(1)

        try:
            print(
                f"Operator running: namespace={namespace} "
                f"interval={interval}s")
            print("Press Ctrl+C to stop")
            stop_event.wait()
        finally:
            operator.stop()
            print("Operator stopped.")
        return

    if op_cmd == "status":
        # Construct a detached instance purely to report K8s availability
        operator = MemoptOperator(dry_run=True)
        stats = operator.stats()
        print("memopt operator status:")
        for key, val in stats.items():
            print(f"  {key}: {val}")
        return


def cmd_pod_controller(args):
    """Start or manage the pod controller."""
    import os
    import signal
    import threading

    pod_cmd = getattr(args, "pod_command", None)
    if not pod_cmd:
        print("Usage: memopt pod-controller start --pod-id POD_ID")
        return

    if pod_cmd == "start":
        # Apply CLI args to environment
        if getattr(args, "pod_id", ""):
            os.environ["MEMOPT_POD_ID"] = args.pod_id
        if getattr(args, "control_plane", ""):
            os.environ["MEMOPT_CONTROL_PLANE_URL"] = args.control_plane
        if getattr(args, "redis_url", ""):
            os.environ["REDIS_URL"] = args.redis_url

        from memopt.vmm.pod_controller import PodController, PodConfig

        config = PodConfig.from_env()

        if not config.pod_id:
            print("ERROR: pod-id required. "
                  "Set --pod-id or MEMOPT_POD_ID env var.")
            sys.exit(1)

        controller = PodController(config)
        stop_event = threading.Event()

        def handle_signal(sig, frame):
            print("\nStopping pod controller...")
            stop_event.set()

        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

        try:
            controller.start()
            print(f"Pod controller started: pod={config.pod_id}")
            print("Press Ctrl+C to stop")
            stop_event.wait()
        finally:
            controller.stop()
            print("Pod controller stopped.")


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

    # certificates command
    cert_parser = subparsers.add_parser(
        "certificates", help="List and inspect optimization certificates"
    )
    cert_sub = cert_parser.add_subparsers(dest="cert_command")
    cert_list = cert_sub.add_parser("list", help="List recent certificates")
    cert_list.add_argument("--limit", type=int, default=50,
                           help="Max records to show (default: 50)")
    cert_list.add_argument("--model", dest="model_name", default=None,
                           help="Filter by model name")
    cert_show = cert_sub.add_parser("show", help="Show a single certificate by ID")
    cert_show.add_argument("id", help="Certificate ID or prefix")
    cert_parser.set_defaults(func=cmd_certificates, cert_command="list")

    # serve command
    serve_parser = subparsers.add_parser(
        "serve", help="Start OpenAI-compatible HTTP serving server"
    )
    serve_parser.add_argument("--model",     required=True, help="Path to model file")
    serve_parser.add_argument("--tokenizer", default=None,  help="HuggingFace tokenizer name/path")
    serve_parser.add_argument("--port",      type=int, default=8001, help="Port (default: 8001)")
    serve_parser.add_argument("--host",      default="0.0.0.0", help="Host (default: 0.0.0.0)")
    serve_parser.add_argument("--device",    default=None, help="Device: cuda or cpu")
    serve_parser.add_argument("--max-batch", type=int, default=8,    dest="max_batch",
                              help="Max batch size (default: 8)")
    serve_parser.add_argument("--max-seq",   type=int, default=2048, dest="max_seq",
                              help="Max sequence length (default: 2048)")
    serve_parser.set_defaults(func=cmd_serve)

    # control-plane command
    cp_parser = subparsers.add_parser("control-plane", help="Start the memopt control plane server")
    cp_sub = cp_parser.add_subparsers(dest="cp_command")
    cp_start = cp_sub.add_parser("start", help="Start the control plane server")
    cp_start.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    cp_start.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    cp_parser.set_defaults(func=cmd_control_plane)

    # cluster command
    cluster_parser = subparsers.add_parser("cluster", help="Cluster management commands")
    cluster_sub = cluster_parser.add_subparsers(dest="cluster_command")
    cluster_sub.add_parser("status", help="Show cluster-wide status and ROI")
    cluster_sub.add_parser("nodes", help="List all nodes and their state")
    events_p = cluster_sub.add_parser("events", help="Show recent optimization events")
    events_p.add_argument("--limit", type=int, default=20, help="Number of events to show (default: 20)")
    cluster_parser.set_defaults(func=cmd_cluster)

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

    # certify command (pillar 6 — silicon certification)
    certify_parser = subparsers.add_parser(
        "certify", help="Run silicon certification suite and produce a signed certificate"
    )
    certify_parser.add_argument(
        "--node-id", dest="node_id", default="",
        help="Node identifier to embed in the certificate (default: empty)"
    )
    certify_parser.add_argument(
        "--output-dir", dest="output_dir", default="/tmp/memopt_certs",
        help="Directory to save the certificate JSON (default: /tmp/memopt_certs)"
    )
    certify_parser.add_argument(
        "--no-save", dest="no_save", action="store_true",
        help="Print results but do not save the certificate to disk"
    )
    certify_parser.set_defaults(func=cmd_certify)

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

    # pod-controller command
    pod_parser = subparsers.add_parser(
        "pod-controller",
        help="Pod controller for multi-node oracle aggregation")
    pod_sub = pod_parser.add_subparsers(dest="pod_command")
    pod_start = pod_sub.add_parser(
        "start", help="Start the pod controller (foreground)")
    pod_start.add_argument(
        "--pod-id", dest="pod_id", default="",
        help="Pod identifier (or set MEMOPT_POD_ID)")
    pod_start.add_argument(
        "--control-plane", dest="control_plane", default="",
        help="Control plane URL (or set MEMOPT_CONTROL_PLANE_URL)")
    pod_start.add_argument(
        "--redis-url", dest="redis_url", default="",
        help="Redis URL (or set REDIS_URL)")
    pod_parser.set_defaults(func=cmd_pod_controller)

    # operator command
    operator_parser = subparsers.add_parser(
        "operator",
        help="Kubernetes operator controller loop")
    operator_sub = operator_parser.add_subparsers(
        dest="operator_command")
    op_start = operator_sub.add_parser(
        "start",
        help="Start the memopt Kubernetes operator")
    op_start.add_argument(
        "--namespace", dest="namespace", default="",
        help="Kubernetes namespace to watch "
             "(default: memopt)")
    op_start.add_argument(
        "--interval", dest="interval",
        type=float, default=30.0,
        help="Reconcile interval in seconds "
             "(default: 30)")
    op_start.add_argument(
        "--dry-run", dest="dry_run", action="store_true",
        help="Log actions without calling the K8s API")
    operator_sub.add_parser(
        "status",
        help="Show operator statistics")
    operator_parser.set_defaults(func=cmd_operator)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
