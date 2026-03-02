"""
memopt cluster CLI commands.

  memopt control-plane start [--port 8080]
  memopt cluster status
  memopt cluster nodes
  memopt cluster events [--limit 20]
"""
import sys
import json
import urllib.request
import urllib.error
from memopt.auth.api_key import load_key


def get_control_plane_url() -> str:
    import os
    url = os.getenv("MEMOPT_CONTROL_PLANE", "http://localhost:8080")
    return url.rstrip("/")


def fetch(path: str) -> dict:
    url = get_control_plane_url() + path
    try:
        headers = {}
        key = load_key()
        if key:
            headers["X-Memopt-API-Key"] = key
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except urllib.error.URLError:
        print(f"Cannot reach control plane at {get_control_plane_url()}")
        print("Set MEMOPT_CONTROL_PLANE env var or start with:")
        print("  memopt control-plane start")
        sys.exit(1)


def cmd_start(args):
    print(f"Starting memopt control plane on port {args.port}...")
    from memopt.control_plane.server import run_server
    run_server(host="0.0.0.0", port=args.port)


def cmd_status(args):
    s = fetch("/api/v1/status")
    print(f"""
memopt Cluster Status
{'='*50}
Nodes:          {s['online_nodes']}/{s['total_nodes']} online
GPUs:           {s['total_gpus']} ({s['total_vram_gb']:.0f} GB VRAM)
Active jobs:    {s['active_processes']}
Optimizations:  {s['total_optimizations_applied']} applied (all time)

Savings:
  Last 24h:     ${s['dollar_saved_last_24h']:,.2f}
  Annual rate:  ${s['dollar_saved_per_year_estimate']:,.0f}/year
{'='*50}
""")


def cmd_nodes(args):
    data = fetch("/api/v1/nodes")
    nodes = data["nodes"]
    if not nodes:
        print("No nodes reporting to control plane yet.")
        return

    print(f"\n{'NODE':<20} {'STATUS':<10} {'GPUs':>5} "
          f"{'VRAM':>8} {'JOBS':>5} {'OPTS':>5} {'SAVED TODAY':>12}")
    print("-" * 75)
    for n in nodes:
        print(
            f"{n['node_name']:<20} "
            f"{n['status']:<10} "
            f"{n['gpu_count']:>5} "
            f"{n['total_vram_gb']:>6.0f}GB "
            f"{n['active_processes']:>5} "
            f"{n['optimizations_applied']:>5} "
            f"${n['dollar_saved_today']:>10.2f}"
        )
    print()


def cmd_events(args):
    data = fetch(f"/api/v1/events?limit={args.limit}")
    events = data["events"]
    if not events:
        print("No optimization events yet.")
        return

    print(f"\n{'TIME':<10} {'NODE':<20} {'MODEL':<12} "
          f"{'STATUS':<12} {'SPEEDUP':<10} {'$/HR':>8}")
    print("-" * 75)
    import time
    for e in events:
        t = time.strftime("%H:%M:%S", time.localtime(e["timestamp"]))
        print(
            f"{t:<10} "
            f"{e['node_name']:<20} "
            f"{e['model_family']:<12} "
            f"{e['status']:<12} "
            f"{e['speedup_min']:.1f}-{e['speedup_max']:.1f}x{'':<3} "
            f"${e['dollar_saved_per_hour']:>6.2f}"
        )
    print()
