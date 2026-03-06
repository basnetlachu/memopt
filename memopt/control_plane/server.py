"""
memopt control plane — FastAPI server.

Endpoints:
  POST /api/v1/report              ← nodes POST here every 60s  [auth required]
  GET  /api/v1/status              ← cluster summary             [auth required]
  GET  /api/v1/nodes               ← all nodes                   [auth required]
  GET  /api/v1/nodes/{name}        ← single node detail          [auth required]
  GET  /api/v1/events              ← event history               [auth required]
  GET  /api/v1/alerts              ← drift alerts (active by default) [auth required]
  POST /api/v1/alerts/{id}/resolve ← mark alert resolved         [auth required]
  GET  /health                     ← liveness check              [NO auth]
  GET  /                           ← HTML dashboard              [auth required]

Run:
  python -m memopt.control_plane.server
  OR
  memopt control-plane start --port 8080
"""
import time
import json
import logging
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query, Security
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from memopt.control_plane.database import Database, NodeRecord, EventRecord
from memopt.alerts.alert_store import AlertStore
from memopt.fleet.intelligence import FleetIntelligence
from memopt.auth.api_key import get_or_create_key, verify_key, mask_key

log = logging.getLogger(__name__)

app = FastAPI(
    title="memopt Control Plane",
    description="Centralized management for memopt GPU optimization",
    version="1.0.0",
)

# Single database instance — initialized on startup
db = Database()
alert_store = AlertStore()
_fleet = FleetIntelligence(
    db_path=str(Path.home() / ".memopt" / "fleet.db"),
    auto_remediate=False,
)

# API key — loaded/created at module import time so the dependency closure captures it
_API_KEY: str = ""
_api_key_header = APIKeyHeader(name="X-Memopt-API-Key", auto_error=False)


def verify_api_key(key: str = Security(_api_key_header)) -> str:
    """FastAPI dependency — rejects requests without a valid API key."""
    if not verify_key(key, _API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key


@app.on_event("startup")
def startup():
    global _API_KEY
    _API_KEY = get_or_create_key()
    log.info("memopt control plane started (key: %s)", mask_key(_API_KEY))
    db.init()


# ─────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────

class WorkloadInfo(BaseModel):
    model_family: str
    mode: str
    speedup_applied: float = 1.0
    gpu_ids: List[int] = []


class OptimizationEventPayload(BaseModel):
    timestamp: float
    pid: int
    model_family: str
    gpu_ids: List[int]
    optimizations_applied: List[str]
    speedup_min: float
    speedup_max: float
    status: str
    dollar_saved_per_hour: float = 0.0


class NodeReport(BaseModel):
    """
    Payload each node sends every 60 seconds.
    Matches ZeroTouchDaemon.get_summary() output.
    """
    node_name: str
    timestamp: float
    gpu_count: int = 0
    total_vram_gb: float = 0.0
    active_processes: int = 0
    optimizations_applied: int = 0
    dollar_saved_today: float = 0.0
    dollar_saved_total: float = 0.0
    current_workloads: List[WorkloadInfo] = []
    new_events: List[OptimizationEventPayload] = []


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.post("/api/v1/report")
def receive_report(report: NodeReport, _: str = Security(verify_api_key)):
    """
    Receive node report.
    Called by ZeroTouchDaemon every scan cycle.
    Upserts node record, appends new events.
    """
    # Upsert node
    node_record = NodeRecord(
        node_name=report.node_name,
        last_seen=report.timestamp,
        gpu_count=report.gpu_count,
        total_vram_gb=report.total_vram_gb,
        active_processes=report.active_processes,
        optimizations_applied=report.optimizations_applied,
        dollar_saved_today=report.dollar_saved_today,
        dollar_saved_total=report.dollar_saved_total,
        status="online",
        current_workloads=json.dumps([w.dict() for w in report.current_workloads]),
    )
    db.upsert_node(node_record)

    # Insert new events
    for event in report.new_events:
        db.insert_event(EventRecord(
            id=None,
            timestamp=event.timestamp,
            node_name=report.node_name,
            pid=event.pid,
            model_family=event.model_family,
            gpu_ids=json.dumps(event.gpu_ids),
            optimizations=json.dumps(event.optimizations_applied),
            speedup_min=event.speedup_min,
            speedup_max=event.speedup_max,
            status=event.status,
            dollar_saved_per_hour=event.dollar_saved_per_hour,
        ))

    # Mark stale nodes offline
    db.mark_offline_nodes(timeout_seconds=180)

    return {"ok": True, "node": report.node_name}


@app.get("/api/v1/status")
def cluster_status(_: str = Security(verify_api_key)):
    """Cluster-wide summary. Used by dashboard and CLI."""
    summary = db.get_cluster_summary()
    summary["power_metrics"] = {
        "avg_power_baseline_watts":   _fleet.get_avg_power_baseline(),
        "avg_power_optimized_watts":  _fleet.get_avg_power_optimized(),
        "power_reduction_pct":        _fleet.get_power_reduction_pct(),
        "electricity_savings_24h_usd": _fleet.get_electricity_savings(),
    }
    return summary


@app.get("/api/v1/nodes")
def list_nodes(_: str = Security(verify_api_key)):
    """All nodes with current state."""
    nodes = db.get_all_nodes()
    for node in nodes:
        try:
            node["current_workloads"] = json.loads(
                node.get("current_workloads", "[]")
            )
        except Exception:
            node["current_workloads"] = []
    return {"nodes": nodes, "total": len(nodes)}


@app.get("/api/v1/nodes/{node_name}")
def get_node(node_name: str, _: str = Security(verify_api_key)):
    """Single node detail."""
    node = db.get_node(node_name)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node {node_name} not found")
    try:
        node["current_workloads"] = json.loads(node.get("current_workloads", "[]"))
    except Exception:
        node["current_workloads"] = []
    events = db.get_recent_events(limit=20, node_name=node_name)
    node["recent_events"] = events
    return node


@app.get("/api/v1/events")
def list_events(
    limit: int = Query(100, le=1000),
    node_name: Optional[str] = None,
    status: Optional[str] = None,
    _: str = Security(verify_api_key),
):
    """Optimization event history."""
    events = db.get_recent_events(
        limit=limit, node_name=node_name, status=status
    )
    return {"events": events, "total": len(events)}


@app.post("/api/v1/events")
def record_event(
    payload: dict,
    _: str = Security(verify_api_key),
):
    """Record an optimization event posted by memopt apply."""
    import json as _json
    record = EventRecord(
        id=None,
        timestamp=time.time(),
        node_name=payload.get("node", "unknown"),
        pid=int(payload.get("pid", 0)),
        model_family=payload.get("model", "unknown"),
        gpu_ids=_json.dumps([]),
        optimizations=_json.dumps(payload.get("optimizations", [])),
        speedup_min=float(payload.get("speedup") or 0.0),
        speedup_max=float(payload.get("speedup") or 0.0),
        status=payload.get("status", "applied"),
        dollar_saved_per_hour=float(payload.get("saved_per_hour") or 0.0),
    )
    db.insert_event(record)
    return {"ok": True}


@app.get("/api/v1/alerts")
def list_alerts(
    node_name: Optional[str] = None,
    severity: Optional[str] = None,
    include_resolved: bool = False,
    _: str = Security(verify_api_key),
):
    """
    Return drift alerts from both FleetIntelligence and legacy AlertStore.

    By default returns only unresolved alerts, newest first.
    Pass include_resolved=true to see full history.
    """
    # Fleet drift events (from daemon's FleetIntelligence SQLite)
    fleet_drifts = _fleet.get_recent_drift_events(limit=200)
    if node_name:
        fleet_drifts = [d for d in fleet_drifts if d.get("node_name") == node_name]
    if severity:
        fleet_drifts = [d for d in fleet_drifts if d.get("severity") == severity]

    # Legacy alert store (backward compat)
    if include_resolved:
        legacy = alert_store.get_all_alerts(limit=200)
    else:
        legacy = alert_store.get_active_alerts(node_name=node_name, severity=severity)

    all_alerts = fleet_drifts + legacy
    fleet_counts = alert_store.count_active_by_severity()
    for d in fleet_drifts:
        sev = d.get("severity", "info")
        fleet_counts[sev] = fleet_counts.get(sev, 0) + 1

    return {
        "alerts": all_alerts,
        "total": len(all_alerts),
        "active_counts": {
            "critical": fleet_counts.get("critical", 0),
            "warning":  fleet_counts.get("warning", 0),
            "info":     fleet_counts.get("info", 0),
        },
    }


@app.post("/api/v1/alerts/{alert_id}/resolve")
def resolve_alert(alert_id: int, _: str = Security(verify_api_key)):
    """
    Mark a drift alert as resolved.

    Call after verifying the issue and re-optimizing (or confirming false positive).
    """
    alert_store.resolve_alert(alert_id)
    return {"ok": True, "resolved_id": alert_id}


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Serve the HTML dashboard (public — no auth required)."""
    html_path = Path(__file__).parent / "dashboard.html"
    if html_path.exists():
        return html_path.read_text()
    return HTMLResponse("<h1>memopt Control Plane</h1><p>Dashboard not found.</p>")


def run_server(host: str = "0.0.0.0", port: int = 8080):
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_server()
