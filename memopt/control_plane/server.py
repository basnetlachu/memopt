"""
memopt control plane — FastAPI server.

Endpoints:
  POST /api/v1/report              ← nodes POST here every 60s  [auth required]
  GET  /api/v1/status              ← cluster summary             [auth required]
  GET  /api/v1/nodes               ← all nodes                   [auth required]
  GET  /api/v1/nodes/{name}        ← single node detail          [auth required]
  GET  /api/v1/events              ← event history               [auth required]
  GET  /health                     ← liveness check              [NO auth]
  GET  /                           ← HTML dashboard              [auth required]

Run:
  python -m memopt.control_plane.server
  OR
  memopt control-plane start --port 8080
"""
import os
import time
import json
import logging
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query, Security
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from memopt.control_plane.database import Database, NodeRecord, EventRecord
from memopt.auth.api_key import get_or_create_key, verify_key, mask_key

log = logging.getLogger(__name__)

app = FastAPI(
    title="memopt Control Plane",
    description="Centralized management for memopt GPU optimization",
    version="1.0.0",
)

# Single database instance — initialized on startup
db = Database()

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
    return db.get_cluster_summary()


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


# ─────────────────────────────────────────────
# EXECUTIVE DASHBOARD
# ─────────────────────────────────────────────

@app.get("/executive", response_class=HTMLResponse)
def executive_dashboard():
    """Serve the Executive ROI dashboard (public page — API calls require auth)."""
    html_path = Path(__file__).parent / "executive.html"
    if html_path.exists():
        return html_path.read_text()
    return HTMLResponse("<h1>Executive Dashboard</h1><p>Dashboard not found.</p>")


# ─────────────────────────────────────────────
# PRE-FLIGHT GRAPH ANALYSIS
# ─────────────────────────────────────────────

@app.post("/api/v1/preflight")
async def preflight_analysis(
    body: dict,
    _: str = Security(verify_api_key),
):
    """
    Run pre-flight static graph analysis on a model.
    Loads model structure only (no weights, no VRAM).
    Returns optimization opportunities before first run.

    Body: {"model_name": "gpt2", "gpu_index": 0}
    """
    from memopt.profiler.graph_analyzer import StaticGraphAnalyzer

    model_name = body.get("model_name", "")
    gpu_index  = int(body.get("gpu_index", 0))

    if not model_name:
        raise HTTPException(status_code=400, detail="model_name required")

    try:
        from transformers import AutoConfig, AutoModelForCausalLM
        import torch
        from memopt.profiler.roofline import RooflineProfiler

        config = AutoConfig.from_pretrained(model_name)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config)

        profiler = RooflineProfiler()
        hw       = profiler.profile_gpu(gpu_index)

        analyzer = StaticGraphAnalyzer()
        result   = analyzer.analyze(
            model=model,
            gpu_name=hw.gpu_name,
            ridge_point=hw.ridge_point_flops_per_byte,
        )

        return {
            "model_name":         result.model_name,
            "gpu_name":           result.gpu_name,
            "overall_bottleneck": result.overall_bottleneck.value,
            "overall_intensity":  result.overall_intensity,
            "total_params":       result.total_params,
            "top_opportunities":  result.top_opportunities,
            "preflight_config":   result.preflight_config,
            "estimated_speedup":  result.estimated_speedup,
            "analysis_time_ms":   result.analysis_time_ms,
        }

    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Missing dependency: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# OPTIMIZATION CERTIFICATES
# ─────────────────────────────────────────────

@app.get("/api/v1/certificates")
def list_certificates(
    limit: int = Query(50, le=500),
    model_name: Optional[str] = None,
    optimization_type: Optional[str] = None,
    _: str = Security(verify_api_key),
):
    """
    Return optimization certificates issued by this node (or a shared store).

    Each certificate is a signed audit record for a committed speedup.
    """
    from memopt.certificates import CertificateStore
    store = CertificateStore()
    certs = store.list_certs(
        limit=limit,
        model_name=model_name,
        optimization_type=optimization_type,
    )
    return {
        "certificates": [c.as_dict() for c in certs],
        "total": len(certs),
    }


# ─────────────────────────────────────────────
# SERVING ENGINE REGISTRY
# ─────────────────────────────────────────────

# In-memory registry: {"{host}:{port}": {host, port, model_path, engine_type, pid, registered_at}}
_serving_registry: dict = {}


class ServingEngineRegistration(BaseModel):
    host: str
    port: int
    model_path: str
    engine_type: str = "continuous_batching"
    pid: int = 0


@app.get("/api/v1/serving/status")
def serving_status(_: str = Security(verify_api_key)):
    """
    Return all registered serving engines.
    Engines self-register on startup via POST /api/v1/serving/register.
    """
    engines = list(_serving_registry.values())
    return {
        "engines": engines,
        "total": len(engines),
        "timestamp": time.time(),
    }


@app.post("/api/v1/serving/register", status_code=201)
def serving_register(body: ServingEngineRegistration):
    """
    Register a serving engine.
    Called by memopt-serve on startup when MEMOPT_CONTROL_PLANE env var is set.
    No auth required so engines can self-register without pre-sharing keys.
    """
    key = f"{body.host}:{body.port}"
    _serving_registry[key] = {
        "host":         body.host,
        "port":         body.port,
        "model_path":   body.model_path,
        "engine_type":  body.engine_type,
        "pid":          body.pid,
        "registered_at": time.time(),
    }
    log.info("Serving engine registered: %s (model=%s)", key, body.model_path)
    return {"ok": True, "key": key}


def run_server(host: str = "0.0.0.0", port: int = 8080):
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_server()
