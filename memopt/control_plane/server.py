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

# Gossip knowledge base — fleet-wide optimization recipe store
from memopt.fleet.gossip import GossipKnowledgeBase, OptimizationRecipe  # noqa: E402
_gossip_kb = GossipKnowledgeBase(
    db_path=str(Path.home() / ".memopt" / "gossip.db")
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


@app.get("/api/v1/executive/summary")
def executive_summary(
    hours: float = Query(24.0, ge=1.0, le=8760.0),
    _: str = Security(verify_api_key),
):
    """CFO-facing ROI summary — dollar savings, GPU utilisation uplift, fleet health."""
    savings = _fleet.calculate_savings(hours=hours)
    electricity_savings = _fleet.get_electricity_savings(
        hours=hours,
        kwh_cost=float(os.environ.get("ELECTRICITY_COST_PER_KWH", "0.12")),
    )
    power_reduction_pct = _fleet.get_power_reduction_pct(hours=hours)
    return {
        "period_hours":            hours,
        "period_start":            savings.period_start.isoformat() if savings.period_start else None,
        "period_end":              savings.period_end.isoformat()   if savings.period_end   else None,
        "total_nodes":             savings.total_nodes,
        "total_gpus":              savings.total_gpus,
        "optimized_gpus":          savings.optimized_gpus,
        "throughput_multiplier":   round(savings.throughput_multiplier, 3),
        "gpu_hours_saved":         round(savings.gpu_hours_saved, 2),
        "dollar_savings":          round(savings.dollar_savings, 2),
        "dollar_savings_annual":   round(savings.dollar_savings_annual, 0),
        "gpu_cost_per_hour":       savings.gpu_cost_per_hour,
        "electricity_savings_usd": round(electricity_savings, 2),
        "power_reduction_pct":     round(power_reduction_pct, 1),
        "top_savings_nodes":       savings.top_savings_nodes,
    }


@app.get("/api/v1/executive/carbon")
def executive_carbon(
    hours: float = Query(24.0, ge=1.0, le=8760.0),
    _: str = Security(verify_api_key),
):
    """Carbon and Green-AI metrics for ESG reporting."""
    baseline_w          = _fleet.get_avg_power_baseline(hours=hours)
    optimized_w         = _fleet.get_avg_power_optimized(hours=hours)
    power_reduction_pct = _fleet.get_power_reduction_pct(hours=hours)

    # kWh saved — per-GPU watt difference × GPU count × hours
    all_nodes  = db.get_all_nodes()
    total_gpus = sum(n.get("gpu_count", 0) for n in all_nodes) or 1
    watts_saved_per_gpu = max(baseline_w - optimized_w, 0.0)
    kwh_saved = (watts_saved_per_gpu * total_gpus * hours) / 1000.0

    # Carbon intensity — live WattTime or US average fallback
    kg_co2_per_kwh, carbon_source = _carbon_intensity()
    kg_co2_saved     = kwh_saved * kg_co2_per_kwh
    tonnes_co2_saved = kg_co2_saved / 1000.0

    return {
        "period_hours":        hours,
        "power_reduction_pct": round(power_reduction_pct, 1),
        "kwh_saved":           round(kwh_saved, 2),
        "kg_co2_saved":        round(kg_co2_saved, 2),
        "tonnes_co2_saved":    round(tonnes_co2_saved, 4),
        "carbon_source":       carbon_source,
        "equivalencies": {
            "cars_removed":    round(tonnes_co2_saved / 4.6,  2),
            "trees_planted":   round(kg_co2_saved     / 21,   1),
            "flights_avoided": round(kg_co2_saved     / 986,  2),
        },
        "annual_projection": {
            "kwh_saved":        round(kwh_saved        * 8760 / max(hours, 1), 1),
            "kg_co2_saved":     round(kg_co2_saved     * 8760 / max(hours, 1), 1),
            "tonnes_co2_saved": round(tonnes_co2_saved * 8760 / max(hours, 1), 2),
        },
    }


def _carbon_intensity() -> tuple:
    """Return (kg_CO2_per_kWh, source_name). Falls back to US average (0.386)."""
    user   = os.environ.get("WATTTIME_USERNAME", "")
    passwd = os.environ.get("WATTTIME_PASSWORD", "")
    region = os.environ.get("WATTTIME_REGION",   "")
    if user and passwd and region:
        try:
            import base64
            import urllib.request
            creds = base64.b64encode(f"{user}:{passwd}".encode()).decode()
            req = urllib.request.Request(
                "https://api.watttime.org/login",
                headers={"Authorization": f"Basic {creds}"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                token = json.loads(resp.read())["token"]
            req2 = urllib.request.Request(
                f"https://api.watttime.org/v3/signal-types/co2_moer/forecasts?"
                f"region={region}&signal_type=co2_moer",
                headers={"Authorization": f"Bearer {token}"},
            )
            with urllib.request.urlopen(req2, timeout=5) as resp2:
                data = json.loads(resp2.read())
            moer_lbs_mwh = data["data"][0]["value"]   # lbs CO2/MWh
            return moer_lbs_mwh * 0.453592 / 1000.0, "watttime"
        except Exception as exc:
            log.warning("WattTime API unavailable (%s), using US average", exc)
    return 0.386, "us_average"


# ─────────────────────────────────────────────
# GOSSIP KNOWLEDGE BASE
# ─────────────────────────────────────────────

@app.get("/api/v1/gossip/recipes")
def list_recipes(_: str = Security(verify_api_key)):
    """Return all known optimization recipes, newest first."""
    recipes = _gossip_kb.all_recipes()
    return {
        "count":   len(recipes),
        "recipes": [asdict(r) for r in recipes],
    }


@app.get("/api/v1/gossip/recipes/{recipe_hash}")
def get_recipe(recipe_hash: str, _: str = Security(verify_api_key)):
    """Return a single recipe by its 16-char hash. 404 if not found."""
    with sqlite3.connect(_gossip_kb.db_path) as conn:
        row = conn.execute(
            "SELECT * FROM recipes WHERE recipe_hash = ?",
            (recipe_hash,)
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Recipe not found")

    return {
        "recipe_hash":        row[0],  "model_family":     row[1],
        "gpu_family":         row[2],  "dtype":            row[3],
        "backend":            row[4],  "batch_size":       row[5],
        "max_model_len":      row[6],  "flash_attention":  row[7],
        "gpu_memory_util":    row[8],  "tensor_parallel":  row[9],
        "extra_flags":        json.loads(row[10] or "{}"),
        "measured_speedup":   row[11], "measured_tps":     row[12],
        "verified_by_node":   row[13], "verified_at":      row[14],
        "verification_count": row[15],
    }


@app.post("/api/v1/gossip/recipes", status_code=201)
def publish_recipe(recipe: dict, _: str = Security(verify_api_key)):
    """
    Node publishes a verified optimization recipe.
    If the recipe hash already exists, increment verification_count
    (another node confirmed the same config works).
    """
    final_count = _gossip_kb.upsert_publish(recipe)
    return {"status": "stored", "recipe_hash": recipe["recipe_hash"],
            "verification_count": final_count}


@app.get("/api/v1/gossip/stats")
def gossip_stats(_: str = Security(verify_api_key)):
    """Fleet-wide gossip knowledge base statistics."""
    return _gossip_kb.stats()


# ─────────────────────────────────────────────
# FLEET RISK (PREDICTIVE MIGRATION)
# ─────────────────────────────────────────────

@app.get("/api/v1/fleet/risk")
def fleet_risk(_: str = Security(verify_api_key)):
    """
    Current predictive migration risk scores.
    Reads the most recent node_metrics entries (last 5 minutes)
    to report how many GPUs are at elevated risk.
    """
    cutoff = time.time() - 300
    with sqlite3.connect(str(Path.home() / ".memopt" / "fleet.db")) as conn:
        try:
            rows = conn.execute("""
                SELECT node_name, gpu_index, timestamp, optimization_applied
                FROM node_metrics
                WHERE timestamp > ?
                ORDER BY timestamp DESC
                LIMIT 100
            """, (cutoff,)).fetchall()
        except sqlite3.OperationalError:
            rows = []

    return {
        "monitored_gpus": len(rows),
        "high_risk":      sum(1 for r in rows if not r[3]),
        "timestamp":      time.time(),
    }


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


def run_server(host: str = "0.0.0.0", port: int = 8080):
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_server()
