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
import threading
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

# Global oracle — top tier of three-tier hierarchy
_global_oracle = None

# Canary controller — progressive rollout orchestrator (Phase 7b)
_canary = None

# Heartbeat batcher — coalesces degradation-status writes at scale
_heartbeat_batcher = None


# ─────────────────────────────────────────────
# HEARTBEAT BATCHING (Phase 8a)
# ─────────────────────────────────────────────

class HeartbeatBatcher:
    """
    Batches node degradation-status writes.

    At 1M nodes with drift checks every few minutes, the naive
    "one INSERT/UPDATE per report" pattern saturates a single
    control-plane DB writer. HeartbeatBatcher queues writes in
    memory and flushes on a timer (default 1s) — the total data
    written is unchanged but the DB roundtrip count drops by
    the batch size.

    Thread-safe. Daemon thread flushes periodically; final flush
    on stop(). Never raises — flush errors are logged and counted
    so callers can't bring the endpoint down.
    """

    import os as _os
    BATCH_INTERVAL_S = float(_os.getenv(
        "MEMOPT_HEARTBEAT_BATCH_S", "1.0"))

    def __init__(self, db):
        self._db = db
        self._pending = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._flushed = 0
        self._errors = 0

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._flush_loop,
            name="heartbeat-batcher",
            daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        self._flush()  # final flush

    def record(
        self,
        node_id: str,
        healthy: bool,
        degraded: bool,
        drift_pct: float,
        reason: str,
    ) -> None:
        """Queue a heartbeat write. Returns immediately. Never raises."""
        try:
            entry = (
                node_id, healthy, degraded,
                drift_pct, reason, time.time())
            with self._lock:
                self._pending.append(entry)
        except Exception as e:
            log.debug("Heartbeat queue error: %s", e)
            # Fall through to immediate write so the heartbeat is
            # not silently dropped if the queue rejects it.
            try:
                self._db.update_node_status(
                    node_id, healthy, degraded, drift_pct, reason)
            except Exception:
                pass

    def stats(self) -> dict:
        with self._lock:
            pending = len(self._pending)
        return {
            "pending":  pending,
            "flushed":  self._flushed,
            "errors":   self._errors,
            "interval": self.BATCH_INTERVAL_S,
        }

    def _flush_loop(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(timeout=self.BATCH_INTERVAL_S)
            self._flush()

    def _flush(self) -> None:
        with self._lock:
            if not self._pending:
                return
            batch = list(self._pending)
            self._pending.clear()

        try:
            for (
                node_id, healthy, degraded,
                drift_pct, reason, _ts,
            ) in batch:
                self._db.update_node_status(
                    node_id, healthy, degraded, drift_pct, reason)
            self._flushed += len(batch)
        except Exception as e:
            self._errors += 1
            log.error("Heartbeat flush error: %s", e)

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
    global _API_KEY, _global_oracle
    _API_KEY = get_or_create_key()
    log.info("memopt control plane started (key: %s)", mask_key(_API_KEY))
    db.init()

    # Start global oracle if configured
    if os.getenv("MEMOPT_GLOBAL_ORACLE", "false").lower() == "true":
        try:
            from memopt.vmm.global_oracle import (
                GlobalOracle, GlobalOracleConfig)
            config = GlobalOracleConfig()
            _global_oracle = GlobalOracle(config)
            _global_oracle.start()
            log.info("Global oracle started")
        except Exception as e:
            log.warning("Global oracle failed to start: %s", e)

    # Start canary controller — always fresh per startup so test
    # contexts (and restarts in general) reset in-memory rollout state.
    global _canary
    try:
        from memopt.canary.controller import CanaryController
        if _canary is not None:
            try:
                _canary.stop()
            except Exception:
                pass
        _canary = CanaryController(db)
        _canary.start()
        log.info("Canary controller started")
    except Exception as e:
        log.warning("Canary controller failed: %s", e)

    # Start heartbeat batcher — always fresh per startup.
    global _heartbeat_batcher
    try:
        if _heartbeat_batcher is not None:
            try:
                _heartbeat_batcher.stop()
            except Exception:
                pass
        _heartbeat_batcher = HeartbeatBatcher(db)
        _heartbeat_batcher.start()
        log.info(
            "Heartbeat batcher started (interval=%ss)",
            HeartbeatBatcher.BATCH_INTERVAL_S)
    except Exception as e:
        log.warning("Heartbeat batcher failed: %s", e)


@app.on_event("shutdown")
def shutdown():
    global _canary, _global_oracle, _heartbeat_batcher
    if _canary is not None:
        try:
            _canary.stop()
        except Exception:
            pass
        _canary = None
    if _heartbeat_batcher is not None:
        try:
            _heartbeat_batcher.stop()
        except Exception:
            pass
        _heartbeat_batcher = None
    if _global_oracle is not None:
        _global_oracle.stop()
        _global_oracle = None


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


@app.post("/api/v1/nodes/{node_name}/status")
def update_node_status(node_name: str, payload: dict, _: str = Security(verify_api_key)):
    """
    Update degradation status for a node.
    Called by CertifyDaemon when drift or certification failure is detected.
    """
    try:
        drift_pct = float(payload.get("drift_pct", 0.0))
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="drift_pct must be numeric")

    healthy = bool(payload.get("healthy", True))
    degraded = bool(payload.get("degraded", False))
    reason = str(payload.get("reason", ""))

    # Prefer the batcher at scale — it queues the write and flushes
    # on a timer. Falls through to a direct write when batching is
    # unavailable (startup race, batcher not configured in tests, etc.)
    try:
        if _heartbeat_batcher is not None:
            _heartbeat_batcher.record(
                node_id=node_name,
                healthy=healthy,
                degraded=degraded,
                drift_pct=drift_pct,
                reason=reason)
        else:
            db.update_node_status(
                node_name=node_name,
                healthy=healthy,
                degraded=degraded,
                drift_pct=drift_pct,
                reason=reason)
        return {"ok": True, "node": node_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/heartbeat-batcher/stats")
def heartbeat_batcher_stats(_: str = Security(verify_api_key)):
    """HeartbeatBatcher observability — pending / flushed / errors."""
    if _heartbeat_batcher is None:
        return {"active": False}
    return {"active": True, **_heartbeat_batcher.stats()}


@app.get("/api/v1/database/health")
def database_health(_: str = Security(verify_api_key)):
    """
    Database-tier health: dialect, read/ping latency, batcher stats.

    - `dialect`: "sqlite" | "postgresql" | "cockroachdb"
    - `read_ms`: time for `get_all_nodes()` (rough read latency)
    - `ping_ms`: time for `SELECT 1` (raw connection latency)
    - `batcher`: `HeartbeatBatcher.stats()` if active, else None
    """
    try:
        dialect = db.dialect

        t0 = time.time()
        nodes = db.get_all_nodes()
        read_ms = (time.time() - t0) * 1000

        t0 = time.time()
        db._db.execute("SELECT 1")
        ping_ms = (time.time() - t0) * 1000

        return {
            "status":     "ok",
            "dialect":    dialect,
            "read_ms":    round(read_ms, 2),
            "ping_ms":    round(ping_ms, 2),
            "node_count": len(nodes),
            "batcher": (
                _heartbeat_batcher.stats()
                if _heartbeat_batcher is not None
                else None),
        }
    except Exception as e:
        log.error("Database health check: %s", e)
        return {
            "status":  "error",
            "dialect": "unknown",
            "error":   str(e),
        }


@app.get("/api/v1/nodes/degraded")
def list_degraded_nodes(_: str = Security(verify_api_key)):
    """Return all nodes currently flagged as degraded."""
    nodes = db.get_degraded_nodes()
    return {"degraded_nodes": nodes, "total": len(nodes)}


_cp_start_time = time.time()


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/healthz")
def healthz():
    """Liveness probe for Kubernetes. No auth."""
    try:
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "node_id": "control-plane",
                "uptime_seconds": round(
                    time.time() - _cp_start_time, 1),
                "checks": {"database": "ok"},
            })
    except Exception:
        return JSONResponse(
            status_code=503,
            content={"status": "error"})


@app.get("/readyz")
def readyz():
    """Readiness probe for Kubernetes. No auth."""
    try:
        # Check database accessible
        try:
            db.get_cluster_summary()
            db_ok = True
        except Exception:
            db_ok = False

        ready = db_ok
        return JSONResponse(
            status_code=200 if ready else 503,
            content={
                "status": "ready" if ready else "not_ready",
                "node_id": "control-plane",
                "checks": {
                    "database": "ok" if db_ok else "error",
                },
            })
    except Exception:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "reason": "internal_error",
            })


# ─────────────────────────────────────────────
# PXE BOOT ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/boot/config/{mac_address}")
def get_boot_config(mac_address: str):
    """
    Return boot configuration for a node identified by MAC address.

    Called by iPXE during PXE boot. No authentication — boot network
    only. Unknown MACs get a default config pointing at the "latest"
    image so a brand-new node can still come online.
    """
    mac = mac_address.lower().replace("-", ":").replace(" ", "")

    image_version = os.environ.get(
        "MEMOPT_GOLDEN_IMAGE_VERSION", "latest")
    image_registry = os.environ.get(
        "MEMOPT_IMAGE_REGISTRY",
        "registry.example.com/memopt")
    boot_server = os.environ.get(
        "MEMOPT_BOOT_SERVER_URL", "")
    control_plane = os.environ.get(
        "MEMOPT_CONTROL_PLANE_URL",
        "http://localhost:8765")
    redis_url = os.environ.get("REDIS_URL", "")

    rack = "unknown"
    pod = "unknown"
    region = "unknown"
    node_id = mac  # default: use MAC as id

    try:
        node = db.get_node_by_mac(mac)
        if node:
            rack = node.get("rack") or "unknown"
            pod = node.get("pod") or "unknown"
            region = node.get("region") or "unknown"
            node_id = node.get("node_name") or mac
    except Exception:
        pass

    # Apply pending rollback if any — overrides image_version so the
    # node boots the target version on its next reboot.
    try:
        pending = db.get_pending_rollback(node_id)
        if pending:
            image_version = pending["target_version"]
            log.info(
                "Boot config: applying rollback for %s to %s",
                node_id, image_version)
    except Exception:
        pass

    config = {
        "node_id":       node_id,
        "mac_address":   mac,
        "image_url":     f"{image_registry}/memopt-golden:{image_version}",
        "image_version": image_version,
        "rack":          rack,
        "pod":           pod,
        "region":        region,
        "kernel_args": [
            f"memopt.node_id={node_id}",
            f"memopt.rack={rack}",
            f"memopt.pod={pod}",
            f"memopt.region={region}",
            f"memopt.boot_server={boot_server}",
            f"memopt.control_plane={control_plane}",
            f"memopt.redis_url={redis_url}",
        ],
        "generated_at": time.time(),
    }

    log.info(
        "Boot config requested: mac=%s node_id=%s version=%s",
        mac, node_id, image_version)

    return config


@app.post("/boot/callback")
def boot_callback(payload: dict):
    """
    Called by first_boot.sh after successful boot.
    Records boot event in the database. No auth — boot network only.

    Body:
      {
        "node_id":           str,
        "mac_address":       str (optional),
        "image_version":     str,
        "boot_time_seconds": float,
        "gpu_count":         int,
        "cert_status":       "PASSED" | "FAILED" | "SKIPPED"
      }
    """
    node_id = payload.get("node_id", "")
    image_version = payload.get("image_version", "unknown")
    cert_status = payload.get("cert_status", "UNKNOWN")

    if not node_id:
        return {"status": "error",
                "reason": "node_id required"}

    try:
        db.record_boot_event(
            node_id=node_id,
            image_version=image_version,
            cert_status=cert_status,
            gpu_count=int(payload.get("gpu_count", 0)),
            boot_time_seconds=float(
                payload.get("boot_time_seconds", 0.0)),
        )
        log.info(
            "Boot callback: node=%s version=%s cert=%s",
            node_id, image_version, cert_status)
        return {"status": "ok", "node_id": node_id}
    except Exception as e:
        log.error("Boot callback error: %s", e)
        return {"status": "error", "reason": str(e)}


@app.get("/boot/status")
def boot_status(_: str = Security(verify_api_key)):
    """
    Cluster-wide boot status — booted node count and version
    distribution. Useful for monitoring rolling image upgrades.
    """
    try:
        return db.get_boot_status()
    except Exception as e:
        log.error("Boot status error: %s", e)
        return {
            "total_nodes":         0,
            "booted_nodes":        0,
            "current_version":     "unknown",
            "version_distribution": {},
            "failed_certs":        0,
        }


# ─────────────────────────────────────────────
# IMAGE VERSION REGISTRY
# ─────────────────────────────────────────────

@app.post("/api/v1/images")
def register_image(
    payload: dict,
    _: str = Security(verify_api_key),
):
    """
    Register a new image version. Called by CI/CD after a successful
    golden image build.
    """
    version = payload.get("version", "")
    if not version:
        raise HTTPException(
            status_code=400, detail="version required")

    try:
        db.register_image_version(
            version=version,
            git_commit=payload.get("git_commit", ""),
            build_date=payload.get("build_date", ""),
            cuda_version=payload.get("cuda_version", ""),
            sm_targets=payload.get("sm_targets", ""),
            image_url=payload.get("image_url", ""),
            digest=payload.get("digest", ""),
        )
        return {"status": "ok", "version": version}
    except Exception as e:
        log.error("Image registration failed: %s", e)
        raise HTTPException(
            status_code=500, detail=str(e))


@app.get("/api/v1/images")
def list_images(_: str = Security(verify_api_key)):
    """List all registered image versions, with the current stable."""
    versions = db.list_image_versions()
    return {
        "versions":       versions,
        "stable_version": db.get_stable_version(),
        "total":          len(versions),
    }


@app.post("/api/v1/images/{version}/stable")
def mark_image_stable(
    version: str,
    _: str = Security(verify_api_key),
):
    """Mark an image version as stable — served to new nodes."""
    db.mark_version_stable(version)
    return {"status": "ok",
            "version": version,
            "is_stable": True}


@app.post("/api/v1/images/{version}/deprecated")
def mark_image_deprecated(
    version: str,
    _: str = Security(verify_api_key),
):
    """Mark an image version as deprecated — triggers alerts."""
    db.mark_version_deprecated(version)
    return {"status": "ok",
            "version": version,
            "is_deprecated": True}


@app.get("/api/v1/images/{version}/nodes")
def list_nodes_on_version(
    version: str,
    _: str = Security(verify_api_key),
):
    """Return node_ids whose latest boot is on this version."""
    nodes = db.get_nodes_on_version(version)
    return {
        "version": version,
        "nodes":   nodes,
        "total":   len(nodes),
    }


# ─────────────────────────────────────────────
# ROLLBACK
# ─────────────────────────────────────────────

@app.post("/api/v1/nodes/{node_id}/rollback")
def rollback_node(
    node_id: str,
    payload: dict,
    _: str = Security(verify_api_key),
):
    """
    Request a rollback for a node to a previous image version.

    The rollback records intent — it does NOT reboot the node.
    On next reboot, /boot/config/{mac} returns the target version.
    For the design-partner phase, the reboot itself is manual
    (SSH + reboot, or IPMI if configured).
    """
    target_version = payload.get("target_version", "")
    reason = payload.get("reason", "")

    if not target_version:
        raise HTTPException(
            status_code=400,
            detail="target_version required")

    # Verify the target version is registered
    versions = db.list_image_versions()
    version_ids = [v["version"] for v in versions]
    if target_version not in version_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Version {target_version} not registered")

    db.record_rollback_intent(
        node_id=node_id,
        target_version=target_version,
        reason=reason)

    log.warning(
        "Rollback requested: node=%s target=%s reason=%s",
        node_id, target_version, reason)

    return {
        "status":         "rollback_scheduled",
        "node_id":        node_id,
        "target_version": target_version,
        "note": (
            "Node will boot target version on next reboot. "
            "Manual reboot required unless IPMI configured."),
    }


# ─────────────────────────────────────────────
# CANARY ROLLOUTS
# ─────────────────────────────────────────────

@app.post("/api/v1/rollouts")
def start_rollout(
    payload: dict,
    _: str = Security(verify_api_key),
):
    """Begin a new progressive (1-10-100-All) rollout."""
    target = payload.get("target_version", "")
    current = payload.get("current_version", "")

    if not target or not current:
        raise HTTPException(
            status_code=400,
            detail="target_version and current_version required")

    # Verify target is registered
    versions = db.list_image_versions()
    version_ids = [v["version"] for v in versions]
    if target not in version_ids:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Version {target} not registered. "
                f"Register it first via POST /api/v1/images"))

    if _canary is None:
        raise HTTPException(
            status_code=503,
            detail="Canary controller not available")

    try:
        plan = _canary.begin_rollout(
            target_version=target,
            current_version=current,
            created_by="api")
        return {
            "rollout_id":     plan.rollout_id,
            "target_version": plan.target_version,
            "status":         "started",
        }
    except ValueError as e:
        raise HTTPException(
            status_code=409, detail=str(e))


@app.get("/api/v1/rollouts/active")
def get_active_rollout(_: str = Security(verify_api_key)):
    """Return the currently-active rollout's state, or active=False."""
    if _canary is None:
        return {"active": False}
    status = _canary.get_status()
    if status is None:
        return {"active": False}
    return {"active": True, **status}


@app.post("/api/v1/rollouts/pause")
def pause_rollout(
    payload: dict,
    _: str = Security(verify_api_key),
):
    """Pause the currently-active rollout."""
    if _canary is None:
        raise HTTPException(
            status_code=503, detail="Canary not available")
    reason = payload.get("reason", "manual pause")
    _canary.pause_rollout(reason)
    return {"status": "paused", "reason": reason}


@app.post("/api/v1/rollouts/resume")
def resume_rollout(_: str = Security(verify_api_key)):
    """Resume a paused rollout."""
    if _canary is None:
        raise HTTPException(
            status_code=503, detail="Canary not available")
    _canary.resume_rollout()
    return {"status": "resumed"}


@app.post("/api/v1/rollouts/abort")
def abort_rollout(
    payload: dict,
    _: str = Security(verify_api_key),
):
    """Abort (fail) the active rollout."""
    if _canary is None:
        raise HTTPException(
            status_code=503, detail="Canary not available")
    reason = payload.get("reason", "manual abort")
    _canary.abort_rollout(reason)
    return {"status": "aborted", "reason": reason}


@app.get("/api/v1/rollouts/{rollout_id}/events")
def get_rollout_events_endpoint(
    rollout_id: str,
    _: str = Security(verify_api_key),
):
    """Audit log for a rollout (ordered oldest → newest)."""
    try:
        events = db.get_rollout_events(rollout_id)
        return {
            "rollout_id": rollout_id,
            "events":     events,
            "total":      len(events),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# POD CONTROLLER ENDPOINTS
# ─────────────────────────────────────────────

@app.post("/api/v1/pods/{pod_id}")
def receive_pod_report(pod_id: str, payload: dict,
                       _: str = Security(verify_api_key)):
    """Accept pod health report. Persisted to database."""
    try:
        db.upsert_pod(pod_id, {
            "node_count": payload.get("node_count", 0),
            "healthy_nodes": payload.get("healthy_nodes", 0),
            "pod_oracle_size": payload.get("pod_oracle_size", 0),
            "avg_hbm_free_gb": payload.get("avg_hbm_free_gb", 0.0),
            "gkd_hit_rate_pct": payload.get("gkd_hit_rate_pct", 0.0),
            "reported_at": payload.get("reported_at", time.time()),
        })

        # Register pod URL with global oracle for transition pulling
        if _global_oracle is not None:
            pod_url = payload.get("pod_url", "")
            if pod_url:
                _global_oracle.register_pod(pod_id, pod_url)

        return {"status": "ok", "pod_id": pod_id}
    except Exception as e:
        log.error("Pod report failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to store pod data")


@app.get("/api/v1/pods")
def list_pods(_: str = Security(verify_api_key)):
    """List all registered pods from database."""
    try:
        pods = db.list_pods()
        return {"pods": pods, "total": len(pods)}
    except Exception as e:
        log.error("List pods failed: %s", e)
        return {"pods": [], "total": 0}


@app.get("/api/v1/pods/{pod_id}")
def get_pod(pod_id: str, _: str = Security(verify_api_key)):
    """Single pod detail from database."""
    pod = db.get_pod(pod_id)
    if not pod:
        raise HTTPException(
            status_code=404, detail=f"Pod {pod_id} not found")
    return pod


# ─────────────────────────────────────────────
# GLOBAL ORACLE
# ─────────────────────────────────────────────

@app.get("/api/v1/global-oracle/stats")
def global_oracle_stats(_: str = Security(verify_api_key)):
    """Global oracle statistics and top transitions."""
    if _global_oracle is None:
        return {
            "enabled": False,
            "reason": "MEMOPT_GLOBAL_ORACLE not set to true",
        }
    return {
        "enabled": True,
        **_global_oracle.stats(),
    }


@app.get("/api/v1/global-oracle/transitions")
def global_oracle_transitions(
    top_k: int = Query(100, le=1000),
    _: str = Security(verify_api_key),
):
    """Top global transitions with pod coverage information."""
    if _global_oracle is None:
        return {"transitions": [], "enabled": False}
    return {
        "transitions": _global_oracle.get_global_transitions(
            top_k=top_k),
        "enabled": True,
        "exported_at": time.time(),
    }


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
