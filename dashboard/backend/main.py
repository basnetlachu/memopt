"""
Memopt Dashboard - FastAPI Backend

Provides REST API and WebSocket for real-time GPU monitoring.
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import func, select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from database import (
    Alert,
    GPU,
    MetricSnapshot,
    OptimizationSession,
    Server,
    TrainingRun,
    get_session,
    init_db,
)

app = FastAPI(
    title="Memopt Dashboard API",
    description="Real-time GPU optimization monitoring",
    version="1.0.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# WebSocket Manager
# =============================================================================

class ConnectionManager:
    """Manage WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast(self, message: dict):
        """Send message to all connected clients."""
        dead_connections = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.add(connection)

        # Clean up dead connections
        self.active_connections -= dead_connections


manager = ConnectionManager()


# =============================================================================
# Pydantic Models
# =============================================================================

class ServerCreate(BaseModel):
    hostname: str
    ip_address: str
    gpu_count: int = 1
    gpu_model: str = ""
    total_memory_gb: float = 0


class ServerUpdate(BaseModel):
    status: Optional[str] = None
    gpu_count: Optional[int] = None


class GPUUpdate(BaseModel):
    memory_used_mb: float
    utilization_pct: float
    temperature_c: int


class SessionCreate(BaseModel):
    session_id: str
    server_hostname: Optional[str] = None
    model_name: str
    model_type: str = "inference"
    gpu_index: int = 0


class SessionUpdate(BaseModel):
    status: Optional[str] = None
    speedup: Optional[float] = None
    memory_saved_mb: Optional[float] = None
    bandwidth_saved_gbps: Optional[float] = None
    memory_bound_pct: Optional[float] = None
    optimizations_applied: Optional[int] = None
    rollback_count: Optional[int] = None
    details: Optional[str] = None


class TrainingRunCreate(BaseModel):
    run_id: str
    server_hostname: Optional[str] = None
    model_name: str
    total_epochs: int = 0
    total_steps: int = 0


class TrainingRunUpdate(BaseModel):
    current_epoch: Optional[int] = None
    current_step: Optional[int] = None
    current_loss: Optional[float] = None
    speedup: Optional[float] = None
    optimizations_applied: Optional[int] = None
    rollbacks: Optional[int] = None
    status: Optional[str] = None


class AlertCreate(BaseModel):
    server_hostname: Optional[str] = None
    level: str
    title: str
    message: str
    source: str = "system"


class FleetOverview(BaseModel):
    total_servers: int
    online_servers: int
    total_gpus: int
    total_memory_gb: float
    total_speedup: float
    total_bandwidth_saved_gbps: float
    active_training_runs: int
    total_optimizations: int


# =============================================================================
# Startup/Shutdown
# =============================================================================

@app.on_event("startup")
async def startup():
    await init_db()


# =============================================================================
# Fleet Overview
# =============================================================================

@app.get("/api/fleet/overview", response_model=FleetOverview)
async def get_fleet_overview(db: AsyncSession = Depends(get_session)):
    """Get aggregate fleet statistics."""
    # Server counts
    total_servers = await db.scalar(select(func.count(Server.id)))
    online_servers = await db.scalar(
        select(func.count(Server.id)).where(Server.status == "online")
    )

    # GPU stats
    total_gpus = await db.scalar(select(func.sum(Server.gpu_count)))
    total_memory = await db.scalar(select(func.sum(Server.total_memory_gb)))

    # Optimization stats
    speedup_result = await db.scalar(
        select(func.avg(OptimizationSession.speedup))
        .where(OptimizationSession.status == "completed")
    )
    bandwidth_result = await db.scalar(
        select(func.sum(OptimizationSession.bandwidth_saved_gbps))
    )
    total_opts = await db.scalar(select(func.count(OptimizationSession.id)))

    # Training runs
    active_runs = await db.scalar(
        select(func.count(TrainingRun.id)).where(TrainingRun.status == "running")
    )

    return FleetOverview(
        total_servers=total_servers or 0,
        online_servers=online_servers or 0,
        total_gpus=total_gpus or 0,
        total_memory_gb=total_memory or 0,
        total_speedup=speedup_result or 1.0,
        total_bandwidth_saved_gbps=bandwidth_result or 0,
        active_training_runs=active_runs or 0,
        total_optimizations=total_opts or 0,
    )


# =============================================================================
# Servers
# =============================================================================

@app.get("/api/servers")
async def list_servers(db: AsyncSession = Depends(get_session)):
    """List all registered servers."""
    result = await db.execute(select(Server).order_by(desc(Server.last_heartbeat)))
    servers = result.scalars().all()
    return [
        {
            "id": s.id,
            "hostname": s.hostname,
            "ip_address": s.ip_address,
            "gpu_count": s.gpu_count,
            "gpu_model": s.gpu_model,
            "total_memory_gb": s.total_memory_gb,
            "status": s.status,
            "last_heartbeat": s.last_heartbeat.isoformat() if s.last_heartbeat else None,
        }
        for s in servers
    ]


@app.post("/api/servers")
async def register_server(server: ServerCreate, db: AsyncSession = Depends(get_session)):
    """Register a new GPU server."""
    existing = await db.execute(
        select(Server).where(Server.hostname == server.hostname)
    )
    if existing.scalar():
        raise HTTPException(status_code=400, detail="Server already registered")

    new_server = Server(
        hostname=server.hostname,
        ip_address=server.ip_address,
        gpu_count=server.gpu_count,
        gpu_model=server.gpu_model,
        total_memory_gb=server.total_memory_gb,
        status="online",
    )
    db.add(new_server)
    await db.commit()
    await db.refresh(new_server)

    # Broadcast update
    await manager.broadcast({
        "type": "server_registered",
        "server": {"id": new_server.id, "hostname": new_server.hostname},
    })

    return {"id": new_server.id, "hostname": new_server.hostname}


@app.post("/api/servers/{hostname}/heartbeat")
async def server_heartbeat(
    hostname: str,
    gpus: Optional[List[GPUUpdate]] = None,
    db: AsyncSession = Depends(get_session),
):
    """Update server heartbeat and GPU metrics."""
    result = await db.execute(select(Server).where(Server.hostname == hostname))
    server = result.scalar()

    if not server:
        raise HTTPException(status_code=404, detail="Server not found")

    server.last_heartbeat = datetime.utcnow()
    server.status = "online"

    # Update GPU metrics if provided
    if gpus:
        for i, gpu_data in enumerate(gpus):
            gpu_result = await db.execute(
                select(GPU).where(GPU.server_id == server.id, GPU.gpu_index == i)
            )
            gpu = gpu_result.scalar()

            if gpu:
                gpu.memory_used_mb = gpu_data.memory_used_mb
                gpu.utilization_pct = gpu_data.utilization_pct
                gpu.temperature_c = gpu_data.temperature_c
                gpu.updated_at = datetime.utcnow()
            else:
                new_gpu = GPU(
                    server_id=server.id,
                    gpu_index=i,
                    name=server.gpu_model,
                    memory_total_mb=server.total_memory_gb * 1024 / server.gpu_count,
                    memory_used_mb=gpu_data.memory_used_mb,
                    utilization_pct=gpu_data.utilization_pct,
                    temperature_c=gpu_data.temperature_c,
                )
                db.add(new_gpu)

            # Store metric snapshot
            snapshot = MetricSnapshot(
                server_id=server.id,
                gpu_index=i,
                memory_used_mb=gpu_data.memory_used_mb,
                utilization_pct=gpu_data.utilization_pct,
                temperature_c=gpu_data.temperature_c,
            )
            db.add(snapshot)

    await db.commit()

    # Broadcast GPU update
    await manager.broadcast({
        "type": "gpu_update",
        "server": hostname,
        "gpus": [g.dict() for g in gpus] if gpus else [],
    })

    return {"status": "ok"}


@app.get("/api/servers/{hostname}")
async def get_server(hostname: str, db: AsyncSession = Depends(get_session)):
    """Get server details with GPU info."""
    result = await db.execute(select(Server).where(Server.hostname == hostname))
    server = result.scalar()

    if not server:
        raise HTTPException(status_code=404, detail="Server not found")

    gpu_result = await db.execute(
        select(GPU).where(GPU.server_id == server.id).order_by(GPU.gpu_index)
    )
    gpus = gpu_result.scalars().all()

    return {
        "id": server.id,
        "hostname": server.hostname,
        "ip_address": server.ip_address,
        "gpu_count": server.gpu_count,
        "gpu_model": server.gpu_model,
        "total_memory_gb": server.total_memory_gb,
        "status": server.status,
        "last_heartbeat": server.last_heartbeat.isoformat() if server.last_heartbeat else None,
        "gpus": [
            {
                "index": g.gpu_index,
                "name": g.name,
                "memory_total_mb": g.memory_total_mb,
                "memory_used_mb": g.memory_used_mb,
                "utilization_pct": g.utilization_pct,
                "temperature_c": g.temperature_c,
            }
            for g in gpus
        ],
    }


@app.get("/api/servers/{hostname}/metrics")
async def get_server_metrics(
    hostname: str,
    hours: int = Query(default=24, ge=1, le=168),
    db: AsyncSession = Depends(get_session),
):
    """Get historical metrics for a server."""
    result = await db.execute(select(Server).where(Server.hostname == hostname))
    server = result.scalar()

    if not server:
        raise HTTPException(status_code=404, detail="Server not found")

    since = datetime.utcnow() - timedelta(hours=hours)
    metrics_result = await db.execute(
        select(MetricSnapshot)
        .where(MetricSnapshot.server_id == server.id)
        .where(MetricSnapshot.timestamp >= since)
        .order_by(MetricSnapshot.timestamp)
    )
    metrics = metrics_result.scalars().all()

    return [
        {
            "timestamp": m.timestamp.isoformat(),
            "gpu_index": m.gpu_index,
            "memory_used_mb": m.memory_used_mb,
            "utilization_pct": m.utilization_pct,
            "temperature_c": m.temperature_c,
        }
        for m in metrics
    ]


# =============================================================================
# Optimization Sessions
# =============================================================================

@app.get("/api/sessions")
async def list_sessions(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    server: Optional[str] = None,
    model: Optional[str] = None,
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_session),
):
    """List optimization sessions with filtering."""
    query = select(OptimizationSession).order_by(desc(OptimizationSession.started_at))

    if server:
        server_result = await db.execute(select(Server).where(Server.hostname == server))
        server_obj = server_result.scalar()
        if server_obj:
            query = query.where(OptimizationSession.server_id == server_obj.id)

    if model:
        query = query.where(OptimizationSession.model_name.ilike(f"%{model}%"))

    if status:
        query = query.where(OptimizationSession.status == status)

    query = query.limit(limit).offset(offset)
    result = await db.execute(query)
    sessions = result.scalars().all()

    return [
        {
            "id": s.id,
            "session_id": s.session_id,
            "model_name": s.model_name,
            "model_type": s.model_type,
            "gpu_index": s.gpu_index,
            "speedup": s.speedup,
            "memory_saved_mb": s.memory_saved_mb,
            "bandwidth_saved_gbps": s.bandwidth_saved_gbps,
            "memory_bound_pct": s.memory_bound_pct,
            "status": s.status,
            "optimizations_applied": s.optimizations_applied,
            "rollback_count": s.rollback_count,
            "started_at": s.started_at.isoformat() if s.started_at else None,
            "completed_at": s.completed_at.isoformat() if s.completed_at else None,
        }
        for s in sessions
    ]


@app.post("/api/sessions")
async def create_session(session: SessionCreate, db: AsyncSession = Depends(get_session)):
    """Create a new optimization session."""
    server_id = None
    if session.server_hostname:
        server_result = await db.execute(
            select(Server).where(Server.hostname == session.server_hostname)
        )
        server = server_result.scalar()
        if server:
            server_id = server.id

    new_session = OptimizationSession(
        session_id=session.session_id,
        server_id=server_id,
        model_name=session.model_name,
        model_type=session.model_type,
        gpu_index=session.gpu_index,
        status="profiling",
    )
    db.add(new_session)
    await db.commit()
    await db.refresh(new_session)

    # Broadcast
    await manager.broadcast({
        "type": "session_created",
        "session": {"id": new_session.session_id, "model": new_session.model_name},
    })

    return {"id": new_session.id, "session_id": new_session.session_id}


@app.patch("/api/sessions/{session_id}")
async def update_session(
    session_id: str,
    update: SessionUpdate,
    db: AsyncSession = Depends(get_session),
):
    """Update an optimization session."""
    result = await db.execute(
        select(OptimizationSession).where(OptimizationSession.session_id == session_id)
    )
    session = result.scalar()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    for field, value in update.dict(exclude_unset=True).items():
        setattr(session, field, value)

    if update.status == "completed":
        session.completed_at = datetime.utcnow()

    await db.commit()

    # Broadcast optimization event
    if update.status == "completed" and update.speedup and update.speedup > 1.0:
        await manager.broadcast({
            "type": "optimization_applied",
            "session_id": session_id,
            "model": session.model_name,
            "speedup": update.speedup,
        })

    if update.rollback_count and update.rollback_count > 0:
        await manager.broadcast({
            "type": "rollback_triggered",
            "session_id": session_id,
            "model": session.model_name,
        })

    return {"status": "updated"}


@app.get("/api/sessions/{session_id}")
async def get_session_detail(session_id: str, db: AsyncSession = Depends(get_session)):
    """Get session details."""
    result = await db.execute(
        select(OptimizationSession).where(OptimizationSession.session_id == session_id)
    )
    session = result.scalar()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "id": session.id,
        "session_id": session.session_id,
        "model_name": session.model_name,
        "model_type": session.model_type,
        "gpu_index": session.gpu_index,
        "speedup": session.speedup,
        "memory_saved_mb": session.memory_saved_mb,
        "bandwidth_saved_gbps": session.bandwidth_saved_gbps,
        "memory_bound_pct": session.memory_bound_pct,
        "status": session.status,
        "optimizations_applied": session.optimizations_applied,
        "rollback_count": session.rollback_count,
        "started_at": session.started_at.isoformat() if session.started_at else None,
        "completed_at": session.completed_at.isoformat() if session.completed_at else None,
        "details": json.loads(session.details) if session.details else None,
    }


# =============================================================================
# Training Runs
# =============================================================================

@app.get("/api/training")
async def list_training_runs(
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_session),
):
    """List training runs."""
    query = select(TrainingRun).order_by(desc(TrainingRun.updated_at))

    if status:
        query = query.where(TrainingRun.status == status)

    result = await db.execute(query)
    runs = result.scalars().all()

    return [
        {
            "id": r.id,
            "run_id": r.run_id,
            "model_name": r.model_name,
            "current_epoch": r.current_epoch,
            "total_epochs": r.total_epochs,
            "current_step": r.current_step,
            "total_steps": r.total_steps,
            "current_loss": r.current_loss,
            "baseline_loss": r.baseline_loss,
            "speedup": r.speedup,
            "optimizations_applied": r.optimizations_applied,
            "rollbacks": r.rollbacks,
            "status": r.status,
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "updated_at": r.updated_at.isoformat() if r.updated_at else None,
        }
        for r in runs
    ]


@app.post("/api/training")
async def create_training_run(run: TrainingRunCreate, db: AsyncSession = Depends(get_session)):
    """Create a new training run."""
    server_id = None
    if run.server_hostname:
        server_result = await db.execute(
            select(Server).where(Server.hostname == run.server_hostname)
        )
        server = server_result.scalar()
        if server:
            server_id = server.id

    new_run = TrainingRun(
        run_id=run.run_id,
        server_id=server_id,
        model_name=run.model_name,
        total_epochs=run.total_epochs,
        total_steps=run.total_steps,
    )
    db.add(new_run)
    await db.commit()

    await manager.broadcast({
        "type": "training_started",
        "run_id": run.run_id,
        "model": run.model_name,
    })

    return {"id": new_run.id, "run_id": new_run.run_id}


@app.patch("/api/training/{run_id}")
async def update_training_run(
    run_id: str,
    update: TrainingRunUpdate,
    db: AsyncSession = Depends(get_session),
):
    """Update a training run."""
    result = await db.execute(select(TrainingRun).where(TrainingRun.run_id == run_id))
    run = result.scalar()

    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")

    for field, value in update.dict(exclude_unset=True).items():
        setattr(run, field, value)

    run.updated_at = datetime.utcnow()
    await db.commit()

    # Broadcast progress
    await manager.broadcast({
        "type": "training_progress",
        "run_id": run_id,
        "epoch": run.current_epoch,
        "total_epochs": run.total_epochs,
        "loss": run.current_loss,
        "speedup": run.speedup,
    })

    return {"status": "updated"}


# =============================================================================
# Alerts
# =============================================================================

@app.get("/api/alerts")
async def list_alerts(
    unacknowledged_only: bool = False,
    limit: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(get_session),
):
    """List alerts."""
    query = select(Alert).order_by(desc(Alert.created_at))

    if unacknowledged_only:
        query = query.where(Alert.acknowledged == False)

    query = query.limit(limit)
    result = await db.execute(query)
    alerts = result.scalars().all()

    return [
        {
            "id": a.id,
            "level": a.level,
            "title": a.title,
            "message": a.message,
            "source": a.source,
            "acknowledged": a.acknowledged,
            "created_at": a.created_at.isoformat() if a.created_at else None,
        }
        for a in alerts
    ]


@app.post("/api/alerts")
async def create_alert(alert: AlertCreate, db: AsyncSession = Depends(get_session)):
    """Create a new alert."""
    server_id = None
    if alert.server_hostname:
        server_result = await db.execute(
            select(Server).where(Server.hostname == alert.server_hostname)
        )
        server = server_result.scalar()
        if server:
            server_id = server.id

    new_alert = Alert(
        server_id=server_id,
        level=alert.level,
        title=alert.title,
        message=alert.message,
        source=alert.source,
    )
    db.add(new_alert)
    await db.commit()

    # Broadcast alert
    await manager.broadcast({
        "type": "alert",
        "level": alert.level,
        "title": alert.title,
        "message": alert.message,
    })

    return {"id": new_alert.id}


@app.patch("/api/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: int, db: AsyncSession = Depends(get_session)):
    """Acknowledge an alert."""
    result = await db.execute(select(Alert).where(Alert.id == alert_id))
    alert = result.scalar()

    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    alert.acknowledged = True
    await db.commit()

    return {"status": "acknowledged"}


# =============================================================================
# Bottleneck Analysis (Phase 1 Enhancement)
# =============================================================================

class BottleneckData(BaseModel):
    kernel_name: str
    bottleneck_type: str
    confidence: float
    gpu_time_pct: float
    memory_stall_pct: float
    dram_traffic_gb: float
    recoverable_pct: float
    recommendations: List[dict]


class BottleneckReport(BaseModel):
    server_hostname: Optional[str] = None
    total_kernels: int
    optimizable_kernels: int
    total_gpu_time_ms: float
    total_recoverable_ms: float
    avg_memory_stall_pct: float
    bottlenecks: List[dict]


@app.post("/api/bottleneck/report")
async def submit_bottleneck_report(report: BottleneckReport):
    """Submit a bottleneck analysis report from a daemon."""
    # Store report in session details for historical tracking
    # Broadcast to dashboard for real-time display
    await manager.broadcast({
        "type": "bottleneck_report",
        "server": report.server_hostname,
        "total_kernels": report.total_kernels,
        "optimizable_kernels": report.optimizable_kernels,
        "total_gpu_time_ms": report.total_gpu_time_ms,
        "total_recoverable_ms": report.total_recoverable_ms,
        "avg_memory_stall_pct": report.avg_memory_stall_pct,
        "bottlenecks": report.bottlenecks[:10],  # Top 10
    })

    return {"status": "received", "kernels_analyzed": report.total_kernels}


@app.get("/api/bottleneck/types")
async def get_bottleneck_types():
    """Get bottleneck type definitions."""
    return {
        "types": [
            {
                "id": "memory_bound_dram",
                "name": "Memory Bound (DRAM)",
                "description": ">70% stalls on main memory. High DRAM traffic, poor cache reuse.",
                "color": "#ef4444",  # red
            },
            {
                "id": "memory_bound_cache",
                "name": "Memory Bound (Cache)",
                "description": "High L2 miss rate, cache thrashing. Working set exceeds cache.",
                "color": "#f97316",  # orange
            },
            {
                "id": "compute_bound",
                "name": "Compute Bound",
                "description": "Low stalls, high arithmetic intensity. OPTIMAL - no optimization needed.",
                "color": "#22c55e",  # green
            },
            {
                "id": "pipeline_bound_occupancy",
                "name": "Pipeline Bound",
                "description": "Low warp occupancy, underutilized SMs.",
                "color": "#eab308",  # yellow
            },
            {
                "id": "mixed",
                "name": "Mixed",
                "description": "Multiple bottlenecks present requiring multi-faceted approach.",
                "color": "#8b5cf6",  # purple
            },
        ]
    }


class KernelCounters(BaseModel):
    kernel_name: str
    duration_us: float
    dram_bytes_read: int
    dram_bytes_write: int
    memory_stall_ratio: float
    sm_efficiency: float
    achieved_occupancy: float
    l2_hit_rate: float


@app.post("/api/bottleneck/analyze")
async def analyze_kernel_bottleneck(counters: KernelCounters):
    """Analyze a single kernel's bottleneck type (stateless analysis)."""
    # Classification thresholds
    MEMORY_BOUND_DRAM_THRESHOLD = 0.70
    MEMORY_BOUND_CACHE_THRESHOLD = 0.40
    COMPUTE_BOUND_THRESHOLD = 0.60

    stall_ratio = counters.memory_stall_ratio
    l2_miss_rate = 1.0 - counters.l2_hit_rate
    occupancy = counters.achieved_occupancy
    sm_efficiency = counters.sm_efficiency

    # Determine bottleneck type
    if sm_efficiency > COMPUTE_BOUND_THRESHOLD and stall_ratio < 0.30:
        bottleneck_type = "compute_bound"
        confidence = sm_efficiency
        recoverable_pct = 0.0
    elif stall_ratio > MEMORY_BOUND_DRAM_THRESHOLD:
        bottleneck_type = "memory_bound_dram"
        confidence = stall_ratio
        recoverable_pct = stall_ratio * 50
    elif l2_miss_rate > MEMORY_BOUND_CACHE_THRESHOLD and stall_ratio > 0.30:
        bottleneck_type = "memory_bound_cache"
        confidence = l2_miss_rate
        recoverable_pct = l2_miss_rate * 40
    elif occupancy < 0.30:
        bottleneck_type = "pipeline_bound_occupancy"
        confidence = 1.0 - occupancy
        recoverable_pct = (1.0 - occupancy) * 30
    else:
        bottleneck_type = "mixed"
        confidence = 0.6
        recoverable_pct = stall_ratio * 35

    dram_traffic_gb = (counters.dram_bytes_read + counters.dram_bytes_write) / (1024**3)
    gpu_time_ms = counters.duration_us / 1000

    return {
        "kernel_name": counters.kernel_name,
        "bottleneck_type": bottleneck_type,
        "confidence": confidence,
        "gpu_time_ms": gpu_time_ms,
        "memory_stall_pct": stall_ratio * 100,
        "dram_traffic_gb": dram_traffic_gb,
        "recoverable_pct": recoverable_pct,
        "recoverable_ms": gpu_time_ms * (recoverable_pct / 100),
    }


# =============================================================================
# WebSocket
# =============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, handle incoming messages
            data = await websocket.receive_text()
            # Client can send ping/subscribe messages
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# =============================================================================
# Export
# =============================================================================

@app.get("/api/export/sessions")
async def export_sessions(
    format: str = Query(default="json", regex="^(json|csv)$"),
    db: AsyncSession = Depends(get_session),
):
    """Export all sessions."""
    result = await db.execute(
        select(OptimizationSession).order_by(desc(OptimizationSession.started_at))
    )
    sessions = result.scalars().all()

    if format == "csv":
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "session_id", "model_name", "model_type", "gpu_index",
            "speedup", "memory_saved_mb", "bandwidth_saved_gbps",
            "status", "started_at", "completed_at"
        ])

        for s in sessions:
            writer.writerow([
                s.session_id, s.model_name, s.model_type, s.gpu_index,
                s.speedup, s.memory_saved_mb, s.bandwidth_saved_gbps,
                s.status, s.started_at, s.completed_at
            ])

        return JSONResponse(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=memopt_sessions.csv"}
        )

    return [
        {
            "session_id": s.session_id,
            "model_name": s.model_name,
            "model_type": s.model_type,
            "gpu_index": s.gpu_index,
            "speedup": s.speedup,
            "memory_saved_mb": s.memory_saved_mb,
            "bandwidth_saved_gbps": s.bandwidth_saved_gbps,
            "status": s.status,
            "started_at": s.started_at.isoformat() if s.started_at else None,
            "completed_at": s.completed_at.isoformat() if s.completed_at else None,
        }
        for s in sessions
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
