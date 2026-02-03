"""
Database models and connection for Memopt Dashboard.
"""

import os
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./memopt_dashboard.db")

engine = create_async_engine(DATABASE_URL, echo=False)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()


class Server(Base):
    """GPU server running Memopt daemon."""
    __tablename__ = "servers"

    id = Column(Integer, primary_key=True, index=True)
    hostname = Column(String(255), unique=True, index=True)
    ip_address = Column(String(45))
    gpu_count = Column(Integer, default=1)
    gpu_model = Column(String(255))
    total_memory_gb = Column(Float)
    status = Column(String(50), default="offline")  # online, offline, error
    last_heartbeat = Column(DateTime, default=datetime.utcnow)
    registered_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    gpus = relationship("GPU", back_populates="server", cascade="all, delete-orphan")
    sessions = relationship("OptimizationSession", back_populates="server")
    training_runs = relationship("TrainingRun", back_populates="server")


class GPU(Base):
    """Individual GPU on a server."""
    __tablename__ = "gpus"

    id = Column(Integer, primary_key=True, index=True)
    server_id = Column(Integer, ForeignKey("servers.id"))
    gpu_index = Column(Integer)
    name = Column(String(255))
    memory_total_mb = Column(Float)
    memory_used_mb = Column(Float, default=0)
    utilization_pct = Column(Float, default=0)
    temperature_c = Column(Integer, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow)

    server = relationship("Server", back_populates="gpus")


class OptimizationSession(Base):
    """Optimization session from daemon or training wrapper."""
    __tablename__ = "optimization_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), unique=True, index=True)
    server_id = Column(Integer, ForeignKey("servers.id"), nullable=True)
    model_name = Column(String(255))
    model_type = Column(String(100))  # inference, training
    gpu_index = Column(Integer, default=0)

    # Metrics
    speedup = Column(Float, default=1.0)
    memory_saved_mb = Column(Float, default=0)
    bandwidth_saved_gbps = Column(Float, default=0)
    memory_bound_pct = Column(Float, default=0)

    # Status
    status = Column(String(50), default="pending")  # pending, profiling, optimizing, completed, failed, rolled_back
    optimizations_applied = Column(Integer, default=0)
    rollback_count = Column(Integer, default=0)

    # Timing
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Details
    details = Column(Text, nullable=True)  # JSON string with full details

    server = relationship("Server", back_populates="sessions")


class TrainingRun(Base):
    """Active training run using Memopt."""
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(100), unique=True, index=True)
    server_id = Column(Integer, ForeignKey("servers.id"), nullable=True)
    model_name = Column(String(255))

    # Progress
    current_epoch = Column(Integer, default=0)
    total_epochs = Column(Integer, default=0)
    current_step = Column(Integer, default=0)
    total_steps = Column(Integer, default=0)

    # Metrics
    current_loss = Column(Float, nullable=True)
    baseline_loss = Column(Float, nullable=True)
    speedup = Column(Float, default=1.0)
    optimizations_applied = Column(Integer, default=0)
    rollbacks = Column(Integer, default=0)

    # Status
    status = Column(String(50), default="running")  # running, paused, completed, failed
    started_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    server = relationship("Server", back_populates="training_runs")


class Alert(Base):
    """System alerts and notifications."""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    server_id = Column(Integer, ForeignKey("servers.id"), nullable=True)
    level = Column(String(20))  # info, warning, error, success
    title = Column(String(255))
    message = Column(Text)
    source = Column(String(100))  # daemon, training, system
    acknowledged = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class MetricSnapshot(Base):
    """Time-series metrics for graphs."""
    __tablename__ = "metric_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    server_id = Column(Integer, ForeignKey("servers.id"))
    gpu_index = Column(Integer, default=0)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # GPU metrics
    memory_used_mb = Column(Float)
    utilization_pct = Column(Float)
    temperature_c = Column(Integer)

    # Memopt metrics
    active_optimizations = Column(Integer, default=0)
    bandwidth_saved_gbps = Column(Float, default=0)


async def init_db():
    """Initialize database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:
    """Get database session."""
    async with async_session() as session:
        yield session
