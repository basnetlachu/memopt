"""
SQLite database for memopt control plane.
Three tables: nodes, events, metrics.
No ORM — plain sqlite3 for zero dependencies.
Thread-safe via check_same_thread=False + connection-per-request pattern.
"""
import sqlite3
import time
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass

log = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path.home() / ".memopt" / "control_plane" / "memopt.db"


@dataclass
class NodeRecord:
    node_name: str
    last_seen: float          # unix timestamp
    gpu_count: int
    total_vram_gb: float
    active_processes: int
    optimizations_applied: int
    dollar_saved_today: float
    dollar_saved_total: float
    status: str               # online / offline / idle
    current_workloads: str    # JSON list of {model, mode, speedup}


@dataclass
class EventRecord:
    id: Optional[int]
    timestamp: float
    node_name: str
    pid: int
    model_family: str
    gpu_ids: str              # JSON list
    optimizations: str        # JSON list
    speedup_min: float
    speedup_max: float
    status: str               # applied / recommended / failed
    dollar_saved_per_hour: float


class Database:
    """
    SQLite database for control plane.

    Usage:
        db = Database()
        db.init()
        db.upsert_node(record)
        db.insert_event(record)
        events = db.get_recent_events(limit=100)
    """

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=10,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")  # concurrent reads
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def init(self):
        """Create tables if they don't exist."""
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS nodes (
                    node_name TEXT PRIMARY KEY,
                    last_seen REAL NOT NULL,
                    gpu_count INTEGER DEFAULT 0,
                    total_vram_gb REAL DEFAULT 0,
                    active_processes INTEGER DEFAULT 0,
                    optimizations_applied INTEGER DEFAULT 0,
                    dollar_saved_today REAL DEFAULT 0,
                    dollar_saved_total REAL DEFAULT 0,
                    status TEXT DEFAULT 'online',
                    current_workloads TEXT DEFAULT '[]',
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    node_name TEXT NOT NULL,
                    pid INTEGER NOT NULL,
                    model_family TEXT NOT NULL,
                    gpu_ids TEXT NOT NULL,
                    optimizations TEXT NOT NULL,
                    speedup_min REAL DEFAULT 1.0,
                    speedup_max REAL DEFAULT 1.0,
                    status TEXT NOT NULL,
                    dollar_saved_per_hour REAL DEFAULT 0.0
                );

                CREATE INDEX IF NOT EXISTS idx_events_timestamp
                    ON events(timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_events_node
                    ON events(node_name);
                CREATE INDEX IF NOT EXISTS idx_events_status
                    ON events(status);

                CREATE TABLE IF NOT EXISTS metrics (
                    hour_bucket INTEGER NOT NULL,
                    node_name TEXT NOT NULL,
                    total_dollar_saved REAL DEFAULT 0,
                    total_optimizations INTEGER DEFAULT 0,
                    avg_speedup REAL DEFAULT 1.0,
                    PRIMARY KEY (hour_bucket, node_name)
                );
            """)
        log.info(f"Database initialized: {self.db_path}")

    def upsert_node(self, record: NodeRecord):
        """Insert or update node record."""
        now = time.time()
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO nodes (
                    node_name, last_seen, gpu_count, total_vram_gb,
                    active_processes, optimizations_applied,
                    dollar_saved_today, dollar_saved_total,
                    status, current_workloads, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(node_name) DO UPDATE SET
                    last_seen=excluded.last_seen,
                    gpu_count=excluded.gpu_count,
                    total_vram_gb=excluded.total_vram_gb,
                    active_processes=excluded.active_processes,
                    optimizations_applied=excluded.optimizations_applied,
                    dollar_saved_today=excluded.dollar_saved_today,
                    dollar_saved_total=excluded.dollar_saved_total,
                    status=excluded.status,
                    current_workloads=excluded.current_workloads,
                    updated_at=excluded.updated_at
            """, (
                record.node_name, record.last_seen, record.gpu_count,
                record.total_vram_gb, record.active_processes,
                record.optimizations_applied, record.dollar_saved_today,
                record.dollar_saved_total, record.status,
                record.current_workloads, now,
            ))

    def insert_event(self, record: EventRecord):
        """Append optimization event."""
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO events (
                    timestamp, node_name, pid, model_family,
                    gpu_ids, optimizations, speedup_min, speedup_max,
                    status, dollar_saved_per_hour
                ) VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                record.timestamp, record.node_name, record.pid,
                record.model_family, record.gpu_ids, record.optimizations,
                record.speedup_min, record.speedup_max,
                record.status, record.dollar_saved_per_hour,
            ))

    def mark_offline_nodes(self, timeout_seconds: int = 180):
        """Mark nodes offline if not seen recently."""
        cutoff = time.time() - timeout_seconds
        with self._connect() as conn:
            conn.execute("""
                UPDATE nodes SET status='offline'
                WHERE last_seen < ? AND status != 'offline'
            """, (cutoff,))

    def get_all_nodes(self) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT * FROM nodes ORDER BY node_name
            """).fetchall()
        return [dict(r) for r in rows]

    def get_node(self, node_name: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM nodes WHERE node_name=?", (node_name,)
            ).fetchone()
        return dict(row) if row else None

    def get_recent_events(
        self, limit: int = 100, node_name: str = None, status: str = None
    ) -> List[dict]:
        query = "SELECT * FROM events"
        params = []
        conditions = []
        if node_name:
            conditions.append("node_name=?")
            params.append(node_name)
        if status:
            conditions.append("status=?")
            params.append(status)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_cluster_summary(self) -> dict:
        """Aggregate cluster-wide metrics."""
        with self._connect() as conn:
            nodes = conn.execute("SELECT * FROM nodes").fetchall()
            total_events = conn.execute(
                "SELECT COUNT(*) as c FROM events WHERE status='applied'"
            ).fetchone()["c"]
            total_saved = conn.execute(
                "SELECT COALESCE(SUM(dollar_saved_per_hour),0) as s "
                "FROM events WHERE status='applied' "
                "AND timestamp > ?", (time.time() - 86400,)
            ).fetchone()["s"]
            total_saved_all = conn.execute(
                "SELECT COALESCE(SUM(dollar_saved_total),0) as s FROM nodes"
            ).fetchone()["s"]

        online = sum(1 for n in nodes if n["status"] == "online")
        total_gpus = sum(n["gpu_count"] for n in nodes)
        total_vram = sum(n["total_vram_gb"] for n in nodes)
        active_procs = sum(n["active_processes"] for n in nodes)

        return {
            "total_nodes": len(nodes),
            "online_nodes": online,
            "offline_nodes": len(nodes) - online,
            "total_gpus": total_gpus,
            "total_vram_gb": round(total_vram, 1),
            "active_processes": active_procs,
            "total_optimizations_applied": total_events,
            "dollar_saved_last_24h": round(total_saved, 2),
            "dollar_saved_total": round(total_saved_all, 2),
            "dollar_saved_per_year_estimate": round(total_saved * 365, 2),
        }
