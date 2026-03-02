"""
Persistent storage for drift alerts.
Appends a drift_alerts table to the existing SQLite database used by
memopt/control_plane/database.py.  Same file, same WAL journal — no
extra process or DB file needed.
"""
import json
import sqlite3
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Inline the path rather than importing through control_plane (whose __init__
# pulls in FastAPI via server.py, making alert_store unnecessarily heavy).
_DEFAULT_DB_PATH = Path.home() / ".memopt" / "control_plane" / "memopt.db"

log = logging.getLogger(__name__)


@dataclass
class DriftAlert:
    pid: int
    node_name: str
    model_family: str
    gpu_ids: List[int]
    optimization_timestamp: float       # when optimization was applied
    detection_timestamp: float          # when drift was detected
    baseline_util_pct: float            # utilization right after optimization
    current_util_pct: float             # utilization at detection time
    util_drop_pct: float                # percentage drop (0–100)
    original_speedup_min: float
    original_speedup_max: float
    optimizations_originally_applied: List[str]
    severity: str                       # info / warning / critical
    recommended_action: str
    resolved: bool = False
    id: Optional[int] = None


class AlertStore:
    """
    Stores drift alerts in SQLite.

    Uses the same database as control_plane for simplicity — no
    second DB file, no second connection pool.  All tables use WAL mode.
    """

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or _DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_table()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_table(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS drift_alerts (
                    id                              INTEGER PRIMARY KEY AUTOINCREMENT,
                    pid                             INTEGER NOT NULL,
                    node_name                       TEXT    NOT NULL,
                    model_family                    TEXT    NOT NULL,
                    gpu_ids                         TEXT    NOT NULL,
                    optimization_timestamp          REAL    NOT NULL,
                    detection_timestamp             REAL    NOT NULL,
                    baseline_util_pct               REAL    NOT NULL,
                    current_util_pct                REAL    NOT NULL,
                    util_drop_pct                   REAL    NOT NULL,
                    original_speedup_min            REAL    NOT NULL,
                    original_speedup_max            REAL    NOT NULL,
                    optimizations_originally_applied TEXT   NOT NULL,
                    severity                        TEXT    NOT NULL,
                    recommended_action              TEXT    NOT NULL,
                    resolved                        INTEGER DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_drift_detection
                    ON drift_alerts(detection_timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_drift_resolved
                    ON drift_alerts(resolved, severity);
            """)

    def save_alert(self, alert: DriftAlert) -> int:
        """Insert alert, return its row id."""
        with self._connect() as conn:
            cursor = conn.execute("""
                INSERT INTO drift_alerts (
                    pid, node_name, model_family, gpu_ids,
                    optimization_timestamp, detection_timestamp,
                    baseline_util_pct, current_util_pct, util_drop_pct,
                    original_speedup_min, original_speedup_max,
                    optimizations_originally_applied,
                    severity, recommended_action, resolved
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                alert.pid, alert.node_name, alert.model_family,
                json.dumps(alert.gpu_ids),
                alert.optimization_timestamp, alert.detection_timestamp,
                alert.baseline_util_pct, alert.current_util_pct,
                alert.util_drop_pct,
                alert.original_speedup_min, alert.original_speedup_max,
                json.dumps(alert.optimizations_originally_applied),
                alert.severity, alert.recommended_action,
                int(alert.resolved),
            ))
            return cursor.lastrowid

    def get_active_alerts(
        self, node_name: str = None, severity: str = None
    ) -> List[dict]:
        """Return unresolved alerts, newest first."""
        query = "SELECT * FROM drift_alerts WHERE resolved=0"
        params: list = []
        if node_name:
            query += " AND node_name=?"
            params.append(node_name)
        if severity:
            query += " AND severity=?"
            params.append(severity)
        query += " ORDER BY detection_timestamp DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def resolve_alert(self, alert_id: int):
        """Mark alert resolved (operator acknowledged + acted)."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE drift_alerts SET resolved=1 WHERE id=?", (alert_id,)
            )

    def get_all_alerts(self, limit: int = 100) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM drift_alerts ORDER BY detection_timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def count_active_by_severity(self) -> dict:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT severity, COUNT(*) as count
                FROM drift_alerts WHERE resolved=0
                GROUP BY severity
            """).fetchall()
        return {r["severity"]: r["count"] for r in rows}
