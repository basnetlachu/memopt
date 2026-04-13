"""
Database for memopt control plane.

Supports SQLite (development) and PostgreSQL (production).
Selection via DATABASE_URL environment variable:
  Not set or sqlite:///path → SQLite
  postgresql://user:pass@host/db → PostgreSQL

Thread-safe. Falls back to SQLite if PostgreSQL unavailable.
"""
import os
import re
import sqlite3
import time
import json
import logging
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Dict
from dataclasses import dataclass

log = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path.home() / ".memopt" / "control_plane" / "memopt.db"


@dataclass
class NodeRecord:
    node_name: str
    last_seen: float
    gpu_count: int
    total_vram_gb: float
    active_processes: int
    optimizations_applied: int
    dollar_saved_today: float
    dollar_saved_total: float
    status: str
    current_workloads: str


@dataclass
class EventRecord:
    id: Optional[int]
    timestamp: float
    node_name: str
    pid: int
    model_family: str
    gpu_ids: str
    optimizations: str
    speedup_min: float
    speedup_max: float
    status: str
    dollar_saved_per_hour: float


# ═══════════════════════════════════════════════════════════════════════════
# Database Backend Abstraction
# ═══════════════════════════════════════════════════════════════════════════

class DatabaseBackend(ABC):
    """Abstract database backend."""

    @abstractmethod
    def execute(self, query: str, params: tuple = ()) -> None:
        ...

    @abstractmethod
    def executescript(self, script: str) -> None:
        ...

    @abstractmethod
    def fetchall(self, query: str, params: tuple = ()) -> List[dict]:
        ...

    @abstractmethod
    def fetchone(self, query: str, params: tuple = ()) -> Optional[dict]:
        ...

    @abstractmethod
    def close(self) -> None:
        ...


def _pg_to_sqlite(query: str) -> str:
    """Convert $1,$2 placeholders to ? for SQLite."""
    return re.sub(r'\$\d+', '?', query)


class SQLiteBackend(DatabaseBackend):
    """SQLite backend — development and single-node."""

    def __init__(self, db_path: str = ""):
        if not db_path or db_path == ":memory:":
            self._db_path = db_path or ":memory:"
        else:
            self._db_path = db_path
            os.makedirs(os.path.dirname(
                os.path.abspath(db_path)), exist_ok=True)

        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            self._db_path, check_same_thread=False, timeout=10)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

    def execute(self, query, params=()):
        query = _pg_to_sqlite(query)
        with self._lock:
            self._conn.execute(query, params)
            self._conn.commit()

    def executescript(self, script):
        with self._lock:
            self._conn.executescript(script)
            self._conn.commit()

    def fetchall(self, query, params=()):
        query = _pg_to_sqlite(query)
        with self._lock:
            cur = self._conn.execute(query, params)
            return [dict(row) for row in cur.fetchall()]

    def fetchone(self, query, params=()):
        query = _pg_to_sqlite(query)
        with self._lock:
            cur = self._conn.execute(query, params)
            row = cur.fetchone()
            return dict(row) if row else None

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass


class PostgreSQLBackend(DatabaseBackend):
    """PostgreSQL backend — production scale."""

    def __init__(
        self,
        database_url: str,
        min_conn: int = 2,
        max_conn: int = 20,
    ):
        try:
            import psycopg2
            import psycopg2.pool
            import psycopg2.extras
            self._psycopg2 = psycopg2
            self._extras = psycopg2.extras
        except ImportError:
            raise ImportError(
                "psycopg2-binary required for PostgreSQL. "
                "Install: pip install psycopg2-binary")

        self._pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=min_conn, maxconn=max_conn, dsn=database_url)

    def execute(self, query, params=()):
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def executescript(self, script):
        """Execute multi-statement script."""
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(script)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def fetchall(self, query, params=()):
        conn = self._pool.getconn()
        try:
            with conn.cursor(
                    cursor_factory=
                    self._extras.RealDictCursor) as cur:
                cur.execute(query, params)
                return [dict(r) for r in cur.fetchall()]
        finally:
            self._pool.putconn(conn)

    def fetchone(self, query, params=()):
        conn = self._pool.getconn()
        try:
            with conn.cursor(
                    cursor_factory=
                    self._extras.RealDictCursor) as cur:
                cur.execute(query, params)
                row = cur.fetchone()
                return dict(row) if row else None
        finally:
            self._pool.putconn(conn)

    def close(self):
        try:
            self._pool.closeall()
        except Exception:
            pass


class CockroachDBBackend(DatabaseBackend):
    """
    CockroachDB backend for global-scale control plane.

    Wire-compatible with PostgreSQL via psycopg2; the `cockroachdb://`
    URL scheme is normalized to `postgresql://` so psycopg2 accepts
    it. The schema and query patterns differ from PostgreSQL:

      - UUID primary keys (not SERIAL) — prevents insert hotspots
      - No cross-region foreign keys
      - Geo-partitioning applied separately (multi-region Enterprise)

    Use `memopt.control_plane.crdb_schema.apply_schema()` to
    provision the schema against a new CRDB cluster; the normal
    SQLite/PostgreSQL migration runner is NOT used with CRDB.

    Requires: pip install psycopg2-binary.
    """

    def __init__(
        self,
        database_url: str,
        min_conn: int = 2,
        max_conn: int = 20,
        application_name: str = "memopt",
    ):
        try:
            import psycopg2
            import psycopg2.pool
            import psycopg2.extras
        except ImportError:
            raise ImportError(
                "psycopg2-binary required: "
                "pip install psycopg2-binary")

        # psycopg2 does not understand cockroachdb:// — normalize.
        normalized = database_url.replace(
            "cockroachdb://", "postgresql://")

        sep = "&" if "?" in normalized else "?"
        normalized = f"{normalized}{sep}application_name={application_name}"

        self._pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=min_conn, maxconn=max_conn, dsn=normalized)
        self._extras = psycopg2.extras
        self._psycopg2 = psycopg2

        self._verify_connection()

        log.info(
            "CockroachDBBackend connected: pool=%d-%d",
            min_conn, max_conn)

    def _verify_connection(self) -> None:
        """Log the server version, raise on failure."""
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT version()")
                row = cur.fetchone()
                version = row[0] if row else ""
            if "CockroachDB" in version:
                log.info("CockroachDB: %s", version[:80])
            else:
                # Wire-compatible PostgreSQL is acceptable too.
                log.info("Connected: %s", version[:50])
        finally:
            self._pool.putconn(conn)

    def execute(self, query: str, params: tuple = ()) -> None:
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def executescript(self, script: str) -> None:
        """
        Execute a multi-statement script. CockroachDB supports
        multi-statement transactions over the wire, but some
        schema statements (CREATE DATABASE / USE) must be run
        individually — prefer `crdb_schema.apply_schema()` for
        DDL bootstrap.
        """
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(script)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def executemany(
        self,
        query: str,
        params_list: list,
    ) -> None:
        """
        Bulk write helper. One transaction for the whole batch —
        dramatically fewer roundtrips than per-row execute().
        """
        if not params_list:
            return
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.executemany(query, params_list)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def fetchall(
        self, query: str, params: tuple = (),
    ) -> List[dict]:
        conn = self._pool.getconn()
        try:
            with conn.cursor(
                    cursor_factory=self._extras.RealDictCursor,
            ) as cur:
                cur.execute(query, params)
                return [dict(r) for r in cur.fetchall()]
        finally:
            self._pool.putconn(conn)

    def fetchone(
        self, query: str, params: tuple = (),
    ) -> Optional[dict]:
        conn = self._pool.getconn()
        try:
            with conn.cursor(
                    cursor_factory=self._extras.RealDictCursor,
            ) as cur:
                cur.execute(query, params)
                row = cur.fetchone()
                return dict(row) if row else None
        finally:
            self._pool.putconn(conn)

    def close(self) -> None:
        try:
            self._pool.closeall()
        except Exception:
            pass


def _make_sqlite_default() -> "SQLiteBackend":
    """SQLite at the default path, creating parent dirs as needed."""
    path = str(DEFAULT_DB_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    log.info("Control plane: SQLite (default) at %s", path)
    return SQLiteBackend(path)


def make_backend(database_url: str = "") -> DatabaseBackend:
    """
    Create a backend from a URL.

    Priority:
      cockroachdb:// → CockroachDBBackend
      postgresql:// / postgres:// → PostgreSQLBackend
      sqlite:// / empty → SQLiteBackend

    Falls back to SQLiteBackend on any connection / import failure.
    Never raises.
    """
    if not database_url:
        database_url = os.getenv("DATABASE_URL", "")

    # CockroachDB
    if database_url.startswith("cockroachdb"):
        try:
            backend = CockroachDBBackend(database_url)
            log.info("Control plane: CockroachDB")
            return backend
        except ImportError as e:
            log.warning(
                "CockroachDB: psycopg2 not installed (%s). "
                "Falling back to SQLite.", e)
        except Exception as e:
            log.warning(
                "CockroachDB connection failed: %s. "
                "Falling back to SQLite.", e)
        return _make_sqlite_default()

    # PostgreSQL
    if (database_url.startswith("postgresql")
            or database_url.startswith("postgres")):
        try:
            backend = PostgreSQLBackend(database_url)
            log.info("Control plane: PostgreSQL")
            return backend
        except ImportError as e:
            log.warning(
                "PostgreSQL: psycopg2 not installed (%s). "
                "Falling back to SQLite.", e)
        except Exception as e:
            log.warning(
                "PostgreSQL connection failed: %s. "
                "Falling back to SQLite.", e)
        return _make_sqlite_default()

    # SQLite explicit
    if database_url.startswith("sqlite"):
        path = database_url.replace("sqlite:///", "").strip()
        if not path:
            return _make_sqlite_default()
        os.makedirs(
            os.path.dirname(os.path.abspath(path)),
            exist_ok=True)
        log.info("Control plane: SQLite at %s", path)
        return SQLiteBackend(path)

    # Empty / unknown scheme → default SQLite
    if not database_url:
        return _make_sqlite_default()

    log.warning(
        "Unknown DATABASE_URL scheme, falling back to SQLite: %s",
        database_url[:50])
    return _make_sqlite_default()


def make_backend_for_scale(
    database_url: str,
    expected_nodes: int = 1000,
) -> DatabaseBackend:
    """
    Create a database backend whose connection pool is sized for
    the expected node count.

    Pool sizing (min_conn, max_conn):
      expected_nodes < 100:    (2, 5)
      expected_nodes < 1,000:  (5, 20)
      expected_nodes < 10,000: (10, 50)
      expected_nodes >= 10,000: (20, 100)

    SQLite ignores the sizes (it's a single-file lock). PostgreSQL
    and CockroachDB honor them. The caps are chosen to stay under
    CockroachDB's default max_connections across a typical cluster
    (5 control-plane replicas × 100 = 500).

    Falls back to SQLite on any connection / import failure.
    Never raises.
    """
    if expected_nodes < 100:
        min_conn, max_conn = 2, 5
    elif expected_nodes < 1_000:
        min_conn, max_conn = 5, 20
    elif expected_nodes < 10_000:
        min_conn, max_conn = 10, 50
    else:
        min_conn, max_conn = 20, 100

    if database_url.startswith("cockroachdb"):
        try:
            return CockroachDBBackend(
                database_url,
                min_conn=min_conn,
                max_conn=max_conn)
        except ImportError as e:
            log.warning(
                "CockroachDB: psycopg2 not installed (%s). "
                "Falling back to SQLite.", e)
        except Exception as e:
            log.warning(
                "CockroachDB connection failed: %s. "
                "Falling back to SQLite.", e)
        return _make_sqlite_default()

    if (database_url.startswith("postgresql")
            or database_url.startswith("postgres")):
        try:
            return PostgreSQLBackend(
                database_url,
                min_conn=min_conn,
                max_conn=max_conn)
        except ImportError as e:
            log.warning(
                "PostgreSQL: psycopg2 not installed (%s). "
                "Falling back to SQLite.", e)
        except Exception as e:
            log.warning(
                "PostgreSQL connection failed: %s. "
                "Falling back to SQLite.", e)
        return _make_sqlite_default()

    # SQLite or empty: pool sizing is a no-op, delegate.
    return make_backend(database_url)


# ═══════════════════════════════════════════════════════════════════════════
# Migrations
# ═══════════════════════════════════════════════════════════════════════════

MIGRATIONS = [
    {
        "name": "001_initial_schema",
        "sql": """
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
                is_degraded INTEGER DEFAULT 0,
                degraded_since REAL,
                drift_pct REAL DEFAULT 0,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                id SERIAL PRIMARY KEY,
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
            CREATE TABLE IF NOT EXISTS metrics (
                hour_bucket INTEGER NOT NULL,
                node_name TEXT NOT NULL,
                total_dollar_saved REAL DEFAULT 0,
                total_optimizations INTEGER DEFAULT 0,
                avg_speedup REAL DEFAULT 1.0,
                PRIMARY KEY (hour_bucket, node_name)
            );
        """,
    },
    {
        "name": "002_add_pods_table",
        "sql": """
            CREATE TABLE IF NOT EXISTS pods (
                pod_id TEXT PRIMARY KEY,
                node_count INTEGER DEFAULT 0,
                healthy_nodes INTEGER DEFAULT 0,
                pod_oracle_size INTEGER DEFAULT 0,
                avg_hbm_free_gb REAL DEFAULT 0,
                gkd_hit_rate_pct REAL DEFAULT 0,
                last_reported_at REAL,
                created_at REAL
            );
        """,
    },
    {
        "name": "003_add_boot_events_table",
        "sql": """
            CREATE TABLE IF NOT EXISTS boot_events (
                id SERIAL PRIMARY KEY,
                node_id TEXT NOT NULL,
                image_version TEXT,
                cert_status TEXT,
                gpu_count INTEGER DEFAULT 0,
                boot_time_seconds REAL DEFAULT 0,
                booted_at REAL
            );
            CREATE INDEX IF NOT EXISTS idx_boot_events_node_id
                ON boot_events(node_id);
            CREATE INDEX IF NOT EXISTS idx_boot_events_version
                ON boot_events(image_version);
        """,
        # SQLite ALTER TABLE has no IF NOT EXISTS for columns. The
        # migration runner applies each entry below and tolerates a
        # "duplicate column" error so reruns after partial success
        # (or a hand-modified DB) do not break.
        "alter_columns": [
            ("nodes", "mac_address",   "TEXT"),
            ("nodes", "image_version", "TEXT"),
            ("nodes", "rack",          "TEXT"),
            ("nodes", "pod",           "TEXT"),
            ("nodes", "region",        "TEXT"),
        ],
    },
    {
        "name": "004_add_image_versions_table",
        "sql": """
            CREATE TABLE IF NOT EXISTS image_versions (
                version       TEXT PRIMARY KEY,
                git_commit    TEXT,
                build_date    TEXT,
                cuda_version  TEXT,
                sm_targets    TEXT,
                image_url     TEXT,
                digest        TEXT,
                registered_at REAL,
                is_stable     INTEGER DEFAULT 0,
                is_deprecated INTEGER DEFAULT 0,
                node_count    INTEGER DEFAULT 0
            );
        """,
    },
    {
        "name": "005_add_rollback_intents",
        "sql": """
            CREATE TABLE IF NOT EXISTS rollback_intents (
                id             SERIAL PRIMARY KEY,
                node_id        TEXT NOT NULL,
                target_version TEXT NOT NULL,
                reason         TEXT,
                requested_at   REAL,
                executed_at    REAL,
                status         TEXT DEFAULT 'pending'
            );
            CREATE INDEX IF NOT EXISTS idx_rollback_node_id
                ON rollback_intents(node_id);
            CREATE INDEX IF NOT EXISTS idx_rollback_status
                ON rollback_intents(status);
        """,
    },
    {
        "name": "006_add_canary_tables",
        "sql": """
            CREATE TABLE IF NOT EXISTS rollout_events (
                id             SERIAL PRIMARY KEY,
                rollout_id     TEXT NOT NULL,
                event          TEXT NOT NULL,
                stage          TEXT,
                target_version TEXT,
                details_json   TEXT,
                recorded_at    REAL
            );
            CREATE INDEX IF NOT EXISTS idx_rollout_events_id
                ON rollout_events(rollout_id);

            CREATE TABLE IF NOT EXISTS canary_baselines (
                id          SERIAL PRIMARY KEY,
                metric      TEXT NOT NULL,
                value       REAL NOT NULL,
                recorded_at REAL
            );
            CREATE INDEX IF NOT EXISTS idx_canary_baselines_metric
                ON canary_baselines(metric);
        """,
    },
    {
        "name": "007_add_hot_path_indexes",
        "sql": """
            -- Composite index powering the "latest boot per node"
            -- queries (get_boot_status, get_nodes_by_version,
            -- get_version_node_count, get_nodes_on_version).
            CREATE INDEX IF NOT EXISTS idx_boot_events_node_time
                ON boot_events(node_id, booted_at DESC);

            -- MAC address lookup on every PXE boot.
            CREATE INDEX IF NOT EXISTS idx_nodes_mac_address
                ON nodes(mac_address);
        """,
        # Pre-migration-003 dev DBs may be missing the degradation
        # columns that migration 001 now declares. Ensure they exist
        # before the idx_nodes_is_degraded index is created.
        "alter_columns": [
            ("nodes", "is_degraded",    "INTEGER DEFAULT 0"),
            ("nodes", "degraded_since", "REAL"),
            ("nodes", "drift_pct",      "REAL DEFAULT 0"),
        ],
        "post_sql": """
            -- Degraded-node listing (small cardinality after
            -- filter, big cardinality before). Runs after the
            -- degradation columns are guaranteed to exist.
            CREATE INDEX IF NOT EXISTS idx_nodes_is_degraded
                ON nodes(is_degraded, degraded_since DESC);
        """,
    },
]

# SQLite uses INTEGER PRIMARY KEY AUTOINCREMENT, not SERIAL
_SQLITE_MIGRATIONS = []
for m in MIGRATIONS:
    sqlite_m = {
        "name": m["name"],
        "sql": m["sql"].replace("SERIAL PRIMARY KEY",
                                "INTEGER PRIMARY KEY AUTOINCREMENT"),
    }
    if "alter_columns" in m:
        sqlite_m["alter_columns"] = m["alter_columns"]
    if "post_sql" in m:
        sqlite_m["post_sql"] = m["post_sql"]
    _SQLITE_MIGRATIONS.append(sqlite_m)


# ═══════════════════════════════════════════════════════════════════════════
# Database class — public API unchanged
# ═══════════════════════════════════════════════════════════════════════════

class Database:
    """
    Database for control plane.
    Uses DatabaseBackend abstraction for SQLite/PostgreSQL.
    All public methods unchanged from original.
    """

    def __init__(self, db_path: Path = None, backend: DatabaseBackend = None):
        if backend is not None:
            self._db = backend
            self._is_pg = isinstance(backend, PostgreSQLBackend)
        else:
            url = os.getenv("DATABASE_URL", "")
            if url and url.startswith("postgresql"):
                self._db = make_backend(url)
                self._is_pg = True
            else:
                path = str(db_path or DEFAULT_DB_PATH)
                self._db = SQLiteBackend(path)
                self._is_pg = False

    @property
    def dialect(self) -> str:
        """
        Return the SQL dialect name for the backing store.

          "sqlite"       — SQLiteBackend
          "cockroachdb"  — CockroachDBBackend
          "postgresql"   — PostgreSQLBackend (or anything else)

        Used by hot-path queries that emit different SQL per
        dialect (e.g. DISTINCT ON is Postgres/CRDB-only).
        """
        if isinstance(self._db, SQLiteBackend):
            return "sqlite"
        # CockroachDBBackend subclasses nothing else — but check
        # first because it wraps psycopg2 just like PostgreSQLBackend.
        try:
            if isinstance(self._db, CockroachDBBackend):
                return "cockroachdb"
        except NameError:
            pass
        return "postgresql"

    def init(self):
        """Create tables via migration system."""
        self._run_migrations()
        log.info("Database initialized")

    def _run_migrations(self):
        """Run schema migrations. Idempotent."""
        # Create migrations table
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS migrations (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                applied_at REAL NOT NULL
            );
        """)

        migrations = MIGRATIONS if self._is_pg else _SQLITE_MIGRATIONS

        for m in migrations:
            existing = self._db.fetchone(
                "SELECT id FROM migrations WHERE name = ?",
                (m["name"],))
            if not existing:
                self._db.executescript(m["sql"])
                # Apply per-column ALTER TABLE entries, tolerating
                # "duplicate column" errors (no IF NOT EXISTS on
                # SQLite ALTER TABLE ADD COLUMN).
                for table, col, col_type in m.get("alter_columns", []):
                    try:
                        self._db.execute(
                            f"ALTER TABLE {table} "
                            f"ADD COLUMN {col} {col_type}")
                    except Exception as e:
                        if "duplicate column" in str(e).lower():
                            pass  # already applied
                        else:
                            raise
                # Optional post-DDL (runs after alter_columns so
                # indexes can reference just-added columns).
                if m.get("post_sql"):
                    self._db.executescript(m["post_sql"])
                self._db.execute(
                    "INSERT INTO migrations (name, applied_at) "
                    "VALUES (?, ?)",
                    (m["name"], time.time()))
                log.info("Applied migration: %s", m["name"])

    # Alias for backward compatibility
    migrate = _run_migrations

    # ── Node operations ───────────────────────────────────────────────

    def upsert_node(self, record: NodeRecord):
        now = time.time()
        self._db.execute("""
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
        self._db.execute("""
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
        cutoff = time.time() - timeout_seconds
        self._db.execute("""
            UPDATE nodes SET status='offline'
            WHERE last_seen < ? AND status != 'offline'
        """, (cutoff,))

    def get_all_nodes(self) -> List[dict]:
        return self._db.fetchall(
            "SELECT * FROM nodes ORDER BY node_name")

    def get_node(self, node_name: str) -> Optional[dict]:
        return self._db.fetchone(
            "SELECT * FROM nodes WHERE node_name=?",
            (node_name,))

    def get_recent_events(
        self, limit: int = 100,
        node_name: str = None, status: str = None
    ) -> List[dict]:
        query = "SELECT * FROM events"
        params: list = []
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
        return self._db.fetchall(query, tuple(params))

    def get_cluster_summary(self) -> dict:
        nodes = self._db.fetchall("SELECT * FROM nodes")
        total_events_row = self._db.fetchone(
            "SELECT COUNT(*) as c FROM events "
            "WHERE status='applied'")
        total_events = total_events_row["c"] if total_events_row else 0

        cutoff = time.time() - 86400
        saved_row = self._db.fetchone(
            "SELECT COALESCE(SUM(dollar_saved_per_hour),0) as s "
            "FROM events WHERE status='applied' "
            "AND timestamp > ?", (cutoff,))
        total_saved = saved_row["s"] if saved_row else 0

        saved_all_row = self._db.fetchone(
            "SELECT COALESCE(SUM(dollar_saved_total),0) as s "
            "FROM nodes")
        total_saved_all = saved_all_row["s"] if saved_all_row else 0

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
            "dollar_saved_per_year_estimate":
                round(total_saved * 365, 2),
        }

    def update_node_status(
        self, node_name: str, healthy: bool,
        degraded: bool, drift_pct: float = 0.0,
        reason: str = "",
    ) -> bool:
        now = time.time()
        status = "online" if healthy else (
            "degraded" if degraded else "offline")
        degraded_since = now if degraded else None

        existing = self._db.fetchone(
            "SELECT node_name FROM nodes WHERE node_name=?",
            (node_name,))

        if existing:
            self._db.execute("""
                UPDATE nodes SET
                    status=?, is_degraded=?,
                    degraded_since=CASE WHEN ?
                        THEN COALESCE(degraded_since, ?)
                        ELSE NULL END,
                    drift_pct=?, updated_at=?
                WHERE node_name=?
            """, (status, 1 if degraded else 0,
                  degraded, degraded_since,
                  drift_pct, now, node_name))
        else:
            self._db.execute("""
                INSERT INTO nodes (
                    node_name, last_seen, gpu_count, total_vram_gb,
                    active_processes, optimizations_applied,
                    dollar_saved_today, dollar_saved_total,
                    status, current_workloads,
                    is_degraded, degraded_since, drift_pct,
                    updated_at
                ) VALUES (?,?,0,0,0,0,0,0,?,?,?,?,?,?)
            """, (node_name, now, status, "[]",
                  1 if degraded else 0, degraded_since,
                  drift_pct, now))
        return True

    def get_degraded_nodes(self) -> List[dict]:
        return self._db.fetchall(
            "SELECT * FROM nodes WHERE is_degraded = 1 "
            "ORDER BY degraded_since DESC")

    # ── Pod operations ────────────────────────────────────────────────

    def upsert_pod(self, pod_id: str, stats: dict):
        now = time.time()
        existing = self._db.fetchone(
            "SELECT pod_id FROM pods WHERE pod_id=?",
            (pod_id,))
        if existing:
            self._db.execute("""
                UPDATE pods SET
                    node_count=?, healthy_nodes=?,
                    pod_oracle_size=?, avg_hbm_free_gb=?,
                    gkd_hit_rate_pct=?, last_reported_at=?
                WHERE pod_id=?
            """, (
                stats.get("node_count", 0),
                stats.get("healthy_nodes", 0),
                stats.get("pod_oracle_size", 0),
                stats.get("avg_hbm_free_gb", 0),
                stats.get("gkd_hit_rate_pct", 0),
                stats.get("reported_at", now),
                pod_id,
            ))
        else:
            self._db.execute("""
                INSERT INTO pods (
                    pod_id, node_count, healthy_nodes,
                    pod_oracle_size, avg_hbm_free_gb,
                    gkd_hit_rate_pct, last_reported_at,
                    created_at
                ) VALUES (?,?,?,?,?,?,?,?)
            """, (
                pod_id,
                stats.get("node_count", 0),
                stats.get("healthy_nodes", 0),
                stats.get("pod_oracle_size", 0),
                stats.get("avg_hbm_free_gb", 0),
                stats.get("gkd_hit_rate_pct", 0),
                stats.get("reported_at", now),
                now,
            ))

    def get_pod(self, pod_id: str) -> Optional[dict]:
        return self._db.fetchone(
            "SELECT * FROM pods WHERE pod_id=?",
            (pod_id,))

    def list_pods(self) -> List[dict]:
        return self._db.fetchall(
            "SELECT * FROM pods ORDER BY pod_id")

    # ── Boot / PXE operations ─────────────────────────────────────────

    def record_boot_event(
        self,
        node_id: str,
        image_version: str,
        cert_status: str,
        gpu_count: int,
        boot_time_seconds: float,
    ) -> None:
        """Record a node boot event from /boot/callback."""
        self._db.execute(
            """INSERT INTO boot_events
               (node_id, image_version, cert_status,
                gpu_count, boot_time_seconds, booted_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (node_id, image_version, cert_status,
             int(gpu_count), float(boot_time_seconds),
             time.time()))

    def get_boot_status(self) -> dict:
        """Aggregate boot status across cluster (latest boot per node)."""
        try:
            if self.dialect == "sqlite":
                # SQLite lacks DISTINCT ON. With the composite
                # (node_id, booted_at DESC) index from migration
                # 007, the MAX(id) subquery is covering and cheap.
                rows = self._db.fetchall("""
                    SELECT node_id, image_version, cert_status
                    FROM boot_events
                    WHERE id IN (
                        SELECT MAX(id)
                        FROM boot_events
                        GROUP BY node_id
                    )
                """)
            else:
                # Postgres / CockroachDB: index-only skip scan.
                rows = self._db.fetchall("""
                    SELECT DISTINCT ON (node_id)
                        node_id, image_version, cert_status
                    FROM boot_events
                    ORDER BY node_id, booted_at DESC
                """)

            version_dist: Dict[str, int] = {}
            failed_certs = 0

            for row in rows:
                v = row.get("image_version") or "unknown"
                version_dist[v] = version_dist.get(v, 0) + 1
                if row.get("cert_status") == "FAILED":
                    failed_certs += 1

            current = (
                max(version_dist, key=version_dist.get)
                if version_dist else "unknown")

            return {
                "total_nodes":         len(rows),
                "booted_nodes":        len(rows),
                "current_version":     current,
                "version_distribution": version_dist,
                "failed_certs":        failed_certs,
            }
        except Exception as e:
            log.debug("get_boot_status failed: %s", e)
            return {
                "total_nodes":         0,
                "booted_nodes":        0,
                "current_version":     "unknown",
                "version_distribution": {},
                "failed_certs":        0,
            }

    def get_node_by_mac(self, mac: str) -> Optional[dict]:
        """Look up node by MAC address. Returns None if not found."""
        try:
            return self._db.fetchone(
                "SELECT * FROM nodes WHERE mac_address = ?",
                (mac,))
        except Exception:
            return None

    # ── Image version registry ────────────────────────────────────────

    def register_image_version(
        self,
        version: str,
        git_commit: str,
        build_date: str,
        cuda_version: str,
        sm_targets: str,
        image_url: str,
        digest: str = "",
    ) -> None:
        """Register a new image version (upsert)."""
        self._db.execute(
            """INSERT OR REPLACE INTO image_versions
               (version, git_commit, build_date, cuda_version,
                sm_targets, image_url, digest, registered_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (version, git_commit, build_date, cuda_version,
             sm_targets, image_url, digest, time.time()))

    def mark_version_stable(self, version: str) -> None:
        """Mark an image version as stable."""
        self._db.execute(
            "UPDATE image_versions SET is_stable = 1 "
            "WHERE version = ?",
            (version,))

    def mark_version_deprecated(self, version: str) -> None:
        """Mark version as deprecated."""
        self._db.execute(
            "UPDATE image_versions SET is_deprecated = 1 "
            "WHERE version = ?",
            (version,))

    def get_stable_version(self) -> Optional[str]:
        """Latest stable, non-deprecated image version."""
        row = self._db.fetchone(
            """SELECT version FROM image_versions
               WHERE is_stable = 1 AND is_deprecated = 0
               ORDER BY registered_at DESC
               LIMIT 1""")
        return row["version"] if row else None

    def get_version_node_count(self, version: str) -> int:
        """Number of nodes currently running this version."""
        try:
            if self.dialect == "sqlite":
                row = self._db.fetchone(
                    """SELECT COUNT(*) as cnt
                       FROM boot_events
                       WHERE image_version = ?
                       AND id IN (
                           SELECT MAX(id) FROM boot_events
                           GROUP BY node_id
                       )""",
                    (version,))
                return int(row["cnt"]) if row else 0

            # Postgres / CockroachDB: DISTINCT ON + subquery count.
            row = self._db.fetchone(
                """SELECT COUNT(*) AS cnt FROM (
                       SELECT DISTINCT ON (node_id)
                           image_version
                       FROM boot_events
                       ORDER BY node_id, booted_at DESC
                   ) t
                   WHERE image_version = ?""",
                (version,))
            return int(row["cnt"]) if row else 0
        except Exception:
            return 0

    def get_nodes_on_version(self, version: str) -> List[str]:
        """Return the node_ids whose latest boot is on this version."""
        try:
            if self.dialect == "sqlite":
                rows = self._db.fetchall(
                    """SELECT node_id FROM boot_events
                       WHERE image_version = ?
                       AND id IN (
                           SELECT MAX(id) FROM boot_events
                           GROUP BY node_id
                       )""",
                    (version,))
                return [r["node_id"] for r in rows]

            rows = self._db.fetchall(
                """SELECT node_id FROM (
                       SELECT DISTINCT ON (node_id)
                           node_id, image_version
                       FROM boot_events
                       ORDER BY node_id, booted_at DESC
                   ) t
                   WHERE image_version = ?""",
                (version,))
            return [r["node_id"] for r in rows]
        except Exception:
            return []

    def list_image_versions(self) -> List[dict]:
        """List all registered image versions, newest first."""
        return self._db.fetchall(
            "SELECT * FROM image_versions "
            "ORDER BY registered_at DESC")

    # ── Rollback intents ──────────────────────────────────────────────

    def record_rollback_intent(
        self,
        node_id: str,
        target_version: str,
        reason: str,
    ) -> None:
        """Record pending rollback intent for a node."""
        self._db.execute(
            """INSERT INTO rollback_intents
               (node_id, target_version, reason, requested_at, status)
               VALUES (?, ?, ?, ?, 'pending')""",
            (node_id, target_version, reason, time.time()))

    def get_pending_rollback(
        self, node_id: str,
    ) -> Optional[dict]:
        """
        Get pending rollback for a node.
        Called by /boot/config/{mac} to check whether this node
        should boot a different version on its next reboot.
        """
        try:
            return self._db.fetchone(
                """SELECT * FROM rollback_intents
                   WHERE node_id = ? AND status = 'pending'
                   ORDER BY requested_at DESC
                   LIMIT 1""",
                (node_id,))
        except Exception:
            return None

    # ── Canary rollout operations ─────────────────────────────────────

    def record_rollout_event(
        self,
        rollout_id: str,
        event: str,
        stage: str,
        target_version: str,
        details: dict,
    ) -> None:
        """Append a rollout audit event."""
        self._db.execute(
            """INSERT INTO rollout_events
               (rollout_id, event, stage, target_version,
                details_json, recorded_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (rollout_id, event, stage, target_version,
             json.dumps(details or {}), time.time()))

    def get_rollout_events(self, rollout_id: str) -> List[dict]:
        """All audit events for a rollout, oldest first."""
        try:
            return self._db.fetchall(
                """SELECT * FROM rollout_events
                   WHERE rollout_id = ?
                   ORDER BY recorded_at ASC""",
                (rollout_id,))
        except Exception:
            return []

    def get_nodes_by_version(
        self, image_version: str,
    ) -> List[dict]:
        """
        Nodes whose latest boot event is on `image_version`.
        Returns list of {node_id, hostname}. We use node_id as
        hostname since the boot callback doesn't carry a separate
        hostname field yet.
        """
        try:
            if self.dialect == "sqlite":
                return self._db.fetchall(
                    """SELECT node_id,
                              node_id AS hostname
                       FROM boot_events
                       WHERE image_version = ?
                       AND id IN (
                           SELECT MAX(id) FROM boot_events
                           GROUP BY node_id
                       )""",
                    (image_version,))

            return self._db.fetchall(
                """SELECT node_id, node_id AS hostname
                   FROM (
                       SELECT DISTINCT ON (node_id)
                           node_id, image_version
                       FROM boot_events
                       ORDER BY node_id, booted_at DESC
                   ) t
                   WHERE image_version = ?""",
                (image_version,))
        except Exception:
            return []


# Alias used by newer code / tests — same class.
ControlPlaneDB = Database
