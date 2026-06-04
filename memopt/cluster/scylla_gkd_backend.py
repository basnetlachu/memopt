"""
ScyllaDB-backed GKD store for large scale deployments (100K+ nodes).

Uses cassandra-driver (compatible with ScyllaDB and Apache Cassandra).
Falls back to Redis or local if driver not installed or connection fails.

Requires: pip install cassandra-driver

Configuration via environment:
  SCYLLA_HOSTS         comma-separated contact points (e.g. 10.0.0.1,10.0.0.2)
  SCYLLA_PORT          CQL port (default 9042)
  SCYLLA_KEYSPACE      keyspace name (default memopt)
  SCYLLA_CONSISTENCY   ONE | QUORUM | LOCAL_ONE | LOCAL_QUORUM (default ONE)
"""
from __future__ import annotations

import json
import logging
import time
from typing import List, Optional

logger = logging.getLogger(__name__)


class ScyllaGKDBackend:
    """
    ScyllaDB/Cassandra backend for cluster-wide GKD deduplication.

    Implements the same interface as LocalGKDBackend and RedisGKDBackend
    so GKDStore can use it transparently.

    Data model:
      gkd_entries:  (content_hash TEXT, data TEXT) with TTL
      Uses a single wide table for simplicity — content_hash is partition key.
      All GKD keys (gkd:v1:*, pfx:*) stored as rows.
    """

    _KEY_PREFIX = "gkd:v1:"

    def __init__(
        self,
        hosts: List[str],
        port: int = 9042,
        keyspace: str = "memopt",
        consistency: str = "ONE",
        ttl_s: int = 3600,
        connect_timeout: float = 5.0,
    ):
        from cassandra.cluster import Cluster
        from cassandra.policies import (
            DCAwareRoundRobinPolicy, RetryPolicy)
        from cassandra import ConsistencyLevel

        self._hosts = hosts
        self._keyspace = keyspace
        self._ttl_s = ttl_s
        self._degraded = False
        self._degraded_since: Optional[float] = None
        self._failures = 0

        self._cluster = Cluster(
            contact_points=hosts,
            port=port,
            connect_timeout=connect_timeout,
            load_balancing_policy=DCAwareRoundRobinPolicy(),
            default_retry_policy=RetryPolicy(),
        )
        self._session = self._cluster.connect()

        # Consistency level
        consistency_map = {
            "ONE": ConsistencyLevel.ONE,
            "QUORUM": ConsistencyLevel.QUORUM,
            "LOCAL_ONE": ConsistencyLevel.LOCAL_ONE,
            "LOCAL_QUORUM": ConsistencyLevel.LOCAL_QUORUM,
        }
        self._consistency = consistency_map.get(
            consistency.upper(), ConsistencyLevel.ONE)

        self._ensure_schema()

        # Prepared statements (compiled once by ScyllaDB)
        self._prep_get = self._session.prepare(
            f"SELECT data FROM {keyspace}.gkd_kv WHERE key = ?")
        self._prep_set = self._session.prepare(
            f"INSERT INTO {keyspace}.gkd_kv (key, data) "
            f"VALUES (?, ?) USING TTL ?")
        self._prep_delete = self._session.prepare(
            f"DELETE FROM {keyspace}.gkd_kv WHERE key = ?")

        self._prep_get.consistency_level = self._consistency
        self._prep_set.consistency_level = self._consistency

        logger.info(
            "ScyllaGKDBackend connected: hosts=%s consistency=%s",
            hosts, consistency)

    def _ensure_schema(self) -> None:
        """Create keyspace and table if not exist. Idempotent."""
        self._session.execute(f"""
            CREATE KEYSPACE IF NOT EXISTS {self._keyspace}
            WITH replication = {{
                'class': 'SimpleStrategy',
                'replication_factor': 1
            }}
        """)
        self._session.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._keyspace}.gkd_kv (
                key TEXT PRIMARY KEY,
                data TEXT
            ) WITH default_time_to_live = {self._ttl_s}
              AND compaction = {{
                'class': 'LeveledCompactionStrategy'
              }}
        """)

    # ── GKDBackend interface ──────────────────────────────────────────

    def get(self, content_hash: str) -> Optional[dict]:
        """Look up by key. Returns parsed dict or None."""
        try:
            key = self._make_key(content_hash)
            rows = self._session.execute(
                self._prep_get, (key,))
            row = rows.one()
            if row is None:
                return None
            return json.loads(row.data)
        except Exception as e:
            self._record_failure(e)
            return None

    def set(self, content_hash: str, entry,
            ttl_seconds: int = 0) -> None:
        """Store entry with TTL."""
        try:
            key = self._make_key(content_hash)
            if isinstance(entry, str):
                data = entry
            else:
                data = json.dumps(entry)
            ttl = ttl_seconds or self._ttl_s
            self._session.execute(
                self._prep_set, (key, data, ttl))
        except Exception as e:
            self._record_failure(e)

    def increment_hit(self, content_hash: str) -> None:
        """Increment hit count. Best-effort read-modify-write."""
        try:
            existing = self.get(content_hash)
            if existing and isinstance(existing, dict):
                existing["hit_count"] = \
                    existing.get("hit_count", 0) + 1
                self.set(content_hash, existing)
        except Exception:
            pass

    def delete(self, content_hash: str) -> None:
        """Delete a key."""
        try:
            key = self._make_key(content_hash)
            self._session.execute(
                self._prep_delete, (key,))
        except Exception as e:
            self._record_failure(e)

    def size(self) -> int:
        """Approximate row count. Expensive — use sparingly."""
        try:
            rows = self._session.execute(
                f"SELECT COUNT(*) as c "
                f"FROM {self._keyspace}.gkd_kv")
            return rows.one().c
        except Exception:
            return -1

    def clear(self) -> None:
        """Truncate the table. Use in tests only."""
        try:
            self._session.execute(
                f"TRUNCATE {self._keyspace}.gkd_kv")
        except Exception as e:
            self._record_failure(e)

    def pipeline_get(self, keys: List[str]) -> List[Optional[str]]:
        """
        Batch get using concurrent execution.
        ScyllaDB equivalent of Redis pipeline.
        Returns results in same order as keys.
        """
        if not keys:
            return []
        try:
            from cassandra.concurrent import \
                execute_concurrent_with_args

            full_keys = [self._make_key(k) for k in keys]
            results = execute_concurrent_with_args(
                self._session,
                self._prep_get,
                [(k,) for k in full_keys],
                concurrency=50,
                raise_on_first_error=False)

            output: List[Optional[str]] = []
            for success, result in results:
                if success and result:
                    row = result.one()
                    if row:
                        output.append(row.data)
                    else:
                        output.append(None)
                else:
                    output.append(None)
            return output

        except Exception as e:
            self._record_failure(e)
            return [None] * len(keys)

    # ── Health and stats ──────────────────────────────────────────────

    @property
    def degraded(self) -> bool:
        return self._degraded

    @property
    def degraded_since(self) -> Optional[float]:
        return self._degraded_since

    def is_healthy(self) -> bool:
        try:
            self._session.execute(
                "SELECT now() FROM system.local")
            if self._degraded:
                logger.info("ScyllaDB: recovered")
                self._degraded = False
                self._degraded_since = None
            return True
        except Exception:
            return False

    def stats(self) -> dict:
        return {
            "backend": "scylladb",
            "hosts": self._hosts,
            "keyspace": self._keyspace,
            "degraded": self._degraded,
            "degraded_since": self._degraded_since,
            "failure_count": self._failures,
        }

    def close(self) -> None:
        try:
            self._cluster.shutdown()
        except Exception:
            pass

    # ── Private ───────────────────────────────────────────────────────

    def _make_key(self, content_hash: str) -> str:
        """Prefix key if not already prefixed."""
        if content_hash.startswith("gkd:") or \
           content_hash.startswith("pfx:"):
            return content_hash
        return f"{self._KEY_PREFIX}{content_hash}"

    def _record_failure(self, error: Exception) -> None:
        self._failures += 1
        if not self._degraded:
            self._degraded = True
            self._degraded_since = time.time()
            logger.error("ScyllaDB degraded: %s", error)

    @staticmethod
    def _parse_key_static(key: str) -> dict:
        """Parse key format. Static for testing without connection."""
        try:
            parts = key.split(":")
            if len(parts) >= 3:
                return {"hash": parts[1],
                        "length": int(parts[2])}
        except Exception:
            pass
        return {"hash": key, "length": 0}
