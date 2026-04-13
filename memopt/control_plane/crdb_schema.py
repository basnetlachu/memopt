"""
CockroachDB schema for the memopt control plane.

Applied once against a new CRDB cluster:

    python -m memopt.control_plane.crdb_schema \
        --database-url cockroachdb://root@host:26257/memopt

The normal migration system (001-006) targets SQLite and PostgreSQL.
At CRDB scale the SERIAL primary keys those migrations produce become
range-leader hotspots — every INSERT goes to the same replica. This
module supplies the target schema with:

  - UUID primary keys (gen_random_uuid) — inserts spread evenly
  - Secondary indexes for the hot query paths
  - Columns ready for later PARTITION BY LIST(region) geo-partitioning
    (commented, applied separately on multi-region Enterprise clusters)

Safe to re-run: every statement uses IF NOT EXISTS.
"""
from __future__ import annotations

import logging
import sys

from memopt.control_plane.database import CockroachDBBackend

logger = logging.getLogger(__name__)


CRDB_SCHEMA = """
-- memopt control plane — CockroachDB target schema.
-- Optimized for high-throughput writes at 1M-node scale.

CREATE DATABASE IF NOT EXISTS memopt;
USE memopt;

-- nodes: one row per GPU node. High read, medium write (heartbeats
-- batched upstream so this table sees 1-per-second per batch).
CREATE TABLE IF NOT EXISTS nodes (
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    name            TEXT NOT NULL,
    hostname        TEXT,
    gpu_count       INT  DEFAULT 0,
    mac_address     TEXT,
    image_version   TEXT,
    rack            TEXT,
    pod             TEXT,
    region          TEXT,
    last_heartbeat  FLOAT,
    is_degraded     BOOL DEFAULT FALSE,
    degraded_since  FLOAT,
    drift_pct       FLOAT DEFAULT 0,
    created_at      FLOAT,
    UNIQUE INDEX idx_nodes_name  (name),
    INDEX        idx_nodes_rack   (rack),
    INDEX        idx_nodes_pod    (pod),
    INDEX        idx_nodes_region (region),
    INDEX        idx_nodes_mac    (mac_address)
);

-- pods: low-volume aggregate rows (one per 10K-node pod).
CREATE TABLE IF NOT EXISTS pods (
    pod_id          TEXT PRIMARY KEY,
    node_count      INT  DEFAULT 0,
    healthy_nodes   INT  DEFAULT 0,
    pod_oracle_size INT  DEFAULT 0,
    avg_hbm_free_gb FLOAT DEFAULT 0,
    gkd_hit_rate_pct FLOAT DEFAULT 0,
    last_reported_at FLOAT,
    created_at      FLOAT
);

-- events: append-only optimization audit log.
-- UUID key prevents insert hotspots.
CREATE TABLE IF NOT EXISTS events (
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    node_name       TEXT,
    event_type      TEXT,
    status          TEXT,
    metrics_before  TEXT,
    metrics_after   TEXT,
    created_at      FLOAT,
    INDEX idx_events_node (node_name),
    INDEX idx_events_type (event_type),
    INDEX idx_events_time (created_at DESC)
);

-- boot_events: high write volume during rollouts.
CREATE TABLE IF NOT EXISTS boot_events (
    id                UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    node_id           TEXT NOT NULL,
    image_version     TEXT,
    cert_status       TEXT,
    gpu_count         INT   DEFAULT 0,
    boot_time_seconds FLOAT DEFAULT 0,
    booted_at         FLOAT,
    INDEX idx_boot_node      (node_id),
    INDEX idx_boot_version   (image_version),
    INDEX idx_boot_time      (booted_at DESC),
    -- For "latest boot per node" queries driven by /boot/status:
    INDEX idx_boot_node_time (node_id, booted_at DESC)
);

-- image_versions: low volume, high read.
CREATE TABLE IF NOT EXISTS image_versions (
    version         TEXT PRIMARY KEY,
    git_commit      TEXT,
    build_date      TEXT,
    cuda_version    TEXT,
    sm_targets      TEXT,
    image_url       TEXT,
    digest          TEXT,
    registered_at   FLOAT,
    is_stable       BOOL DEFAULT FALSE,
    is_deprecated   BOOL DEFAULT FALSE,
    node_count      INT  DEFAULT 0
);

-- rollout_events: append-only canary rollout audit log.
CREATE TABLE IF NOT EXISTS rollout_events (
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    rollout_id      TEXT NOT NULL,
    event           TEXT NOT NULL,
    stage           TEXT,
    target_version  TEXT,
    details_json    TEXT,
    recorded_at     FLOAT,
    INDEX idx_rollout_id   (rollout_id),
    INDEX idx_rollout_time (recorded_at DESC)
);

-- rollback_intents: per-node pending rollbacks.
CREATE TABLE IF NOT EXISTS rollback_intents (
    id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    node_id         TEXT NOT NULL,
    target_version  TEXT NOT NULL,
    reason          TEXT,
    requested_at    FLOAT,
    executed_at     FLOAT,
    status          TEXT DEFAULT 'pending',
    INDEX idx_rollback_node   (node_id),
    INDEX idx_rollback_status (status)
);

-- canary_baselines: reference values for gate evaluation.
CREATE TABLE IF NOT EXISTS canary_baselines (
    id          UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    metric      TEXT NOT NULL,
    value       FLOAT NOT NULL,
    recorded_at FLOAT,
    INDEX idx_baseline_metric (metric)
);

-- metrics: rolling time-series (keep a TTL sweep job externally).
CREATE TABLE IF NOT EXISTS metrics (
    id          UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    node_name   TEXT,
    gpu_util    FLOAT,
    memory_used FLOAT,
    recorded_at FLOAT,
    INDEX idx_metrics_node (node_name),
    INDEX idx_metrics_time (recorded_at DESC)
);
"""

# Geo-partitioning: apply AFTER the schema above, only on multi-region
# Enterprise clusters. Keeps node data in the correct region for
# latency and GDPR. Documented here; not executed automatically.
#
#   ALTER TABLE nodes PARTITION BY LIST (region) (
#     PARTITION us_east VALUES IN ('us-east-1'),
#     PARTITION us_west VALUES IN ('us-west-2'),
#     PARTITION eu      VALUES IN ('eu-west-1'),
#     PARTITION default VALUES IN (DEFAULT)
#   );


def _split_statements(schema: str) -> list:
    """
    Split the schema into individual statements while preserving
    multi-line DDL. Strips SQL comments and empty statements.
    """
    out = []
    for raw in schema.split(";"):
        # Drop pure comments and blanks line-by-line
        lines = [
            ln for ln in raw.splitlines()
            if ln.strip() and not ln.strip().startswith("--")
        ]
        stmt = "\n".join(lines).strip()
        if stmt:
            out.append(stmt)
    return out


def apply_schema(database_url: str) -> None:
    """
    Apply CockroachDB schema to a cluster. Idempotent (IF NOT EXISTS).

    Raises on connection failure or a genuine SQL error; tolerates
    "already exists" errors because CRDB sometimes returns those
    even with IF NOT EXISTS when an index is racing.
    """
    backend = CockroachDBBackend(database_url)
    try:
        for stmt in _split_statements(CRDB_SCHEMA):
            try:
                backend.execute(stmt)
                logger.debug("Applied: %s...", stmt[:60])
            except Exception as e:
                if "already exists" in str(e).lower():
                    continue
                raise
        logger.info("CockroachDB schema applied")
    finally:
        backend.close()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply CockroachDB schema to a cluster")
    parser.add_argument(
        "--database-url", required=True,
        help="cockroachdb://user:pass@host:26257/db")
    args = parser.parse_args()

    try:
        apply_schema(args.database_url)
        print("Schema applied successfully")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
