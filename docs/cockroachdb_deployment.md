# CockroachDB Deployment Guide

Real operational doc — not marketing. Everything here is either
verified in code or flagged as unverified.

---

## When to use CockroachDB

Use CockroachDB when at least one is true:

- Node count exceeds ~10,000
- Sustained write throughput exceeds ~1,000 writes/sec
- Multi-region deployment is required (GDPR, latency)
- A single PostgreSQL instance is already the bottleneck

Keep **SQLite** when:

- Single-node deployment (local dev, single-box prod)
- Running the test suite
- Node count < 1,000

Keep **PostgreSQL** when:

- Single-region, < 10,000 nodes
- Operations team already runs Postgres at scale
- CockroachDB operational complexity isn't justified

memopt works with all three backends via `DATABASE_URL` — no code
changes when switching tiers.

---

## Architecture

A CockroachDB cluster needs **at least 3 nodes**. A single-node
CRDB has no replication and is strictly worse than Postgres.

Minimal start-up example (paraphrased from the CRDB docs):

```bash
# Node 1
cockroach start \
  --listen-addr=node1:26257 \
  --join=node1,node2,node3 \
  --store=/var/lib/cockroach

# Node 2
cockroach start \
  --listen-addr=node2:26257 \
  --join=node1,node2,node3 \
  --store=/var/lib/cockroach

# Node 3
cockroach start \
  --listen-addr=node3:26257 \
  --join=node1,node2,node3 \
  --store=/var/lib/cockroach

# On one node, initialize the cluster:
cockroach init --host=node1:26257
```

For memopt the DB is deployed alongside the control plane —
typically in the same region(s) but decoupled enough to failover
independently.

---

## Schema differences from SQLite / PostgreSQL

See `memopt/control_plane/crdb_schema.py` for the full target
schema. Key differences from the migration-001..007 chain used
by SQLite and PostgreSQL:

| Concern | SQLite / PG migrations | CRDB schema |
|:--------|:-----------------------|:------------|
| Primary keys | `INTEGER PRIMARY KEY AUTOINCREMENT` / `SERIAL` | `UUID DEFAULT gen_random_uuid()` |
| Latest-per-group queries | `WHERE id IN (SELECT MAX(id)…)` | `SELECT DISTINCT ON (node_id)…` |
| Composite index on boot_events | added by migration 007 | declared inline |
| Multi-statement bootstrap | `executescript()` | statement-by-statement |

The `Database.dialect` property (`"sqlite" | "postgresql" | "cockroachdb"`)
drives the branch in the four hot queries (`get_boot_status`,
`get_nodes_by_version`, `get_version_node_count`,
`get_nodes_on_version`). See `docs/query_optimization.md` for the
full query audit.

**UUID vs SERIAL:** at CRDB scale `SERIAL` (or `INTEGER PRIMARY KEY`)
causes an insert hotspot — every new row goes to the same range
leader. `UUID DEFAULT gen_random_uuid()` spreads inserts uniformly
across ranges. This matters for `boot_events`, `events`,
`rollout_events`, `metrics`, `rollback_intents`, and
`canary_baselines`. Natural-key tables (`pods.pod_id`,
`image_versions.version`) keep their `TEXT PRIMARY KEY`.

**DISTINCT ON vs MAX(id):** `MAX(id) GROUP BY node_id` forces a
full scan of `boot_events`. `DISTINCT ON (node_id) … ORDER BY
node_id, booted_at DESC` uses the composite index as an index-only
skip scan — O(distinct nodes) instead of O(total rows).

**No multi-statement transactions in schema:** CRDB accepts
multi-statement scripts over the wire, but `CREATE DATABASE` /
`USE` must be run individually. `crdb_schema.apply_schema()`
splits on `;` and runs each statement under its own transaction.

**Geo-partitioning:** available in CRDB Enterprise. The schema
declares `rack`, `pod`, `region` columns; the actual `PARTITION BY
LIST (region)` DDL is commented in `crdb_schema.py` and must be
applied separately on multi-region Enterprise clusters.

---

## Connection string

```text
DATABASE_URL=cockroachdb://memopt:PASSWORD@crdb-gateway:26257/memopt?sslmode=verify-full
```

- `cockroachdb://` is normalized to `postgresql://` inside
  `CockroachDBBackend.__init__` — psycopg2 doesn't understand the
  `cockroachdb://` scheme.
- `sslmode=verify-full` in production. `sslmode=disable` for local
  insecure clusters.
- `application_name=memopt` is appended automatically (or merged
  with an existing `?` query string) for server-side
  observability.

---

## Applying the schema

Once on a fresh CRDB cluster:

```bash
python -m memopt.control_plane.crdb_schema \
    --database-url cockroachdb://root@crdb:26257/memopt
```

Safe to re-run — every statement uses `IF NOT EXISTS` and the
wrapper tolerates `already exists` errors.

---

## Migrating from an existing deployment

```bash
# Dry-run first (no target writes)
./scripts/migrate_to_crdb.sh \
    --source sqlite:///$HOME/.memopt/control_plane/memopt.db \
    --target cockroachdb://root@crdb:26257/memopt \
    --dry-run

# If the dry-run report looks right, remove --dry-run
./scripts/migrate_to_crdb.sh \
    --source postgresql://memopt@pg:5432/memopt \
    --target cockroachdb://root@crdb:26257/memopt
```

Behavior:

- **Never** writes to the source.
- Applies the CRDB schema to the target (idempotent).
- Exports each table row-by-row from the source, inserts into the
  target, tolerating duplicate-key errors on re-runs.
- Verifies target row counts are ≥ source row counts per table;
  fewer rows exits with code 2.

The script runs a small Python payload inline — all real DB work
happens through `memopt.control_plane.database` which already
understands every dialect.

---

## Connection pool sizing

Use `make_backend_for_scale(url, expected_nodes)` when wiring the
control plane. Tier sizing:

| Expected nodes | min_conn | max_conn |
|:---------------|---------:|---------:|
| `< 100`        |        2 |        5 |
| `< 1,000`      |        5 |       20 |
| `< 10,000`     |       10 |       50 |
| `>= 10,000`    |       20 |      100 |

The upper cap (100 per control-plane replica) is chosen to stay
under CRDB's default `max_connections` across a typical 5-replica
control-plane deployment (5 × 100 = 500).

---

## Monitoring

Minimum metrics to watch per CRDB node:

- p99 query latency — target `< 10 ms` for point lookups
- Replication lag between replicas — should stay below a few
  seconds under normal load
- Connection pool utilization on each control-plane replica
- Write throughput vs cluster capacity
- Range count and rebalancing activity

Endpoints the control plane itself exposes that correlate with
DB health:

- `GET /api/v1/database/health` — dialect, read latency, ping
  latency, heartbeat batcher stats (auth required)
- `GET /api/v1/heartbeat-batcher/stats` — pending / flushed /
  errors / interval
- `GET /healthz` — overall control-plane liveness
- `GET /readyz` — rejects traffic when DB is unreachable

---

## Known limitations

These are honest, not a checklist of excuses.

1. **Geo-partitioning requires a CRDB Enterprise license.** The
   column set (`region`) is in place, but the `PARTITION BY LIST`
   DDL is documented (commented) rather than executed.
2. **Multi-region demands 3+ regions with sub-50ms inter-region
   latency.** Single-region "multi-AZ" isn't the same as CRDB
   multi-region — replication semantics differ.
3. **Schema migrations need care.** CRDB's `ALTER TABLE ADD
   COLUMN` is online but writes to every row. We guard with
   `IF NOT EXISTS` where CRDB supports it; the migration runner
   tolerates `duplicate column` errors on re-runs.
4. **Not yet tested at 1M nodes.** The query optimizations (DISTINCT
   ON + composite indexes) and connection pool sizing are correct
   in design but unmeasured at that scale. The design-partner phase
   with Nebius will validate; until then, treat the throughput
   numbers as predictions.
5. **`UUID` primary keys complicate some admin tooling.** Any
   script that hand-writes `WHERE id = 42` needs to use
   `WHERE id = '...'::uuid` or the textual cast form. The
   migration script uses `SELECT *` and column-name inserts so it
   sidesteps this.
6. **`executescript()` semantics differ by dialect.** SQLite runs
   a multi-statement script as one call; CRDB accepts it but
   not for `CREATE DATABASE` / `USE`. `apply_schema()` splits
   statements; ordinary migration SQL uses `executescript()` and
   works on both. Authors of new migrations should test both
   dialects.

---

## Appendix: minimum CRDB version

Tested against CockroachDB v23.1+. Earlier versions may lack
`gen_random_uuid()` as a built-in (use `uuid_v4()` instead) or
have different `DISTINCT ON` cost semantics.
