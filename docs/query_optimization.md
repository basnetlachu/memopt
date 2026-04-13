# Query Optimization for Global Scale

Every hot control-plane query audited against 1M-node load.
"Full scan" below means every row of the table must be read,
which at 1M nodes × 100 boots = 100M rows for `boot_events`
is not survivable on a single-writer DB.

The audit drives three code changes:

1. `Database.dialect` property: returns `"sqlite"`, `"postgresql"`,
   or `"cockroachdb"` so query methods can emit the right SQL.
2. Hot queries that use `MAX(id) GROUP BY node_id` now emit
   `DISTINCT ON (node_id) ... ORDER BY node_id, booted_at DESC`
   on Postgres/CRDB, which uses the composite index for O(k)
   work where k = number of distinct nodes.
3. Migration `007_add_hot_path_indexes` creates the composite
   `(node_id, booted_at DESC)` index on SQLite so the SQLite
   fallback path also stays O(k) for the same queries.

---

## Hot Queries (called on every request / heartbeat)

### Query: latest boot per node

**Location:** `get_boot_status()`, `get_nodes_by_version()`,
`get_version_node_count()`, `get_nodes_on_version()`.

**Before:**
```sql
SELECT node_id, image_version, cert_status
FROM boot_events
WHERE id IN (
    SELECT MAX(id) FROM boot_events
    GROUP BY node_id
)
```

**Why slow:** `MAX(id) GROUP BY node_id` requires a full
`boot_events` scan followed by an `id IN (...)` lookup.
At 100M rows this is ~30 s+ on commodity DBs.

**After (Postgres / CockroachDB):**
```sql
SELECT DISTINCT ON (node_id)
    node_id, image_version, cert_status
FROM boot_events
ORDER BY node_id, booted_at DESC
```
With the `(node_id, booted_at DESC)` index this is an
index-only skip scan — O(k) where k is the number of
distinct nodes.

**After (SQLite — fallback for dev/test):**
Keeps the `MAX(id)` pattern because SQLite lacks `DISTINCT ON`,
but the new `idx_boot_events_node_time` composite index
(migration 007) turns the subquery into a covering index
scan.

**Index required:** `(node_id, booted_at DESC)` on
`boot_events` — present in the CRDB schema and added to
SQLite/Postgres via migration 007.

---

### Query: node lookup by MAC address

**Location:** `get_node_by_mac()` — called from
`/boot/config/{mac}` on every PXE boot.

**Before:**
```sql
SELECT * FROM nodes WHERE mac_address = ?
```

**Why slow:** Full scan without an index on `mac_address`.
At 1M nodes, every PXE boot does a full-table scan.

**After:** Same query — but migration 007 adds
`idx_nodes_mac_address ON nodes(mac_address)`,
which makes the lookup O(log n).

**Index required:** `mac_address` on `nodes`.

---

### Query: degraded-node listing

**Location:** `get_degraded_nodes()` — called by health
dashboards and the degradation cron.

**Before:**
```sql
SELECT * FROM nodes WHERE is_degraded = 1
ORDER BY degraded_since DESC
```

**Why slow at scale:** Without an index, the query scans
every node row; at 1M nodes most of which are healthy
this is wasted work.

**After:** Same query — migration 007 adds a partial
index `idx_nodes_degraded ON nodes(is_degraded, degraded_since DESC) WHERE is_degraded = 1`
on Postgres/CRDB; on SQLite a non-partial index covers it.

**Index required:** `(is_degraded, degraded_since DESC)` on `nodes`.

---

### Query: node count

**Location:** `get_cluster_summary()` reads
`SELECT COUNT(*) FROM nodes`.

**Why slow at scale:** `COUNT(*)` without a WHERE clause
is a full row count. Postgres does a seqscan; CRDB
distributes across ranges.

**After:** No query change needed today — the summary
endpoint is admin-only and reads ~infrequently. Leaving
as-is. If it becomes hot, switch to an
`approximate_count()` table statistic on CRDB or a
cached count maintained by the heartbeat batcher.

---

## Event queries

### Query: recent events

**Location:** `get_recent_events()`.

**Before:**
```sql
SELECT * FROM events
ORDER BY timestamp DESC
LIMIT ?
```

**Why fast enough:** `LIMIT ?` + `ORDER BY indexed col`
becomes an index range scan. With an index on `timestamp`
or a time-series partitioning scheme, this stays O(limit).

**No change required.** Migration 001 creates `events`
but no explicit `timestamp` index; the table's implicit
insertion order gives acceptable performance for the
development dataset. For CRDB production we rely on the
`idx_events_time (created_at DESC)` in the CRDB schema.

---

### Query: rollout events for a rollout ID

**Location:** `get_rollout_events()`.

**Before:**
```sql
SELECT * FROM rollout_events
WHERE rollout_id = ?
ORDER BY recorded_at ASC
```

**Why fast enough:** Migration 006 already creates
`idx_rollout_events_id ON rollout_events(rollout_id)`
so the `WHERE` clause hits the index. ORDER BY on an
indexed secondary column is acceptable for the low
cardinality per-rollout (dozens of events).

**No change required.**

---

## Non-hot queries (kept as-is)

- `get_node(node_name)` — PK lookup, already O(log n).
- `get_pod(pod_id)` / `list_pods()` — small table.
- `get_recent_events(limit)` — LIMIT-bounded.
- `list_image_versions()` — small table (hundreds of rows).
- `get_pending_rollback(node_id)` — indexed by
  `idx_rollback_node_id` (migration 005).
- `get_stable_version()` — LIMIT 1 on small table.

---

## Connection pool sizing

`make_backend_for_scale(url, expected_nodes)` picks pool
sizes based on expected scale:

| expected_nodes | min | max |
|:---------------|:----|:----|
| < 100          | 2   | 5   |
| < 1,000        | 5   | 20  |
| < 10,000       | 10  | 50  |
| >= 10,000      | 20  | 100 |

**Rationale:**
- Too small: concurrent heartbeats block on pool exhaustion
  and time out.
- Too large: exceeds CockroachDB `max_connections` (default
  ~500 across the cluster). Running 100 backends × 100-slot
  pools exhausts the server well before the clients notice.

SQLite backend ignores the sizing (single-file lock anyway)
but the helper keeps a uniform interface regardless of
target backend.
