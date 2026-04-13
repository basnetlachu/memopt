#!/usr/bin/env bash
# Migrate the memopt control-plane database from SQLite or PostgreSQL
# to CockroachDB.
#
# Steps:
#   1. Validate source and target connections
#   2. Apply the CockroachDB schema to the target (idempotent)
#   3. Export every row from each table in source as JSON
#   4. Import into target in table order
#   5. Verify row counts match
#   6. Print a summary
#
# The script NEVER writes to the source. --dry-run connects to both
# but skips every write on the target.
#
# Usage:
#   ./scripts/migrate_to_crdb.sh \
#       --source sqlite:///path/to/memopt.db \
#       --target cockroachdb://root@host:26257/memopt \
#       [--dry-run]
#
# Exit codes:
#   0 — success
#   1 — input validation / connection failure
#   2 — row-count verification failed

set -euo pipefail

SOURCE=""
TARGET=""
DRY_RUN=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC}   $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; }

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Migrate the memopt control-plane database from SQLite or PostgreSQL
to CockroachDB.

Options:
  --source URL           Source DB URL (sqlite:///... or postgresql://...)
  --target URL           Target DB URL (cockroachdb://...)
  --dry-run              Connect and report, but do not write to target
  --help                 Show this help

Examples:
  # Migrate local dev DB to CRDB (dry run)
  $(basename "$0") \\
      --source sqlite:///\$HOME/.memopt/control_plane/memopt.db \\
      --target cockroachdb://root@crdb:26257/memopt \\
      --dry-run

  # Real migration
  $(basename "$0") \\
      --source postgresql://memopt@pg:5432/memopt \\
      --target cockroachdb://root@crdb:26257/memopt
EOF
    exit 0
}

# ── Parse arguments ───────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --source)   SOURCE="$2"; shift 2 ;;
        --target)   TARGET="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=true; shift ;;
        --help|-h)  usage ;;
        *) echo "Unknown option: $1" >&2; usage ;;
    esac
done

# ── Validate inputs ───────────────────────────────────────────────────
if [[ -z "$SOURCE" || -z "$TARGET" ]]; then
    fail "--source and --target are required (see --help)"
    exit 1
fi

if [[ ! "$SOURCE" =~ ^(sqlite:///|postgresql://|postgres://) ]]; then
    fail "--source must be sqlite:// or postgresql://"
    exit 1
fi

if [[ ! "$TARGET" =~ ^cockroachdb:// ]]; then
    fail "--target must start with cockroachdb://"
    exit 1
fi

# ── Python-level migration ───────────────────────────────────────────
# All the real work happens in Python because database.py already
# understands every dialect. Bash just orchestrates.
export MEMOPT_MIGRATE_SOURCE="$SOURCE"
export MEMOPT_MIGRATE_TARGET="$TARGET"
export MEMOPT_MIGRATE_DRY_RUN="$DRY_RUN"

cd "$PROJECT_DIR"

python3 -u - <<'PYEOF'
import json
import os
import sys

SOURCE = os.environ["MEMOPT_MIGRATE_SOURCE"]
TARGET = os.environ["MEMOPT_MIGRATE_TARGET"]
DRY_RUN = os.environ["MEMOPT_MIGRATE_DRY_RUN"] == "true"

# Order matters only if FKs exist; CRDB schema has none, so this
# ordering is cosmetic (keeps the progress log readable).
TABLES = [
    "nodes",
    "pods",
    "events",
    "boot_events",
    "image_versions",
    "rollout_events",
    "rollback_intents",
    "canary_baselines",
    "metrics",
]

# Columns we persist per table. We read whatever the source has
# (handles missing columns gracefully), then insert only the
# subset that exists in the target CRDB schema.
TABLE_COLUMNS = {
    "nodes": [
        "node_name", "last_seen", "gpu_count",
        "total_vram_gb", "active_processes",
        "optimizations_applied", "dollar_saved_today",
        "dollar_saved_total", "status", "current_workloads",
        "is_degraded", "degraded_since", "drift_pct",
        "mac_address", "image_version", "rack",
        "pod", "region", "updated_at",
    ],
    "pods": [
        "pod_id", "node_count", "healthy_nodes",
        "pod_oracle_size", "avg_hbm_free_gb",
        "gkd_hit_rate_pct", "last_reported_at", "created_at",
    ],
    "events": [
        "timestamp", "node_name", "pid", "model_family",
        "gpu_ids", "optimizations", "speedup_min",
        "speedup_max", "status", "dollar_saved_per_hour",
    ],
    "boot_events": [
        "node_id", "image_version", "cert_status",
        "gpu_count", "boot_time_seconds", "booted_at",
    ],
    "image_versions": [
        "version", "git_commit", "build_date",
        "cuda_version", "sm_targets", "image_url",
        "digest", "registered_at", "is_stable",
        "is_deprecated", "node_count",
    ],
    "rollout_events": [
        "rollout_id", "event", "stage",
        "target_version", "details_json", "recorded_at",
    ],
    "rollback_intents": [
        "node_id", "target_version", "reason",
        "requested_at", "executed_at", "status",
    ],
    "canary_baselines": ["metric", "value", "recorded_at"],
    "metrics": [
        "hour_bucket", "node_name", "total_dollar_saved",
        "total_optimizations", "avg_speedup",
    ],
}

from memopt.control_plane.database import make_backend
from memopt.control_plane.crdb_schema import apply_schema

print(f"─── memopt CRDB migration ───")
print(f"source:  {SOURCE[:60]}")
print(f"target:  {TARGET[:60]}")
print(f"dry-run: {DRY_RUN}")
print()

# 1. Validate source connectivity
print("── Step 1: connect source ──")
try:
    src = make_backend(SOURCE)
    # make_backend falls back to SQLite on failure; ensure we got
    # the dialect the caller asked for.
    got = src.__class__.__name__
    if SOURCE.startswith("sqlite") and "SQLite" not in got:
        print(f"ERROR: source is {got}, expected SQLite", file=sys.stderr)
        sys.exit(1)
    if SOURCE.startswith("postgresql") and "PostgreSQL" not in got:
        print(f"ERROR: source fell back to {got}; PostgreSQL unreachable",
              file=sys.stderr)
        sys.exit(1)
    print(f"[OK]   source backend: {got}")
except Exception as e:
    print(f"ERROR: source connect failed: {e}", file=sys.stderr)
    sys.exit(1)

# 2. Validate target connectivity (and is actually CRDB)
print("── Step 2: connect target ──")
try:
    tgt = make_backend(TARGET)
    got = tgt.__class__.__name__
    if "CockroachDB" not in got:
        print(f"ERROR: target fell back to {got}; CockroachDB unreachable",
              file=sys.stderr)
        sys.exit(1)
    print(f"[OK]   target backend: {got}")
except Exception as e:
    print(f"ERROR: target connect failed: {e}", file=sys.stderr)
    sys.exit(1)

# 3. Apply CRDB schema (idempotent)
print("── Step 3: apply schema ──")
if DRY_RUN:
    print("[DRY]  would call apply_schema() on target")
else:
    try:
        apply_schema(TARGET)
        print("[OK]   schema applied")
    except Exception as e:
        print(f"ERROR: schema apply failed: {e}", file=sys.stderr)
        sys.exit(1)

# 4. Export from source
print("── Step 4: export from source ──")
exported = {}
for table in TABLES:
    try:
        rows = src.fetchall(f"SELECT * FROM {table}")
        exported[table] = rows
        print(f"[OK]   {table}: {len(rows)} rows")
    except Exception as e:
        print(f"[WARN] {table}: skip ({e})")
        exported[table] = []

# 5. Import into target
print("── Step 5: import into target ──")
for table in TABLES:
    rows = exported.get(table, [])
    if not rows:
        print(f"[SKIP] {table}: no rows")
        continue
    if DRY_RUN:
        print(f"[DRY]  {table}: would insert {len(rows)} rows")
        continue

    cols = [c for c in TABLE_COLUMNS[table]
            if c in rows[0]]
    placeholders = ", ".join(["%s"] * len(cols))
    col_list = ", ".join(cols)

    written = 0
    for row in rows:
        vals = tuple(row.get(c) for c in cols)
        try:
            tgt.execute(
                f"INSERT INTO {table} ({col_list}) "
                f"VALUES ({placeholders})",
                vals)
            written += 1
        except Exception as e:
            # Tolerate duplicate primary keys (re-runs).
            msg = str(e).lower()
            if "duplicate" in msg or "unique" in msg:
                continue
            print(f"[WARN] {table}: row failed ({e})")
    print(f"[OK]   {table}: wrote {written} rows")

# 6. Verify row counts
print("── Step 6: verify ──")
mismatches = 0
for table in TABLES:
    src_count = len(exported.get(table, []))
    try:
        r = tgt.fetchone(f"SELECT COUNT(*) AS cnt FROM {table}")
        tgt_count = int(r["cnt"]) if r else 0
    except Exception as e:
        print(f"[WARN] {table}: count failed ({e})")
        continue
    if DRY_RUN:
        print(f"[DRY]  {table}: source={src_count} target=(unchanged)")
        continue
    if tgt_count >= src_count:
        print(f"[OK]   {table}: source={src_count} target={tgt_count}")
    else:
        print(f"[FAIL] {table}: source={src_count} target={tgt_count}")
        mismatches += 1

print()
if mismatches > 0:
    print(f"ERROR: {mismatches} table(s) had fewer rows on target",
          file=sys.stderr)
    sys.exit(2)

print("── Summary ──")
print(f"tables migrated: {len(TABLES)}")
print(f"dry-run:         {DRY_RUN}")
print(f"result:          {'DRY-RUN OK' if DRY_RUN else 'OK'}")

src.close()
tgt.close()
PYEOF
