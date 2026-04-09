"""
Oracle data cleaner — 7-step cleaning pipeline for access log data.

Validates, deduplicates, filters, and normalises raw JSONL access events
before they are fed to the MemoryOracle.  Pure Python only.
"""
from __future__ import annotations

import collections
import glob
import json
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CleaningStats:
    raw_events_read: int = 0
    events_after_dedup: int = 0
    events_after_sequence_filter: int = 0
    events_after_outlier_filter: int = 0
    final_events: int = 0
    sequences_kept: int = 0
    sequences_dropped: int = 0
    drop_reasons: dict = field(default_factory=dict)


class OracleDataCleaner:

    def __init__(
        self,
        min_sequence_length: int = 3,
        max_sequence_length: int = 10_000,
        min_block_index: int = 0,
        max_block_index: int = 1_000_000,
        max_promotion_latency_ms: float = 10_000.0,
        valid_tiers: set = None,
        deduplicate_window: int = 3,
    ) -> None:
        self._min_seq_len = min_sequence_length
        self._max_seq_len = max_sequence_length
        self._min_block = min_block_index
        self._max_block = max_block_index
        self._max_latency = max_promotion_latency_ms
        self._valid_tiers = valid_tiers or {"hbm", "dram", "nvme", "unknown"}
        self._dedup_window = deduplicate_window

    # ── Public API ────────────────────────────────────────────────────────

    def clean_file(
        self, jsonl_path: str,
    ) -> tuple[list[dict], CleaningStats]:
        stats = CleaningStats()

        # ── Step 1: Parse ─────────────────────────────────────────────
        raw_events: list[dict] = []
        try:
            with open(jsonl_path) as f:
                for line in f:
                    stats.raw_events_read += 1
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw_events.append(json.loads(line))
                    except json.JSONDecodeError:
                        stats.drop_reasons["malformed_json"] = (
                            stats.drop_reasons.get("malformed_json", 0) + 1
                        )
                    except Exception:
                        stats.drop_reasons["parse_error"] = (
                            stats.drop_reasons.get("parse_error", 0) + 1
                        )
        except (OSError, IOError):
            return [], stats

        # ── Step 2: Field validation ──────────────────────────────────
        valid: list[dict] = []
        for ev in raw_events:
            if self._validate_event(ev):
                valid.append(ev)
            else:
                stats.drop_reasons["invalid_fields"] = (
                    stats.drop_reasons.get("invalid_fields", 0) + 1
                )

        # ── Step 3: Group by sequence_id, sort by token_position ──────
        groups: dict[str, list[dict]] = {}
        for ev in valid:
            groups.setdefault(ev["sequence_id"], []).append(ev)

        for sid in groups:
            normal = [e for e in groups[sid] if e["token_position"] != -1]
            unknown = [e for e in groups[sid] if e["token_position"] == -1]
            normal.sort(key=lambda e: e["token_position"])
            groups[sid] = normal + unknown

        # ── Step 4: Deduplicate within each sequence ──────────────────
        total_after_dedup = 0
        for sid in groups:
            seen: collections.deque = collections.deque()
            deduped: list[dict] = []
            for ev in groups[sid]:
                block = ev["block_index"]
                while len(seen) >= self._dedup_window:
                    seen.popleft()
                if block in seen:
                    stats.drop_reasons["duplicate_in_window"] = (
                        stats.drop_reasons.get("duplicate_in_window", 0) + 1
                    )
                else:
                    deduped.append(ev)
                seen.append(block)
            groups[sid] = deduped
            total_after_dedup += len(deduped)
        stats.events_after_dedup = total_after_dedup

        # ── Step 5: Filter short sequences ────────────────────────────
        total_after_seq = 0
        to_drop = []
        for sid, evts in groups.items():
            if len(evts) < self._min_seq_len:
                stats.drop_reasons["sequence_too_short"] = (
                    stats.drop_reasons.get("sequence_too_short", 0)
                    + len(evts)
                )
                stats.sequences_dropped += 1
                to_drop.append(sid)
            else:
                total_after_seq += len(evts)
        for sid in to_drop:
            del groups[sid]
        stats.events_after_sequence_filter = total_after_seq

        # ── Step 6: Truncate long sequences ───────────────────────────
        for sid in groups:
            if len(groups[sid]) > self._max_seq_len:
                groups[sid] = groups[sid][-self._max_seq_len:]

        # ── Step 7: Outlier filter on promotion_latency_ms ────────────
        all_latencies = sorted(
            e["promotion_latency_ms"]
            for evts in groups.values()
            for e in evts
        )
        p99 = 0.0
        if all_latencies:
            idx = int(len(all_latencies) * 0.99)
            p99 = all_latencies[min(idx, len(all_latencies) - 1)]

        total_after_outlier = 0
        for sid in list(groups.keys()):
            filtered = []
            for ev in groups[sid]:
                lat = ev["promotion_latency_ms"]
                if lat > p99 and lat > 1000.0:
                    stats.drop_reasons["latency_outlier"] = (
                        stats.drop_reasons.get("latency_outlier", 0) + 1
                    )
                else:
                    filtered.append(ev)
            groups[sid] = filtered
            total_after_outlier += len(filtered)
        stats.events_after_outlier_filter = total_after_outlier

        # ── Step 8: Flatten ───────────────────────────────────────────
        result: list[dict] = []
        for sid in sorted(groups.keys()):
            evts = groups[sid]
            evts.sort(key=lambda e: (
                e["token_position"]
                if e["token_position"] != -1
                else float("inf")
            ))
            result.extend(evts)

        stats.final_events = len(result)
        stats.sequences_kept = len(groups)
        return result, stats

    def clean_directory(
        self, log_dir: str,
    ) -> tuple[list[dict], CleaningStats]:
        merged = CleaningStats()
        merged_events: list[dict] = []

        try:
            files = sorted(glob.glob(os.path.join(log_dir, "*.jsonl")))
        except OSError:
            return [], merged

        for path in files:
            events, st = self.clean_file(path)
            merged_events.extend(events)
            merged.raw_events_read += st.raw_events_read
            merged.events_after_dedup += st.events_after_dedup
            merged.events_after_sequence_filter += st.events_after_sequence_filter
            merged.events_after_outlier_filter += st.events_after_outlier_filter
            merged.final_events += st.final_events
            merged.sequences_kept += st.sequences_kept
            merged.sequences_dropped += st.sequences_dropped
            for reason, count in st.drop_reasons.items():
                merged.drop_reasons[reason] = (
                    merged.drop_reasons.get(reason, 0) + count
                )

        return merged_events, merged

    def stats_summary(self, stats: CleaningStats) -> str:
        raw = stats.raw_events_read
        final = stats.final_events
        pct = (final / raw * 100) if raw > 0 else 0.0
        kept = stats.sequences_kept
        dropped = stats.sequences_dropped

        if stats.drop_reasons:
            top_3 = sorted(
                stats.drop_reasons.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:3]
            top_str = ", ".join(f"{r}({c})" for r, c in top_3)
        else:
            top_str = "none"

        return (
            f"OracleDataCleaner: read {raw} events, kept {final} "
            f"({pct:.1f}%). {kept} sequences kept, {dropped} dropped. "
            f"Top drop reasons: {top_str}"
        )

    # ── Internal ──────────────────────────────────────────────────────

    def _validate_event(self, ev: dict) -> bool:
        try:
            sid = ev.get("sequence_id")
            if not isinstance(sid, str) or not sid:
                return False
            bidx = ev.get("block_index")
            if not isinstance(bidx, int) or not (
                self._min_block <= bidx <= self._max_block
            ):
                return False
            tpos = ev.get("token_position")
            if not isinstance(tpos, int) or tpos < -1:
                return False
            tier = ev.get("tier_at_access")
            if tier not in self._valid_tiers:
                return False
            lat = ev.get("promotion_latency_ms")
            if not isinstance(lat, (int, float)) or not (
                0 <= lat <= self._max_latency
            ):
                return False
            tid = ev.get("tenant_id")
            if not isinstance(tid, str) or not tid:
                return False
            return True
        except (TypeError, ValueError):
            return False
