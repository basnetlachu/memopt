"""
Tests for vmm.oracle, vmm.oracle_data_cleaner, and vmm.oracle_trainer.
All pass on CPU — no torch, no numpy, no external deps.
"""
import json
import os
import tempfile
import time

import pytest

from memopt.vmm.oracle import BlockPrediction, MemoryOracle, OracleStats
from memopt.vmm.oracle_data_cleaner import CleaningStats, OracleDataCleaner
from memopt.vmm.oracle_trainer import OracleTrainer


# ── Helpers ───────────────────────────────────────────────────────────────


def _write_jsonl(path, events):
    with open(path, "w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


def _valid_event(seq="s1", block=0, tpos=0, tier="dram",
                 latency=0.0, tenant="_default"):
    return {
        "sequence_id": seq,
        "block_index": block,
        "token_position": tpos,
        "attention_layer": 0,
        "timestamp": time.perf_counter(),
        "tier_at_access": tier,
        "promotion_latency_ms": latency,
        "tenant_id": tenant,
    }


# ══════════════════════════════════════════════════════════════════════════
#  Oracle Tests (15)
# ══════════════════════════════════════════════════════════════════════════


def test_oracle_instantiates_with_defaults():
    oracle = MemoryOracle()
    assert oracle._horizon == 50
    assert oracle._min_confidence == 0.3
    assert oracle._max_transitions == 100_000


def test_observe_updates_transitions():
    oracle = MemoryOracle()
    oracle.observe("s1", 0, step=1)
    oracle.observe("s1", 1, step=2)
    oracle.observe("s1", 2, step=3)
    # 0→1 and 1→2 should be recorded
    assert 1 in oracle._transitions[0]
    assert 2 in oracle._transitions[1]


def test_predict_returns_block_predictions():
    oracle = MemoryOracle()
    oracle.observe("s1", 0, step=1)
    oracle.observe("s1", 1, step=2)
    preds = oracle.predict("s1", current_block=1)
    assert len(preds) > 0
    assert all(isinstance(p, BlockPrediction) for p in preds)


def test_predict_ordered_by_confidence_descending():
    oracle = MemoryOracle()
    for i in range(20):
        oracle.observe("s1", i, step=i)
    preds = oracle.predict("s1", current_block=10, top_k=10)
    confs = [p.confidence for p in preds]
    assert confs == sorted(confs, reverse=True)


def test_sequential_fallback_always_present():
    oracle = MemoryOracle()
    preds = oracle.predict("new_seq", current_block=5)
    assert len(preds) >= 1
    blocks = [p.block_index for p in preds]
    assert 6 in blocks  # sequential +1


def test_confidence_between_zero_and_one():
    oracle = MemoryOracle()
    for i in range(10):
        oracle.observe("s1", i, step=i)
    preds = oracle.predict("s1", current_block=5, top_k=20)
    for p in preds:
        assert 0.0 <= p.confidence <= 1.0


def test_min_confidence_filters_predictions():
    oracle = MemoryOracle(min_confidence=0.99)
    oracle.observe("s1", 0, step=1)
    oracle.observe("s1", 1, step=2)
    oracle.observe("s1", 0, step=3)
    oracle.observe("s1", 2, step=4)
    preds = oracle.predict("s1", current_block=0, top_k=10)
    # Only fallback can survive since all transition/sequential confs < 0.99
    for p in preds:
        assert p.confidence >= 0.99 or p.source == "fallback"


def test_record_outcome_updates_accuracy():
    oracle = MemoryOracle()
    oracle.observe("s1", 0, step=1)
    oracle.observe("s1", 1, step=2)
    preds = oracle.predict("s1", current_block=1)
    assert len(preds) > 0
    oracle.record_outcome("s1", preds[0].block_index)
    s = oracle.stats()
    assert s.total_predictions_correct >= 1
    assert s.accuracy_pct > 0


def test_stats_has_all_required_fields():
    oracle = MemoryOracle()
    s = oracle.stats()
    assert isinstance(s, OracleStats)
    for field in (
        "total_predictions_made", "total_predictions_correct",
        "accuracy_pct", "transitions_learned", "sequences_tracked",
        "horizon", "uptime_seconds",
    ):
        assert hasattr(s, field)


def test_reset_sequence_clears_state():
    oracle = MemoryOracle()
    oracle.observe("s1", 0, step=1)
    oracle.observe("s1", 1, step=2)
    oracle.observe("s2", 0, step=1)
    oracle.reset(sequence_id="s1")
    s = oracle.stats()
    assert s.sequences_tracked == 1  # only s2 remains


def test_reset_all_clears_everything():
    oracle = MemoryOracle()
    oracle.observe("s1", 0, step=1)
    oracle.observe("s1", 1, step=2)
    oracle.reset()
    s = oracle.stats()
    assert s.transitions_learned == 0
    assert s.sequences_tracked == 0


def test_warm_from_log_replays_events():
    oracle = MemoryOracle()
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False,
    ) as f:
        for i in range(10):
            f.write(json.dumps({
                "sequence_id": "s1",
                "block_index": i,
                "token_position": i,
            }) + "\n")
        path = f.name
    try:
        count = oracle.warm_from_log(path)
        assert count == 10
        assert oracle.stats().transitions_learned >= 1
    finally:
        os.unlink(path)


def test_warm_from_log_handles_missing_file():
    oracle = MemoryOracle()
    count = oracle.warm_from_log("/nonexistent/path.jsonl")
    assert count == 0


def test_oracle_trainer_starts_and_stops():
    oracle = MemoryOracle()
    with tempfile.TemporaryDirectory() as td:
        trainer = OracleTrainer(oracle, td, poll_interval_s=0.1)
        trainer.start()
        time.sleep(0.3)
        trainer.stop()
        s = trainer.stats()
        assert "events_trained" in s
        assert "oracle_stats" in s


def test_oracle_trainer_ingests_new_events():
    oracle = MemoryOracle()
    with tempfile.TemporaryDirectory() as td:
        # Write initial log with enough events to pass cleaning
        events = [_valid_event("s1", i, i) for i in range(10)]
        _write_jsonl(os.path.join(td, "access_log_20260406.jsonl"), events)

        trainer = OracleTrainer(oracle, td, poll_interval_s=0.2)
        trainer.start()
        time.sleep(0.5)

        # Append new events to same file
        with open(
            os.path.join(td, "access_log_20260406.jsonl"), "a",
        ) as f:
            for i in range(10):
                f.write(json.dumps(_valid_event("s2", i, i)) + "\n")

        time.sleep(1.0)
        trainer.stop()

        s = trainer.stats()
        assert s["events_trained"] >= 10  # at least initial warm


# ══════════════════════════════════════════════════════════════════════════
#  Cleaner Tests (8)
# ══════════════════════════════════════════════════════════════════════════


def test_cleaner_rejects_malformed_json():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False,
    ) as f:
        for i in range(5):
            f.write(json.dumps(_valid_event("s1", i, i)) + "\n")
        f.write("not json\n")
        f.write("{bad json\n")
        f.write("also not json\n")
        path = f.name
    try:
        cleaner = OracleDataCleaner()
        _, stats = cleaner.clean_file(path)
        assert stats.drop_reasons.get("malformed_json", 0) == 3
    finally:
        os.unlink(path)


def test_cleaner_rejects_invalid_block_index():
    events = [
        _valid_event("s1", 0, 0),
        _valid_event("s1", 1, 1),
        _valid_event("s1", 2, 2),
        _valid_event("s1", 3, 3),
        _valid_event("s1", 4, 4),
    ]
    events[1]["block_index"] = -1
    events[3]["block_index"] = 2_000_000
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False,
    ) as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
        path = f.name
    try:
        cleaner = OracleDataCleaner()
        _, stats = cleaner.clean_file(path)
        assert stats.drop_reasons.get("invalid_fields", 0) >= 2
    finally:
        os.unlink(path)


def test_cleaner_drops_short_sequences():
    # 10 events alternating 2 blocks → after dedup only 2 events < min 3
    events = [_valid_event("s1", i % 2, i) for i in range(10)]
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False,
    ) as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
        path = f.name
    try:
        cleaner = OracleDataCleaner(min_sequence_length=3)
        clean_events, stats = cleaner.clean_file(path)
        assert stats.sequences_dropped >= 1
        assert len(clean_events) == 0
    finally:
        os.unlink(path)


def test_cleaner_deduplicates_window():
    blocks = [1, 1, 1, 2, 3, 1, 1, 4]
    events = [_valid_event("s1", b, i) for i, b in enumerate(blocks)]
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False,
    ) as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
        path = f.name
    try:
        cleaner = OracleDataCleaner(deduplicate_window=3)
        clean_events, stats = cleaner.clean_file(path)
        assert len(clean_events) == 5
    finally:
        os.unlink(path)


def test_cleaner_truncates_long_sequences():
    events = [_valid_event("s1", i, i) for i in range(1000)]
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False,
    ) as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
        path = f.name
    try:
        cleaner = OracleDataCleaner(max_sequence_length=100)
        clean_events, stats = cleaner.clean_file(path)
        seq_events = [
            e for e in clean_events if e["sequence_id"] == "s1"
        ]
        assert len(seq_events) == 100
    finally:
        os.unlink(path)


def test_cleaner_stats_summary_returns_string():
    cleaner = OracleDataCleaner()
    stats = CleaningStats(
        raw_events_read=100, final_events=80,
        sequences_kept=5, sequences_dropped=2,
        drop_reasons={"malformed_json": 10},
    )
    summary = cleaner.stats_summary(stats)
    assert isinstance(summary, str)
    assert "kept" in summary
    assert "sequences" in summary


def test_cleaner_handles_missing_file():
    cleaner = OracleDataCleaner()
    clean_events, stats = cleaner.clean_file("/nonexistent/path.jsonl")
    assert clean_events == []
    assert stats.raw_events_read == 0


def test_cleaner_clean_directory_merges_stats():
    with tempfile.TemporaryDirectory() as td:
        for j in range(3):
            events = [_valid_event(f"s{j}", i, i) for i in range(10)]
            _write_jsonl(os.path.join(td, f"log_{j}.jsonl"), events)
        cleaner = OracleDataCleaner()
        _, stats = cleaner.clean_directory(td)
        assert stats.raw_events_read == 30
