"""Tests for vmm.access_log — runs on CPU with no external dependencies."""
import dataclasses
import json
import os
import tempfile
import threading
import time


from memopt.vmm.access_log import AccessLog, BlockAccessEvent


def _make_event(**overrides) -> BlockAccessEvent:
    defaults = dict(
        sequence_id="seq_0",
        block_index=0,
        token_position=1,
        attention_layer=2,
        timestamp=time.perf_counter(),
        tier_at_access="dram",
        promotion_latency_ms=0.0,
        tenant_id="_default",
    )
    defaults.update(overrides)
    return BlockAccessEvent(**defaults)


# ─────────────────────────────────────────────────────────────────────────


def test_record_creates_jsonl_file():
    with tempfile.TemporaryDirectory() as td:
        log = AccessLog(log_dir=td)
        log.record(_make_event())
        log.shutdown()
        files = [f for f in os.listdir(td) if f.endswith(".jsonl")]
        assert len(files) == 1


def test_record_fields_persisted_correctly():
    with tempfile.TemporaryDirectory() as td:
        log = AccessLog(log_dir=td)
        evt = _make_event(
            sequence_id="s1",
            block_index=7,
            token_position=42,
            attention_layer=3,
            tier_at_access="hbm",
            promotion_latency_ms=1.5,
            tenant_id="alice",
        )
        log.record(evt)
        log.shutdown()

        path = log.stats()["log_file_path"]
        with open(path) as f:
            obj = json.loads(f.readline())
        assert obj["sequence_id"] == "s1"
        assert obj["block_index"] == 7
        assert obj["token_position"] == 42
        assert obj["attention_layer"] == 3
        assert obj["tier_at_access"] == "hbm"
        assert obj["promotion_latency_ms"] == 1.5
        assert obj["tenant_id"] == "alice"


def test_stats_has_required_keys():
    with tempfile.TemporaryDirectory() as td:
        log = AccessLog(log_dir=td)
        s = log.stats()
        assert "events_recorded" in s
        assert "events_dropped" in s
        assert "log_file_path" in s
        assert "log_size_bytes" in s
        log.shutdown()


def test_shutdown_flushes_queue():
    with tempfile.TemporaryDirectory() as td:
        log = AccessLog(log_dir=td)
        for i in range(50):
            log.record(_make_event(block_index=i))
        log.shutdown()
        s = log.stats()
        assert s["events_recorded"] == 50


def test_never_raises_on_bad_log_dir():
    # Pass a path that can't be created (nested under a file)
    bad_dir = "/dev/null/impossible/path"
    log = AccessLog(log_dir=bad_dir)
    log.record(_make_event())  # should not raise
    log.shutdown()


def test_thread_safe_concurrent_writes():
    with tempfile.TemporaryDirectory() as td:
        log = AccessLog(log_dir=td)
        n_threads = 10
        n_events = 100
        barrier = threading.Barrier(n_threads)

        def writer(tid):
            barrier.wait()
            for i in range(n_events):
                log.record(_make_event(sequence_id=f"t{tid}", block_index=i))

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        log.shutdown()
        s = log.stats()
        assert s["events_recorded"] == n_threads * n_events


def test_dropped_count_increments_on_full_queue():
    with tempfile.TemporaryDirectory() as td:
        log = AccessLog(log_dir=td)
        # Stop the writer thread so the queue can never drain
        log._queue.put(AccessLog._SENTINEL)
        log._writer_thread.join(timeout=5.0)

        # Fill the queue to capacity (no consumer now)
        for _ in range(AccessLog._QUEUE_MAXSIZE):
            log._queue.put_nowait(_make_event())

        # These records should be dropped
        for _ in range(500):
            log.record(_make_event())

        assert log.stats()["events_dropped"] >= 500


def test_event_dataclass_has_all_fields():
    expected = {
        "sequence_id", "block_index", "token_position", "attention_layer",
        "timestamp", "tier_at_access", "promotion_latency_ms", "tenant_id",
    }
    actual = {f.name for f in dataclasses.fields(BlockAccessEvent)}
    assert actual == expected
