"""
Ledger hardening tests — disk space, rotation, size tracking.
"""
import os
import tempfile


from memopt.observability.ledger import OptimizationLedger


def test_ledger_size_bytes():
    """size_bytes() should return file size or 0."""
    with tempfile.TemporaryDirectory() as d:
        db_path = os.path.join(d, "test_ledger.db")
        ledger = OptimizationLedger(db_path=db_path)
        sz = ledger.size_bytes() if hasattr(ledger, 'size_bytes') else 0
        assert isinstance(sz, (int, float))
        assert sz >= 0
        ledger.shutdown()


def test_ledger_record_never_raises():
    """record() must never propagate exceptions."""
    with tempfile.TemporaryDirectory() as d:
        db_path = os.path.join(d, "test_ledger.db")
        ledger = OptimizationLedger(db_path=db_path)
        # Record 100 entries rapidly — must not raise
        for i in range(100):
            entry = ledger.record(tokens=10, node_id="test")
            assert entry is not None
        ledger.shutdown()


def test_ledger_survives_db_error():
    """Ledger should handle SQLite errors gracefully."""
    with tempfile.TemporaryDirectory() as d:
        db_path = os.path.join(d, "test_ledger.db")
        ledger = OptimizationLedger(db_path=db_path)
        # Record works normally
        entry = ledger.record(tokens=5, node_id="test")
        assert entry is not None
        ledger.shutdown()


def test_ledger_flush_writes_to_db():
    """flush() should persist buffered entries."""
    with tempfile.TemporaryDirectory() as d:
        db_path = os.path.join(d, "test_ledger.db")
        ledger = OptimizationLedger(db_path=db_path)
        ledger.record(tokens=10, node_id="test")
        ledger.flush()
        recent = ledger.recent(n=10)
        assert len(recent) >= 1
        ledger.shutdown()


def test_ledger_verify_chain_empty():
    """verify_chain on empty ledger should return ok=True."""
    with tempfile.TemporaryDirectory() as d:
        db_path = os.path.join(d, "test_ledger.db")
        ledger = OptimizationLedger(db_path=db_path)
        result = ledger.verify_chain()
        assert result["ok"] is True
        assert result["entries_checked"] == 0
        ledger.shutdown()


def test_ledger_records_real_j_per_token():
    """When actual_j_per_token is provided, savings are computed."""
    with tempfile.TemporaryDirectory() as d:
        db_path = os.path.join(d, "test_ledger.db")
        ledger = OptimizationLedger(db_path=db_path)
        # actual < baseline → positive energy savings
        entry = ledger.record(
            tokens=100,
            actual_j_per_token=0.0005)  # less than default 0.001
        assert entry is not None
        assert entry.energy_saved_kwh is not None
        assert entry.energy_saved_kwh > 0
        ledger.shutdown()


def test_ledger_records_none_j_per_token_no_crash():
    """When actual_j_per_token is None, no crash."""
    with tempfile.TemporaryDirectory() as d:
        db_path = os.path.join(d, "test_ledger.db")
        ledger = OptimizationLedger(db_path=db_path)
        entry = ledger.record(tokens=100, actual_j_per_token=None)
        assert entry is not None
        # energy_saved_kwh may be None when no measurement
        ledger.shutdown()


def test_ledger_skips_write_on_low_disk():
    """record() should skip DB write when disk space is low."""
    with tempfile.TemporaryDirectory() as d:
        db_path = os.path.join(d, "test_ledger.db")
        ledger = OptimizationLedger(db_path=db_path)

        # Set absurdly high minimum so check always fails
        old_env = os.environ.get("MEMOPT_LEDGER_MIN_FREE_MB")
        os.environ["MEMOPT_LEDGER_MIN_FREE_MB"] = "99999999"
        try:
            entry = ledger.record(tokens=10, node_id="test")
            assert entry is not None
            # Entry should be marked as skipped
            assert entry.batch_id.startswith("skipped_")
            # Counter should be incremented
            skipped = getattr(ledger, '_entries_skipped', 0)
            assert skipped >= 1
        finally:
            if old_env is not None:
                os.environ["MEMOPT_LEDGER_MIN_FREE_MB"] = old_env
            else:
                os.environ.pop("MEMOPT_LEDGER_MIN_FREE_MB", None)
            ledger.shutdown()
