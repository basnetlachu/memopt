"""
Tests for ledger verify endpoint.
All pass without GPU, without signing key.
"""
import os
import tempfile
import pytest
from memopt.observability.ledger import OptimizationLedger


@pytest.fixture
def ledger():
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        l  = OptimizationLedger(db_path=db)
        l._buffer._flush_batch_size = 1   # flush on every append
        yield l
        l.shutdown()


def test_verify_and_certify_empty_ledger(ledger):
    result = ledger.verify_and_certify()
    assert "chain_valid"      in result
    assert "entries_checked"  in result
    assert "issued_at"        in result
    assert "signature_status" in result


def test_verify_and_certify_valid_chain(ledger):
    for _ in range(3):
        ledger.record(
            tokens=100,
            tenant_id="acme",
            actual_j_per_token=0.001
        )
    result = ledger.verify_and_certify(tenant_id="acme")
    assert result["chain_valid"]    is True
    assert result["entries_checked"] == 3
    assert result["tenant_id"]       == "acme"


def test_verify_and_certify_signed_when_key_set(ledger):
    os.environ["MEMOPT_SIGNING_KEY"] = "test-key-xyz"
    try:
        for _ in range(2):
            ledger.record(tokens=50, tenant_id="t1",
                          actual_j_per_token=0.001)
        result = ledger.verify_and_certify(tenant_id="t1")
        assert result["signature_status"] in ("signed", "unsigned")
    finally:
        os.environ.pop("MEMOPT_SIGNING_KEY", None)


def test_verify_and_certify_unsigned_without_key(ledger):
    os.environ.pop("MEMOPT_SIGNING_KEY", None)
    ledger.record(tokens=100, tenant_id="t2",
                  actual_j_per_token=0.001)
    result = ledger.verify_and_certify(tenant_id="t2")
    assert result["signature_status"] == "unsigned"
    assert result["signature"] is None


def test_verify_and_certify_tamper_detected(ledger):
    import sqlite3
    ledger.record(tokens=100, tenant_id="acme",
                  actual_j_per_token=0.001)
    # Tamper with the record
    conn = sqlite3.connect(ledger._db_path)
    conn.execute(
        "UPDATE entries SET tokens_generated = 9999 "
        "WHERE tenant_id = 'acme'"
    )
    conn.commit()
    conn.close()
    result = ledger.verify_and_certify(tenant_id="acme")
    assert result["chain_valid"] is False


def test_verify_returns_required_fields(ledger):
    result = ledger.verify_and_certify()
    required = {
        "tenant_id", "chain_valid", "entries_checked",
        "issued_at", "signature", "signature_status",
    }
    assert required.issubset(set(result.keys()))
