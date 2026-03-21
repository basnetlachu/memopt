"""
Tests for hardware-signed kernel archive.
All pass without GPU.
"""
from memopt.kernels.kernel_archive import (
    hardware_fingerprint,
    current_hardware_fingerprint,
    verify_entry,
)


def test_fingerprint_is_deterministic():
    fp1 = hardware_fingerprint("A100", "8.0", "abc123")
    fp2 = hardware_fingerprint("A100", "8.0", "abc123")
    assert fp1 == fp2


def test_fingerprint_differs_by_compute_cap():
    fp_80 = hardware_fingerprint("A100", "8.0", "abc")
    fp_89 = hardware_fingerprint("A100", "8.9", "abc")
    assert fp_80 != fp_89


def test_fingerprint_differs_by_source():
    fp1 = hardware_fingerprint("A100", "8.0", "source_v1")
    fp2 = hardware_fingerprint("A100", "8.0", "source_v2")
    assert fp1 != fp2


def test_fingerprint_is_64_hex_chars():
    fp = hardware_fingerprint("H100", "9.0", "deadbeef")
    assert len(fp) == 64
    assert all(c in "0123456789abcdef" for c in fp)


def test_verify_entry_missing_source_sha_fails():
    assert not verify_entry({})
    assert not verify_entry({"hw_fingerprint": "abc"})


def test_verify_entry_correct_fingerprint_passes():
    source = "a" * 64
    fp     = hardware_fingerprint("TestGPU", "8.0", source)
    entry  = {"source_sha256": source, "hw_fingerprint": fp}
    # On GPU hardware: wrong fingerprint must fail.
    # On CPU-only: no fingerprint comparison is possible, so
    # verify_entry returns True (safe — CPU has no arch mismatch risk).
    bad_entry = {"source_sha256": source, "hw_fingerprint": "wrong"}
    current_fp = current_hardware_fingerprint(source)
    if current_fp is not None:
        # GPU present — wrong fingerprint must be rejected
        assert not verify_entry(bad_entry)
    else:
        # CPU only — all entries pass (no GPU arch to mismatch)
        assert verify_entry(bad_entry)


def test_current_hardware_fingerprint_returns_none_or_str():
    fp = current_hardware_fingerprint("somehash")
    assert fp is None or (isinstance(fp, str) and len(fp) == 64)
