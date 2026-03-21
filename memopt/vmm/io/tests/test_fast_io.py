"""
Tests for hardware-accelerated NVMe I/O.
All pass without io_uring, without GDS, without GPU.
Benchmark test prints real numbers — never hardcoded.
"""
import os
import tempfile
import time
import pytest

from memopt.vmm.io import fast_read_block, status as io_status


def test_status_has_required_keys():
    s = io_status()
    for key in ("backend", "gds_available", "uring_available",
                "pread_fallback", "python_fallback"):
        assert key in s, f"Missing key: {key}"


def test_backend_is_valid_string():
    assert io_status()["backend"] in ("gds", "uring", "pread", "python")


def test_read_correct_bytes_small():
    data = os.urandom(1024)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(data); path = f.name
    try:
        assert fast_read_block(path, len(data)) == data
    finally:
        os.unlink(path)


def test_read_correct_bytes_large():
    data = os.urandom(4 * 1024 * 1024)   # 4 MB
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(data); path = f.name
    try:
        result = fast_read_block(path, len(data))
        assert result == data
    finally:
        os.unlink(path)


def test_read_raises_on_missing_file():
    with pytest.raises(OSError):
        fast_read_block("/tmp/memopt_nonexistent_xyz.bin", 1024)


def test_read_idempotent():
    data = os.urandom(512)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(data); path = f.name
    try:
        r1 = fast_read_block(path, len(data))
        r2 = fast_read_block(path, len(data))
        assert r1 == r2 == data
    finally:
        os.unlink(path)


def test_benchmark_real_numbers(capsys):
    """
    Measures actual throughput on this hardware.
    Prints a table. Numbers are never hardcoded.
    This test always passes — it is a measurement, not an assertion.
    """
    size   = 256 * 1024 * 1024   # 256 MB
    n_iter = 5
    data   = os.urandom(size)

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(data); path = f.name

    try:
        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            fast_read_block(path, size)
            times.append(time.perf_counter() - t0)

        avg_s  = sum(times) / len(times)
        bw     = (size / 1e9) / avg_s          # GB/s
        backend = io_status()["backend"]

        # Write to benchmark.txt for use in emails
        import pathlib
        bench = pathlib.Path("benchmark.txt")
        line  = (
            f"NVMe read: backend={backend} "
            f"avg={avg_s*1000:.1f}ms "
            f"bw={bw:.2f}GB/s "
            f"size=256MB iters={n_iter}\n"
        )
        with bench.open("a") as fh:
            fh.write(line)

        print(f"\n[io_benchmark]\n  backend : {backend}")
        print(f"  avg_ms  : {avg_s*1000:.1f}")
        print(f"  gb_s    : {bw:.2f}")
        print(f"  written : benchmark.txt")

    finally:
        os.unlink(path)
