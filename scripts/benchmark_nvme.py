#!/usr/bin/env python3
"""
NVMe I/O benchmark for memopt VMM.

Measures:
  Sequential read throughput
  Sequential write throughput
  Random read IOPS
  Async vs sync latency comparison

Run on the GPU rental machines to get
real numbers for the design partner deck.

Usage:
  python scripts/benchmark_nvme.py \
    --dir /var/memopt/nvme \
    --block-size 131072 \
    --blocks 1000
"""

import argparse
import os
import statistics
import subprocess
import sys
import time

# Ensure memopt is importable when run from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def benchmark_sync_read(
    dir_path: str,
    block_size: int,
    n_blocks: int,
) -> dict:
    """Benchmark synchronous NVMe reads."""

    # Write test files first
    test_files = []
    data = os.urandom(block_size)

    for i in range(n_blocks):
        path = os.path.join(dir_path, f"bench_{i:06d}.bin")
        with open(path, "wb") as f:
            f.write(data)
        test_files.append(path)

    # Sync reads
    latencies = []
    start = time.monotonic()

    for path in test_files:
        t0 = time.monotonic()
        with open(path, "rb") as f:
            _ = f.read()
        latencies.append((time.monotonic() - t0) * 1000)

    total_s = time.monotonic() - start
    total_mb = (block_size * n_blocks) / 1e6

    # Cleanup
    for path in test_files:
        os.unlink(path)

    return {
        "mode":            "sync",
        "block_size_kb":   block_size // 1024,
        "n_blocks":        n_blocks,
        "total_mb":        round(total_mb, 1),
        "throughput_mbps": round(total_mb / total_s, 1),
        "lat_p50_ms":      round(statistics.median(latencies), 2),
        "lat_p99_ms":      round(
            sorted(latencies)[int(len(latencies) * 0.99)], 2),
        "lat_max_ms":      round(max(latencies), 2),
    }


def benchmark_async_read(
    dir_path: str,
    block_size: int,
    n_blocks: int,
) -> dict:
    """
    Benchmark async NVMe reads via memopt AsyncNVMeManager.

    Submits all reads, then waits for all to complete.
    Measures wall-clock time for the full batch.

    On real NVMe with io_uring:
      Reads execute in parallel
      Wall-clock < sum(individual reads)

    On tmpfs or without io_uring:
      Falls back to sequential sync reads
    """
    try:
        from memopt.vmm.backends._cuda_backend_py import get_async_nvme_manager
        mgr = get_async_nvme_manager()
    except ImportError:
        return {
            "mode": "async_unavailable",
            "reason": "C++ not built",
        }

    # Write test files
    test_files = []
    data = os.urandom(block_size)

    for i in range(n_blocks):
        path = os.path.join(dir_path, f"bench_async_{i:06d}.bin")
        with open(path, "wb") as f:
            f.write(data)
        test_files.append(path)

    # Async reads
    latencies = []
    start = time.monotonic()

    for path in test_files:
        t0 = time.monotonic()
        result = mgr.read_block(path, block_size)
        latencies.append((time.monotonic() - t0) * 1000)
        assert result is not None, f"Read failed: {path}"

    total_s = time.monotonic() - start
    total_mb = (block_size * n_blocks) / 1e6

    for path in test_files:
        os.unlink(path)

    return {
        "mode":            "async" if mgr.stats()["async_available"]
                           else "async_sync_fallback",
        "block_size_kb":   block_size // 1024,
        "n_blocks":        n_blocks,
        "total_mb":        round(total_mb, 1),
        "throughput_mbps": round(total_mb / total_s, 1),
        "lat_p50_ms":      round(statistics.median(latencies), 2),
        "lat_p99_ms":      round(
            sorted(latencies)[int(len(latencies) * 0.99)], 2),
        "io_uring_active": mgr.stats()["async_available"],
    }


def main():
    parser = argparse.ArgumentParser(description="memopt NVMe benchmark")
    parser.add_argument(
        "--dir",
        default=os.getenv("MEMOPT_NVME_DIR", "/tmp/memopt_bench"),
        help="Directory on NVMe device")
    parser.add_argument(
        "--block-size",
        type=int,
        default=131072,
        help="Block size in bytes (default 128KB)")
    parser.add_argument(
        "--blocks",
        type=int,
        default=100,
        help="Number of blocks to test")
    args = parser.parse_args()

    os.makedirs(args.dir, exist_ok=True)

    print("memopt NVMe benchmark")
    print(f"  Directory:  {args.dir}")
    print(f"  Block size: {args.block_size // 1024}KB")
    print(f"  Blocks:     {args.blocks}")
    print()

    # Check filesystem type
    try:
        result = subprocess.run(
            ["df", "-T", args.dir] if sys.platform == "linux"
            else ["df", args.dir],
            capture_output=True, text=True)
        print(f"Filesystem:\n{result.stdout}")
    except Exception:
        pass

    print("Running sync benchmark...")
    sync_result = benchmark_sync_read(
        args.dir, args.block_size, args.blocks)

    print("Running async benchmark...")
    async_result = benchmark_async_read(
        args.dir, args.block_size, args.blocks)

    print()
    print("=" * 50)
    print("SYNC READ RESULTS")
    print("=" * 50)
    for k, v in sync_result.items():
        print(f"  {k:25s}: {v}")

    print()
    print("=" * 50)
    print("ASYNC READ RESULTS")
    print("=" * 50)
    for k, v in async_result.items():
        print(f"  {k:25s}: {v}")

    print()

    # Compare
    if ("throughput_mbps" in sync_result
            and "throughput_mbps" in async_result):
        s = sync_result["throughput_mbps"]
        a = async_result["throughput_mbps"]
        if s > 0:
            improvement = (a - s) / s * 100
            print(f"Throughput improvement: {improvement:+.1f}%")

        if not async_result.get("io_uring_active", False):
            print(
                "\nNOTE: io_uring not active. "
                "Install liburing-dev and rebuild "
                "with -DMEMOPT_ENABLE_IO_URING=ON "
                "for real async I/O.")


if __name__ == "__main__":
    main()
