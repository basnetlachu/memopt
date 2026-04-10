// test_cuda_backend.cpp — GoogleTest for stream pool, NVMe I/O, GDS stubs.
//
// All tests run without GPU, without CUDA. Stream pool tests that
// require CUDA are skipped via is_available() check.

#include "gds.h"
#include "nvme_io.h"
#include "stream_pool.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using namespace memopt::cuda_backend;
namespace fs = std::filesystem;

// ═══════════════════════════════════════════════════════════════════════════
// Helper: create temp directory
// ═══════════════════════════════════════════════════════════════════════════
class TempDir {
public:
    TempDir() {
        char tpl[] = "/tmp/memopt_test_XXXXXX";
        path_ = ::mkdtemp(tpl);
    }
    ~TempDir() {
        fs::remove_all(path_);
    }
    const std::string& path() const { return path_; }
private:
    std::string path_;
};

// ═══════════════════════════════════════════════════════════════════════════
// 1. Stream pool — no CUDA
// ═══════════════════════════════════════════════════════════════════════════

TEST(StreamPool, ConstructionWithoutCUDA) {
    CUDAStreamPool pool(0);
    // On CPU-only machine, pool is in degraded mode
    // is_available() returns false, acquire() returns nullptr
    if (!pool.is_available()) {
        EXPECT_EQ(pool.acquire(), nullptr);
    }
    // Either way, no crash
}

TEST(StreamPool, SyncAllNoCrash) {
    CUDAStreamPool pool(0);
    pool.sync_all();  // no-op if not available
}

TEST(StreamPool, DeviceId) {
    CUDAStreamPool pool(0);
    EXPECT_EQ(pool.device_id(), 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Stream pool — round-robin (GPU only)
// ═══════════════════════════════════════════════════════════════════════════

TEST(StreamPool, RoundRobinDistribution) {
    CUDAStreamPool pool(0);
    if (!pool.is_available()) {
        GTEST_SKIP() << "No CUDA — skipping round-robin test";
    }

    // Acquire 16 streams — each of 8 should be acquired twice
    for (int i = 0; i < 16; ++i) {
        auto s = pool.acquire();
        EXPECT_NE(s, nullptr);
    }

    auto counts = pool.use_counts();
    for (int i = 0; i < CUDAStreamPool::NUM_STREAMS; ++i) {
        EXPECT_EQ(counts[i], 2u) << "Stream " << i
            << " acquired " << counts[i] << " times, expected 2";
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Stream pool thread safety (GPU only)
// ═══════════════════════════════════════════════════════════════════════════

TEST(StreamPool, ThreadSafeAcquire) {
    CUDAStreamPool pool(0);
    if (!pool.is_available()) {
        GTEST_SKIP() << "No CUDA — skipping thread safety test";
    }

    constexpr int NUM_THREADS = 16;
    constexpr int ITERS = 1000;
    std::vector<std::thread> threads;

    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&pool]() {
            for (int i = 0; i < ITERS; ++i) {
                auto s = pool.acquire();
                EXPECT_NE(s, nullptr);
            }
        });
    }
    for (auto& t : threads) t.join();

    uint64_t total = 0;
    for (auto c : pool.use_counts()) total += c;
    EXPECT_EQ(total, static_cast<uint64_t>(NUM_THREADS * ITERS));
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. write_block_atomic — success path
// ═══════════════════════════════════════════════════════════════════════════

TEST(NVMeIO, WriteBlockAtomicSuccess) {
    TempDir td;
    std::string path = td.path() + "/test.vmm_block";

    std::vector<char> data(1024 * 1024, 'A');  // 1MB
    auto result = write_block_atomic(path, data.data(), data.size());

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.errno_val, 0);
    EXPECT_EQ(result.bytes_written, static_cast<int64_t>(data.size()));

    // Verify file exists and no .tmp remains
    EXPECT_TRUE(fs::exists(path));
    EXPECT_FALSE(fs::exists(path + ".tmp"));

    // Verify content
    std::ifstream f(path, std::ios::binary);
    std::vector<char> readback(data.size());
    f.read(readback.data(), readback.size());
    EXPECT_EQ(readback, data);
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. write_block_atomic — crash safety (.tmp cleanup)
// ═══════════════════════════════════════════════════════════════════════════

TEST(NVMeIO, RecoverRemovesTmpFiles) {
    TempDir td;

    // Create a committed file and two .tmp files
    std::string committed = td.path() + "/a.vmm_block";
    std::string tmp1 = td.path() + "/b.vmm_block.tmp";
    std::string tmp2 = td.path() + "/c.vmm_block.tmp";

    { std::ofstream(committed) << "committed"; }
    { std::ofstream(tmp1) << "orphan1"; }
    { std::ofstream(tmp2) << "orphan2"; }

    int removed = recover_nvme_dir(td.path());

    EXPECT_EQ(removed, 2);
    EXPECT_TRUE(fs::exists(committed));
    EXPECT_FALSE(fs::exists(tmp1));
    EXPECT_FALSE(fs::exists(tmp2));
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. write_block_atomic — concurrent writes
// ═══════════════════════════════════════════════════════════════════════════

TEST(NVMeIO, ConcurrentWrites) {
    TempDir td;
    constexpr int NUM_THREADS = 8;

    std::vector<std::thread> threads;
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&td, t]() {
            std::string path = td.path() + "/block_" +
                               std::to_string(t) + ".vmm_block";
            std::vector<char> data(4096, static_cast<char>('0' + t));
            auto result = write_block_atomic(path, data.data(), data.size());
            EXPECT_TRUE(result.success) << "Thread " << t << " write failed";
        });
    }
    for (auto& t : threads) t.join();

    // Verify all files exist and content matches
    for (int t = 0; t < NUM_THREADS; ++t) {
        std::string path = td.path() + "/block_" +
                           std::to_string(t) + ".vmm_block";
        EXPECT_TRUE(fs::exists(path));
        std::ifstream f(path, std::ios::binary);
        std::vector<char> readback(4096);
        f.read(readback.data(), readback.size());
        EXPECT_EQ(readback[0], static_cast<char>('0' + t));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. read_block — round-trip
// ═══════════════════════════════════════════════════════════════════════════

TEST(NVMeIO, ReadBlockRoundTrip) {
    TempDir td;
    std::string path = td.path() + "/rt.vmm_block";

    std::string data = "Hello NVMe block I/O round-trip test!";
    auto wr = write_block_atomic(path, data.data(), data.size());
    ASSERT_TRUE(wr.success);

    std::vector<char> buf(1024);
    auto rd = read_block(path, buf.data(), buf.size());

    EXPECT_TRUE(rd.success);
    EXPECT_EQ(rd.bytes_read, static_cast<int64_t>(data.size()));
    EXPECT_EQ(std::string(buf.data(), rd.bytes_read), data);
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. read_block — missing file
// ═══════════════════════════════════════════════════════════════════════════

TEST(NVMeIO, ReadBlockMissingFile) {
    std::vector<char> buf(1024);
    auto result = read_block("/nonexistent/path.vmm_block",
                              buf.data(), buf.size());
    EXPECT_FALSE(result.success);
    EXPECT_NE(result.errno_val, 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. recover_nvme_dir — missing directory
// ═══════════════════════════════════════════════════════════════════════════

TEST(NVMeIO, RecoverMissingDir) {
    int removed = recover_nvme_dir("/nonexistent/dir/path");
    EXPECT_EQ(removed, 0);  // no crash
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. hbm_used_bytes — no crash on CPU
// ═══════════════════════════════════════════════════════════════════════════
// (This tests the C++ function directly, not through pybind11)

// Can't test hbm_used_bytes directly without CUDA headers in test,
// but we verify the NVMe I/O functions don't crash on CPU-only.

// ═══════════════════════════════════════════════════════════════════════════
// 11. gds_is_available — no crash without GDS
// ═══════════════════════════════════════════════════════════════════════════

TEST(GDS, IsAvailableNoCrash) {
    bool avail = gds_is_available();
    // On most test machines, GDS is not available
    // Just verify it doesn't crash
    (void)avail;
}

// ═══════════════════════════════════════════════════════════════════════════
// 12. Benchmark: write_block_atomic throughput
// ═══════════════════════════════════════════════════════════════════════════

TEST(NVMeIO, BenchmarkWriteThroughput) {
    TempDir td;
    constexpr int NUM_WRITES = 500;
    constexpr int BLOCK_SIZE = 128 * 1024;  // 128KB

    std::vector<char> data(BLOCK_SIZE, 'X');

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_WRITES; ++i) {
        std::string path = td.path() + "/bench_" +
                           std::to_string(i) + ".vmm_block";
        auto result = write_block_atomic(path, data.data(), data.size());
        EXPECT_TRUE(result.success);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start).count();

    double total_mb = static_cast<double>(NUM_WRITES) * BLOCK_SIZE / (1024.0 * 1024.0);
    double mbps = (ms > 0) ? total_mb / (ms / 1000.0) : 0;

    // Report throughput — no pass/fail threshold (NVMe speed varies)
    std::printf("NVMe write throughput: %.1f MB/s (%d × %dKB in %ldms)\n",
                mbps, NUM_WRITES, BLOCK_SIZE / 1024,
                static_cast<long>(ms));
}

// ═══════════════════════════════════════════════════════════════════════════
// 13. write_block_atomic — empty data
// ═══════════════════════════════════════════════════════════════════════════

TEST(NVMeIO, WriteBlockAtomicEmpty) {
    TempDir td;
    std::string path = td.path() + "/empty.vmm_block";

    auto result = write_block_atomic(path, nullptr, 0);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.bytes_written, 0);
    EXPECT_TRUE(fs::exists(path));
    EXPECT_EQ(fs::file_size(path), 0u);
}
