// test_block_pool.cpp — GoogleTest for C++ block pool.
//
// x86-64 with -mcx16: tests the lock-free CAS path.
// Other platforms:     tests the mutex-protected path.
// All tests run without GPU, without Python, without CUDA.
// Concurrent tests designed to trigger TSan/ASan reports if races exist.

#include "block_pool.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <set>
#include <thread>
#include <vector>

using namespace memopt::paged;

// ═══════════════════════════════════════════════════════════════════════════
// 1. Allocate all blocks — all return distinct IDs in [0, capacity)
// ═══════════════════════════════════════════════════════════════════════════

TEST(BlockPool, AllocateAllBlocks) {
    constexpr int N = 100;
    BlockPool pool(N);

    std::set<int32_t> ids;
    for (int i = 0; i < N; ++i) {
        int32_t id = pool.allocate();
        ASSERT_GE(id, 0) << "Allocation " << i << " failed unexpectedly";
        ASSERT_LT(id, N) << "Block ID out of range: " << id;
        ids.insert(id);
    }

    // All IDs must be distinct
    EXPECT_EQ(ids.size(), static_cast<size_t>(N));

    // Pool must be exhausted
    EXPECT_EQ(pool.allocate(), -1);
    EXPECT_EQ(pool.num_free_blocks(), 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Free and reallocate
// ═══════════════════════════════════════════════════════════════════════════

TEST(BlockPool, FreeAndReallocate) {
    constexpr int N = 100;
    BlockPool pool(N);

    // Allocate all
    std::vector<int32_t> allocated;
    for (int i = 0; i < N; ++i) {
        allocated.push_back(pool.allocate());
    }

    // Free 50
    for (int i = 0; i < 50; ++i) {
        pool.free(allocated[i]);
    }
    EXPECT_EQ(pool.num_free_blocks(), 50);

    // Reallocate 50 — all must succeed
    for (int i = 0; i < 50; ++i) {
        int32_t id = pool.allocate();
        EXPECT_GE(id, 0) << "Reallocation " << i << " failed";
    }
    EXPECT_EQ(pool.num_free_blocks(), 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Capacity invariant
// ═══════════════════════════════════════════════════════════════════════════

TEST(BlockPool, CapacityInvariant) {
    constexpr int N = 200;
    BlockPool pool(N);

    int32_t allocated_count = 0;

    // Mixed alloc/free sequence
    for (int round = 0; round < 5; ++round) {
        // Allocate 40
        std::vector<int32_t> batch;
        for (int i = 0; i < 40 && allocated_count < N; ++i) {
            int32_t id = pool.allocate();
            if (id >= 0) {
                batch.push_back(id);
                ++allocated_count;
            }
        }

        EXPECT_EQ(pool.num_free_blocks() + allocated_count, N)
            << "Invariant violated after allocating in round " << round;

        // Free half
        int to_free = static_cast<int>(batch.size()) / 2;
        for (int i = 0; i < to_free; ++i) {
            pool.free(batch[i]);
            --allocated_count;
        }

        EXPECT_EQ(pool.num_free_blocks() + allocated_count, N)
            << "Invariant violated after freeing in round " << round;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Concurrent alloc/free — TSan safety test
// ═══════════════════════════════════════════════════════════════════════════

TEST(BlockPool, ConcurrentAllocFreeNoRace) {
    constexpr int NUM_THREADS = 8;
    constexpr int ITERS = 100000;
    // Pool large enough that threads don't exhaust it
    constexpr int POOL_SIZE = NUM_THREADS * 4;
    BlockPool pool(POOL_SIZE);

    std::vector<std::thread> threads;
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&pool]() {
            for (int i = 0; i < ITERS; ++i) {
                int32_t id = pool.allocate();
                if (id >= 0) {
                    pool.free(id);
                }
            }
        });
    }
    for (auto& t : threads) t.join();

    // All blocks must be returned to pool
    EXPECT_EQ(pool.num_free_blocks(), POOL_SIZE)
        << "Pool leaked blocks under concurrent load";
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Concurrent producer/consumer — no deadlock
// ═══════════════════════════════════════════════════════════════════════════

TEST(BlockPool, ConcurrentProducerConsumer) {
    constexpr int POOL_SIZE = 64;
    constexpr int ROUNDS = 10000;
    BlockPool pool(POOL_SIZE);

    std::atomic<bool> done{false};
    std::atomic<int32_t> total_allocated{0};
    std::atomic<int32_t> total_freed{0};

    // Shared buffer: allocated block IDs waiting to be freed
    std::atomic<int32_t> buffer[64];
    std::atomic<int32_t> buf_count{0};
    for (auto& b : buffer) b.store(-1, std::memory_order_relaxed);

    // Producers: allocate blocks
    std::vector<std::thread> producers;
    for (int t = 0; t < 4; ++t) {
        producers.emplace_back([&]() {
            for (int i = 0; i < ROUNDS && !done.load(); ++i) {
                int32_t id = pool.allocate();
                if (id >= 0) {
                    total_allocated.fetch_add(1);
                    // Try to put in buffer for consumers
                    int32_t idx = buf_count.fetch_add(1) % 64;
                    int32_t expected = -1;
                    if (!buffer[idx].compare_exchange_strong(expected, id)) {
                        // Buffer slot occupied — free immediately
                        pool.free(id);
                        total_freed.fetch_add(1);
                    }
                }
            }
        });
    }

    // Consumers: free blocks from buffer
    std::vector<std::thread> consumers;
    for (int t = 0; t < 4; ++t) {
        consumers.emplace_back([&]() {
            for (int i = 0; i < ROUNDS && !done.load(); ++i) {
                int32_t idx = i % 64;
                int32_t id = buffer[idx].exchange(-1);
                if (id >= 0) {
                    pool.free(id);
                    total_freed.fetch_add(1);
                }
            }
        });
    }

    for (auto& t : producers) t.join();
    done.store(true);
    for (auto& t : consumers) t.join();

    // Drain remaining buffer
    for (auto& b : buffer) {
        int32_t id = b.exchange(-1);
        if (id >= 0) {
            pool.free(id);
            total_freed.fetch_add(1);
        }
    }

    EXPECT_EQ(pool.num_free_blocks(), POOL_SIZE)
        << "Pool leaked blocks in producer/consumer test";
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. Exhaustion is non-fatal
// ═══════════════════════════════════════════════════════════════════════════

TEST(BlockPool, ExhaustionNonFatal) {
    constexpr int N = 50;
    BlockPool pool(N);

    // Allocate all
    for (int i = 0; i < N; ++i) {
        EXPECT_GE(pool.allocate(), 0);
    }

    // 1000 more allocations must all return -1
    for (int i = 0; i < 1000; ++i) {
        EXPECT_EQ(pool.allocate(), -1)
            << "Expected -1 on exhausted pool, iteration " << i;
    }

    // No crash, no throw
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. Single-threaded alloc/free ordering (LIFO verification)
// ═══════════════════════════════════════════════════════════════════════════

TEST(BlockPool, SingleThreadedOrdering) {
    constexpr int N = 10;
    BlockPool pool(N);

    // Allocate 10
    std::vector<int32_t> first_round;
    for (int i = 0; i < N; ++i) {
        first_round.push_back(pool.allocate());
    }

    // Free in reverse order
    for (int i = N - 1; i >= 0; --i) {
        pool.free(first_round[i]);
    }

    // Allocate 10 again — all must be valid IDs
    std::set<int32_t> second_round;
    for (int i = 0; i < N; ++i) {
        int32_t id = pool.allocate();
        EXPECT_GE(id, 0);
        EXPECT_LT(id, N);
        second_round.insert(id);
    }
    // All 10 must be distinct
    EXPECT_EQ(second_round.size(), static_cast<size_t>(N));
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. LIFO — last freed is first allocated
// ═══════════════════════════════════════════════════════════════════════════

TEST(BlockPool, LIFOBehavior) {
    BlockPool pool(10);

    int32_t b1 = pool.allocate();
    int32_t b2 = pool.allocate();

    pool.free(b1);
    pool.free(b2);

    // LIFO: b2 was freed last → should be allocated first
    int32_t b3 = pool.allocate();
    EXPECT_EQ(b3, b2) << "Expected LIFO: last freed should be first allocated";
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. Capacity accessor
// ═══════════════════════════════════════════════════════════════════════════

TEST(BlockPool, CapacityAccessor) {
    BlockPool pool(256);
    EXPECT_EQ(pool.capacity(), 256);
    EXPECT_EQ(pool.num_free_blocks(), 256);

    pool.allocate();
    EXPECT_EQ(pool.capacity(), 256);     // doesn't change
    EXPECT_EQ(pool.num_free_blocks(), 255);
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. Benchmark: concurrent alloc+free throughput
// ═══════════════════════════════════════════════════════════════════════════

TEST(BlockPool, BenchmarkConcurrentThroughput) {
    constexpr int NUM_THREADS = 8;
    constexpr int ITERS = 1000000;
    constexpr int POOL_SIZE = NUM_THREADS * 8;  // enough for all threads
    BlockPool pool(POOL_SIZE);

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&pool]() {
            for (int i = 0; i < ITERS; ++i) {
                int32_t id = pool.allocate();
                if (id >= 0) pool.free(id);
            }
        });
    }
    for (auto& t : threads) t.join();

    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start).count();

    double total_ops = static_cast<double>(NUM_THREADS) * ITERS * 2;
    double ops_per_sec = total_ops / (ms / 1000.0);

    // Must achieve > 5M alloc+free ops/sec aggregate
    EXPECT_GT(ops_per_sec, 5e6)
        << "Throughput: " << ops_per_sec / 1e6
        << "M ops/sec (min 5M/sec), took " << ms << "ms";
}

// ═══════════════════════════════════════════════════════════════════════════
// 11. Implementation honestly named
// ═══════════════════════════════════════════════════════════════════════════

TEST(BlockPool, ImplementationHonestlyNamed) {
#if MEMOPT_BLOCK_POOL_LOCK_FREE
    SUCCEED() << "x86-64: lock-free CAS path (cmpxchg16b)";
    // Verify is_lock_free() matches compile-time selection
    EXPECT_TRUE(BlockPool::is_lock_free());
#else
    SUCCEED() << "non-x86-64: mutex-protected pool";
    EXPECT_FALSE(BlockPool::is_lock_free());
#endif
}
