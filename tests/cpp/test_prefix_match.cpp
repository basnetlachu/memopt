// test_prefix_match.cpp — GoogleTest for prefix matching implementations.
//
// No GPU, no RDMA, runs on any machine.
// AVX2/AVX-512 tests auto-skip if ISA not available.

#include "prefix_match.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

using namespace memopt::simd;

// ═══════════════════════════════════════════════════════════════════════════
// Helper: generate random int32 array
// ═══════════════════════════════════════════════════════════════════════════
static std::vector<int32_t> random_tokens(size_t len, std::mt19937& rng) {
    std::vector<int32_t> v(len);
    std::uniform_int_distribution<int32_t> dist(0, 50000);
    for (auto& x : v) x = dist(rng);
    return v;
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Identical arrays → returns len
// ═══════════════════════════════════════════════════════════════════════════

TEST(PrefixMatch, IdenticalReturnsLen) {
    int32_t a[] = {1, 2, 3, 4, 5};
    EXPECT_EQ(find_lcp_scalar(a, a, 5), 5u);
    EXPECT_EQ(find_lcp(a, a, 5), 5u);
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. First token differs → returns 0
// ═══════════════════════════════════════════════════════════════════════════

TEST(PrefixMatch, FirstDiffersReturns0) {
    int32_t a[] = {1, 2, 3};
    int32_t b[] = {9, 2, 3};
    EXPECT_EQ(find_lcp_scalar(a, b, 3), 0u);
    EXPECT_EQ(find_lcp(a, b, 3), 0u);
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Last token differs → returns len-1
// ═══════════════════════════════════════════════════════════════════════════

TEST(PrefixMatch, LastDiffersReturnsLenMinus1) {
    int32_t a[] = {1, 2, 3, 4, 5};
    int32_t b[] = {1, 2, 3, 4, 9};
    EXPECT_EQ(find_lcp_scalar(a, b, 5), 4u);
    EXPECT_EQ(find_lcp(a, b, 5), 4u);
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Empty arrays → returns 0
// ═══════════════════════════════════════════════════════════════════════════

TEST(PrefixMatch, EmptyReturns0) {
    EXPECT_EQ(find_lcp_scalar(nullptr, nullptr, 0), 0u);
    EXPECT_EQ(find_lcp(nullptr, nullptr, 0), 0u);
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Single token match
// ═══════════════════════════════════════════════════════════════════════════

TEST(PrefixMatch, SingleMatch) {
    int32_t a[] = {42};
    int32_t b[] = {42};
    EXPECT_EQ(find_lcp(a, b, 1), 1u);
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. Single token mismatch
// ═══════════════════════════════════════════════════════════════════════════

TEST(PrefixMatch, SingleMismatch) {
    int32_t a[] = {42};
    int32_t b[] = {43};
    EXPECT_EQ(find_lcp(a, b, 1), 0u);
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. AVX2 vs scalar agreement
// ═══════════════════════════════════════════════════════════════════════════

TEST(PrefixMatch, AVX2MatchesScalar) {
    std::mt19937 rng(42);

    for (int trial = 0; trial < 1000; ++trial) {
        size_t len = 1 + (rng() % 512);
        auto a = random_tokens(len, rng);
        auto b = a;  // start identical

        // Randomly corrupt some suffix
        size_t corrupt_at = rng() % (len + 1);
        for (size_t j = corrupt_at; j < len; ++j) {
            b[j] = a[j] + 1 + (rng() % 100);
        }

        size_t scalar_result = find_lcp_scalar(a.data(), b.data(), len);
        size_t avx2_result   = find_lcp_avx2(a.data(), b.data(), len);

        EXPECT_EQ(scalar_result, avx2_result)
            << "Mismatch at trial=" << trial << " len=" << len
            << " corrupt_at=" << corrupt_at;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. AVX-512 vs scalar agreement
// ═══════════════════════════════════════════════════════════════════════════

TEST(PrefixMatch, AVX512MatchesScalar) {
    std::mt19937 rng(123);

    for (int trial = 0; trial < 1000; ++trial) {
        size_t len = 1 + (rng() % 1024);
        auto a = random_tokens(len, rng);
        auto b = a;

        size_t corrupt_at = rng() % (len + 1);
        for (size_t j = corrupt_at; j < len; ++j) {
            b[j] = a[j] + 1 + (rng() % 100);
        }

        size_t scalar_result = find_lcp_scalar(a.data(), b.data(), len);
        size_t avx512_result = find_lcp_avx512(a.data(), b.data(), len);

        EXPECT_EQ(scalar_result, avx512_result)
            << "Mismatch at trial=" << trial << " len=" << len;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. Boundary conditions — chunk edges
// ═══════════════════════════════════════════════════════════════════════════

TEST(PrefixMatch, BoundaryConditions) {
    // Test lengths around SIMD chunk boundaries
    size_t test_lens[] = {
        1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33,
        63, 64, 65, 127, 128, 129, 255, 256, 257
    };

    for (size_t len : test_lens) {
        // Identical arrays
        std::vector<int32_t> a(len, 42);
        std::vector<int32_t> b(len, 42);

        EXPECT_EQ(find_lcp_scalar(a.data(), b.data(), len), len);
        EXPECT_EQ(find_lcp_avx2(a.data(), b.data(), len), len);
        EXPECT_EQ(find_lcp_avx512(a.data(), b.data(), len), len);

        // Mismatch at position 0
        b[0] = 999;
        EXPECT_EQ(find_lcp_scalar(a.data(), b.data(), len), 0u);
        EXPECT_EQ(find_lcp_avx2(a.data(), b.data(), len), 0u);
        EXPECT_EQ(find_lcp_avx512(a.data(), b.data(), len), 0u);
        b[0] = 42;

        // Mismatch at last position
        b[len - 1] = 999;
        EXPECT_EQ(find_lcp_scalar(a.data(), b.data(), len), len - 1);
        EXPECT_EQ(find_lcp_avx2(a.data(), b.data(), len), len - 1);
        EXPECT_EQ(find_lcp_avx512(a.data(), b.data(), len), len - 1);
        b[len - 1] = 42;

        // Mismatch at midpoint
        if (len > 2) {
            size_t mid = len / 2;
            b[mid] = 999;
            EXPECT_EQ(find_lcp_scalar(a.data(), b.data(), len), mid);
            EXPECT_EQ(find_lcp_avx2(a.data(), b.data(), len), mid);
            EXPECT_EQ(find_lcp_avx512(a.data(), b.data(), len), mid);
            b[mid] = 42;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. Large sequence — 1M tokens
// ═══════════════════════════════════════════════════════════════════════════

TEST(PrefixMatch, LargeSequence1M) {
    constexpr size_t N = 1000000;
    std::vector<int32_t> a(N);
    std::iota(a.begin(), a.end(), 0);  // 0, 1, 2, ..., N-1
    std::vector<int32_t> b = a;        // identical

    // All match
    EXPECT_EQ(find_lcp_scalar(a.data(), b.data(), N), N);
    EXPECT_EQ(find_lcp_avx2(a.data(), b.data(), N), N);
    EXPECT_EQ(find_lcp_avx512(a.data(), b.data(), N), N);

    // Mismatch at last element
    b[N - 1] = -1;
    EXPECT_EQ(find_lcp_scalar(a.data(), b.data(), N), N - 1);
    EXPECT_EQ(find_lcp_avx2(a.data(), b.data(), N), N - 1);
    EXPECT_EQ(find_lcp_avx512(a.data(), b.data(), N), N - 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// 11. Random stress — 10,000 pairs
// ═══════════════════════════════════════════════════════════════════════════

TEST(PrefixMatch, RandomStress10K) {
    std::mt19937 rng(999);

    for (int trial = 0; trial < 10000; ++trial) {
        size_t len = rng() % 2049;  // [0, 2048]
        if (len == 0) {
            EXPECT_EQ(find_lcp(nullptr, nullptr, 0), 0u);
            continue;
        }

        auto a = random_tokens(len, rng);
        auto b = a;

        // Random corruption point
        if (rng() % 2 == 0 && len > 0) {
            size_t pos = rng() % len;
            b[pos] = a[pos] + 1;
        }

        size_t s = find_lcp_scalar(a.data(), b.data(), len);
        size_t v2 = find_lcp_avx2(a.data(), b.data(), len);
        size_t v5 = find_lcp_avx512(a.data(), b.data(), len);

        EXPECT_EQ(s, v2) << "AVX2 disagrees at trial " << trial;
        EXPECT_EQ(s, v5) << "AVX512 disagrees at trial " << trial;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 12. ISA detection — never crashes
// ═══════════════════════════════════════════════════════════════════════════

TEST(PrefixMatch, ISADetection) {
    ISA isa = detect_best_isa();
    // Must be one of the valid values
    EXPECT_TRUE(isa == ISA::SCALAR || isa == ISA::AVX2 ||
                isa == ISA::AVX512);
}

// ═══════════════════════════════════════════════════════════════════════════
// 13. Benchmark: scalar throughput
// ═══════════════════════════════════════════════════════════════════════════

TEST(PrefixMatch, BenchmarkScalar) {
    constexpr size_t LEN = 1024;
    constexpr int ITERS = 100000;

    std::vector<int32_t> a(LEN, 42);
    std::vector<int32_t> b = a;
    b[LEN / 2] = 999;  // mismatch at midpoint

    auto start = std::chrono::high_resolution_clock::now();
    volatile size_t dummy = 0;
    for (int i = 0; i < ITERS; ++i) {
        dummy = find_lcp_scalar(a.data(), b.data(), LEN);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start).count();

    double calls_per_sec = static_cast<double>(ITERS) /
                           (us / 1e6);
    double tokens_per_sec = calls_per_sec * (LEN / 2);  // avg tokens compared

    std::printf("Scalar: %.1fM calls/sec, %.1fM tokens/sec "
                "(%d iters, %ldµs)\n",
                calls_per_sec / 1e6, tokens_per_sec / 1e6,
                ITERS, static_cast<long>(us));

    EXPECT_EQ(dummy, LEN / 2);
}

// ═══════════════════════════════════════════════════════════════════════════
// 14. Benchmark: AVX2 throughput
// ═══════════════════════════════════════════════════════════════════════════

TEST(PrefixMatch, BenchmarkAVX2) {
    constexpr size_t LEN = 1024;
    constexpr int ITERS = 100000;

    std::vector<int32_t> a(LEN, 42);
    std::vector<int32_t> b = a;
    b[LEN / 2] = 999;

    auto start = std::chrono::high_resolution_clock::now();
    volatile size_t dummy = 0;
    for (int i = 0; i < ITERS; ++i) {
        dummy = find_lcp_avx2(a.data(), b.data(), LEN);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start).count();

    double calls_per_sec = static_cast<double>(ITERS) /
                           (us / 1e6);
    std::printf("AVX2:   %.1fM calls/sec (%d iters, %ldµs)\n",
                calls_per_sec / 1e6, ITERS, static_cast<long>(us));

    EXPECT_EQ(dummy, LEN / 2);
}

// ═══════════════════════════════════════════════════════════════════════════
// 15. Benchmark: AVX-512 throughput
// ═══════════════════════════════════════════════════════════════════════════

TEST(PrefixMatch, BenchmarkAVX512) {
    constexpr size_t LEN = 1024;
    constexpr int ITERS = 100000;

    std::vector<int32_t> a(LEN, 42);
    std::vector<int32_t> b = a;
    b[LEN / 2] = 999;

    auto start = std::chrono::high_resolution_clock::now();
    volatile size_t dummy = 0;
    for (int i = 0; i < ITERS; ++i) {
        dummy = find_lcp_avx512(a.data(), b.data(), LEN);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start).count();

    double calls_per_sec = static_cast<double>(ITERS) /
                           (us / 1e6);
    std::printf("AVX512: %.1fM calls/sec (%d iters, %ldµs)\n",
                calls_per_sec / 1e6, ITERS, static_cast<long>(us));

    EXPECT_EQ(dummy, LEN / 2);
}
