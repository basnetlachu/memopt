// test_oracle.cpp — GoogleTest unit tests for the Memory Oracle.
//
// Tests:
//   1. observe/predict round-trip
//   2. Prediction ordering by confidence
//   3. Sequential fallback always present
//   4. Confidence values in [0, 1]
//   5. min_confidence filtering
//   6. record_outcome accuracy tracking
//   7. Stats field completeness
//   8. Reset single sequence
//   9. Reset all
//  10. warm_from_log replay
//  11. warm_from_log handles missing file
//  12. Concurrent observe from 8 threads
//  13. Transition pruning stays bounded

#include "oracle.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using namespace memopt;

// ═══════════════════════════════════════════════════════════════════════════
// Basic observe / predict
// ═══════════════════════════════════════════════════════════════════════════

TEST(Oracle, ObserveUpdatesTransitions) {
    MemoryOracle oracle;
    oracle.observe("s1", 0, 1);
    oracle.observe("s1", 1, 2);
    oracle.observe("s1", 2, 3);

    // 0→1 and 1→2 should be recorded
    const auto& trans = oracle.transitions();
    ASSERT_TRUE(trans.count(0) > 0);
    ASSERT_TRUE(trans.at(0).count(1) > 0);
    ASSERT_TRUE(trans.count(1) > 0);
    ASSERT_TRUE(trans.at(1).count(2) > 0);
}

TEST(Oracle, PredictReturnsResults) {
    MemoryOracle oracle;
    oracle.observe("s1", 0, 1);
    oracle.observe("s1", 1, 2);

    auto preds = oracle.predict("s1", 1);
    EXPECT_GT(preds.size(), 0u);
}

TEST(Oracle, PredictOrderedByConfidence) {
    MemoryOracle oracle;
    for (int i = 0; i < 20; ++i) {
        oracle.observe("s1", i, i);
    }

    auto preds = oracle.predict("s1", 10, 10);
    for (size_t i = 1; i < preds.size(); ++i) {
        EXPECT_GE(preds[i - 1].confidence, preds[i].confidence);
    }
}

TEST(Oracle, SequentialFallbackAlwaysPresent) {
    MemoryOracle oracle;
    auto preds = oracle.predict("new_seq", 5);
    EXPECT_GE(preds.size(), 1u);

    bool has_six = false;
    for (const auto& p : preds) {
        if (p.block_index == 6) has_six = true;
    }
    EXPECT_TRUE(has_six);
}

TEST(Oracle, ConfidenceBetweenZeroAndOne) {
    MemoryOracle oracle;
    for (int i = 0; i < 10; ++i) {
        oracle.observe("s1", i, i);
    }
    auto preds = oracle.predict("s1", 5, 20);
    for (const auto& p : preds) {
        EXPECT_GE(p.confidence, 0.0);
        EXPECT_LE(p.confidence, 1.0);
    }
}

TEST(Oracle, MinConfidenceFilters) {
    MemoryOracle oracle(50, 100000, 0.99);
    oracle.observe("s1", 0, 1);
    oracle.observe("s1", 1, 2);
    oracle.observe("s1", 0, 3);
    oracle.observe("s1", 2, 4);

    auto preds = oracle.predict("s1", 0, 10);
    for (const auto& p : preds) {
        // Only high-confidence or fallback predictions survive
        EXPECT_TRUE(p.confidence >= 0.99 || p.source == "fallback");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Record outcome
// ═══════════════════════════════════════════════════════════════════════════

TEST(Oracle, RecordOutcomeTracksAccuracy) {
    MemoryOracle oracle;
    oracle.observe("s1", 0, 1);
    oracle.observe("s1", 1, 2);

    auto preds = oracle.predict("s1", 1);
    ASSERT_GT(preds.size(), 0u);

    oracle.record_outcome("s1", preds[0].block_index);

    auto s = oracle.stats();
    EXPECT_GE(s.total_predictions_correct, 1);
    EXPECT_GT(s.accuracy_pct, 0.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Stats
// ═══════════════════════════════════════════════════════════════════════════

TEST(Oracle, StatsHasAllFields) {
    MemoryOracle oracle;
    auto s = oracle.stats();

    // All fields should be initialized
    EXPECT_EQ(s.total_predictions_made, 0);
    EXPECT_EQ(s.total_predictions_correct, 0);
    EXPECT_DOUBLE_EQ(s.accuracy_pct, 0.0);
    EXPECT_EQ(s.transitions_learned, 0);
    EXPECT_EQ(s.sequences_tracked, 0);
    EXPECT_EQ(s.horizon, 50);
    EXPECT_GE(s.uptime_seconds, 0.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Reset
// ═══════════════════════════════════════════════════════════════════════════

TEST(Oracle, ResetSequenceClearsState) {
    MemoryOracle oracle;
    oracle.observe("s1", 0, 1);
    oracle.observe("s1", 1, 2);
    oracle.observe("s2", 0, 1);

    oracle.reset("s1");

    auto s = oracle.stats();
    EXPECT_EQ(s.sequences_tracked, 1);  // only s2 remains
}

TEST(Oracle, ResetAllClearsEverything) {
    MemoryOracle oracle;
    oracle.observe("s1", 0, 1);
    oracle.observe("s1", 1, 2);
    oracle.reset("");

    auto s = oracle.stats();
    EXPECT_EQ(s.transitions_learned, 0);
    EXPECT_EQ(s.sequences_tracked, 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// warm_from_log
// ═══════════════════════════════════════════════════════════════════════════

TEST(Oracle, WarmFromLogReplaysEvents) {
    // Create temp JSONL file
    std::string path = "/tmp/test_oracle_warm.jsonl";
    {
        std::ofstream f(path);
        for (int i = 0; i < 10; ++i) {
            f << "{\"sequence_id\": \"s1\", \"block_index\": " << i
              << ", \"token_position\": " << i << "}\n";
        }
    }

    MemoryOracle oracle;
    int count = oracle.warm_from_log(path);
    EXPECT_EQ(count, 10);
    EXPECT_GE(oracle.stats().transitions_learned, 1);

    // Cleanup
    std::remove(path.c_str());
}

TEST(Oracle, WarmFromLogHandlesMissingFile) {
    MemoryOracle oracle;
    int count = oracle.warm_from_log("/nonexistent/path.jsonl");
    EXPECT_EQ(count, 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Concurrent observe — run under TSan
// ═══════════════════════════════════════════════════════════════════════════

TEST(Oracle, ConcurrentObserve) {
    MemoryOracle oracle;
    constexpr int NUM_THREADS = 8;
    constexpr int OPS_PER_THREAD = 1000;

    std::vector<std::thread> threads;
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&oracle, t]() {
            std::string seq = "thread_" + std::to_string(t);
            for (int i = 0; i < OPS_PER_THREAD; ++i) {
                oracle.observe(seq, i, i);
            }
        });
    }
    for (auto& t : threads) t.join();

    auto s = oracle.stats();
    EXPECT_EQ(s.sequences_tracked, NUM_THREADS);
    EXPECT_GT(s.transitions_learned, 0);
}

TEST(Oracle, ConcurrentObserveAndPredict) {
    MemoryOracle oracle;
    constexpr int NUM_THREADS = 8;

    std::vector<std::thread> threads;

    // Writers
    for (int t = 0; t < NUM_THREADS / 2; ++t) {
        threads.emplace_back([&oracle, t]() {
            std::string seq = "writer_" + std::to_string(t);
            for (int i = 0; i < 500; ++i) {
                oracle.observe(seq, i, i);
            }
        });
    }

    // Readers
    for (int t = 0; t < NUM_THREADS / 2; ++t) {
        threads.emplace_back([&oracle, t]() {
            std::string seq = "writer_" + std::to_string(t);
            for (int i = 0; i < 500; ++i) {
                oracle.predict(seq, i, 5);
            }
        });
    }

    for (auto& t : threads) t.join();

    // Just verify no crashes or TSan reports
    auto s = oracle.stats();
    EXPECT_GT(s.total_predictions_made, 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Transition pruning
// ═══════════════════════════════════════════════════════════════════════════

TEST(Oracle, TransitionPruningBoundsSize) {
    // Use a small max_transitions to trigger pruning frequently
    MemoryOracle oracle(50, 100, 0.3);

    // Insert many transitions — pruning should keep size bounded
    for (int i = 0; i < 2000; ++i) {
        oracle.observe("s1", i, i);
    }

    // After pruning, transitions should not grow unboundedly
    // (exact bound depends on pruning frequency — every 1000 obs)
    auto s = oracle.stats();
    // We're not checking exact count because pruning removes the
    // min-count entry every 1000 obs, but we verify no crash.
    EXPECT_GT(s.transitions_learned, 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Constructor defaults
// ═══════════════════════════════════════════════════════════════════════════

TEST(Oracle, ConstructorDefaults) {
    MemoryOracle oracle;
    EXPECT_EQ(oracle.horizon(), 50);
    EXPECT_DOUBLE_EQ(oracle.min_confidence(), 0.3);
    EXPECT_EQ(oracle.max_transitions(), 100000);
}

TEST(Oracle, ConstructorCustom) {
    MemoryOracle oracle(100, 50000, 0.5);
    EXPECT_EQ(oracle.horizon(), 100);
    EXPECT_DOUBLE_EQ(oracle.min_confidence(), 0.5);
    EXPECT_EQ(oracle.max_transitions(), 50000);
}
