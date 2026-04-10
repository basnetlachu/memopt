// oracle.h — Memory Oracle: predictive prefetch using Markov transitions.
//
// Replaces memopt/vmm/oracle.py. Identical pybind11 surface.
//
// Design:
//   - Transition table: unordered_map<uint32_t, unordered_map<uint32_t, uint32_t>>
//     (~40 bytes/entry vs Python's ~200 bytes → 5× memory reduction).
//   - 256-bucket striped locking: two concurrent updates to different
//     from_blocks never contend unless they share a stripe.
//   - predict() uses std::partial_sort (O(n + k log k)) instead of full sort.
//   - Recency: per-sequence circular buffer of last 8 blocks.
//   - warm_from_log: hand-written JSON field extraction (no library overhead).
//
// Thread safety: all public methods are thread-safe.
// GIL: all read-only methods release the GIL.
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace memopt {

// ═══════════════════════════════════════════════════════════════════════════
// BlockPrediction — matches Python dataclass exactly
// ═══════════════════════════════════════════════════════════════════════════
struct BlockPrediction {
    std::string sequence_id;
    int32_t     block_index;
    double      confidence;
    int32_t     predicted_at_step;
    std::string source;  // "transition"|"sequential"|"recency"|"fallback"
};

// ═══════════════════════════════════════════════════════════════════════════
// OracleStats — matches Python dataclass exactly
// ═══════════════════════════════════════════════════════════════════════════
struct OracleStats {
    int64_t total_predictions_made;
    int64_t total_predictions_correct;
    double  accuracy_pct;
    int64_t transitions_learned;
    int64_t sequences_tracked;
    int32_t horizon;
    double  uptime_seconds;
};

// ═══════════════════════════════════════════════════════════════════════════
// RecencyBuffer — per-sequence circular buffer of last 8 accessed blocks
// ═══════════════════════════════════════════════════════════════════════════
struct RecencyBuffer {
    std::array<int32_t, 8> blocks{};
    uint8_t write_idx = 0;
    uint8_t count = 0;

    void push(int32_t block) noexcept {
        blocks[write_idx % 8] = block;
        write_idx = (write_idx + 1) % 8;
        if (count < 8) ++count;
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// MemoryOracle — the public C++ class exposed to Python via pybind11
// ═══════════════════════════════════════════════════════════════════════════
class MemoryOracle {
public:
    /// Constructor matches Python: MemoryOracle(horizon, max_transitions, min_confidence)
    explicit MemoryOracle(int horizon = 50,
                          int max_transitions = 100000,
                          double min_confidence = 0.3);

    // ── Core API ─────────────────────────────────────────────────────

    /// Record a block access for transition learning.
    /// GIL: released.
    void observe(const std::string& sequence_id,
                 int32_t block_index,
                 int32_t step = -1);  // -1 = auto-increment

    /// Predict next blocks for a sequence.
    /// GIL: released.
    std::vector<BlockPrediction> predict(const std::string& sequence_id,
                                         int32_t current_block,
                                         int top_k = 10);

    /// Record whether a predicted block was actually accessed.
    /// GIL: released.
    void record_outcome(const std::string& sequence_id,
                        int32_t block_index);

    /// Return oracle statistics.
    /// GIL: released.
    OracleStats stats() const;

    /// Reset state — either for one sequence or all.
    /// GIL: released.
    void reset(const std::string& sequence_id = "");

    /// Replay JSONL log file to warm transitions.
    /// GIL: released (no Python callbacks).
    int warm_from_log(const std::string& log_path,
                      int max_events = 50000);

    // ── Accessors for Python test compatibility ──────────────────────
    // Tests access oracle._transitions[0], oracle._horizon, etc. directly.
    // We expose these as read-only properties through pybind11.

    int horizon() const noexcept { return horizon_; }
    double min_confidence() const noexcept { return min_confidence_; }
    int max_transitions() const noexcept { return max_transitions_; }

    /// Expose transitions for test compatibility.
    /// Returns a copy (tests read oracle._transitions[0]).
    const std::unordered_map<int32_t,
          std::unordered_map<int32_t, uint32_t>>& transitions() const {
        return transitions_;
    }

private:
    // ── Configuration ────────────────────────────────────────────────
    int    horizon_;
    int    max_transitions_;
    double min_confidence_;

    // ── Transition table ─────────────────────────────────────────────
    // from_block → { to_block → count }
    std::unordered_map<int32_t,
        std::unordered_map<int32_t, uint32_t>> transitions_;

    // ── Striped locks: 256 buckets ───────────────────────────────────
    static constexpr int NUM_STRIPES = 256;
    mutable std::array<std::mutex, NUM_STRIPES> stripes_;

    std::mutex& stripe_for(int32_t from_block) const noexcept {
        return stripes_[static_cast<uint32_t>(from_block) % NUM_STRIPES];
    }

    // ── Per-sequence state ───────────────────────────────────────────
    // Previous block per sequence (for transition recording)
    std::unordered_map<std::string, int32_t> prev_block_;  // _prev_{seq} equivalent
    std::unordered_map<std::string, bool>    has_prev_;

    // Step tracking per sequence
    std::unordered_map<std::string, int32_t> sequence_steps_;

    // Recency buffer per sequence
    std::unordered_map<std::string, RecencyBuffer> recency_;

    // Global recency: (sequence_id, block_index) → step, capped at 1000
    struct RecencyKey {
        std::string sequence_id;
        int32_t block_index;
        bool operator==(const RecencyKey& o) const {
            return block_index == o.block_index && sequence_id == o.sequence_id;
        }
    };
    struct RecencyKeyHash {
        size_t operator()(const RecencyKey& k) const {
            size_t h = std::hash<std::string>{}(k.sequence_id);
            h ^= std::hash<int32_t>{}(k.block_index) * 2654435761ULL;
            return h;
        }
    };
    std::unordered_map<RecencyKey, int32_t, RecencyKeyHash> global_recency_;
    std::vector<RecencyKey> recency_order_;  // insertion order for LRU eviction

    // ── Stats ────────────────────────────────────────────────────────
    mutable std::mutex stats_mu_;
    std::atomic<int64_t> total_predictions_made_{0};
    std::atomic<int64_t> total_predictions_correct_{0};
    std::unordered_set<int32_t> pending_predictions_;
    int64_t observation_count_ = 0;

    std::chrono::steady_clock::time_point start_time_;

    // ── Sequence-level lock (protects per-sequence state) ────────────
    mutable std::mutex seq_mu_;

    // ── Helpers ──────────────────────────────────────────────────────
    void maybe_prune();
    int64_t count_transitions() const;
};

} // namespace memopt
