// oracle.cpp — Implementation of the Memory Oracle.
//
// Expected latency per operation:
//   observe():      ~50ns (striped lock + hash insert + counter increment)
//   predict():      ~200ns (partial sort top-k from transition table)
//   warm_from_log(): ~5µs/event (file I/O + JSON parse + observe())

#include "oracle.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

namespace memopt {

// ═══════════════════════════════════════════════════════════════════════════
// Construction
// ═══════════════════════════════════════════════════════════════════════════

MemoryOracle::MemoryOracle(int horizon, int max_transitions,
                           double min_confidence)
    : horizon_(horizon),
      max_transitions_(max_transitions),
      min_confidence_(min_confidence),
      start_time_(std::chrono::steady_clock::now()) {}

// ═══════════════════════════════════════════════════════════════════════════
// observe — record a block access for transition learning
// ═══════════════════════════════════════════════════════════════════════════

void MemoryOracle::observe(const std::string& sequence_id,
                           int32_t block_index,
                           int32_t step) {
    // ── Update per-sequence state (seq_mu_) ──────────────────────────
    {
        std::lock_guard<std::mutex> seq_lock(seq_mu_);

        // Record transition: prev_block → block_index
        auto has_it = has_prev_.find(sequence_id);
        if (has_it != has_prev_.end() && has_it->second) {
            int32_t prev = prev_block_[sequence_id];

            // Acquire stripe lock for this from_block
            std::lock_guard<std::mutex> stripe_lock(stripe_for(prev));
            transitions_[prev][block_index]++;
            observation_count_++;

            // Periodic pruning (same trigger as Python: every 1000 observations)
            if (observation_count_ % 1000 == 0) {
                maybe_prune();
            }
        }

        prev_block_[sequence_id] = block_index;
        has_prev_[sequence_id] = true;

        // Step tracking
        if (step < 0) {
            auto it = sequence_steps_.find(sequence_id);
            step = (it != sequence_steps_.end()) ? it->second + 1 : 1;
        }
        sequence_steps_[sequence_id] = step;

        // Update global recency map (capped at 1000 entries)
        RecencyKey rk{sequence_id, block_index};
        auto rec_it = global_recency_.find(rk);
        if (rec_it != global_recency_.end()) {
            rec_it->second = step;
            // Remove from order and re-add at end
            auto ord_it = std::find_if(recency_order_.begin(),
                                        recency_order_.end(),
                                        [&](const RecencyKey& k) {
                return k == rk;
            });
            if (ord_it != recency_order_.end()) {
                recency_order_.erase(ord_it);
            }
        } else {
            global_recency_[rk] = step;
        }
        recency_order_.push_back(rk);

        // Cap at 1000 entries (LRU eviction from front)
        while (global_recency_.size() > 1000) {
            auto& oldest = recency_order_.front();
            global_recency_.erase(oldest);
            recency_order_.erase(recency_order_.begin());
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// predict — return top-k block predictions
// ═══════════════════════════════════════════════════════════════════════════

std::vector<BlockPrediction> MemoryOracle::predict(
        const std::string& sequence_id,
        int32_t current_block,
        int top_k) {
    try {
        int32_t current_step = 0;
        {
            std::lock_guard<std::mutex> seq_lock(seq_mu_);
            auto it = sequence_steps_.find(sequence_id);
            if (it != sequence_steps_.end()) {
                current_step = it->second;
            }
        }

        // Collect predictions from all sources
        std::unordered_map<int32_t, BlockPrediction> results;

        // ── Source 1: Transition table ────────────────────────────────
        {
            std::lock_guard<std::mutex> stripe_lock(stripe_for(current_block));
            auto it = transitions_.find(current_block);
            if (it != transitions_.end()) {
                const auto& counter = it->second;
                uint32_t total = 0;
                for (const auto& [blk, cnt] : counter) {
                    total += cnt;
                }

                // Collect all, then partial_sort for top_k
                struct Scored {
                    int32_t block;
                    double  conf;
                };
                std::vector<Scored> scored;
                scored.reserve(counter.size());
                for (const auto& [blk, cnt] : counter) {
                    double conf = (total > 0)
                        ? static_cast<double>(cnt) / static_cast<double>(total)
                        : 0.0;
                    scored.push_back({blk, conf});
                }

                int k = std::min(top_k, static_cast<int>(scored.size()));
                if (k > 0) {
                    std::partial_sort(
                        scored.begin(), scored.begin() + k, scored.end(),
                        [](const Scored& a, const Scored& b) {
                            return a.conf > b.conf;
                        });
                }

                for (int i = 0; i < k; ++i) {
                    if (scored[i].conf >= min_confidence_) {
                        results[scored[i].block] = BlockPrediction{
                            sequence_id, scored[i].block, scored[i].conf,
                            current_step, "transition"
                        };
                    }
                }
            }
        }

        // ── Source 2: Sequential heuristic ────────────────────────────
        struct SeqHint { int delta; double conf; };
        SeqHint seq_hints[] = {{1, 0.6}, {2, 0.4}};
        for (const auto& hint : seq_hints) {
            int32_t b = current_block + hint.delta;
            if (results.find(b) == results.end() &&
                hint.conf >= min_confidence_) {
                results[b] = BlockPrediction{
                    sequence_id, b, hint.conf,
                    current_step, "sequential"
                };
            }
        }

        // ── Source 3: Recency ────────────────────────────────────────
        {
            std::lock_guard<std::mutex> seq_lock(seq_mu_);
            // Find recent blocks for this sequence from global recency
            struct RecentEntry {
                int32_t block;
                int32_t step;
            };
            std::vector<RecentEntry> seq_recent;
            for (const auto& [rk, s] : global_recency_) {
                if (rk.sequence_id == sequence_id) {
                    seq_recent.push_back({rk.block_index, s});
                }
            }
            std::sort(seq_recent.begin(), seq_recent.end(),
                      [](const RecentEntry& a, const RecentEntry& b) {
                          return a.step > b.step;
                      });

            int recency_count = 0;
            for (const auto& re : seq_recent) {
                if (recency_count >= 5) break;
                if (results.find(re.block) == results.end() &&
                    0.35 >= min_confidence_) {
                    results[re.block] = BlockPrediction{
                        sequence_id, re.block, 0.35,
                        current_step, "recency"
                    };
                }
                ++recency_count;
            }
        }

        // ── Source 4: Fallback ────────────────────────────────────────
        if (results.empty()) {
            int32_t b = current_block + 1;
            results[b] = BlockPrediction{
                sequence_id, b, 0.3,
                current_step, "fallback"
            };
        }

        // ── Sort by confidence descending, take top_k ────────────────
        std::vector<BlockPrediction> final_results;
        final_results.reserve(results.size());
        for (auto& [blk, pred] : results) {
            final_results.push_back(std::move(pred));
        }

        int k = std::min(top_k, static_cast<int>(final_results.size()));
        if (k > 0 && k < static_cast<int>(final_results.size())) {
            std::partial_sort(
                final_results.begin(), final_results.begin() + k,
                final_results.end(),
                [](const BlockPrediction& a, const BlockPrediction& b) {
                    return a.confidence > b.confidence;
                });
            final_results.resize(k);
        } else {
            std::sort(final_results.begin(), final_results.end(),
                      [](const BlockPrediction& a, const BlockPrediction& b) {
                          return a.confidence > b.confidence;
                      });
        }

        // ── Update stats ─────────────────────────────────────────────
        {
            std::lock_guard<std::mutex> stats_lock(stats_mu_);
            total_predictions_made_.fetch_add(
                static_cast<int64_t>(final_results.size()),
                std::memory_order_relaxed);
            for (const auto& p : final_results) {
                pending_predictions_.insert(p.block_index);
            }
            if (pending_predictions_.size() > 10000) {
                pending_predictions_.clear();
            }
        }

        return final_results;

    } catch (...) {
        // Mirror Python: predict() never raises, returns empty on error.
        return {};
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// record_outcome
// ═══════════════════════════════════════════════════════════════════════════

void MemoryOracle::record_outcome(const std::string& /*sequence_id*/,
                                   int32_t block_index) {
    std::lock_guard<std::mutex> lock(stats_mu_);
    auto it = pending_predictions_.find(block_index);
    if (it != pending_predictions_.end()) {
        total_predictions_correct_.fetch_add(1, std::memory_order_relaxed);
        pending_predictions_.erase(it);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// stats
// ═══════════════════════════════════════════════════════════════════════════

OracleStats MemoryOracle::stats() const {
    int64_t made = total_predictions_made_.load(std::memory_order_relaxed);
    int64_t correct = total_predictions_correct_.load(
        std::memory_order_relaxed);

    double accuracy = (made > 0)
        ? std::round((static_cast<double>(correct) / made) * 10000.0) / 100.0
        : 0.0;

    int64_t trans_count = count_transitions();

    int64_t seq_count = 0;
    {
        std::lock_guard<std::mutex> lock(seq_mu_);
        for (const auto& [k, v] : sequence_steps_) {
            seq_count++;
        }
    }

    auto now = std::chrono::steady_clock::now();
    double uptime = std::chrono::duration<double>(now - start_time_).count();
    uptime = std::round(uptime * 100.0) / 100.0;

    return OracleStats{
        made, correct, accuracy,
        trans_count, seq_count,
        horizon_, uptime
    };
}

// ═══════════════════════════════════════════════════════════════════════════
// reset
// ═══════════════════════════════════════════════════════════════════════════

void MemoryOracle::reset(const std::string& sequence_id) {
    if (sequence_id.empty()) {
        // Reset everything
        // Lock all stripes (overkill but safe for rare reset-all)
        for (auto& stripe : stripes_) {
            stripe.lock();
        }
        transitions_.clear();
        for (auto& stripe : stripes_) {
            stripe.unlock();
        }

        std::lock_guard<std::mutex> seq_lock(seq_mu_);
        prev_block_.clear();
        has_prev_.clear();
        sequence_steps_.clear();
        recency_.clear();
        global_recency_.clear();
        recency_order_.clear();

        std::lock_guard<std::mutex> stats_lock(stats_mu_);
        pending_predictions_.clear();
    } else {
        // Reset single sequence
        std::lock_guard<std::mutex> seq_lock(seq_mu_);

        prev_block_.erase(sequence_id);
        has_prev_.erase(sequence_id);
        sequence_steps_.erase(sequence_id);
        recency_.erase(sequence_id);

        // Remove from global recency
        auto it = global_recency_.begin();
        while (it != global_recency_.end()) {
            if (it->first.sequence_id == sequence_id) {
                it = global_recency_.erase(it);
            } else {
                ++it;
            }
        }
        recency_order_.erase(
            std::remove_if(recency_order_.begin(), recency_order_.end(),
                           [&](const RecencyKey& k) {
                               return k.sequence_id == sequence_id;
                           }),
            recency_order_.end());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// warm_from_log — replay JSONL file
// ═══════════════════════════════════════════════════════════════════════════

// Minimal JSON field extraction — avoids nlohmann overhead.
// Expects lines like:
//   {"sequence_id": "s1", "block_index": 5, "token_position": 42, ...}
static bool extract_string_field(const std::string& line,
                                  const char* field_name,
                                  std::string& out) {
    std::string needle = std::string("\"") + field_name + "\"";
    auto pos = line.find(needle);
    if (pos == std::string::npos) return false;

    // Find the colon after the field name
    pos = line.find(':', pos + needle.size());
    if (pos == std::string::npos) return false;

    // Skip whitespace
    pos = line.find('"', pos + 1);
    if (pos == std::string::npos) return false;

    auto end = line.find('"', pos + 1);
    if (end == std::string::npos) return false;

    out = line.substr(pos + 1, end - pos - 1);
    return true;
}

static bool extract_int_field(const std::string& line,
                               const char* field_name,
                               int32_t& out) {
    std::string needle = std::string("\"") + field_name + "\"";
    auto pos = line.find(needle);
    if (pos == std::string::npos) return false;

    pos = line.find(':', pos + needle.size());
    if (pos == std::string::npos) return false;

    // Skip whitespace
    ++pos;
    while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t')) ++pos;

    // Parse integer
    char* end_ptr = nullptr;
    long val = std::strtol(line.c_str() + pos, &end_ptr, 10);
    if (end_ptr == line.c_str() + pos) return false;

    out = static_cast<int32_t>(val);
    return true;
}

int MemoryOracle::warm_from_log(const std::string& log_path,
                                 int max_events) {
    std::ifstream file(log_path);
    if (!file.is_open()) {
        return 0;  // Match Python: silently return 0 on missing file
    }

    int count = 0;
    std::string line;
    while (std::getline(file, line) && count < max_events) {
        // Skip empty lines
        if (line.empty() || line[0] != '{') continue;

        std::string seq_id;
        int32_t block_idx = 0;
        int32_t token_pos = -1;

        if (!extract_string_field(line, "sequence_id", seq_id)) continue;
        if (!extract_int_field(line, "block_index", block_idx)) continue;
        extract_int_field(line, "token_position", token_pos);

        observe(seq_id, block_idx, token_pos);
        ++count;
    }

    return count;
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

void MemoryOracle::maybe_prune() {
    // Find and erase the transition entry with minimum count.
    // Called under stripe lock and seq_mu_ — we only touch transitions_.
    if (transitions_.empty()) return;

    int32_t min_from = 0;
    int32_t min_to = 0;
    uint32_t min_count = UINT32_MAX;
    bool found = false;

    for (const auto& [from, tos] : transitions_) {
        for (const auto& [to, count] : tos) {
            if (count < min_count) {
                min_count = count;
                min_from = from;
                min_to = to;
                found = true;
            }
        }
    }

    if (found) {
        auto it = transitions_.find(min_from);
        if (it != transitions_.end()) {
            it->second.erase(min_to);
            if (it->second.empty()) {
                transitions_.erase(it);
            }
        }
    }
}

int64_t MemoryOracle::count_transitions() const {
    int64_t total = 0;
    // We don't lock all stripes for stats — this is a best-effort count.
    // The Python version also reads under its single lock but stats()
    // is called infrequently.
    for (const auto& [from, tos] : transitions_) {
        total += static_cast<int64_t>(tos.size());
    }
    return total;
}

} // namespace memopt
