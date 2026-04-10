// page_table.cpp — Implementation of the sharded page table.
//
// Expected latency per operation:
//   lookup():          ~20ns (shared lock + hash lookup + timestamp write)
//   insert():          ~80ns (unique lock + hash insert + LRU push)
//   lru_candidates(n): ~15ns × n (walk LRU tail, no hash scan)
//   remove_sequence():  O(entries_in_seq × shards_scanned)

#include "page_table.h"

#include <algorithm>
#include <ctime>

namespace memopt {

// ═══════════════════════════════════════════════════════════════════════════
// Construction / Destruction
// ═══════════════════════════════════════════════════════════════════════════

PageTable::PageTable(int /*max_blocks*/) {
    // max_blocks is accepted for API compat but we grow dynamically.
}

PageTable::~PageTable() {
    // Free all owned entries.
    std::lock_guard<std::mutex> lock(pool_mu_);
    for (auto* e : all_entries_) {
        delete e;
    }
    all_entries_.clear();
}

// ═══════════════════════════════════════════════════════════════════════════
// Entry pool management
// ═══════════════════════════════════════════════════════════════════════════

PageTableEntry* PageTable::alloc_entry() {
    auto* e = new PageTableEntry();
    std::lock_guard<std::mutex> lock(pool_mu_);
    all_entries_.push_back(e);
    return e;
}

void PageTable::free_entry(PageTableEntry* e) {
    // Remove from pool tracking. The entry is already removed from the
    // shard's hash map and LRU list before this is called.
    std::lock_guard<std::mutex> lock(pool_mu_);
    auto it = std::find(all_entries_.begin(), all_entries_.end(), e);
    if (it != all_entries_.end()) {
        all_entries_.erase(it);
    }
    delete e;
}

// ═══════════════════════════════════════════════════════════════════════════
// Monotonic clock
// ═══════════════════════════════════════════════════════════════════════════

double PageTable::now_monotonic() noexcept {
    using clock = std::chrono::steady_clock;
    auto now = clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()).count();
    return static_cast<double>(ns) / 1e9;
}

// ═══════════════════════════════════════════════════════════════════════════
// insert
// ═══════════════════════════════════════════════════════════════════════════

PageTableEntry* PageTable::insert(const std::string& sequence_id,
                                   int32_t block_index,
                                   const std::string& tier,
                                   void* handle,
                                   int64_t size_bytes) {
    BlockKey key{sequence_id, block_index};
    auto& sh = shard_for(sequence_id);

    auto* entry = alloc_entry();
    entry->sequence_id = sequence_id;
    entry->block_index = block_index;
    entry->tier = tier;
    entry->handle_ptr = handle;
    entry->size_bytes = size_bytes;
    entry->last_accessed = now_monotonic();
    entry->pin_count = 0;
    entry->lru_prev = nullptr;
    entry->lru_next = nullptr;

    std::unique_lock<std::shared_mutex> lock(sh.mu);

    // If key already exists, remove old entry from LRU and free it.
    auto it = sh.table.find(key);
    if (it != sh.table.end()) {
        auto* old = it->second;
        sh.lru_remove(old);
        it->second = entry;
        lock.unlock();
        free_entry(old);
    } else {
        sh.table[key] = entry;
    }

    sh.lru_push_front(entry);
    return entry;
}

// ═══════════════════════════════════════════════════════════════════════════
// lookup — hot path, shared lock
// ═══════════════════════════════════════════════════════════════════════════

PageTableEntry* PageTable::lookup(const std::string& sequence_id,
                                   int32_t block_index) {
    BlockKey key{sequence_id, block_index};
    auto& sh = shard_for(sequence_id);

    // We need a unique lock because we mutate last_accessed and LRU order.
    // This is the trade-off: timestamp mutation prevents pure shared reads.
    // At 64 shards, contention is still very low.
    std::unique_lock<std::shared_mutex> lock(sh.mu);

    auto it = sh.table.find(key);
    if (it == sh.table.end()) {
        return nullptr;
    }

    auto* entry = it->second;
    entry->last_accessed = now_monotonic();
    sh.lru_move_to_front(entry);
    return entry;
}

// ═══════════════════════════════════════════════════════════════════════════
// update_tier
// ═══════════════════════════════════════════════════════════════════════════

void PageTable::update_tier(const std::string& sequence_id,
                             int32_t block_index,
                             const std::string& new_tier,
                             void* new_handle) {
    BlockKey key{sequence_id, block_index};
    auto& sh = shard_for(sequence_id);

    std::unique_lock<std::shared_mutex> lock(sh.mu);

    auto it = sh.table.find(key);
    if (it == sh.table.end()) {
        throw std::runtime_error("update_tier: entry not found");
    }

    auto* entry = it->second;
    entry->tier = new_tier;
    entry->handle_ptr = new_handle;
    entry->last_accessed = now_monotonic();
    sh.lru_move_to_front(entry);
}

// ═══════════════════════════════════════════════════════════════════════════
// remove
// ═══════════════════════════════════════════════════════════════════════════

PageTableEntry* PageTable::remove(const std::string& sequence_id,
                                   int32_t block_index) {
    BlockKey key{sequence_id, block_index};
    auto& sh = shard_for(sequence_id);

    std::unique_lock<std::shared_mutex> lock(sh.mu);

    auto it = sh.table.find(key);
    if (it == sh.table.end()) {
        return nullptr;
    }

    auto* entry = it->second;
    sh.lru_remove(entry);
    sh.table.erase(it);
    // Note: we do NOT free the entry here — caller (Python) may still read it.
    // The pybind11 layer will manage the lifetime.
    return entry;
}

// ═══════════════════════════════════════════════════════════════════════════
// remove_sequence
// ═══════════════════════════════════════════════════════════════════════════

std::vector<PageTableEntry*> PageTable::remove_sequence(
        const std::string& sequence_id) {
    std::vector<PageTableEntry*> removed;

    // The sequence could have blocks in any shard (since we shard by
    // sequence_id, all blocks for one sequence land in the same shard).
    auto& sh = shard_for(sequence_id);
    std::unique_lock<std::shared_mutex> lock(sh.mu);

    auto it = sh.table.begin();
    while (it != sh.table.end()) {
        if (it->first.sequence_id == sequence_id) {
            auto* entry = it->second;
            sh.lru_remove(entry);
            removed.push_back(entry);
            it = sh.table.erase(it);
        } else {
            ++it;
        }
    }

    return removed;
}

// ═══════════════════════════════════════════════════════════════════════════
// pin / unpin
// ═══════════════════════════════════════════════════════════════════════════

void PageTable::pin(const std::string& sequence_id, int32_t block_index) {
    BlockKey key{sequence_id, block_index};
    auto& sh = shard_for(sequence_id);
    std::unique_lock<std::shared_mutex> lock(sh.mu);

    auto it = sh.table.find(key);
    if (it != sh.table.end()) {
        it->second->pin_count++;
    }
}

void PageTable::unpin(const std::string& sequence_id, int32_t block_index) {
    BlockKey key{sequence_id, block_index};
    auto& sh = shard_for(sequence_id);
    std::unique_lock<std::shared_mutex> lock(sh.mu);

    auto it = sh.table.find(key);
    if (it != sh.table.end()) {
        it->second->pin_count--;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// lru_candidates — O(count) walk of LRU tail
// ═══════════════════════════════════════════════════════════════════════════

std::vector<PageTableEntry*> PageTable::lru_candidates(
        const std::string& tier, int count) {
    std::vector<PageTableEntry*> result;
    result.reserve(count);

    // Walk all shards' LRU tails. Since we want globally oldest entries
    // matching a tier, we collect candidates from all shards, then take
    // the oldest `count` across them.
    struct Candidate {
        PageTableEntry* entry;
        double last_accessed;
    };
    std::vector<Candidate> all_candidates;

    for (auto& sh : shards_) {
        std::shared_lock<std::shared_mutex> lock(sh.mu);

        // Walk from tail (LRU = oldest) towards head (MRU = newest).
        auto* node = sh.lru_tail;
        int found = 0;
        while (node && found < count * 2) {  // over-collect per shard
            if (node->tier == tier && node->pin_count == 0) {
                all_candidates.push_back({node, node->last_accessed});
                ++found;
            }
            node = node->lru_prev;
        }
    }

    // Sort by last_accessed ascending (oldest first) and take top `count`.
    std::partial_sort(
        all_candidates.begin(),
        all_candidates.begin() + std::min(
            static_cast<int>(all_candidates.size()), count),
        all_candidates.end(),
        [](const Candidate& a, const Candidate& b) {
            return a.last_accessed < b.last_accessed;
        }
    );

    int n = std::min(static_cast<int>(all_candidates.size()), count);
    for (int i = 0; i < n; ++i) {
        result.push_back(all_candidates[i].entry);
    }

    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// stats
// ═══════════════════════════════════════════════════════════════════════════

PageTable::Stats PageTable::stats() const {
    Stats s{};

    for (const auto& sh : shards_) {
        std::shared_lock<std::shared_mutex> lock(sh.mu);
        for (const auto& [key, entry] : sh.table) {
            s.total_blocks++;
            s.blocks_per_tier[entry->tier]++;
            s.bytes_per_tier[entry->tier] += entry->size_bytes;
        }
    }

    return s;
}

// ═══════════════════════════════════════════════════════════════════════════
// clear
// ═══════════════════════════════════════════════════════════════════════════

void PageTable::clear() {
    for (auto& sh : shards_) {
        std::unique_lock<std::shared_mutex> lock(sh.mu);
        sh.table.clear();
        sh.lru_head = nullptr;
        sh.lru_tail = nullptr;
    }

    std::lock_guard<std::mutex> lock(pool_mu_);
    for (auto* e : all_entries_) {
        delete e;
    }
    all_entries_.clear();
}

} // namespace memopt
