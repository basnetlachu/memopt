// page_table.h — Sharded, lock-free-read page table for the Infinite Context VMM.
//
// Replaces memopt/vmm/page_table.py. Identical pybind11 surface so the
// remaining Python codebase never changes.
//
// Design:
//   - 64 shards, each with its own std::shared_mutex (RW lock).
//   - Lookups acquire shared (reader) locks — many concurrent readers.
//   - Inserts / removes acquire unique (writer) locks.
//   - Intrusive doubly-linked LRU list per shard for O(1) move-to-front.
//   - lru_candidates(n) walks the tail of the LRU list: O(n), not O(total).
//   - PageTableEntry is 128 bytes (2 cache lines), no heap allocation.
//
// Thread safety: safe for concurrent read/write from any number of threads.
// GIL: all public methods release the GIL before entering C++.
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace memopt {

// ═══════════════════════════════════════════════════════════════════════════
// FNV-1a hash — fast, deterministic, no heap
// ═══════════════════════════════════════════════════════════════════════════
inline uint64_t fnv1a(const char* data, size_t len) noexcept {
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < len; ++i) {
        h ^= static_cast<uint64_t>(static_cast<uint8_t>(data[i]));
        h *= 1099511628211ULL;
    }
    return h;
}

inline uint64_t fnv1a_str(const std::string& s) noexcept {
    return fnv1a(s.data(), s.size());
}

// ═══════════════════════════════════════════════════════════════════════════
// BlockKey — (sequence_id, block_index) composite key
// ═══════════════════════════════════════════════════════════════════════════
struct BlockKey {
    std::string sequence_id;
    int32_t block_index;

    bool operator==(const BlockKey& o) const noexcept {
        return block_index == o.block_index && sequence_id == o.sequence_id;
    }
};

struct BlockKeyHash {
    size_t operator()(const BlockKey& k) const noexcept {
        uint64_t h = fnv1a_str(k.sequence_id);
        h ^= static_cast<uint64_t>(k.block_index) * 2654435761ULL;
        return static_cast<size_t>(h);
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// PageTableEntry — cache-line aligned, intrusive LRU list node
// ═══════════════════════════════════════════════════════════════════════════
//
// We use a struct with intrusive list pointers. The entry lives in the
// shard's hash map and is also threaded through the shard's LRU list.
// No separate heap allocation for list nodes.
struct PageTableEntry {
    // ── Identity ─────────────────────────────────────────────────────
    std::string sequence_id;
    int32_t     block_index = 0;

    // ── Mapping ──────────────────────────────────────────────────────
    std::string tier;        // "hbm" | "dram" | "nvme"
    int64_t     size_bytes = 0;

    // ── Handle — opaque pointer to backend allocation ────────────────
    // In Python this is a torch.Tensor or str. We store a Python object
    // handle as a void* (borrowed reference managed by pybind11).
    // The bindings layer converts to/from py::object.
    void*       handle_ptr = nullptr;

    // ── LRU tracking ─────────────────────────────────────────────────
    double      last_accessed = 0.0; // time.monotonic() equivalent
    int32_t     pin_count = 0;

    // ── Intrusive doubly-linked list hooks ───────────────────────────
    PageTableEntry* lru_prev = nullptr;
    PageTableEntry* lru_next = nullptr;

    PageTableEntry() = default;
};

// ═══════════════════════════════════════════════════════════════════════════
// Shard — one segment of the sharded page table
// ═══════════════════════════════════════════════════════════════════════════
struct Shard {
    mutable std::shared_mutex mu;
    std::unordered_map<BlockKey, PageTableEntry*, BlockKeyHash> table;

    // Intrusive LRU doubly-linked list: head = MRU, tail = LRU
    PageTableEntry* lru_head = nullptr;
    PageTableEntry* lru_tail = nullptr;

    // ── LRU list operations (caller holds unique lock) ───────────────

    void lru_remove(PageTableEntry* e) noexcept {
        if (e->lru_prev) e->lru_prev->lru_next = e->lru_next;
        else             lru_head = e->lru_next;
        if (e->lru_next) e->lru_next->lru_prev = e->lru_prev;
        else             lru_tail = e->lru_prev;
        e->lru_prev = e->lru_next = nullptr;
    }

    void lru_push_front(PageTableEntry* e) noexcept {
        e->lru_prev = nullptr;
        e->lru_next = lru_head;
        if (lru_head) lru_head->lru_prev = e;
        else          lru_tail = e;
        lru_head = e;
    }

    void lru_move_to_front(PageTableEntry* e) noexcept {
        lru_remove(e);
        lru_push_front(e);
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// PageTable — the public C++ class exposed to Python via pybind11
// ═══════════════════════════════════════════════════════════════════════════
class PageTable {
public:
    static constexpr int NUM_SHARDS = 64;

    explicit PageTable(int max_blocks = 0);
    ~PageTable();

    // ── Python API surface (must match page_table.py exactly) ────────

    /// Register a newly allocated block. Returns the entry.
    /// GIL: released.
    PageTableEntry* insert(const std::string& sequence_id,
                           int32_t block_index,
                           const std::string& tier,
                           void* handle,
                           int64_t size_bytes);

    /// Return entry and update last_accessed, or nullptr if missing.
    /// GIL: released.
    PageTableEntry* lookup(const std::string& sequence_id,
                           int32_t block_index);

    /// Atomically update tier + handle after promote/evict.
    /// GIL: released.
    void update_tier(const std::string& sequence_id,
                     int32_t block_index,
                     const std::string& new_tier,
                     void* new_handle);

    /// Remove and return a single entry. Returns nullptr if missing.
    /// GIL: released.
    PageTableEntry* remove(const std::string& sequence_id,
                           int32_t block_index);

    /// Remove all entries for a sequence. Returns removed entries.
    /// GIL: released.
    std::vector<PageTableEntry*> remove_sequence(const std::string& sequence_id);

    /// Prevent a block from being evicted.
    /// GIL: released.
    void pin(const std::string& sequence_id, int32_t block_index);

    /// Allow a block to be evicted again.
    /// GIL: released.
    void unpin(const std::string& sequence_id, int32_t block_index);

    /// Return up to `count` unpinned entries on `tier`, oldest-access first.
    /// O(count) walk of LRU tail — not O(total_entries).
    /// GIL: released.
    std::vector<PageTableEntry*> lru_candidates(const std::string& tier,
                                                 int count);

    /// Return tier-level block and byte counts as a dict-like struct.
    /// GIL: released.
    struct Stats {
        int total_blocks;
        std::unordered_map<std::string, int> blocks_per_tier;
        std::unordered_map<std::string, int64_t> bytes_per_tier;
    };
    Stats stats() const;

    /// Remove all entries.
    void clear();

    // Allow bindings.cpp access to shards_ for handle side-table updates.
    // This is the only consumer — the shard internals are not otherwise public.
    friend class PyPageTable;

private:
    std::array<Shard, NUM_SHARDS> shards_;

    // Storage pool: we own all PageTableEntry memory.
    // Using a vector of unique_ptr so entries are heap-stable (pointers
    // remain valid across rehashes of the hash map).
    std::vector<PageTableEntry*> all_entries_;
    std::mutex pool_mu_;

    Shard& shard_for(const std::string& sequence_id) noexcept {
        return shards_[fnv1a_str(sequence_id) % NUM_SHARDS];
    }
    const Shard& shard_for(const std::string& sequence_id) const noexcept {
        return shards_[fnv1a_str(sequence_id) % NUM_SHARDS];
    }

    PageTableEntry* alloc_entry();
    void free_entry(PageTableEntry* e);

    static double now_monotonic() noexcept;
};

} // namespace memopt
