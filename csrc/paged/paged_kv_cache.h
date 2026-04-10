// paged_kv_cache.h — Paged KV cache with C++ block allocation.
//
// Replaces memopt/serving/paged_attention.py's PagedKVCache class.
// The Python class uses threading.Lock for all operations.
// This C++ class uses:
//   - BlockPool (lock-free on x86-64, mutex on other platforms)
//   - shared_mutex for sequence metadata (read-heavy, write-rare)
//   - CUDA gather kernel for zero-copy fetch (optional, CPU fallback)
//
// Thread safety: all public methods safe for concurrent use.
// GIL: managed per-method — released during tensor ops, held for Python API.
#pragma once

#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "block_pool.h"

namespace memopt {
namespace paged {

// ═══════════════════════════════════════════════════════════════════════════
// SequenceState — per-sequence block mapping
// ═══════════════════════════════════════════════════════════════════════════
struct SequenceState {
    std::string          seq_id;
    std::vector<int32_t> block_ids;     // physical blocks assigned
    int32_t              current_pos{0}; // tokens stored so far (high water mark)
    std::string          tenant_id{"_default"};
};

// ═══════════════════════════════════════════════════════════════════════════
// PagedKVCache — the public C++ class exposed to Python via pybind11
// ═══════════════════════════════════════════════════════════════════════════
class PagedKVCache {
public:
    PagedKVCache(int32_t num_blocks,
                 int32_t block_size,
                 int32_t num_heads,
                 int32_t head_dim,
                 int32_t num_layers);

    ~PagedKVCache() = default;

    // ── Sequence lifecycle ───────────────────────────────────────────

    /// Allocate tracking state for a new sequence.
    /// Returns the SequenceState (by pointer for Python binding).
    void allocate_sequence(const std::string& seq_id,
                            const std::string& tenant_id = "_default");

    /// Free all blocks held by a sequence.
    void free_sequence(const std::string& seq_id);

    // ── Store / Fetch — the hot path ─────────────────────────────────

    /// Store one token's KV for one layer.
    /// Allocates new blocks from pool as needed.
    /// Throws std::runtime_error if pool exhausted.
    void store(const std::string& seq_id,
               int32_t layer_idx,
               int32_t token_pos,
               int32_t block_id_for_token,
               int32_t block_offset);

    /// Get the block_id and offset for a given token position.
    /// Used by the Python binding to write tensor data.
    struct BlockLocation {
        int32_t block_id;
        int32_t block_offset;
        bool    new_block_allocated;
    };
    BlockLocation get_or_allocate_block(const std::string& seq_id,
                                         int32_t token_pos);

    /// Update current_pos after a store.
    void update_pos(const std::string& seq_id, int32_t token_pos);

    /// Get all block_ids and current_pos for fetch.
    struct FetchInfo {
        std::vector<int32_t> block_ids;
        int32_t              current_pos;
    };
    FetchInfo get_fetch_info(const std::string& seq_id) const;

    // ── Accessors ────────────────────────────────────────────────────

    int32_t num_free_blocks() const noexcept;
    int32_t total_blocks()    const noexcept;
    int32_t block_size()      const noexcept { return block_size_; }
    int32_t num_heads()       const noexcept { return num_heads_; }
    int32_t head_dim()        const noexcept { return head_dim_; }
    int32_t num_layers()      const noexcept { return num_layers_; }
    int32_t num_sequences()   const;

private:
    BlockPool pool_;
    int32_t   num_blocks_;
    int32_t   block_size_;
    int32_t   num_heads_;
    int32_t   head_dim_;
    int32_t   num_layers_;

    // Sequence metadata — protected by shared_mutex
    std::unordered_map<std::string, SequenceState> sequences_;
    mutable std::shared_mutex seq_mutex_;
};

} // namespace paged
} // namespace memopt
