// paged_kv_cache.cpp — PagedKVCache implementation.
//
// The C++ layer manages block allocation and sequence metadata.
// Tensor storage (k_blocks, v_blocks) lives in Python as PyTorch tensors.
// The pybind11 bindings coordinate: C++ decides block_id/offset,
// Python writes/reads the tensor at that position.
//
// Expected latency:
//   allocate_sequence(): ~50ns (unique_lock + map insert)
//   get_or_allocate_block(): ~5ns hit (shared_lock), ~15ns miss (unique_lock + pool alloc)
//   free_sequence():     ~20ns × blocks_freed (unique_lock + pool free)

#include "paged_kv_cache.h"

#include <stdexcept>

namespace memopt {
namespace paged {

// ═══════════════════════════════════════════════════════════════════════════
// Constructor
// ═══════════════════════════════════════════════════════════════════════════

PagedKVCache::PagedKVCache(int32_t num_blocks,
                           int32_t block_size,
                           int32_t num_heads,
                           int32_t head_dim,
                           int32_t num_layers)
    : pool_(num_blocks),
      num_blocks_(num_blocks),
      block_size_(block_size),
      num_heads_(num_heads),
      head_dim_(head_dim),
      num_layers_(num_layers) {
}

// ═══════════════════════════════════════════════════════════════════════════
// allocate_sequence
// ═══════════════════════════════════════════════════════════════════════════

void PagedKVCache::allocate_sequence(const std::string& seq_id,
                                      const std::string& tenant_id) {
    std::unique_lock<std::shared_mutex> lock(seq_mutex_);
    if (sequences_.count(seq_id)) return;  // idempotent
    SequenceState state;
    state.seq_id = seq_id;
    state.tenant_id = tenant_id;
    sequences_[seq_id] = std::move(state);
}

// ═══════════════════════════════════════════════════════════════════════════
// free_sequence
// ═══════════════════════════════════════════════════════════════════════════

void PagedKVCache::free_sequence(const std::string& seq_id) {
    std::unique_lock<std::shared_mutex> lock(seq_mutex_);
    auto it = sequences_.find(seq_id);
    if (it == sequences_.end()) return;

    // Return all blocks to pool
    for (int32_t bid : it->second.block_ids) {
        pool_.free(bid);
    }
    sequences_.erase(it);
}

// ═══════════════════════════════════════════════════════════════════════════
// get_or_allocate_block — decide where to store a token
// ═══════════════════════════════════════════════════════════════════════════

PagedKVCache::BlockLocation PagedKVCache::get_or_allocate_block(
        const std::string& seq_id, int32_t token_pos) {
    int32_t block_idx    = token_pos / block_size_;
    int32_t block_offset = token_pos % block_size_;

    // Fast path: block already allocated (shared_lock)
    {
        std::shared_lock<std::shared_mutex> lock(seq_mutex_);
        auto it = sequences_.find(seq_id);
        if (it == sequences_.end()) {
            throw std::runtime_error(
                "get_or_allocate_block: sequence not found: " + seq_id);
        }
        if (block_idx < static_cast<int32_t>(it->second.block_ids.size())) {
            return {it->second.block_ids[block_idx], block_offset, false};
        }
    }

    // Slow path: need to allocate new block(s) (unique_lock)
    std::unique_lock<std::shared_mutex> lock(seq_mutex_);
    auto it = sequences_.find(seq_id);
    if (it == sequences_.end()) {
        throw std::runtime_error(
            "get_or_allocate_block: sequence not found: " + seq_id);
    }

    auto& state = it->second;
    while (static_cast<int32_t>(state.block_ids.size()) <= block_idx) {
        int32_t new_block = pool_.allocate();
        if (new_block < 0) {
            throw std::runtime_error(
                "PagedKVCache: out of blocks. "
                "Increase num_blocks or reduce concurrent sequences.");
        }
        state.block_ids.push_back(new_block);
    }

    return {state.block_ids[block_idx], block_offset, true};
}

// ═══════════════════════════════════════════════════════════════════════════
// update_pos
// ═══════════════════════════════════════════════════════════════════════════

void PagedKVCache::update_pos(const std::string& seq_id,
                               int32_t token_pos) {
    std::unique_lock<std::shared_mutex> lock(seq_mutex_);
    auto it = sequences_.find(seq_id);
    if (it != sequences_.end()) {
        if (token_pos + 1 > it->second.current_pos) {
            it->second.current_pos = token_pos + 1;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// get_fetch_info
// ═══════════════════════════════════════════════════════════════════════════

PagedKVCache::FetchInfo PagedKVCache::get_fetch_info(
        const std::string& seq_id) const {
    std::shared_lock<std::shared_mutex> lock(seq_mutex_);
    auto it = sequences_.find(seq_id);
    if (it == sequences_.end()) {
        return {{}, 0};
    }
    return {it->second.block_ids, it->second.current_pos};
}

// ═══════════════════════════════════════════════════════════════════════════
// Accessors
// ═══════════════════════════════════════════════════════════════════════════

int32_t PagedKVCache::num_free_blocks() const noexcept {
    return pool_.num_free_blocks();
}

int32_t PagedKVCache::total_blocks() const noexcept {
    return num_blocks_;
}

int32_t PagedKVCache::num_sequences() const {
    std::shared_lock<std::shared_mutex> lock(seq_mutex_);
    return static_cast<int32_t>(sequences_.size());
}

} // namespace paged
} // namespace memopt
