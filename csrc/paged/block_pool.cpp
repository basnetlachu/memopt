// block_pool.cpp — LIFO block pool implementation.
//
// x86-64 with -mcx16: true lock-free Treiber stack (cmpxchg16b).
// Other platforms:     mutex-protected LIFO stack.
//
// Both paths produce identical results for any input sequence.

#include "block_pool.h"

namespace memopt {
namespace paged {

// ═══════════════════════════════════════════════════════════════════════════
// Constructor — build initial free list (all blocks available)
// ═══════════════════════════════════════════════════════════════════════════

BlockPool::BlockPool(int32_t num_blocks)
    : storage_(num_blocks), capacity_(num_blocks) {
    for (int32_t i = 0; i < num_blocks; ++i) {
        storage_[i].block_id = i;
    }

#if MEMOPT_BLOCK_POOL_LOCK_FREE
    // Build free list by pushing all nodes. Last pushed = first popped (LIFO).
    // Push in reverse so allocate() returns block_id 0 first.
    TaggedPtr h{nullptr, 0};
    for (int32_t i = num_blocks - 1; i >= 0; --i) {
        storage_[i].next = h.ptr;
        h.ptr = &storage_[i];
        h.tag++;
    }
    head_.store(h, std::memory_order_relaxed);
#else
    // Mutex path: build linked list directly.
    BlockNode* h = nullptr;
    for (int32_t i = num_blocks - 1; i >= 0; --i) {
        storage_[i].next = h;
        h = &storage_[i];
    }
    head_.store(h, std::memory_order_relaxed);
#endif

    free_count_.store(num_blocks, std::memory_order_relaxed);
}

// ═══════════════════════════════════════════════════════════════════════════
// allocate
// ═══════════════════════════════════════════════════════════════════════════

#if MEMOPT_BLOCK_POOL_LOCK_FREE

// x86-64: lock-free CAS loop. No mutex. ABA-safe via tagged pointer.
int32_t BlockPool::allocate() noexcept {
    TaggedPtr old_head = head_.load(std::memory_order_acquire);
    while (true) {
        if (!old_head.ptr) return -1;  // pool exhausted

        TaggedPtr new_head{
            old_head.ptr->next,
            old_head.tag + 1
        };

        if (head_.compare_exchange_weak(
                old_head, new_head,
                std::memory_order_release,
                std::memory_order_acquire)) {
            free_count_.fetch_sub(1, std::memory_order_relaxed);
            old_head.ptr->next = nullptr;
            return old_head.ptr->block_id;
        }
        // CAS failed — old_head reloaded by compare_exchange_weak, retry.
    }
}

#else

// Non-x86-64: mutex-protected pop. Correct, not lock-free.
int32_t BlockPool::allocate() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);

    BlockNode* node = head_.load(std::memory_order_relaxed);
    if (!node) return -1;  // pool exhausted

    head_.store(node->next, std::memory_order_relaxed);
    free_count_.fetch_sub(1, std::memory_order_relaxed);
    node->next = nullptr;
    return node->block_id;
}

#endif

// ═══════════════════════════════════════════════════════════════════════════
// free
// ═══════════════════════════════════════════════════════════════════════════

#if MEMOPT_BLOCK_POOL_LOCK_FREE

// x86-64: lock-free CAS loop.
void BlockPool::free(int32_t block_id) noexcept {
    BlockNode* node = &storage_[block_id];

    TaggedPtr old_head = head_.load(std::memory_order_acquire);
    while (true) {
        node->next = old_head.ptr;
        TaggedPtr new_head{node, old_head.tag + 1};

        if (head_.compare_exchange_weak(
                old_head, new_head,
                std::memory_order_release,
                std::memory_order_acquire)) {
            free_count_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        // CAS failed — old_head reloaded, retry.
    }
}

#else

// Non-x86-64: mutex-protected push.
void BlockPool::free(int32_t block_id) noexcept {
    BlockNode* node = &storage_[block_id];

    std::lock_guard<std::mutex> lock(mutex_);
    node->next = head_.load(std::memory_order_relaxed);
    head_.store(node, std::memory_order_relaxed);
    free_count_.fetch_add(1, std::memory_order_relaxed);
}

#endif

// ═══════════════════════════════════════════════════════════════════════════
// Accessors
// ═══════════════════════════════════════════════════════════════════════════

int32_t BlockPool::num_free_blocks() const noexcept {
    return free_count_.load(std::memory_order_relaxed);
}

int32_t BlockPool::capacity() const noexcept {
    return capacity_;
}

} // namespace paged
} // namespace memopt
