// block_pool.h — LIFO free-list block pool for PagedKVCache.
//
// Replaces Python's threading.Lock + list.pop(0) / list.extend()
// in PagedKVCache's block allocation.
//
// Two implementations, selected at compile time:
//
//   x86-64 with cmpxchg16b (-mcx16):
//     True lock-free Treiber stack using 128-bit compare-and-swap.
//     Tagged pointer (ptr + tag) prevents ABA problem.
//     allocate() and free() are lock-free: CAS loop, no mutex.
//
//   All other platforms (ARM64, non-cmpxchg16b x86):
//     Mutex-protected LIFO stack. Correct, not lock-free.
//     The mutex is held for ~3 instructions (read next, swap head).
//
// Pre-allocated node storage — zero heap allocation after construction.
// All methods noexcept — never throws, never allocates on hot path.
// Thread safety: all methods safe for concurrent use from any threads.
// GIL: not needed — pure C++ atomics / mutex.
#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

namespace memopt {
namespace paged {

// ═══════════════════════════════════════════════════════════════════════════
// BlockNode — one per physical block, lives in pre-allocated storage
// ═══════════════════════════════════════════════════════════════════════════
struct BlockNode {
    int32_t    block_id = -1;
    BlockNode* next     = nullptr;
};

// ═══════════════════════════════════════════════════════════════════════════
// Compile-time path selection
// ═══════════════════════════════════════════════════════════════════════════
//
// __GCC_HAVE_SYNC_COMPARE_AND_SWAP_16 is defined by GCC/Clang when the
// target supports 16-byte CAS (cmpxchg16b on x86-64). Requires -mcx16.
// Without it, std::atomic<TaggedPtr>::is_lock_free() returns false and
// the compiler uses a hidden mutex — worse than our explicit one.

#if defined(__x86_64__) && defined(__GCC_HAVE_SYNC_COMPARE_AND_SWAP_16)
#define MEMOPT_BLOCK_POOL_LOCK_FREE 1
#else
#define MEMOPT_BLOCK_POOL_LOCK_FREE 0
#endif

// ═══════════════════════════════════════════════════════════════════════════
// BlockPool
// ═══════════════════════════════════════════════════════════════════════════
class BlockPool {
public:
    /// Construct a pool with num_blocks blocks, all initially free.
    /// Pre-allocates all node storage. No further heap allocation.
    explicit BlockPool(int32_t num_blocks);
    ~BlockPool() = default;

    // Non-copyable, non-movable
    BlockPool(const BlockPool&) = delete;
    BlockPool& operator=(const BlockPool&) = delete;

    /// Pop a free block. Returns block_id >= 0 on success, -1 if exhausted.
    /// noexcept — never throws, never allocates.
    /// x86-64: lock-free CAS loop. Other: mutex-protected.
    int32_t allocate() noexcept;

    /// Push a block back onto the free list.
    /// Caller must have previously allocated this block_id.
    /// noexcept — undefined behavior only if block_id was never allocated.
    void free(int32_t block_id) noexcept;

    /// Number of blocks currently free.
    int32_t num_free_blocks() const noexcept;

    /// Total capacity (fixed at construction).
    int32_t capacity() const noexcept;

    /// True if this build uses lock-free CAS, false if mutex.
    static constexpr bool is_lock_free() noexcept {
        return MEMOPT_BLOCK_POOL_LOCK_FREE != 0;
    }

private:
    // Pre-allocated storage — all nodes live here.
    std::vector<BlockNode> storage_;

    // Free count — maintained atomically for O(1) num_free_blocks().
    std::atomic<int32_t> free_count_{0};
    int32_t capacity_ = 0;

#if MEMOPT_BLOCK_POOL_LOCK_FREE
    // ── x86-64 lock-free path ────────────────────────────────────────
    // Tagged pointer for ABA protection. 16-byte CAS via cmpxchg16b.
    //
    // ABA problem: thread A reads head→X, preempted; thread B pops X,
    // frees X, pushes X; thread A wakes and CAS sees head==X but
    // X->next changed. Tag counter prevents: {X, tag_old} != {X, tag_new}.
    struct alignas(16) TaggedPtr {
        BlockNode* ptr = nullptr;
        uint64_t   tag = 0;
    };
    static_assert(sizeof(TaggedPtr) == 16);

    std::atomic<TaggedPtr> head_{TaggedPtr{nullptr, 0}};
#else
    // ── Mutex path (ARM64, other) ────────────────────────────────────
    // Honest mutex, not a hidden spinlock. Correct, not lock-free.
    std::atomic<BlockNode*> head_{nullptr};
    std::mutex              mutex_;
#endif
};

} // namespace paged
} // namespace memopt
