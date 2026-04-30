#pragma once
/**
 * memopt VMM Allocator — Milestone 1
 *
 * Decouples CUDA virtual addresses from physical HBM backing.
 * Allows eviction of cold pages to DRAM between decode steps.
 *
 * Page size: 2 MiB (CUDA VMM requirement on Ada/Hopper)
 *
 * Key invariant: eviction ONLY happens at step boundaries
 * (when all GPU streams are quiescent). This avoids the
 * in-flight kernel remapping problem entirely.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MEMOPT_PAGE_SIZE_BYTES (2 * 1024 * 1024ULL)  // 2 MiB
// Capacity is now dynamic, sized at allocator_create from
// pool_size_bytes / MEMOPT_PAGE_SIZE_BYTES, plus 25%
// headroom for sub-2-MiB sub-allocation churn.

// Tier definitions
#define TIER_HBM  0
#define TIER_DRAM 1
#define TIER_NVME 2  // Not in Milestone 1

// Per-page metadata
typedef struct {
    CUdeviceptr  va;          // Virtual address (never changes)
    CUmemGenericAllocationHandle hbm_handle;  // Physical HBM chunk
    void*        dram_mirror; // Pinned host copy when evicted
    size_t       size_bytes;  // Always multiple of MEMOPT_PAGE_SIZE_BYTES
    int          tier;        // Current tier
    uint64_t     last_access; // LRU timestamp
    uint64_t     alloc_seq;   // Monotonic allocation counter
    bool         in_use;      // True if allocated
    char         tag[32];     // Debug label
} MemoptPage;

// Allocator state (opaque)
typedef struct MemoptAllocator MemoptAllocator;

// ============================================================
// Lifecycle
// ============================================================

/**
 * Create allocator.
 * device_idx: CUDA device index (0 for single GPU)
 * pool_size_bytes: Reserve this much VA upfront.
 *   Recommended: 90% of device HBM.
 *   For H100 80GB: 72 * 1024 * 1024 * 1024ULL
 */
MemoptAllocator* memopt_allocator_create(
    int device_idx,
    size_t pool_size_bytes
);

void memopt_allocator_destroy(MemoptAllocator* alloc);

// ============================================================
// Core allocation API
// ============================================================

/**
 * Allocate size_bytes of GPU-accessible memory.
 * Returns a virtual address pointer usable like any CUDA pointer.
 * Rounds up to 2MiB page boundary.
 * Returns NULL on failure.
 */
CUdeviceptr memopt_malloc(
    MemoptAllocator* alloc,
    size_t size_bytes,
    const char* tag
);

/**
 * Free a virtual address previously allocated by memopt_malloc.
 * If the page was evicted to DRAM, frees the DRAM mirror too.
 */
void memopt_free(MemoptAllocator* alloc, CUdeviceptr va);

// ============================================================
// Eviction API — call at step boundaries only
// ============================================================

/**
 * MUST call before any eviction.
 * Synchronizes all GPU streams so no kernel is in flight.
 * Safe to call between decode steps.
 */
void memopt_quiesce(MemoptAllocator* alloc);

/**
 * Evict n_pages LRU pages from HBM to pinned DRAM.
 * MUST call memopt_quiesce() first.
 * Returns number of pages actually evicted.
 */
int memopt_evict_n_pages(
    MemoptAllocator* alloc,
    int n_pages
);

/**
 * Promote a specific VA from DRAM back to HBM.
 * Called automatically when a freed-up VA is accessed
 * after eviction.
 * Returns 0 on success.
 */
int memopt_promote(MemoptAllocator* alloc, CUdeviceptr va);

/**
 * Evict pages until HBM pressure drops below target_fraction.
 * E.g. target_fraction=0.70 means evict until 70% full.
 * Returns bytes freed.
 */
size_t memopt_evict_to_target(
    MemoptAllocator* alloc,
    float target_fraction
);

// ============================================================
// Stats
// ============================================================

typedef struct {
    int      pages_total;
    int      pages_hbm;
    int      pages_dram;
    size_t   bytes_hbm;
    size_t   bytes_dram;
    uint64_t eviction_count;
    uint64_t promotion_count;
    size_t   bytes_evicted_total;
    float    hbm_pressure;  // 0.0 to 1.0
} MemoptStats;

MemoptStats memopt_stats(MemoptAllocator* alloc);

// ============================================================
// Debug
// ============================================================
void memopt_dump_pages(MemoptAllocator* alloc);

#ifdef __cplusplus
}
#endif
