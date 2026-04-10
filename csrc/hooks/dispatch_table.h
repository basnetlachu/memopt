// dispatch_table.h — Lock-free-read kernel dispatch table.
//
// Design:
//   - KernelEntry: cache-line-aligned struct with atomics for lock-free read.
//   - base_entries[3]: one per op, always exist, indexed by OpId (no lookup).
//   - shape_cache: per-shape overrides, protected by shared_mutex (RCU).
//   - Readers (inference path): shared_lock → concurrent, non-blocking.
//   - Writers (synthesis): unique_lock → exclusive but rare (~10/day).
//
// Expected latency:
//   get_kernel():       ~5ns (atomic load or shared_lock + flat_map::find)
//   register_kernel():  ~500ns (unique_lock + flat_map::insert, rare)
//   notify_cpp():       ~1ns  (atomic fetch_add, no lock)
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

#include "key_hash.h"

namespace memopt {
namespace hooks {

// ═══════════════════════════════════════════════════════════════════════════
// Synthesis signaling constants
// ═══════════════════════════════════════════════════════════════════════════
constexpr uint64_t WARM_UP_CALLS = 50;

// ═══════════════════════════════════════════════════════════════════════════
// KernelEntry — one per (op, shape) combination
// ═══════════════════════════════════════════════════════════════════════════
//
// Aligned to cache line. Readers never see partial writes due to atomics.
// Written by synthesis thread (rare), read by inference threads (1.92M/sec).
struct KernelEntry {
    // Python callable for the synthesised kernel (PyObject*)
    // nullptr = no kernel registered (use fallback)
    std::atomic<void*> callable{nullptr};

    // Monotonically increasing version — verify no stale reads after hot-swap
    std::atomic<uint64_t> version{0};

    // Call counter — incremented by notify_cpp() on every token
    std::atomic<uint64_t> call_count{0};

    // Fallback counter — incremented when fused kernel errors or not found
    std::atomic<uint64_t> fallback_count{0};

    // Error counter — incremented when fused kernel throws
    std::atomic<uint64_t> error_count{0};
};

// ═══════════════════════════════════════════════════════════════════════════
// Global state
// ═══════════════════════════════════════════════════════════════════════════

// Base entries: one per op, always exist.
// Indexed by OpId — no hash lookup needed.
extern KernelEntry base_entries[static_cast<int>(OpId::NUM_OPS)];

// Per-shape kernel overrides.
// Written: synthesis thread (rare). Read: inference threads (1.92M/sec).
// Use shared_mutex: readers take shared_lock, writer takes unique_lock.
extern std::unordered_map<uint64_t, KernelEntry*> shape_cache;
extern std::shared_mutex shape_cache_mutex;

// Current architecture — set once at init.
extern ArchId current_arch;

// Synthesis signaling — atomic flags to prevent duplicate synthesis fires
extern std::atomic<bool> synthesis_pending[static_cast<int>(OpId::NUM_OPS)];

// ═══════════════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════════════

/// Initialize the dispatch table. Called once at server startup.
void init_dispatch_table(ArchId arch) noexcept;

/// Register a synthesised kernel for an op (optionally per-shape).
/// GIL must be HELD — stores PyObject*, increments refcount.
/// shape_key == 0 means register as base entry (all shapes for this op).
void register_kernel(uint8_t op_id,
                     uint64_t shape_key,
                     void* callable,
                     uint64_t version) noexcept;

/// Invalidate a registered kernel.
/// GIL must be HELD — decrements refcount.
void invalidate_kernel(uint8_t op_id, uint64_t shape_key) noexcept;

/// Atomic notify — replaces Python AutoOptimizer.notify() hot path.
/// Returns true if synthesis threshold just crossed (fire synthesis).
/// ~1ns: one atomic fetch_add + one branch. No lock. No dict. No string.
bool notify_cpp(uint8_t op_id) noexcept;

/// Get kernel callable for a shape key. Returns nullptr on miss.
/// ~5ns: shared_lock + flat_map::find (concurrent readers non-blocking).
void* get_kernel(uint8_t op_id, uint64_t shape_key) noexcept;

/// Reset all counters and entries. For testing only.
void reset_all_counters() noexcept;

/// Return stats as a structured result (converted to py::dict in bindings).
struct DispatchStats {
    uint64_t call_counts[3];
    uint64_t fallback_counts[3];
    uint64_t error_counts[3];
    uint64_t versions[3];
    int      shape_cache_size;
};
DispatchStats get_dispatch_stats() noexcept;

} // namespace hooks
} // namespace memopt
