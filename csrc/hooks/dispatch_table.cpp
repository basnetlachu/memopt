// dispatch_table.cpp — Implementation of the lock-free kernel dispatch table.
//
// Expected latency:
//   notify_cpp():       ~1ns  (atomic fetch_add)
//   get_kernel():       ~5ns  (shared_lock + unordered_map::find)
//   register_kernel():  ~500ns (unique_lock, rare — synthesis only)

#include "dispatch_table.h"

#include <cstring>

namespace memopt {
namespace hooks {

// ═══════════════════════════════════════════════════════════════════════════
// Global state definitions
// ═══════════════════════════════════════════════════════════════════════════

KernelEntry base_entries[static_cast<int>(OpId::NUM_OPS)];
std::unordered_map<uint64_t, KernelEntry*> shape_cache;
std::shared_mutex shape_cache_mutex;
ArchId current_arch = ArchId::UNKNOWN;
std::atomic<bool> synthesis_pending[static_cast<int>(OpId::NUM_OPS)] = {
    {false}, {false}, {false}
};

// Shape cache entry storage — owned, allocated once per register
static std::vector<KernelEntry*> shape_entry_pool;
static std::mutex pool_mu;

// ═══════════════════════════════════════════════════════════════════════════
// init_dispatch_table
// ═══════════════════════════════════════════════════════════════════════════

void init_dispatch_table(ArchId arch) noexcept {
    current_arch = arch;
    reset_all_counters();
}

// ═══════════════════════════════════════════════════════════════════════════
// register_kernel
// ═══════════════════════════════════════════════════════════════════════════

void register_kernel(uint8_t op_id,
                     uint64_t shape_key,
                     void* callable,
                     uint64_t version) noexcept {
    if (op_id >= static_cast<uint8_t>(OpId::NUM_OPS)) return;

    if (shape_key == 0) {
        // Register as base entry (all shapes for this op)
        auto& entry = base_entries[op_id];
        // Store old callable for refcount management (done by caller in Python)
        entry.callable.store(callable, std::memory_order_release);
        entry.version.fetch_add(1, std::memory_order_release);
    } else {
        // Per-shape override
        std::unique_lock<std::shared_mutex> lock(shape_cache_mutex);
        auto it = shape_cache.find(shape_key);
        if (it != shape_cache.end()) {
            // Hot-swap existing entry
            it->second->callable.store(callable, std::memory_order_release);
            it->second->version.fetch_add(1, std::memory_order_release);
        } else {
            // Allocate new entry
            auto* entry = new KernelEntry();
            entry->callable.store(callable, std::memory_order_release);
            entry->version.store(version, std::memory_order_release);
            shape_cache[shape_key] = entry;

            std::lock_guard<std::mutex> pool_lock(pool_mu);
            shape_entry_pool.push_back(entry);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// invalidate_kernel
// ═══════════════════════════════════════════════════════════════════════════

void invalidate_kernel(uint8_t op_id, uint64_t shape_key) noexcept {
    if (op_id >= static_cast<uint8_t>(OpId::NUM_OPS)) return;

    if (shape_key == 0) {
        base_entries[op_id].callable.store(nullptr,
                                           std::memory_order_release);
    } else {
        std::unique_lock<std::shared_mutex> lock(shape_cache_mutex);
        auto it = shape_cache.find(shape_key);
        if (it != shape_cache.end()) {
            it->second->callable.store(nullptr,
                                        std::memory_order_release);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// notify_cpp — THE hot path. ~1ns. No lock. No dict. No string.
// ═══════════════════════════════════════════════════════════════════════════

bool notify_cpp(uint8_t op_id) noexcept {
    if (op_id >= static_cast<uint8_t>(OpId::NUM_OPS)) return false;

    uint64_t count = base_entries[op_id].call_count
                     .fetch_add(1, std::memory_order_relaxed);

    if (count == WARM_UP_CALLS) {
        // Synthesis threshold crossed — signal once using CAS
        bool expected = false;
        return synthesis_pending[op_id]
            .compare_exchange_strong(expected, true,
                                     std::memory_order_acq_rel);
    }
    return false;
}

// ═══════════════════════════════════════════════════════════════════════════
// get_kernel — ~5ns. Shared lock allows concurrent readers.
// ═══════════════════════════════════════════════════════════════════════════

void* get_kernel(uint8_t op_id, uint64_t shape_key) noexcept {
    if (op_id >= static_cast<uint8_t>(OpId::NUM_OPS)) return nullptr;

    // Check shape cache first (per-shape override)
    if (shape_key != 0) {
        std::shared_lock<std::shared_mutex> lock(shape_cache_mutex);
        auto it = shape_cache.find(shape_key);
        if (it != shape_cache.end()) {
            void* fn = it->second->callable.load(
                std::memory_order_acquire);
            if (fn) return fn;
        }
    }

    // Fall back to base entry for this op
    return base_entries[op_id].callable.load(std::memory_order_acquire);
}

// ═══════════════════════════════════════════════════════════════════════════
// reset_all_counters — for testing only
// ═══════════════════════════════════════════════════════════════════════════

void reset_all_counters() noexcept {
    for (int i = 0; i < static_cast<int>(OpId::NUM_OPS); ++i) {
        base_entries[i].callable.store(nullptr,
                                        std::memory_order_relaxed);
        base_entries[i].version.store(0, std::memory_order_relaxed);
        base_entries[i].call_count.store(0, std::memory_order_relaxed);
        base_entries[i].fallback_count.store(0,
                                              std::memory_order_relaxed);
        base_entries[i].error_count.store(0, std::memory_order_relaxed);
        synthesis_pending[i].store(false, std::memory_order_relaxed);
    }

    {
        std::unique_lock<std::shared_mutex> lock(shape_cache_mutex);
        shape_cache.clear();
    }
    {
        std::lock_guard<std::mutex> lock(pool_mu);
        for (auto* e : shape_entry_pool) delete e;
        shape_entry_pool.clear();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// get_dispatch_stats
// ═══════════════════════════════════════════════════════════════════════════

DispatchStats get_dispatch_stats() noexcept {
    DispatchStats s{};
    for (int i = 0; i < static_cast<int>(OpId::NUM_OPS); ++i) {
        s.call_counts[i] = base_entries[i].call_count
                           .load(std::memory_order_relaxed);
        s.fallback_counts[i] = base_entries[i].fallback_count
                               .load(std::memory_order_relaxed);
        s.error_counts[i] = base_entries[i].error_count
                            .load(std::memory_order_relaxed);
        s.versions[i] = base_entries[i].version
                        .load(std::memory_order_relaxed);
    }
    {
        std::shared_lock<std::shared_mutex> lock(shape_cache_mutex);
        s.shape_cache_size = static_cast<int>(shape_cache.size());
    }
    return s;
}

} // namespace hooks
} // namespace memopt
