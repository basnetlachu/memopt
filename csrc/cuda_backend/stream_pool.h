// stream_pool.h — CUDA stream pool with atomic round-robin selection.
//
// Replaces Python's thread-per-copy pattern with 8 reusable streams.
// Eliminates ~50µs thread creation overhead per async copy.
//
// Degrades gracefully: on CPU-only machines, acquire() returns nullptr
// and all methods are no-ops. The Python shim detects this and uses
// the original Python backend.
//
// Thread safety: acquire() is wait-free (one atomic increment).
// GIL: not needed — pure C++ atomics.
#pragma once

#include <array>
#include <atomic>
#include <cstdint>

// Forward-declare cudaStream_t for CPU-only builds
#ifdef MEMOPT_CUDA_AVAILABLE
#include <cuda_runtime.h>
#else
using cudaStream_t = void*;
#endif

namespace memopt {
namespace cuda_backend {

class CUDAStreamPool {
public:
    static constexpr int NUM_STREAMS = 8;

    /// Create stream pool on the specified CUDA device.
    /// If CUDA unavailable, constructs in degraded mode (all no-ops).
    explicit CUDAStreamPool(int device_id = 0);
    ~CUDAStreamPool();

    // Non-copyable
    CUDAStreamPool(const CUDAStreamPool&) = delete;
    CUDAStreamPool& operator=(const CUDAStreamPool&) = delete;

    /// Acquire next stream via atomic round-robin.
    /// Returns nullptr if CUDA unavailable.
    /// noexcept, wait-free: one atomic fetch_add.
    cudaStream_t acquire() noexcept;

    /// Synchronize all streams. Called at shutdown.
    void sync_all() noexcept;

    /// True if CUDA streams were successfully created.
    bool is_available() const noexcept { return available_; }

    /// Device ID this pool is bound to.
    int device_id() const noexcept { return device_id_; }

    /// Per-stream use counts for observability.
    const std::array<uint64_t, NUM_STREAMS>& use_counts() const noexcept {
        return use_count_;
    }

private:
    std::atomic<uint32_t> counter_{0};
    std::array<cudaStream_t, NUM_STREAMS> streams_{};
    std::array<uint64_t, NUM_STREAMS> use_count_{};
    bool available_ = false;
    int  device_id_ = 0;
};

/// Module-level singleton pointer. Initialized by init_stream_pool().
extern CUDAStreamPool* g_stream_pool;

/// Initialize the global stream pool. Safe to call multiple times.
void init_stream_pool(int device_id = 0);

/// Tear down the global stream pool.
void shutdown_stream_pool();

} // namespace cuda_backend
} // namespace memopt
