// stream_pool.cpp — CUDA stream pool implementation.
//
// Expected latency:
//   acquire(): ~1ns (atomic fetch_add + array index)
//   sync_all(): depends on in-flight work

#include "stream_pool.h"
#include <cstring>

namespace memopt {
namespace cuda_backend {

// ═══════════════════════════════════════════════════════════════════════════
// Global singleton
// ═══════════════════════════════════════════════════════════════════════════
CUDAStreamPool* g_stream_pool = nullptr;

void init_stream_pool(int device_id) {
    if (g_stream_pool) return;  // idempotent
    g_stream_pool = new CUDAStreamPool(device_id);
}

void shutdown_stream_pool() {
    delete g_stream_pool;
    g_stream_pool = nullptr;
}

// ═══════════════════════════════════════════════════════════════════════════
// Constructor / Destructor
// ═══════════════════════════════════════════════════════════════════════════

CUDAStreamPool::CUDAStreamPool(int device_id) : device_id_(device_id) {
    std::memset(streams_.data(), 0, sizeof(streams_));
    std::memset(use_count_.data(), 0, sizeof(use_count_));

#ifdef MEMOPT_CUDA_AVAILABLE
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        available_ = false;
        return;
    }

    for (int i = 0; i < NUM_STREAMS; ++i) {
        err = cudaStreamCreateWithFlags(&streams_[i],
                                         cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            // Clean up already-created streams
            for (int j = 0; j < i; ++j) {
                cudaStreamDestroy(streams_[j]);
                streams_[j] = nullptr;
            }
            available_ = false;
            return;
        }
    }
    available_ = true;
#else
    available_ = false;
#endif
}

CUDAStreamPool::~CUDAStreamPool() {
#ifdef MEMOPT_CUDA_AVAILABLE
    if (available_) {
        for (auto& s : streams_) {
            if (s) cudaStreamDestroy(s);
        }
    }
#endif
}

// ═══════════════════════════════════════════════════════════════════════════
// acquire — wait-free, ~1ns
// ═══════════════════════════════════════════════════════════════════════════

cudaStream_t CUDAStreamPool::acquire() noexcept {
    if (!available_) return nullptr;
    uint32_t idx = counter_.fetch_add(1, std::memory_order_relaxed)
                   % NUM_STREAMS;
    use_count_[idx]++;
    return streams_[idx];
}

// ═══════════════════════════════════════════════════════════════════════════
// sync_all
// ═══════════════════════════════════════════════════════════════════════════

void CUDAStreamPool::sync_all() noexcept {
#ifdef MEMOPT_CUDA_AVAILABLE
    if (!available_) return;
    for (auto& s : streams_) {
        if (s) cudaStreamSynchronize(s);
    }
#endif
}

} // namespace cuda_backend
} // namespace memopt
