// gds.h — cuFile GPUDirect Storage integration (optional).
//
// NVMe → HBM without touching CPU memory. Fully optional:
// if cuFile SDK or GDS driver not available, falls back transparently
// to mmap + cudaMemcpyAsync (nvme_io.h path).
//
// Async behavior of gds_read_async / gds_write_async:
//   CUDA >= 11.8 + GDS driver: truly async via cuFileReadAsync.
//     Returns immediately. Completion signaled through the CUDA stream.
//     The stream parameter is used.
//   CUDA < 11.8 + GDS driver: synchronous via cuFileRead.
//     Blocks the calling CPU thread. The stream parameter is ignored.
//     A one-time warning is logged at startup.
//   GDS unavailable: falls back to mmap + cudaMemcpyAsync.
//     The stream parameter is used for the cudaMemcpyAsync call.
//
// Thread safety: all functions are safe for concurrent use.
// GIL: never held.
#pragma once

#include <cstdint>
#include <string>

// Forward-declare cudaStream_t
#ifdef MEMOPT_CUDA_AVAILABLE
#include <cuda_runtime.h>
#else
using cudaStream_t = void*;
#endif

namespace memopt {
namespace cuda_backend {

/// Runtime check: is GDS driver loaded and cuFile SDK available?
/// Safe to call on any machine. Never throws.
bool gds_is_available() noexcept;

/// NVMe → GPU HBM via GDS.
/// Async when CUDA >= 11.8 and GDS driver loaded.
/// Falls back to synchronous cuFileRead on older CUDA.
/// Falls back to mmap + cudaMemcpyAsync if GDS unavailable.
/// GIL: NOT held.
void gds_read_async(const std::string& path,
                    void* gpu_ptr,
                    size_t size,
                    size_t file_offset,
                    cudaStream_t stream) noexcept;

/// GPU HBM → NVMe via GDS.
/// Async when CUDA >= 11.8 and GDS driver loaded.
/// Falls back to synchronous cuFileWrite on older CUDA.
/// Falls back to cudaMemcpy + write_block_atomic if GDS unavailable.
/// GIL: NOT held.
void gds_write_async(const std::string& path,
                     const void* gpu_ptr,
                     size_t size,
                     size_t file_offset,
                     cudaStream_t stream) noexcept;

} // namespace cuda_backend
} // namespace memopt
