// gather_kernel.cuh — CUDA gather kernel for paged KV cache fetch.
//
// Replaces N torch.cat() calls with a single kernel launch.
// At batch=256, seq=2048, block_size=16: eliminates 32,768 temporary
// GPU allocations per fetch.
//
// Template supports fp16, bf16, fp32 — selected at runtime.
// CPU stub provided for builds without CUDA.
#pragma once

#include <cstdint>

#ifdef MEMOPT_CUDA_AVAILABLE
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

namespace memopt {
namespace paged {

/// Launch the gather kernel to collect KV blocks into a contiguous tensor.
///
/// @param block_storage  Pre-allocated block storage [num_blocks, block_size, num_heads, head_dim]
/// @param block_ids      Per-sequence block ID list [batch_size, max_blocks_per_seq] (GPU tensor, -1 = unused)
/// @param output         Output tensor [batch_size, max_seq_len, num_heads, head_dim]
/// @param batch_size     Number of sequences in this batch
/// @param max_seq_len    Maximum sequence length across batch (= max_blocks_per_seq * block_size)
/// @param num_heads      Number of attention heads
/// @param head_dim       Dimension per head
/// @param block_size     Tokens per block (e.g. 16)
/// @param max_blocks_per_seq Maximum blocks allocated to any single sequence
/// @param stream         CUDA stream for async execution
template<typename T>
void launch_gather_kv_blocks(
    const T*       block_storage,
    const int32_t* block_ids,
    T*             output,
    int32_t        batch_size,
    int32_t        max_seq_len,
    int32_t        num_heads,
    int32_t        head_dim,
    int32_t        block_size,
    int32_t        max_blocks_per_seq,
    void*          stream_ptr  // cudaStream_t cast to void* for portability
);

/// Check whether the gather kernel is available (CUDA compiled in).
bool has_cuda_gather() noexcept;

} // namespace paged
} // namespace memopt
