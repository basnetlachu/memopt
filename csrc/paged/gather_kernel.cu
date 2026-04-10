// gather_kernel.cu — CUDA gather kernel for paged KV cache.
//
// Grid:  (batch_size, num_heads, num_blocks_per_seq)
// Block: (min(head_dim, 256), 1, 1)
//
// Each thread block handles one (batch, head, block) triple.
// Consecutive threads read consecutive head_dim elements → coalesced access.
//
// Memory layout:
//   block_storage: [num_blocks, block_size, num_heads, head_dim] — row-major
//   output:        [batch, seq_len, num_heads, head_dim] — row-major
//   block_ids:     [batch, max_blocks_per_seq] — row-major, -1 = unused

#ifdef MEMOPT_CUDA_AVAILABLE

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "gather_kernel.cuh"

namespace memopt {
namespace paged {

// ═══════════════════════════════════════════════════════════════════════════
// Kernel — one template, three instantiations
// ═══════════════════════════════════════════════════════════════════════════

template<typename T>
__global__ void gather_kv_blocks_kernel(
    const T*       __restrict__ block_storage,
    const int32_t* __restrict__ block_ids,
    T*             __restrict__ output,
    int32_t num_heads,
    int32_t head_dim,
    int32_t block_size,
    int32_t max_blocks_per_seq,
    int32_t max_seq_len
) {
    const int32_t batch_idx    = blockIdx.x;
    const int32_t head_idx     = blockIdx.y;
    const int32_t block_offset = blockIdx.z;  // which block in sequence
    const int32_t dim_idx      = threadIdx.x;

    if (dim_idx >= head_dim) return;

    // Look up physical block ID
    const int32_t block_id = block_ids[
        batch_idx * max_blocks_per_seq + block_offset];

    if (block_id < 0) return;  // padding — sequence shorter than max

    // Copy each token in this block for this (head, dim) element
    for (int32_t tok = 0; tok < block_size; ++tok) {
        const int32_t seq_pos = block_offset * block_size + tok;
        if (seq_pos >= max_seq_len) break;

        // Source: block_storage[block_id, tok, head_idx, dim_idx]
        const int64_t src = static_cast<int64_t>(block_id) * block_size * num_heads * head_dim
                          + tok * num_heads * head_dim
                          + head_idx * head_dim
                          + dim_idx;

        // Dest: output[batch_idx, seq_pos, head_idx, dim_idx]
        const int64_t dst = static_cast<int64_t>(batch_idx) * max_seq_len * num_heads * head_dim
                          + seq_pos * num_heads * head_dim
                          + head_idx * head_dim
                          + dim_idx;

        output[dst] = block_storage[src];
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Launch wrapper
// ═══════════════════════════════════════════════════════════════════════════

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
    void*          stream_ptr
) {
    if (batch_size == 0 || max_seq_len == 0) return;

    const int32_t num_block_offsets = (max_seq_len + block_size - 1) / block_size;

    dim3 grid(batch_size, num_heads, num_block_offsets);
    dim3 block_dim(min(head_dim, 256));

    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    gather_kv_blocks_kernel<T><<<grid, block_dim, 0, stream>>>(
        block_storage, block_ids, output,
        num_heads, head_dim, block_size, max_blocks_per_seq, max_seq_len);
}

// Explicit instantiations
template void launch_gather_kv_blocks<float>(
    const float*, const int32_t*, float*,
    int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, void*);
template void launch_gather_kv_blocks<__half>(
    const __half*, const int32_t*, __half*,
    int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, void*);
template void launch_gather_kv_blocks<__nv_bfloat16>(
    const __nv_bfloat16*, const int32_t*, __nv_bfloat16*,
    int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, void*);

bool has_cuda_gather() noexcept { return true; }

} // namespace paged
} // namespace memopt

#else // !MEMOPT_CUDA_AVAILABLE

// CPU stubs — _memopt_paged builds without CUDA, gather disabled.
#include "gather_kernel.cuh"

namespace memopt {
namespace paged {

template<typename T>
void launch_gather_kv_blocks(
    const T*, const int32_t*, T*,
    int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, void*
) {
    // No-op on CPU builds. Caller falls back to torch.cat().
}

// Explicit instantiations for CPU stub (float only — enough for linking)
template void launch_gather_kv_blocks<float>(
    const float*, const int32_t*, float*,
    int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, void*);

bool has_cuda_gather() noexcept { return false; }

} // namespace paged
} // namespace memopt

#endif // MEMOPT_CUDA_AVAILABLE
