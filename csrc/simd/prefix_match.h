// prefix_match.h — AVX-512 / AVX2 / scalar prefix matching.
//
// Finds the length of the longest common prefix between two int32_t
// token arrays. Used by GKD prefix index for partial cache hit detection.
//
// Three implementations, selected at runtime by ISA detection:
//   AVX-512: 16 tokens per instruction, ~400µs for 1M tokens
//   AVX2:    8 tokens per instruction,  ~800µs for 1M tokens
//   Scalar:  4x-unrolled loop,          ~4ms for 1M tokens
//
// All implementations produce identical results for any input.
// noexcept — never throws, never allocates.
// GIL: not needed — pure arithmetic on int32_t arrays.
#pragma once

#include <cstddef>
#include <cstdint>

namespace memopt {
namespace simd {

// ═══════════════════════════════════════════════════════════════════════════
// ISA detection — called once, cached
// ═══════════════════════════════════════════════════════════════════════════
enum class ISA : uint8_t {
    SCALAR = 0,
    AVX2   = 1,
    AVX512 = 2,
};

/// Detect best SIMD ISA available on this CPU.
/// Result is cached after first call.
ISA detect_best_isa() noexcept;

// ═══════════════════════════════════════════════════════════════════════════
// Implementation functions — use find_lcp() for automatic dispatch
// ═══════════════════════════════════════════════════════════════════════════

/// Scalar 4x-unrolled implementation. Works on all platforms.
size_t find_lcp_scalar(const int32_t* __restrict__ a,
                       const int32_t* __restrict__ b,
                       size_t len) noexcept;

/// AVX2 implementation (8 int32 per iteration).
/// Falls back to scalar if AVX2 not compiled in.
size_t find_lcp_avx2(const int32_t* __restrict__ a,
                     const int32_t* __restrict__ b,
                     size_t len) noexcept;

/// AVX-512 implementation (16 int32 per iteration).
/// Falls back to AVX2 → scalar if not compiled in.
size_t find_lcp_avx512(const int32_t* __restrict__ a,
                       const int32_t* __restrict__ b,
                       size_t len) noexcept;

/// Runtime dispatch — uses the best available ISA.
/// This is the primary entry point. Use this everywhere.
size_t find_lcp(const int32_t* __restrict__ a,
                const int32_t* __restrict__ b,
                size_t len) noexcept;

} // namespace simd
} // namespace memopt
