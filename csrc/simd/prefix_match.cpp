// prefix_match.cpp — Scalar + AVX2 + AVX-512 prefix matching.
//
// Expected throughput (1024-token arrays):
//   Scalar:  ~100M tokens/sec
//   AVX2:    ~400M tokens/sec (4x scalar)
//   AVX-512: ~800M tokens/sec (8x scalar)

#include "prefix_match.h"

// Conditionally include SIMD headers
#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

// For __builtin_cpu_supports on GCC/Clang
#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif

namespace memopt {
namespace simd {

// ═══════════════════════════════════════════════════════════════════════════
// Scalar — all platforms
// ═══════════════════════════════════════════════════════════════════════════

size_t find_lcp_scalar(const int32_t* __restrict__ a,
                       const int32_t* __restrict__ b,
                       size_t len) noexcept {
    size_t i = 0;
    // 4x unroll — helps compiler auto-vectorize on non-SIMD targets
    for (; i + 4 <= len; i += 4) {
        if (a[i]   != b[i])   return i;
        if (a[i+1] != b[i+1]) return i + 1;
        if (a[i+2] != b[i+2]) return i + 2;
        if (a[i+3] != b[i+3]) return i + 3;
    }
    // Scalar tail
    for (; i < len; ++i) {
        if (a[i] != b[i]) return i;
    }
    return len;
}

// ═══════════════════════════════════════════════════════════════════════════
// AVX2 — 8 int32 per iteration
// ═══════════════════════════════════════════════════════════════════════════

#ifdef __AVX2__

size_t find_lcp_avx2(const int32_t* __restrict__ a,
                     const int32_t* __restrict__ b,
                     size_t len) noexcept {
    size_t i = 0;

    for (; i + 8 <= len; i += 8) {
        __m256i va = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(b + i));

        // Compare 8 int32s: result is 0xFFFFFFFF per matching lane
        __m256i eq = _mm256_cmpeq_epi32(va, vb);

        // movemask extracts sign bit of each byte → 32-bit mask
        // Each int32 contributes 4 bits (all 1s if match, all 0s if not)
        int mask = _mm256_movemask_epi8(eq);

        if (mask != -1) {  // -1 = 0xFFFFFFFF = all 32 bits set = all match
            // Find first differing int32
            // ~mask has 1s where bytes differ; find first set bit
            uint32_t neg = ~static_cast<uint32_t>(mask);
            int byte_pos = __builtin_ctz(neg);
            // Each int32 occupies 4 bytes in the mask
            return i + static_cast<size_t>(byte_pos / 4);
        }
    }

    // Handle remainder with scalar
    for (; i < len; ++i) {
        if (a[i] != b[i]) return i;
    }
    return len;
}

#else

size_t find_lcp_avx2(const int32_t* __restrict__ a,
                     const int32_t* __restrict__ b,
                     size_t len) noexcept {
    return find_lcp_scalar(a, b, len);
}

#endif // __AVX2__

// ═══════════════════════════════════════════════════════════════════════════
// AVX-512 — 16 int32 per iteration
// ═══════════════════════════════════════════════════════════════════════════

#ifdef __AVX512F__

size_t find_lcp_avx512(const int32_t* __restrict__ a,
                       const int32_t* __restrict__ b,
                       size_t len) noexcept {
    size_t i = 0;

    for (; i + 16 <= len; i += 16) {
        __m512i va = _mm512_loadu_si512(
            reinterpret_cast<const __m512i*>(a + i));
        __m512i vb = _mm512_loadu_si512(
            reinterpret_cast<const __m512i*>(b + i));

        // Compare 16 int32s: returns 16-bit mask
        // bit k = 1 if a[i+k] == b[i+k]
        __mmask16 eq = _mm512_cmpeq_epi32_mask(va, vb);

        if (eq != 0xFFFF) {
            // Not all 16 match — find first differing position
            uint16_t neq = static_cast<uint16_t>(~eq);
            return i + static_cast<size_t>(__builtin_ctz(neq));
        }
    }

    // Handle remainder with scalar (< 16 tokens)
    for (; i < len; ++i) {
        if (a[i] != b[i]) return i;
    }
    return len;
}

#else

size_t find_lcp_avx512(const int32_t* __restrict__ a,
                       const int32_t* __restrict__ b,
                       size_t len) noexcept {
    // Fall through to AVX2 if available, else scalar
    return find_lcp_avx2(a, b, len);
}

#endif // __AVX512F__

// ═══════════════════════════════════════════════════════════════════════════
// ISA detection
// ═══════════════════════════════════════════════════════════════════════════

ISA detect_best_isa() noexcept {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
    // Use __builtin_cpu_supports on GCC/Clang
    #if defined(__GNUC__) || defined(__clang__)
        #ifdef __AVX512F__
        if (__builtin_cpu_supports("avx512f")) return ISA::AVX512;
        #endif
        #ifdef __AVX2__
        if (__builtin_cpu_supports("avx2")) return ISA::AVX2;
        #endif
    #endif
#endif
    return ISA::SCALAR;
}

// ═══════════════════════════════════════════════════════════════════════════
// Runtime dispatch — cached ISA detection
// ═══════════════════════════════════════════════════════════════════════════

size_t find_lcp(const int32_t* __restrict__ a,
                const int32_t* __restrict__ b,
                size_t len) noexcept {
    // ISA detected once, cached in static variable
    static const ISA isa = detect_best_isa();

    switch (isa) {
        case ISA::AVX512: return find_lcp_avx512(a, b, len);
        case ISA::AVX2:   return find_lcp_avx2(a, b, len);
        default:          return find_lcp_scalar(a, b, len);
    }
}

} // namespace simd
} // namespace memopt
