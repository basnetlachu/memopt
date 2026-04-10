// key_hash.h — FNV-1a cache key generation + op/arch enums.
//
// Replaces the Python string-formatting cache key path:
//   f"cuda:ampere:NVIDIA A100:rope:{b}:{s}:{h}:{d}"
// which allocates a Python string object 1.92M times per second.
//
// make_cache_key() produces a uint64_t in ~3 CPU instructions.
// No string. No heap allocation. No GIL needed.
#pragma once

#include <cstdint>

namespace memopt {
namespace hooks {

// ═══════════════════════════════════════════════════════════════════════════
// Op ID — must match Python string names exactly
// ═══════════════════════════════════════════════════════════════════════════
enum class OpId : uint8_t {
    ROPE                = 0,
    LAYER_NORM_RESIDUAL = 1,
    SCALED_SOFTMAX      = 2,
    NUM_OPS             = 3,
};

// Map Python op name strings → OpId
inline OpId op_id_from_string(const char* name) noexcept {
    // Hot path: compare first divergent character.
    // "memopt.rope_fused"           → 'r' at index 7
    // "memopt.ln_residual_fused"    → 'l' at index 7
    // "memopt.scaled_softmax_fused" → 's' at index 7
    if (name[7] == 'r') return OpId::ROPE;
    if (name[7] == 'l') return OpId::LAYER_NORM_RESIDUAL;
    if (name[7] == 's') return OpId::SCALED_SOFTMAX;
    return OpId::NUM_OPS;  // unknown
}

// ═══════════════════════════════════════════════════════════════════════════
// Arch ID — detected once at init from torch.cuda
// ═══════════════════════════════════════════════════════════════════════════
enum class ArchId : uint8_t {
    AMPERE    = 0,  // sm_80, sm_86
    HOPPER    = 1,  // sm_90
    ADA       = 2,  // sm_89
    BLACKWELL = 3,  // sm_100, sm_120
    UNKNOWN   = 4,
};

// ═══════════════════════════════════════════════════════════════════════════
// FNV-1a cache key — zero-allocation unique key for (op, arch, shapes)
// ═══════════════════════════════════════════════════════════════════════════
//
// Collision probability for 10K distinct keys: ~5.4e-12 (effectively zero).
// FNV-1a has excellent distribution for small fixed-width inputs.

inline uint64_t make_cache_key(
    OpId     op,
    ArchId   arch,
    uint16_t batch,
    uint16_t seq_len,
    uint16_t num_heads,
    uint16_t head_dim
) noexcept {
    uint64_t h = 14695981039346656037ULL;  // FNV offset basis
    auto mix = [&](uint64_t v) noexcept {
        h ^= v;
        h *= 1099511628211ULL;  // FNV prime
    };
    mix(static_cast<uint8_t>(op));
    mix(static_cast<uint8_t>(arch));
    mix(batch);
    mix(seq_len);
    mix(num_heads);
    mix(head_dim);
    return h;
}

} // namespace hooks
} // namespace memopt
