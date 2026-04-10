// block_directory.h — Sharded concurrent block directory.
//
// Replaces Python's LocalBlockDirectory (threading.RLock + dict).
// 16 shards with shared_mutex: concurrent reads don't block each other.
// ~40 bytes per entry (fixed-size fields) vs ~200 bytes in Python dicts.
//
// Thread safety: all methods safe for concurrent use.
// GIL: released on all public methods.
#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>

namespace memopt {

// ═══════════════════════════════════════════════════════════════════════════
// BlockEntryC — fixed-size fields, no heap allocation per entry
// ═══════════════════════════════════════════════════════════════════════════
struct BlockEntryC {
    char    content_hash[65] = {};  // SHA-256 hex + null
    char    node_id[64]      = {};
    char    tier[8]          = {};  // "hbm","dram","nvme"
    char    path[256]        = {};
    int64_t size_bytes       = 0;
    double  registered_at    = 0.0; // unix timestamp
    int32_t lease_count      = 0;
    double  ttl_s            = 0.0; // 0 = use default
};

// ═══════════════════════════════════════════════════════════════════════════
// BlockDirectoryCpp — sharded concurrent map
// ═══════════════════════════════════════════════════════════════════════════
class BlockDirectoryCpp {
public:
    explicit BlockDirectoryCpp(const std::string& node_id,
                                double default_ttl_s = 3600.0);

    void register_block(const BlockEntryC& entry) noexcept;

    /// Returns true if found and not expired. Fills out_entry.
    bool lookup(const std::string& content_hash,
                BlockEntryC* out_entry) noexcept;

    bool acquire_lease(const std::string& content_hash,
                       const std::string& requesting_node) noexcept;

    bool release_lease(const std::string& content_hash,
                       const std::string& requesting_node) noexcept;

    bool can_evict(const std::string& content_hash) noexcept;

    void deregister(const std::string& content_hash) noexcept;

    size_t size() const noexcept;

    /// Count and remove expired entries. Returns count removed.
    size_t expired_count() noexcept;

private:
    std::string node_id_;
    double      default_ttl_s_;

    static constexpr int NUM_SHARDS = 16;

    struct Shard {
        std::unordered_map<std::string, BlockEntryC> entries;
        mutable std::shared_mutex mu;
    };
    std::array<Shard, NUM_SHARDS> shards_;

    int shard_for(const std::string& key) const noexcept {
        return static_cast<int>(
            std::hash<std::string>{}(key) % NUM_SHARDS);
    }

    bool is_expired(const BlockEntryC& e) const noexcept;
    static double now_unix() noexcept;
};

} // namespace memopt
