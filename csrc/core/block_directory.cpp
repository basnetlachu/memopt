// block_directory.cpp — Sharded block directory implementation.

#include "block_directory.h"

namespace memopt {

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

double BlockDirectoryCpp::now_unix() noexcept {
    auto now = std::chrono::system_clock::now();
    return std::chrono::duration<double>(
        now.time_since_epoch()).count();
}

bool BlockDirectoryCpp::is_expired(const BlockEntryC& e) const noexcept {
    double ttl = (e.ttl_s > 0) ? e.ttl_s : default_ttl_s_;
    return (now_unix() - e.registered_at) > ttl;
}

// ═══════════════════════════════════════════════════════════════════════════
// Constructor
// ═══════════════════════════════════════════════════════════════════════════

BlockDirectoryCpp::BlockDirectoryCpp(const std::string& node_id,
                                      double default_ttl_s)
    : node_id_(node_id), default_ttl_s_(default_ttl_s) {}

// ═══════════════════════════════════════════════════════════════════════════
// register_block
// ═══════════════════════════════════════════════════════════════════════════

void BlockDirectoryCpp::register_block(const BlockEntryC& entry) noexcept {
    std::string key(entry.content_hash);
    auto& shard = shards_[shard_for(key)];
    std::unique_lock<std::shared_mutex> lock(shard.mu);
    shard.entries[key] = entry;
}

// ═══════════════════════════════════════════════════════════════════════════
// lookup
// ═══════════════════════════════════════════════════════════════════════════

bool BlockDirectoryCpp::lookup(const std::string& content_hash,
                                BlockEntryC* out_entry) noexcept {
    auto& shard = shards_[shard_for(content_hash)];
    std::shared_lock<std::shared_mutex> lock(shard.mu);
    auto it = shard.entries.find(content_hash);
    if (it == shard.entries.end()) return false;
    if (is_expired(it->second)) return false;
    if (out_entry) *out_entry = it->second;
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// acquire_lease / release_lease
// ═══════════════════════════════════════════════════════════════════════════

bool BlockDirectoryCpp::acquire_lease(
        const std::string& content_hash,
        const std::string& /*requesting_node*/) noexcept {
    auto& shard = shards_[shard_for(content_hash)];
    std::unique_lock<std::shared_mutex> lock(shard.mu);
    auto it = shard.entries.find(content_hash);
    if (it == shard.entries.end()) return false;
    if (is_expired(it->second)) return false;
    // Only NVMe blocks are leasable
    if (std::strncmp(it->second.tier, "nvme", 4) != 0) return false;
    it->second.lease_count++;
    return true;
}

bool BlockDirectoryCpp::release_lease(
        const std::string& content_hash,
        const std::string& /*requesting_node*/) noexcept {
    auto& shard = shards_[shard_for(content_hash)];
    std::unique_lock<std::shared_mutex> lock(shard.mu);
    auto it = shard.entries.find(content_hash);
    if (it == shard.entries.end()) return false;
    if (it->second.lease_count > 0)
        it->second.lease_count--;
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// can_evict / deregister
// ═══════════════════════════════════════════════════════════════════════════

bool BlockDirectoryCpp::can_evict(
        const std::string& content_hash) noexcept {
    auto& shard = shards_[shard_for(content_hash)];
    std::shared_lock<std::shared_mutex> lock(shard.mu);
    auto it = shard.entries.find(content_hash);
    if (it == shard.entries.end()) return true;
    return it->second.lease_count == 0;
}

void BlockDirectoryCpp::deregister(
        const std::string& content_hash) noexcept {
    auto& shard = shards_[shard_for(content_hash)];
    std::unique_lock<std::shared_mutex> lock(shard.mu);
    shard.entries.erase(content_hash);
}

// ═══════════════════════════════════════════════════════════════════════════
// size / expired_count
// ═══════════════════════════════════════════════════════════════════════════

size_t BlockDirectoryCpp::size() const noexcept {
    size_t total = 0;
    for (const auto& shard : shards_) {
        std::shared_lock<std::shared_mutex> lock(shard.mu);
        total += shard.entries.size();
    }
    return total;
}

size_t BlockDirectoryCpp::expired_count() noexcept {
    size_t removed = 0;
    for (auto& shard : shards_) {
        std::unique_lock<std::shared_mutex> lock(shard.mu);
        auto it = shard.entries.begin();
        while (it != shard.entries.end()) {
            if (is_expired(it->second)) {
                it = shard.entries.erase(it);
                ++removed;
            } else {
                ++it;
            }
        }
    }
    return removed;
}

} // namespace memopt
