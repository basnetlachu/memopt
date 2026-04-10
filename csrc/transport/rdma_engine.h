// rdma_engine.h — ibverbs QP + CQ management and CQ poll loop.
//
// Fully optional: degrades to TCP if libibverbs not found at build time
// or IB devices not found at runtime.
//
// QP exchange backends (selected via MEMOPT_QP_EXCHANGE env var):
//   "tcp"  — TCP sideband (default, no dependencies)
//   "file" — atomic file exchange (single-machine testing)
//   "etcd" — etcd v3 REST API (production clusters)
#pragma once

#include "protocol.h"
#include "ring_buffer.h"

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

#ifdef MEMOPT_RDMA_AVAILABLE
#include <infiniband/verbs.h>
#endif

namespace memopt {
namespace transport {

// ═══════════════════════════════════════════════════════════════════════════
// QPInfo — exchanged between nodes during connection setup
// ═══════════════════════════════════════════════════════════════════════════
// QPInfo layout (48 bytes, naturally aligned):
//   qpn(4) + lid(2) + _pad1(2) + gid(16) + psn(4) + rkey(4) + remote_addr(8) + _pad2(8) = 48
struct QPInfo {
    uint32_t qpn;            // Queue Pair Number
    uint16_t lid;            // Local ID (0 for RoCE)
    uint16_t _pad1;
    uint8_t  gid[16];        // Global ID (IPv6 format)
    uint32_t psn;            // Packet Sequence Number
    uint32_t rkey;           // Remote key for RDMA access
    uint64_t remote_addr;    // Base address of remote buffer
};
// 4 + 2 + 2 + 16 + 4 + 4 + 8 = 40 bytes
static_assert(sizeof(QPInfo) == 40, "QPInfo must be 40 bytes");

// ═══════════════════════════════════════════════════════════════════════════
// QP Exchange backend interface
// ═══════════════════════════════════════════════════════════════════════════
enum class QPExchangeBackend : uint8_t {
    TCP  = 0,
    FILE = 1,
    ETCD = 2,
};

/// Detect exchange backend from environment.
QPExchangeBackend detect_qp_exchange_backend() noexcept;

/// Exchange QPInfo with a remote node.
/// Returns true if remote QPInfo was received within timeout.
bool exchange_qp_info_tcp(
    const std::string& local_node_id,
    const std::string& remote_node_id,
    const std::string& remote_host,
    uint16_t exchange_port,
    const QPInfo& local_info,
    QPInfo& remote_info,
    int timeout_s = 10
) noexcept;

bool exchange_qp_info_file(
    const std::string& local_node_id,
    const std::string& remote_node_id,
    const std::string& exchange_dir,
    const QPInfo& local_info,
    QPInfo& remote_info,
    int timeout_s = 10
) noexcept;

#ifdef MEMOPT_RDMA_AVAILABLE

struct RDMAConnection {
    std::string node_id;
    ibv_qp*     qp         = nullptr;
    ibv_mr*     local_mr   = nullptr;
    uint32_t    local_psn  = 0;
    uint32_t    remote_qpn = 0;
    uint16_t    remote_lid = 0;
    uint8_t     remote_gid[16] = {};
    uint32_t    remote_psn = 0;
    uint32_t    remote_rkey = 0;
    uint64_t    remote_addr = 0;
    bool        connected  = false;
};

class RDMAEngine {
public:
    explicit RDMAEngine(const std::string& device_name = "");
    ~RDMAEngine();

    bool init() noexcept;
    bool is_available() const noexcept { return available_; }

    /// Full QP handshake: create QP, exchange info, transition to RTS.
    bool connect(const ConnectRequest& req, ConnectAck& ack) noexcept;

    bool post_read(const ReadRequest& req) noexcept;
    bool post_write(const WriteRequest& req) noexcept;
    bool register_memory(const RegisterMemRequest& req,
                         RegisterAck& ack) noexcept;
    int  poll_completions(RingBuffer& response_buf) noexcept;

    std::string detect_best_tls() noexcept;

    /// Local node identifier (set before connect).
    void set_local_node_id(const std::string& id) { local_node_id_ = id; }

    /// Enable QP transition debug logging to stderr.
    void set_log_transitions(bool v) { log_transitions_ = v; }

private:
    bool log_transitions_ = false;
    ibv_context* ctx_  = nullptr;
    ibv_pd*      pd_   = nullptr;
    ibv_cq*      cq_   = nullptr;
    bool         available_ = false;
    std::string  device_name_;
    std::string  local_node_id_;
    uint8_t      local_port_ = 1;

    std::unordered_map<std::string, RDMAConnection> connections_;
    std::mutex conn_mu_;

    // ── QP state transition helpers ──────────────────────────────────
    bool create_qp(RDMAConnection& conn) noexcept;
    bool modify_qp_to_init(RDMAConnection& conn) noexcept;
    bool modify_qp_to_rtr(RDMAConnection& conn,
                           const QPInfo& remote) noexcept;
    bool modify_qp_to_rts(RDMAConnection& conn) noexcept;

    /// Get local GID for address vector construction.
    bool get_local_gid(uint8_t gid[16]) noexcept;
    uint16_t get_local_lid() noexcept;
    uint32_t generate_psn() noexcept;
};

#else // no RDMA

class RDMAEngine {
public:
    explicit RDMAEngine(const std::string& = "") {}
    bool init()          noexcept { return false; }
    bool is_available()  const noexcept { return false; }
    bool connect(const ConnectRequest&, ConnectAck& ack) noexcept {
        safe_strcpy(ack.error_msg, sizeof(ack.error_msg),
                    "RDMA not compiled");
        ack.success = 0;
        return false;
    }
    bool post_read(const ReadRequest&) noexcept { return false; }
    bool post_write(const WriteRequest&) noexcept { return false; }
    bool register_memory(const RegisterMemRequest&,
                         RegisterAck& ack) noexcept {
        ack.success = 0;
        return false;
    }
    int  poll_completions(RingBuffer&) noexcept { return 0; }
    std::string detect_best_tls() noexcept { return "tcp"; }
    void set_local_node_id(const std::string&) {}
    void set_log_transitions(bool) {}
};

#endif

// ═══════════════════════════════════════════════════════════════════════════
// CQ Poller — dedicated thread for sub-µs completion polling
// ═══════════════════════════════════════════════════════════════════════════
class CQPoller {
public:
    void start(RDMAEngine* engine, RingBuffer* resp_buf,
               int cpu_core = -1);
    void stop();

private:
    std::thread       thread_;
    std::atomic<bool> running_{false};
    void run(RDMAEngine* engine, RingBuffer* resp, int cpu_core);
};

} // namespace transport
} // namespace memopt
