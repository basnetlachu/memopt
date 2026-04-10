// daemon.h — Transport sidecar daemon lifecycle.
#pragma once

#include "rdma_engine.h"
#include "ring_buffer.h"
#include "tcp_fallback.h"

#include <atomic>
#include <string>

namespace memopt {
namespace transport {

struct DaemonConfig {
    std::string node_id;
    std::string shm_path_prefix;        // default: /memopt_transport
    uint16_t    tcp_port          = 18516;
    int         rdma_cq_cpu       = -1;  // -1 = any core
    bool        prefer_rdma       = true;
    std::string rdma_device;             // "" = auto-detect
    int         qp_exchange_timeout_s = 10;  // MEMOPT_QP_TIMEOUT_S
    int         qp_exchange_retries   = 5;   // MEMOPT_QP_RETRIES
    bool        log_qp_transitions    = false; // MEMOPT_QP_DEBUG=1
};

/// Read environment variable overrides into config.
void apply_env_overrides(DaemonConfig& cfg) noexcept;

class TransportDaemon {
public:
    explicit TransportDaemon(DaemonConfig cfg);
    ~TransportDaemon();

    /// Initialize RDMA/TCP, create ring buffers, start threads.
    bool start() noexcept;

    /// Main loop — blocks until SIGTERM or SHUTDOWN message.
    void run();

    /// Graceful shutdown.
    void stop() noexcept;

    /// Global shutdown flag — set by signal handler.
    static std::atomic<bool> shutdown_requested;

private:
    DaemonConfig cfg_;
    RDMAEngine   rdma_;
    TCPMultiplexer tcp_;
    CQPoller     cq_poller_;

    RingBuffer   req_buf_;
    RingBuffer   resp_buf_;

    std::atomic<bool> running_{false};

    void process_requests();
    void dispatch(uint8_t msg_type, const void* payload, uint32_t size);

    static void signal_handler(int sig);
};

} // namespace transport
} // namespace memopt
