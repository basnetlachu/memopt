// daemon.cpp — Transport sidecar daemon implementation.

#include "daemon.h"
#include "protocol.h"

#include <csignal>
#include <cstdio>
#include <cstring>
#include <thread>

namespace memopt {
namespace transport {

std::atomic<bool> TransportDaemon::shutdown_requested{false};

// ═══════════════════════════════════════════════════════════════════════════
// Signal handler
// ═══════════════════════════════════════════════════════════════════════════

void TransportDaemon::signal_handler(int /*sig*/) {
    shutdown_requested.store(true, std::memory_order_release);
}

// ═══════════════════════════════════════════════════════════════════════════
// Constructor / Destructor
// ═══════════════════════════════════════════════════════════════════════════

TransportDaemon::TransportDaemon(DaemonConfig cfg)
    : cfg_(std::move(cfg)),
      rdma_(cfg_.rdma_device),
      tcp_(cfg_.tcp_port) {}

TransportDaemon::~TransportDaemon() {
    stop();
}

// ═══════════════════════════════════════════════════════════════════════════
// start
// ═══════════════════════════════════════════════════════════════════════════

bool TransportDaemon::start() noexcept {
    // Install signal handlers
    std::signal(SIGTERM, signal_handler);
    std::signal(SIGINT, signal_handler);

    // Create ring buffers in /dev/shm
    std::string prefix = cfg_.shm_path_prefix.empty()
        ? "/memopt_transport" : cfg_.shm_path_prefix;
    std::string req_path = prefix + "_" + cfg_.node_id + "_req";
    std::string resp_path = prefix + "_" + cfg_.node_id + "_resp";

    req_buf_ = RingBuffer::create(req_path);
    resp_buf_ = RingBuffer::create(resp_path);

    if (!req_buf_.is_valid() || !resp_buf_.is_valid()) {
        std::fprintf(stderr, "Failed to create ring buffers at %s\n",
                     req_path.c_str());
        return false;
    }

    // Wire debug logging and node ID into RDMA engine
    rdma_.set_local_node_id(cfg_.node_id);
    rdma_.set_log_transitions(cfg_.log_qp_transitions);

    // Try RDMA init
    if (cfg_.prefer_rdma) {
        if (rdma_.init()) {
            std::fprintf(stderr, "RDMA engine initialized (%s)\n",
                         rdma_.detect_best_tls().c_str());
            // Start CQ poller on dedicated core
            cq_poller_.start(&rdma_, &resp_buf_, cfg_.rdma_cq_cpu);
        } else {
            std::fprintf(stderr, "RDMA unavailable — TCP only\n");
        }
    }

    // Start TCP multiplexer
    if (!tcp_.start()) {
        std::fprintf(stderr, "TCP multiplexer start failed\n");
        return false;
    }

    running_.store(true, std::memory_order_release);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// run — main event loop, blocks until shutdown
// ═══════════════════════════════════════════════════════════════════════════

void TransportDaemon::run() {
    process_requests();
}

// ═══════════════════════════════════════════════════════════════════════════
// stop
// ═══════════════════════════════════════════════════════════════════════════

void TransportDaemon::stop() noexcept {
    running_.store(false, std::memory_order_release);
    cq_poller_.stop();
    tcp_.stop();
}

// ═══════════════════════════════════════════════════════════════════════════
// process_requests — dispatch loop
// ═══════════════════════════════════════════════════════════════════════════

void TransportDaemon::process_requests() {
    uint8_t  payload[RB_SLOT_SIZE];
    uint8_t  msg_type;
    uint32_t size;

    while (running_.load(std::memory_order_acquire) &&
           !shutdown_requested.load(std::memory_order_acquire)) {
        if (req_buf_.try_pop(&msg_type, payload, &size)) {
            dispatch(msg_type, payload, size);
        } else {
            std::this_thread::yield();
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// dispatch
// ═══════════════════════════════════════════════════════════════════════════

void TransportDaemon::dispatch(uint8_t msg_type,
                                const void* payload,
                                uint32_t size) {
    auto mt = static_cast<MessageType>(msg_type);

    switch (mt) {
    case MessageType::PING: {
        // Respond with PONG
        resp_buf_.try_push(
            static_cast<uint8_t>(MessageType::PONG), nullptr, 0);
        break;
    }

    case MessageType::CONNECT: {
        if (size < sizeof(ConnectRequest)) break;
        ConnectRequest req{};
        std::memcpy(&req, payload, sizeof(req));
        ConnectAck ack{};

        // Try RDMA first, fall back to TCP
        bool ok = false;
        if (cfg_.prefer_rdma && rdma_.is_available()) {
            ok = rdma_.connect(req, ack);
            if (ok) ack.used_rdma = 1;
        }
        if (!ok) {
            ok = tcp_.connect(req, ack);
            ack.used_rdma = 0;
        }

        resp_buf_.try_push(
            static_cast<uint8_t>(MessageType::CONNECT_ACK),
            &ack, sizeof(ack));
        break;
    }

    case MessageType::READ: {
        if (size < sizeof(ReadRequest)) break;
        ReadRequest req{};
        std::memcpy(&req, payload, sizeof(req));

        if (rdma_.is_available()) {
            rdma_.post_read(req);
            // Completion via CQ poller → resp_buf_
        } else {
            // TCP path: send read request via TCP
            // (simplified: actual implementation would use
            //  TCP protocol to request data from remote)
            ReadComplete rc{};
            rc.request_id = req.request_id;
            rc.success = 0;  // TCP read path not fully implemented
            resp_buf_.try_push(
                static_cast<uint8_t>(MessageType::READ_COMPLETE),
                &rc, sizeof(rc));
        }
        break;
    }

    case MessageType::WRITE: {
        if (size < sizeof(WriteRequest)) break;
        WriteRequest req{};
        std::memcpy(&req, payload, sizeof(req));

        if (rdma_.is_available()) {
            rdma_.post_write(req);
        } else {
            WriteComplete wc{};
            wc.request_id = req.request_id;
            wc.success = 0;
            resp_buf_.try_push(
                static_cast<uint8_t>(MessageType::WRITE_COMPLETE),
                &wc, sizeof(wc));
        }
        break;
    }

    case MessageType::REGISTER_MEM: {
        if (size < sizeof(RegisterMemRequest)) break;
        RegisterMemRequest req{};
        std::memcpy(&req, payload, sizeof(req));
        RegisterAck ack{};

        rdma_.register_memory(req, ack);

        resp_buf_.try_push(
            static_cast<uint8_t>(MessageType::REGISTER_ACK),
            &ack, sizeof(ack));
        break;
    }

    case MessageType::SHUTDOWN:
        running_.store(false, std::memory_order_release);
        break;

    default:
        // Unknown message — log and ignore
        break;
    }
}

} // namespace transport
} // namespace memopt
