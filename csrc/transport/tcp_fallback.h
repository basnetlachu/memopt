// tcp_fallback.h — epoll-based TCP multiplexer for non-RDMA nodes.
//
// Replaces Python's thread-per-connection TCPTransport server.
// Uses epoll on Linux, kqueue on macOS for I/O multiplexing.
// Wire format: 4-byte big-endian length prefix + payload.
// Compatible with existing Python TCPTransport.
#pragma once

#include "protocol.h"
#include "ring_buffer.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace memopt {
namespace transport {

struct TCPConnection {
    int         fd = -1;
    std::string node_id;
    bool        connected = false;
    // Partial recv buffer (TCP is stream-oriented)
    std::vector<uint8_t> recv_buf;
    uint32_t             recv_expected = 0;
};

class TCPMultiplexer {
public:
    explicit TCPMultiplexer(uint16_t listen_port);
    ~TCPMultiplexer();

    bool start() noexcept;
    void stop() noexcept;

    bool connect(const ConnectRequest& req, ConnectAck& ack) noexcept;

    /// Send data to a connected node. Returns true on success.
    bool send_data(const std::string& node_id,
                   const void* data, uint32_t size) noexcept;

    /// Receive data for processing — called from event loop.
    /// Writes completed messages into response ring buffer.
    void process_incoming(RingBuffer& resp_buf);

    /// Stats
    uint64_t bytes_sent() const noexcept {
        return bytes_sent_.load(std::memory_order_relaxed);
    }
    uint64_t bytes_recv() const noexcept {
        return bytes_recv_.load(std::memory_order_relaxed);
    }

private:
    uint16_t listen_port_;
    int      listen_fd_ = -1;

    // Platform-specific I/O multiplexer fd
    int      mux_fd_ = -1;

    std::unordered_map<std::string, TCPConnection> connections_;
    std::unordered_map<int, std::string>           fd_to_node_;
    mutable std::mutex conn_mu_;

    std::thread       event_thread_;
    std::atomic<bool> running_{false};

    std::atomic<uint64_t> bytes_sent_{0};
    std::atomic<uint64_t> bytes_recv_{0};

    void event_loop();
    void handle_accept();
    void handle_readable(int fd);

    // Create platform-appropriate multiplexer
    int  create_multiplexer() noexcept;
    void add_fd_to_mux(int fd) noexcept;
};

} // namespace transport
} // namespace memopt
