// tcp_fallback.cpp — TCP multiplexer implementation.
//
// Uses epoll on Linux, kqueue on macOS.
// Wire format: [4-byte big-endian length][payload bytes]
// Compatible with Python TCPTransport framing.

#include "tcp_fallback.h"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#ifdef __linux__
#include <sys/epoll.h>
#elif defined(__APPLE__)
#include <sys/event.h>
#endif

namespace memopt {
namespace transport {

// ═══════════════════════════════════════════════════════════════════════════
// Constructor / Destructor
// ═══════════════════════════════════════════════════════════════════════════

TCPMultiplexer::TCPMultiplexer(uint16_t listen_port)
    : listen_port_(listen_port) {}

TCPMultiplexer::~TCPMultiplexer() {
    stop();
}

// ═══════════════════════════════════════════════════════════════════════════
// Platform multiplexer creation
// ═══════════════════════════════════════════════════════════════════════════

int TCPMultiplexer::create_multiplexer() noexcept {
#ifdef __linux__
    return ::epoll_create1(0);
#elif defined(__APPLE__)
    return ::kqueue();
#else
    return -1;
#endif
}

void TCPMultiplexer::add_fd_to_mux(int fd) noexcept {
#ifdef __linux__
    epoll_event ev{};
    ev.events = EPOLLIN;
    ev.data.fd = fd;
    ::epoll_ctl(mux_fd_, EPOLL_CTL_ADD, fd, &ev);
#elif defined(__APPLE__)
    struct kevent ev;
    EV_SET(&ev, fd, EVFILT_READ, EV_ADD | EV_ENABLE, 0, 0, nullptr);
    ::kevent(mux_fd_, &ev, 1, nullptr, 0, nullptr);
#endif
}

// ═══════════════════════════════════════════════════════════════════════════
// start / stop
// ═══════════════════════════════════════════════════════════════════════════

bool TCPMultiplexer::start() noexcept {
    if (running_.load()) return true;

    // Create listen socket
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) return false;

    int opt = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(listen_port_);

    if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr),
               sizeof(addr)) != 0) {
        ::close(listen_fd_);
        listen_fd_ = -1;
        return false;
    }

    if (::listen(listen_fd_, 128) != 0) {
        ::close(listen_fd_);
        listen_fd_ = -1;
        return false;
    }

    // Set non-blocking
    int flags = ::fcntl(listen_fd_, F_GETFL, 0);
    ::fcntl(listen_fd_, F_SETFL, flags | O_NONBLOCK);

    // Create multiplexer
    mux_fd_ = create_multiplexer();
    if (mux_fd_ < 0) {
        ::close(listen_fd_);
        listen_fd_ = -1;
        return false;
    }

    add_fd_to_mux(listen_fd_);

    running_.store(true, std::memory_order_release);
    event_thread_ = std::thread(&TCPMultiplexer::event_loop, this);
    return true;
}

void TCPMultiplexer::stop() noexcept {
    running_.store(false, std::memory_order_release);
    if (event_thread_.joinable()) event_thread_.join();

    std::lock_guard<std::mutex> lock(conn_mu_);
    for (auto& [id, conn] : connections_) {
        if (conn.fd >= 0) ::close(conn.fd);
    }
    connections_.clear();
    fd_to_node_.clear();

    if (listen_fd_ >= 0) { ::close(listen_fd_); listen_fd_ = -1; }
    if (mux_fd_ >= 0)    { ::close(mux_fd_); mux_fd_ = -1; }
}

// ═══════════════════════════════════════════════════════════════════════════
// event_loop
// ═══════════════════════════════════════════════════════════════════════════

void TCPMultiplexer::event_loop() {
    while (running_.load(std::memory_order_acquire)) {
#ifdef __linux__
        epoll_event events[64];
        int n = ::epoll_wait(mux_fd_, events, 64, 10);  // 10ms timeout
        for (int i = 0; i < n; ++i) {
            if (events[i].data.fd == listen_fd_)
                handle_accept();
            else if (events[i].events & EPOLLIN)
                handle_readable(events[i].data.fd);
        }
#elif defined(__APPLE__)
        struct kevent events[64];
        struct timespec ts{0, 10000000};  // 10ms
        int n = ::kevent(mux_fd_, nullptr, 0, events, 64, &ts);
        for (int i = 0; i < n; ++i) {
            int fd = static_cast<int>(events[i].ident);
            if (fd == listen_fd_)
                handle_accept();
            else if (events[i].filter == EVFILT_READ)
                handle_readable(fd);
        }
#endif
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// handle_accept / handle_readable
// ═══════════════════════════════════════════════════════════════════════════

void TCPMultiplexer::handle_accept() {
    struct sockaddr_in addr{};
    socklen_t len = sizeof(addr);
    int fd = ::accept(listen_fd_, reinterpret_cast<sockaddr*>(&addr), &len);
    if (fd < 0) return;

    // Set non-blocking
    int flags = ::fcntl(fd, F_GETFL, 0);
    ::fcntl(fd, F_SETFL, flags | O_NONBLOCK);

    // Disable Nagle's algorithm
    int opt = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

    add_fd_to_mux(fd);

    // Assign temporary node_id from remote address
    char ip[INET_ADDRSTRLEN];
    ::inet_ntop(AF_INET, &addr.sin_addr, ip, sizeof(ip));
    std::string temp_id = std::string(ip) + ":" +
                          std::to_string(ntohs(addr.sin_port));

    std::lock_guard<std::mutex> lock(conn_mu_);
    TCPConnection conn;
    conn.fd = fd;
    conn.node_id = temp_id;
    conn.connected = true;
    connections_[temp_id] = std::move(conn);
    fd_to_node_[fd] = temp_id;
}

void TCPMultiplexer::handle_readable(int fd) {
    char buf[65536];
    ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
    if (n <= 0) {
        // Connection closed or error
        ::close(fd);
        std::lock_guard<std::mutex> lock(conn_mu_);
        auto it = fd_to_node_.find(fd);
        if (it != fd_to_node_.end()) {
            connections_.erase(it->second);
            fd_to_node_.erase(it);
        }
        return;
    }
    bytes_recv_.fetch_add(static_cast<uint64_t>(n),
                          std::memory_order_relaxed);
}

// ═══════════════════════════════════════════════════════════════════════════
// connect — outgoing connection to remote node
// ═══════════════════════════════════════════════════════════════════════════

bool TCPMultiplexer::connect(const ConnectRequest& req,
                              ConnectAck& ack) noexcept {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        safe_strcpy(ack.error_msg, sizeof(ack.error_msg), "socket() failed");
        ack.success = 0;
        return false;
    }

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(req.port);

    if (::inet_pton(AF_INET, req.host, &addr.sin_addr) <= 0) {
        // Try DNS resolution
        struct addrinfo hints{}, *result;
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        if (::getaddrinfo(req.host, nullptr, &hints, &result) != 0) {
            ::close(fd);
            safe_strcpy(ack.error_msg, sizeof(ack.error_msg),
                        "DNS resolution failed");
            ack.success = 0;
            return false;
        }
        addr.sin_addr = reinterpret_cast<sockaddr_in*>(
            result->ai_addr)->sin_addr;
        ::freeaddrinfo(result);
    }

    // Set connect timeout via SO_SNDTIMEO
    struct timeval tv{2, 0};  // 2 seconds
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr),
                  sizeof(addr)) != 0) {
        ::close(fd);
        safe_strcpy(ack.error_msg, sizeof(ack.error_msg),
                    "connect() failed");
        ack.success = 0;
        return false;
    }

    // Disable Nagle
    int opt = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

    std::string node_id(req.node_id);

    std::lock_guard<std::mutex> lock(conn_mu_);
    TCPConnection conn;
    conn.fd = fd;
    conn.node_id = node_id;
    conn.connected = true;
    connections_[node_id] = std::move(conn);
    fd_to_node_[fd] = node_id;

    safe_strcpy(ack.node_id, sizeof(ack.node_id), req.node_id);
    ack.success = 1;
    ack.used_rdma = 0;
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// send_data
// ═══════════════════════════════════════════════════════════════════════════

bool TCPMultiplexer::send_data(const std::string& node_id,
                                const void* data, uint32_t size) noexcept {
    std::lock_guard<std::mutex> lock(conn_mu_);
    auto it = connections_.find(node_id);
    if (it == connections_.end() || it->second.fd < 0) return false;

    int fd = it->second.fd;

    // Length-prefixed frame: 4-byte big-endian + payload
    uint32_t net_len = htonl(size);
    if (::send(fd, &net_len, 4, MSG_NOSIGNAL) != 4) return false;

    const auto* ptr = static_cast<const char*>(data);
    uint32_t remaining = size;
    while (remaining > 0) {
        ssize_t n = ::send(fd, ptr, remaining, MSG_NOSIGNAL);
        if (n <= 0) return false;
        ptr += n;
        remaining -= static_cast<uint32_t>(n);
    }

    bytes_sent_.fetch_add(size + 4, std::memory_order_relaxed);
    return true;
}

} // namespace transport
} // namespace memopt
