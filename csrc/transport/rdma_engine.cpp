// rdma_engine.cpp — RDMA engine + QP exchange + CQ poller.
//
// QP exchange backends: TCP sideband, file-based, etcd REST.
// QP state transitions: RESET → INIT → RTR → RTS.

#include "rdma_engine.h"

#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

#ifdef __x86_64__
#include <immintrin.h>
#endif

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <pthread.h>
#include <sched.h>

namespace memopt {
namespace transport {

// ═══════════════════════════════════════════════════════════════════════════
// QP Exchange backend detection
// ═══════════════════════════════════════════════════════════════════════════

QPExchangeBackend detect_qp_exchange_backend() noexcept {
    const char* env = std::getenv("MEMOPT_QP_EXCHANGE");
    if (!env) return QPExchangeBackend::TCP;
    if (std::strcmp(env, "file") == 0) return QPExchangeBackend::FILE;
    if (std::strcmp(env, "etcd") == 0) return QPExchangeBackend::ETCD;
    return QPExchangeBackend::TCP;
}

// ═══════════════════════════════════════════════════════════════════════════
// TCP sideband exchange
// ═══════════════════════════════════════════════════════════════════════════

bool exchange_qp_info_tcp(
    const std::string& local_node_id,
    const std::string& remote_node_id,
    const std::string& remote_host,
    uint16_t exchange_port,
    const QPInfo& local_info,
    QPInfo& remote_info,
    int timeout_s
) noexcept {
    // Role: lower node_id lexicographically is client (connects first)
    bool is_client = (local_node_id < remote_node_id);

    if (is_client) {
        // Client: connect to remote and exchange
        int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) return false;

        struct sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(exchange_port);
        if (::inet_pton(AF_INET, remote_host.c_str(),
                        &addr.sin_addr) <= 0) {
            // Try DNS
            struct addrinfo hints{}, *res;
            hints.ai_family = AF_INET;
            hints.ai_socktype = SOCK_STREAM;
            if (::getaddrinfo(remote_host.c_str(), nullptr,
                              &hints, &res) != 0) {
                ::close(fd);
                return false;
            }
            addr.sin_addr = reinterpret_cast<sockaddr_in*>(
                res->ai_addr)->sin_addr;
            ::freeaddrinfo(res);
        }

        // Retry connect with timeout
        struct timeval tv{timeout_s, 0};
        ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

        auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(timeout_s);
        bool connected = false;
        while (std::chrono::steady_clock::now() < deadline) {
            if (::connect(fd, reinterpret_cast<sockaddr*>(&addr),
                          sizeof(addr)) == 0) {
                connected = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            // Re-create socket for retry (connect on failed socket is UB)
            ::close(fd);
            fd = ::socket(AF_INET, SOCK_STREAM, 0);
            if (fd < 0) return false;
            ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
            ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        }
        if (!connected) { ::close(fd); return false; }

        // Send local QPInfo
        if (::send(fd, &local_info, sizeof(local_info), 0)
            != sizeof(local_info)) {
            ::close(fd);
            return false;
        }

        // Receive remote QPInfo
        ssize_t n = ::recv(fd, &remote_info, sizeof(remote_info), MSG_WAITALL);
        ::close(fd);
        return n == sizeof(remote_info);

    } else {
        // Server: listen, accept one connection, exchange
        int listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd < 0) return false;

        int opt = 1;
        ::setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        struct sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(exchange_port);

        if (::bind(listen_fd, reinterpret_cast<sockaddr*>(&addr),
                   sizeof(addr)) != 0) {
            ::close(listen_fd);
            return false;
        }
        ::listen(listen_fd, 1);

        // Set accept timeout
        struct timeval tv{timeout_s, 0};
        ::setsockopt(listen_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

        struct sockaddr_in peer{};
        socklen_t peer_len = sizeof(peer);
        int fd = ::accept(listen_fd, reinterpret_cast<sockaddr*>(&peer),
                          &peer_len);
        ::close(listen_fd);
        if (fd < 0) return false;

        ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

        // Receive remote QPInfo
        ssize_t n = ::recv(fd, &remote_info, sizeof(remote_info),
                           MSG_WAITALL);
        if (n != sizeof(remote_info)) {
            ::close(fd);
            return false;
        }

        // Send local QPInfo
        if (::send(fd, &local_info, sizeof(local_info), 0)
            != sizeof(local_info)) {
            ::close(fd);
            return false;
        }

        ::close(fd);
        return true;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// File-based exchange (single-machine testing)
// ═══════════════════════════════════════════════════════════════════════════

bool exchange_qp_info_file(
    const std::string& local_node_id,
    const std::string& remote_node_id,
    const std::string& exchange_dir,
    const QPInfo& local_info,
    QPInfo& remote_info,
    int timeout_s
) noexcept {
    // Create directory if needed
    ::mkdir(exchange_dir.c_str(), 0700);

    // Write local info atomically
    std::string local_path = exchange_dir + "/" +
        local_node_id + "_" + remote_node_id + ".qpinfo";
    std::string tmp_path = local_path + ".tmp";

    int fd = ::open(tmp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) return false;
    ::write(fd, &local_info, sizeof(local_info));
#ifdef __APPLE__
    ::fcntl(fd, F_FULLFSYNC);
#else
    ::fdatasync(fd);
#endif
    ::close(fd);
    ::rename(tmp_path.c_str(), local_path.c_str());

    // Poll for remote info
    std::string remote_path = exchange_dir + "/" +
        remote_node_id + "_" + local_node_id + ".qpinfo";

    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::seconds(timeout_s);
    while (std::chrono::steady_clock::now() < deadline) {
        int rfd = ::open(remote_path.c_str(), O_RDONLY);
        if (rfd >= 0) {
            ssize_t n = ::read(rfd, &remote_info, sizeof(remote_info));
            ::close(rfd);
            if (n == sizeof(remote_info)) {
                // Clean up both files
                ::unlink(local_path.c_str());
                ::unlink(remote_path.c_str());
                return true;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Timeout — clean up our file
    ::unlink(local_path.c_str());
    return false;
}

// ═══════════════════════════════════════════════════════════════════════════
// RDMA Engine — full ibverbs implementation
// ═══════════════════════════════════════════════════════════════════════════

#ifdef MEMOPT_RDMA_AVAILABLE

RDMAEngine::RDMAEngine(const std::string& device_name)
    : device_name_(device_name) {}

RDMAEngine::~RDMAEngine() {
    // Destroy QPs before CQ and PD
    for (auto& [id, conn] : connections_) {
        if (conn.local_mr) ibv_dereg_mr(conn.local_mr);
        if (conn.qp) ibv_destroy_qp(conn.qp);
    }
    if (cq_) ibv_destroy_cq(cq_);
    if (pd_) ibv_dealloc_pd(pd_);
    if (ctx_) ibv_close_device(ctx_);
}

bool RDMAEngine::init() noexcept {
    int num_devices = 0;
    ibv_device** dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list || num_devices == 0) {
        available_ = false;
        return false;
    }

    ibv_device* dev = nullptr;
    for (int i = 0; i < num_devices; ++i) {
        if (device_name_.empty() ||
            device_name_ == ibv_get_device_name(dev_list[i])) {
            dev = dev_list[i];
            break;
        }
    }

    if (!dev) {
        ibv_free_device_list(dev_list);
        available_ = false;
        return false;
    }

    ctx_ = ibv_open_device(dev);
    ibv_free_device_list(dev_list);
    if (!ctx_) return false;

    pd_ = ibv_alloc_pd(ctx_);
    if (!pd_) return false;

    cq_ = ibv_create_cq(ctx_, 256, nullptr, nullptr, 0);
    if (!cq_) return false;

    available_ = true;
    return true;
}

// ── QP creation ──────────────────────────────────────────────────────────

bool RDMAEngine::create_qp(RDMAConnection& conn) noexcept {
    ibv_qp_init_attr attr{};
    attr.send_cq = cq_;
    attr.recv_cq = cq_;
    attr.cap.max_send_wr = 16;
    attr.cap.max_recv_wr = 16;
    attr.cap.max_send_sge = 1;
    attr.cap.max_recv_sge = 1;
    attr.qp_type = IBV_QPT_RC;

    conn.qp = ibv_create_qp(pd_, &attr);
    return conn.qp != nullptr;
}

// ── QP state transitions (with structured error messages) ────────────────

bool RDMAEngine::modify_qp_to_init(RDMAConnection& conn) noexcept {
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = local_port_;
    attr.qp_access_flags = IBV_ACCESS_REMOTE_READ |
                           IBV_ACCESS_REMOTE_WRITE |
                           IBV_ACCESS_LOCAL_WRITE;

    int flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    int ret = ibv_modify_qp(conn.qp, &attr, flags);
    if (ret != 0) {
        std::fprintf(stderr,
            "memopt-transport: ibv_modify_qp INIT failed: "
            "ret=%d errno=%d (%s) port=%u qpn=%u\n",
            ret, errno, strerror(errno),
            local_port_, conn.qp->qp_num);
        return false;
    }
    if (log_transitions_) {
        std::fprintf(stderr,
            "memopt-transport: QP[%s] RESET → INIT: ok (qpn=%u)\n",
            conn.node_id.c_str(), conn.qp->qp_num);
    }
    return true;
}

bool RDMAEngine::modify_qp_to_rtr(RDMAConnection& conn,
                                    const QPInfo& remote) noexcept {
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_4096;
    attr.dest_qp_num = remote.qpn;
    attr.rq_psn = remote.psn;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;

    attr.ah_attr.dlid = remote.lid;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = local_port_;

    bool has_gid = false;
    for (int i = 0; i < 16; ++i) {
        if (remote.gid[i] != 0) { has_gid = true; break; }
    }
    if (has_gid) {
        attr.ah_attr.is_global = 1;
        std::memcpy(&attr.ah_attr.grh.dgid, remote.gid, 16);
        attr.ah_attr.grh.sgid_index = 0;
        attr.ah_attr.grh.hop_limit = 64;
        attr.ah_attr.grh.traffic_class = 0;
    }

    int flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
    int ret = ibv_modify_qp(conn.qp, &attr, flags);
    if (ret != 0) {
        std::fprintf(stderr,
            "memopt-transport: ibv_modify_qp RTR failed: "
            "ret=%d errno=%d (%s) "
            "remote_qpn=%u remote_lid=%u remote_psn=%u "
            "has_gid=%d\n",
            ret, errno, strerror(errno),
            remote.qpn, remote.lid, remote.psn, has_gid);
        return false;
    }
    if (log_transitions_) {
        std::fprintf(stderr,
            "memopt-transport: QP[%s] INIT → RTR: ok "
            "(remote_qpn=%u lid=%u psn=%u gid=%s)\n",
            conn.node_id.c_str(),
            remote.qpn, remote.lid, remote.psn,
            has_gid ? "yes" : "no");
    }
    return true;
}

bool RDMAEngine::modify_qp_to_rts(RDMAConnection& conn) noexcept {
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.sq_psn = conn.local_psn;
    attr.max_rd_atomic = 1;

    int flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                IBV_QP_MAX_QP_RD_ATOMIC;
    int ret = ibv_modify_qp(conn.qp, &attr, flags);
    if (ret != 0) {
        std::fprintf(stderr,
            "memopt-transport: ibv_modify_qp RTS failed: "
            "ret=%d errno=%d (%s) local_psn=%u\n",
            ret, errno, strerror(errno), conn.local_psn);
        return false;
    }
    if (log_transitions_) {
        std::fprintf(stderr,
            "memopt-transport: QP[%s] RTR → RTS: ok "
            "(local_psn=%u)\n",
            conn.node_id.c_str(), conn.local_psn);
    }
    return true;
}

// ── Helpers ──────────────────────────────────────────────────────────────

bool RDMAEngine::get_local_gid(uint8_t gid[16]) noexcept {
    union ibv_gid ibv_gid{};
    if (ibv_query_gid(ctx_, local_port_, 0, &ibv_gid) != 0) {
        std::memset(gid, 0, 16);
        return false;
    }
    std::memcpy(gid, ibv_gid.raw, 16);
    return true;
}

uint16_t RDMAEngine::get_local_lid() noexcept {
    ibv_port_attr pattr{};
    if (ibv_query_port(ctx_, local_port_, &pattr) != 0) return 0;
    return pattr.lid;
}

uint32_t RDMAEngine::generate_psn() noexcept {
    return static_cast<uint32_t>(
        std::chrono::steady_clock::now().time_since_epoch().count()) & 0xFFFFFF;
}

// ═══════════════════════════════════════════════════════════════════════════
// connect() — full QP handshake
// ═══════════════════════════════════════════════════════════════════════════

bool RDMAEngine::connect(const ConnectRequest& req,
                          ConnectAck& ack) noexcept {
    safe_strcpy(ack.node_id, sizeof(ack.node_id), req.node_id);
    ack.used_rdma = 0;

    if (!available_) {
        ack.success = 0;
        safe_strcpy(ack.error_msg, sizeof(ack.error_msg),
                    "RDMA not available");
        return false;
    }

    std::string remote_node(req.node_id);
    std::string remote_host(req.host);

    // 1. Create QP
    RDMAConnection conn;
    conn.node_id = remote_node;
    conn.local_psn = generate_psn();

    if (!create_qp(conn)) {
        ack.success = 0;
        safe_strcpy(ack.error_msg, sizeof(ack.error_msg),
                    "QP creation failed");
        return false;
    }

    // 2. Transition to INIT
    if (!modify_qp_to_init(conn)) {
        ibv_destroy_qp(conn.qp);
        ack.success = 0;
        safe_strcpy(ack.error_msg, sizeof(ack.error_msg),
                    "QP INIT transition failed");
        return false;
    }

    // 3. Build local QPInfo
    QPInfo local_info{};
    local_info.qpn = conn.qp->qp_num;
    local_info.lid = get_local_lid();
    get_local_gid(local_info.gid);
    local_info.psn = conn.local_psn;
    // rkey and remote_addr filled after memory registration

    // 4. Exchange QPInfo using selected backend
    QPInfo remote_info{};
    auto backend = detect_qp_exchange_backend();
    bool exchanged = false;

    uint16_t exchange_port = 18517;
    const char* port_env = std::getenv("MEMOPT_QP_EXCHANGE_PORT");
    if (port_env) exchange_port = static_cast<uint16_t>(std::atoi(port_env));

    switch (backend) {
    case QPExchangeBackend::TCP:
        exchanged = exchange_qp_info_tcp(
            local_node_id_, remote_node, remote_host,
            exchange_port, local_info, remote_info);
        break;

    case QPExchangeBackend::FILE: {
        const char* dir = std::getenv("MEMOPT_QP_EXCHANGE_DIR");
        std::string exchange_dir = dir ? dir : "/tmp/memopt_qp_exchange";
        exchanged = exchange_qp_info_file(
            local_node_id_, remote_node,
            exchange_dir, local_info, remote_info);
        break;
    }

    case QPExchangeBackend::ETCD:
        // etcd backend not fully implemented — fall back to TCP.
        // Log explicitly so operators know what happened.
        std::fprintf(stderr,
            "memopt-transport: MEMOPT_QP_EXCHANGE=etcd requested "
            "but etcd backend not fully implemented. "
            "Falling back to TCP sideband exchange. "
            "Set MEMOPT_QP_EXCHANGE=tcp to suppress this warning.\n");
        exchanged = exchange_qp_info_tcp(
            local_node_id_, remote_node, remote_host,
            exchange_port, local_info, remote_info);
        break;
    }

    if (!exchanged) {
        ibv_destroy_qp(conn.qp);
        ack.success = 0;
        char emsg[128];
        std::snprintf(emsg, sizeof(emsg),
            "QP info exchange failed: backend=%d host=%s port=%u",
            static_cast<int>(backend),
            remote_host.c_str(), exchange_port);
        safe_strcpy(ack.error_msg, sizeof(ack.error_msg), emsg);
        return false;
    }

    // 5. Store remote info in connection
    conn.remote_qpn = remote_info.qpn;
    conn.remote_lid = remote_info.lid;
    std::memcpy(conn.remote_gid, remote_info.gid, 16);
    conn.remote_psn = remote_info.psn;
    conn.remote_rkey = remote_info.rkey;
    conn.remote_addr = remote_info.remote_addr;

    // 6. Transition to RTR
    if (!modify_qp_to_rtr(conn, remote_info)) {
        ibv_destroy_qp(conn.qp);
        ack.success = 0;
        char emsg[128];
        std::snprintf(emsg, sizeof(emsg),
            "QP RTR failed: remote_qpn=%u lid=%u "
            "— check ibv_devinfo on both nodes",
            remote_info.qpn, remote_info.lid);
        safe_strcpy(ack.error_msg, sizeof(ack.error_msg), emsg);
        return false;
    }

    // 7. Transition to RTS
    if (!modify_qp_to_rts(conn)) {
        ibv_destroy_qp(conn.qp);
        ack.success = 0;
        char emsg[128];
        std::snprintf(emsg, sizeof(emsg),
            "QP RTS failed: local_psn=%u "
            "— check IB port state on this node",
            conn.local_psn);
        safe_strcpy(ack.error_msg, sizeof(ack.error_msg), emsg);
        return false;
    }

    conn.connected = true;

    // Store in connection map
    {
        std::lock_guard<std::mutex> lock(conn_mu_);
        connections_[remote_node] = std::move(conn);
    }

    ack.success = 1;
    ack.used_rdma = 1;
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// post_read / post_write — ibv_post_send
// ═══════════════════════════════════════════════════════════════════════════

bool RDMAEngine::post_read(const ReadRequest& req) noexcept {
    if (!available_) return false;

    std::lock_guard<std::mutex> lock(conn_mu_);
    auto it = connections_.find(std::string(req.node_id));
    if (it == connections_.end() || !it->second.connected) return false;

    auto& conn = it->second;

    ibv_sge sge{};
    sge.addr = req.local_addr;
    sge.length = req.size;
    // lkey would come from registered memory — caller must register first

    ibv_send_wr wr{};
    wr.wr_id = req.request_id;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = req.remote_addr;
    wr.wr.rdma.rkey = req.rkey;

    ibv_send_wr* bad_wr = nullptr;
    return ibv_post_send(conn.qp, &wr, &bad_wr) == 0;
}

bool RDMAEngine::post_write(const WriteRequest& req) noexcept {
    if (!available_) return false;

    std::lock_guard<std::mutex> lock(conn_mu_);
    auto it = connections_.find(std::string(req.node_id));
    if (it == connections_.end() || !it->second.connected) return false;

    auto& conn = it->second;

    ibv_sge sge{};
    sge.addr = req.local_addr;
    sge.length = req.size;

    ibv_send_wr wr{};
    wr.wr_id = req.request_id;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = req.remote_addr;
    wr.wr.rdma.rkey = req.rkey;

    ibv_send_wr* bad_wr = nullptr;
    return ibv_post_send(conn.qp, &wr, &bad_wr) == 0;
}

bool RDMAEngine::register_memory(const RegisterMemRequest& req,
                                  RegisterAck& ack) noexcept {
    if (!available_) { ack.success = 0; return false; }

    ibv_mr* mr = ibv_reg_mr(pd_,
                             reinterpret_cast<void*>(req.addr),
                             req.size,
                             IBV_ACCESS_LOCAL_WRITE |
                             IBV_ACCESS_REMOTE_READ |
                             IBV_ACCESS_REMOTE_WRITE);
    if (!mr) { ack.success = 0; return false; }

    ack.request_id = req.request_id;
    ack.addr = req.addr;
    ack.lkey = mr->lkey;
    ack.rkey = mr->rkey;
    ack.success = 1;
    return true;
}

int RDMAEngine::poll_completions(RingBuffer& resp) noexcept {
    if (!available_ || !cq_) return 0;

    ibv_wc wc[16];
    int n = ibv_poll_cq(cq_, 16, wc);
    if (n <= 0) return 0;

    for (int i = 0; i < n; ++i) {
        if (wc[i].opcode == IBV_WC_RDMA_READ) {
            ReadComplete rc{};
            rc.request_id = wc[i].wr_id;
            rc.success = (wc[i].status == IBV_WC_SUCCESS) ? 1 : 0;
            rc.bytes_read = wc[i].byte_len;
            resp.try_push(static_cast<uint8_t>(MessageType::READ_COMPLETE),
                          &rc, sizeof(rc));
        } else if (wc[i].opcode == IBV_WC_RDMA_WRITE) {
            WriteComplete wc_resp{};
            wc_resp.request_id = wc[i].wr_id;
            wc_resp.success = (wc[i].status == IBV_WC_SUCCESS) ? 1 : 0;
            wc_resp.bytes_written = wc[i].byte_len;
            resp.try_push(
                static_cast<uint8_t>(MessageType::WRITE_COMPLETE),
                &wc_resp, sizeof(wc_resp));
        }
    }
    return n;
}

std::string RDMAEngine::detect_best_tls() noexcept {
    if (available_) return "rc";
    return "tcp";
}

#endif // MEMOPT_RDMA_AVAILABLE

// ═══════════════════════════════════════════════════════════════════════════
// CQ Poller
// ═══════════════════════════════════════════════════════════════════════════

void CQPoller::start(RDMAEngine* engine, RingBuffer* resp_buf,
                      int cpu_core) {
    if (running_.load()) return;
    running_.store(true, std::memory_order_release);
    thread_ = std::thread(&CQPoller::run, this, engine, resp_buf, cpu_core);
}

void CQPoller::stop() {
    running_.store(false, std::memory_order_release);
    if (thread_.joinable()) thread_.join();
}

void CQPoller::run(RDMAEngine* engine, RingBuffer* resp,
                    int cpu_core) {
#ifdef __linux__
    if (cpu_core >= 0) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_core, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    }
    struct sched_param param{};
    param.sched_priority = 50;
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
#endif

    while (running_.load(std::memory_order_acquire)) {
        engine->poll_completions(*resp);
#ifdef __x86_64__
        _mm_pause();
#else
        std::this_thread::yield();
#endif
    }
}

} // namespace transport
} // namespace memopt
