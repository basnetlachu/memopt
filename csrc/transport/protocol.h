// protocol.h — Request/response message types for the transport sidecar.
//
// All structs are POD — trivially copyable, fixed size, 8-byte aligned.
// Safe to memcpy directly into ring buffer slots.
#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>

namespace memopt {
namespace transport {

// ═══════════════════════════════════════════════════════════════════════════
// Message types
// ═══════════════════════════════════════════════════════════════════════════
enum class MessageType : uint8_t {
    // Python → Daemon
    CONNECT        = 1,
    READ           = 2,
    WRITE          = 3,
    REGISTER_MEM   = 4,
    DEREGISTER_MEM = 5,
    CLOSE          = 6,
    SHUTDOWN       = 7,
    PING           = 8,

    // Daemon → Python
    CONNECT_ACK    = 64,
    READ_COMPLETE  = 65,
    WRITE_COMPLETE = 66,
    REGISTER_ACK   = 67,
    ERROR          = 68,
    PONG           = 69,
};

// ═══════════════════════════════════════════════════════════════════════════
// Request structs (Python → Daemon)
// ═══════════════════════════════════════════════════════════════════════════

struct ConnectRequest {
    char     node_id[64];
    char     host[256];
    uint16_t port;
    uint8_t  prefer_rdma;
    uint8_t  _pad[5];
};
static_assert(std::is_trivially_copyable_v<ConnectRequest>);

struct ReadRequest {
    char     node_id[64];
    uint64_t remote_addr;
    uint32_t rkey;
    uint32_t _pad1;
    uint64_t local_addr;
    uint32_t size;
    uint32_t _pad2;
    uint64_t request_id;
};
static_assert(std::is_trivially_copyable_v<ReadRequest>);

struct WriteRequest {
    char     node_id[64];
    uint64_t remote_addr;
    uint32_t rkey;
    uint32_t _pad1;
    uint64_t local_addr;
    uint32_t size;
    uint32_t _pad2;
    uint64_t request_id;
};
static_assert(std::is_trivially_copyable_v<WriteRequest>);

struct RegisterMemRequest {
    uint64_t addr;
    uint64_t size;
    uint64_t request_id;
};
static_assert(std::is_trivially_copyable_v<RegisterMemRequest>);

// ═══════════════════════════════════════════════════════════════════════════
// Response structs (Daemon → Python)
// ═══════════════════════════════════════════════════════════════════════════

struct ConnectAck {
    char    node_id[64];
    uint8_t success;
    uint8_t used_rdma;
    uint8_t _pad[6];
    char    error_msg[128];
};
static_assert(std::is_trivially_copyable_v<ConnectAck>);

struct ReadComplete {
    uint64_t request_id;
    uint8_t  success;
    uint8_t  _pad[3];
    uint32_t bytes_read;
    float    latency_us;
    uint32_t _pad2;
};
static_assert(std::is_trivially_copyable_v<ReadComplete>);

struct WriteComplete {
    uint64_t request_id;
    uint8_t  success;
    uint8_t  _pad[3];
    uint32_t bytes_written;
    float    latency_us;
    uint32_t _pad2;
};
static_assert(std::is_trivially_copyable_v<WriteComplete>);

struct RegisterAck {
    uint64_t request_id;
    uint64_t addr;
    uint32_t lkey;
    uint32_t rkey;
    uint8_t  success;
    uint8_t  _pad[7];
};
static_assert(std::is_trivially_copyable_v<RegisterAck>);

struct ErrorResponse {
    uint64_t request_id;
    uint8_t  error_code;
    uint8_t  _pad[7];
    char     message[128];
};
static_assert(std::is_trivially_copyable_v<ErrorResponse>);

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

inline void safe_strcpy(char* dst, size_t dst_size, const char* src) {
    std::strncpy(dst, src, dst_size - 1);
    dst[dst_size - 1] = '\0';
}

} // namespace transport
} // namespace memopt
