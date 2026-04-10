// ring_buffer.h — Lock-free SPSC ring buffer in shared memory.
//
// Python process (producer) writes requests to /dev/shm/memopt_transport_{node}_req.
// C++ daemon (consumer) reads requests, executes RDMA/TCP, writes responses
// to /dev/shm/memopt_transport_{node}_resp.
//
// SPSC guarantee: one producer, one consumer. No mutex needed.
// Memory ordering: ready=1 store(release) pairs with ready==1 load(acquire).
//
// Thread safety: one producer thread, one consumer thread. Multiple Python
// threads must serialize through a mutex before calling try_push().
#pragma once

#include <atomic>
#include <cstdint>
#include <string>

namespace memopt {
namespace transport {

static constexpr size_t RB_CAPACITY   = 4096;
static constexpr size_t RB_SLOT_SIZE  = 65536;  // 64KB max payload

struct SlotHeader {
    std::atomic<uint8_t> ready{0};   // 1=written by producer, 0=consumed
    uint32_t             size{0};    // actual payload bytes
    uint8_t              msg_type{0};
    uint8_t              _pad[2]{};
};

struct Slot {
    SlotHeader header;
    uint8_t    data[RB_SLOT_SIZE];
};

// Layout at start of shared memory region
struct RingBufferHeader {
    alignas(64) std::atomic<uint64_t> write_head{0};
    char _pad1[56];
    alignas(64) std::atomic<uint64_t> read_head{0};
    char _pad2[56];
    alignas(64) std::atomic<uint32_t> producer_pid{0};
    alignas(64) std::atomic<uint32_t> consumer_pid{0};
};

class RingBuffer {
public:
    /// Create a new ring buffer backed by shared memory.
    /// The creator owns the shm file and unlinks on destruction.
    static RingBuffer create(const std::string& shm_path);

    /// Attach to an existing ring buffer in shared memory.
    /// Does not own the file — will not unlink on destruction.
    static RingBuffer open(const std::string& shm_path);

    ~RingBuffer();

    // Move-only (owns mmap'd region)
    RingBuffer(RingBuffer&& other) noexcept;
    RingBuffer& operator=(RingBuffer&& other) noexcept;
    RingBuffer(const RingBuffer&) = delete;
    RingBuffer& operator=(const RingBuffer&) = delete;

    /// Producer: write a message. Returns false if full (non-blocking).
    bool try_push(uint8_t msg_type,
                  const void* payload,
                  uint32_t size) noexcept;

    /// Consumer: read a message. Returns false if empty (non-blocking).
    bool try_pop(uint8_t* out_msg_type,
                 void* out_payload,
                 uint32_t* out_size) noexcept;

    bool is_full()  const noexcept;
    bool is_empty() const noexcept;

    /// Conventional shm path for a node.
    static std::string shm_path(const std::string& node_id,
                                 bool is_request);

    /// Total size of the shared memory region.
    static size_t total_shm_size() noexcept;

    /// Whether this instance is valid (mmap succeeded).
    bool is_valid() const noexcept { return shm_ptr_ != nullptr; }

private:
    RingBuffer() = default;

    void* shm_ptr_    = nullptr;
    size_t shm_size_  = 0;
    std::string path_;
    bool is_owner_    = false;
    int fd_           = -1;

    RingBufferHeader* header() noexcept;
    const RingBufferHeader* header() const noexcept;
    Slot* slots() noexcept;
    const Slot* slots() const noexcept;
};

} // namespace transport
} // namespace memopt
