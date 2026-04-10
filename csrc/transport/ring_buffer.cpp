// ring_buffer.cpp — SPSC shared memory ring buffer implementation.

#include "ring_buffer.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace memopt {
namespace transport {

// ═══════════════════════════════════════════════════════════════════════════
// Size calculation
// ═══════════════════════════════════════════════════════════════════════════

size_t RingBuffer::total_shm_size() noexcept {
    return sizeof(RingBufferHeader) + RB_CAPACITY * sizeof(Slot);
}

std::string RingBuffer::shm_path(const std::string& node_id,
                                  bool is_request) {
    return "/memopt_transport_" + node_id +
           (is_request ? "_req" : "_resp");
}

// ═══════════════════════════════════════════════════════════════════════════
// create — producer side
// ═══════════════════════════════════════════════════════════════════════════

RingBuffer RingBuffer::create(const std::string& shm_path) {
    RingBuffer rb;
    rb.path_ = shm_path;
    rb.is_owner_ = true;
    rb.shm_size_ = total_shm_size();

    // shm_open creates /dev/shm/<name>
    rb.fd_ = ::shm_open(shm_path.c_str(),
                         O_CREAT | O_RDWR | O_TRUNC, 0600);
    if (rb.fd_ < 0) return rb;  // is_valid() == false

    if (::ftruncate(rb.fd_, static_cast<off_t>(rb.shm_size_)) != 0) {
        ::close(rb.fd_);
        ::shm_unlink(shm_path.c_str());
        rb.fd_ = -1;
        return rb;
    }

    rb.shm_ptr_ = ::mmap(nullptr, rb.shm_size_,
                          PROT_READ | PROT_WRITE,
                          MAP_SHARED, rb.fd_, 0);
    if (rb.shm_ptr_ == MAP_FAILED) {
        rb.shm_ptr_ = nullptr;
        ::close(rb.fd_);
        ::shm_unlink(shm_path.c_str());
        rb.fd_ = -1;
        return rb;
    }

    // Zero-initialize
    std::memset(rb.shm_ptr_, 0, rb.shm_size_);

    // Initialize header with placement new
    new (rb.header()) RingBufferHeader{};
    rb.header()->producer_pid.store(
        static_cast<uint32_t>(::getpid()),
        std::memory_order_relaxed);

    return rb;
}

// ═══════════════════════════════════════════════════════════════════════════
// open — consumer side
// ═══════════════════════════════════════════════════════════════════════════

RingBuffer RingBuffer::open(const std::string& shm_path) {
    RingBuffer rb;
    rb.path_ = shm_path;
    rb.is_owner_ = false;
    rb.shm_size_ = total_shm_size();

    rb.fd_ = ::shm_open(shm_path.c_str(), O_RDWR, 0);
    if (rb.fd_ < 0) return rb;

    rb.shm_ptr_ = ::mmap(nullptr, rb.shm_size_,
                          PROT_READ | PROT_WRITE,
                          MAP_SHARED, rb.fd_, 0);
    if (rb.shm_ptr_ == MAP_FAILED) {
        rb.shm_ptr_ = nullptr;
        ::close(rb.fd_);
        rb.fd_ = -1;
        return rb;
    }

    rb.header()->consumer_pid.store(
        static_cast<uint32_t>(::getpid()),
        std::memory_order_relaxed);

    return rb;
}

// ═══════════════════════════════════════════════════════════════════════════
// Destructor + move
// ═══════════════════════════════════════════════════════════════════════════

RingBuffer::~RingBuffer() {
    if (shm_ptr_) {
        ::munmap(shm_ptr_, shm_size_);
    }
    if (fd_ >= 0) {
        ::close(fd_);
    }
    if (is_owner_ && !path_.empty()) {
        ::shm_unlink(path_.c_str());
    }
}

RingBuffer::RingBuffer(RingBuffer&& other) noexcept
    : shm_ptr_(other.shm_ptr_),
      shm_size_(other.shm_size_),
      path_(std::move(other.path_)),
      is_owner_(other.is_owner_),
      fd_(other.fd_) {
    other.shm_ptr_ = nullptr;
    other.fd_ = -1;
    other.is_owner_ = false;
}

RingBuffer& RingBuffer::operator=(RingBuffer&& other) noexcept {
    if (this != &other) {
        // Clean up current
        if (shm_ptr_) ::munmap(shm_ptr_, shm_size_);
        if (fd_ >= 0) ::close(fd_);
        if (is_owner_ && !path_.empty()) ::shm_unlink(path_.c_str());

        shm_ptr_ = other.shm_ptr_;
        shm_size_ = other.shm_size_;
        path_ = std::move(other.path_);
        is_owner_ = other.is_owner_;
        fd_ = other.fd_;

        other.shm_ptr_ = nullptr;
        other.fd_ = -1;
        other.is_owner_ = false;
    }
    return *this;
}

// ═══════════════════════════════════════════════════════════════════════════
// Accessors
// ═══════════════════════════════════════════════════════════════════════════

RingBufferHeader* RingBuffer::header() noexcept {
    return reinterpret_cast<RingBufferHeader*>(shm_ptr_);
}
const RingBufferHeader* RingBuffer::header() const noexcept {
    return reinterpret_cast<const RingBufferHeader*>(shm_ptr_);
}

Slot* RingBuffer::slots() noexcept {
    auto* base = static_cast<char*>(shm_ptr_) + sizeof(RingBufferHeader);
    return reinterpret_cast<Slot*>(base);
}
const Slot* RingBuffer::slots() const noexcept {
    auto* base = static_cast<const char*>(shm_ptr_) + sizeof(RingBufferHeader);
    return reinterpret_cast<const Slot*>(base);
}

// ═══════════════════════════════════════════════════════════════════════════
// try_push — producer side, SPSC
// ═══════════════════════════════════════════════════════════════════════════

bool RingBuffer::try_push(uint8_t msg_type,
                           const void* payload,
                           uint32_t size) noexcept {
    if (!shm_ptr_ || size > RB_SLOT_SIZE) return false;

    uint64_t wh = header()->write_head.load(std::memory_order_relaxed);
    uint64_t idx = wh % RB_CAPACITY;
    Slot& slot = slots()[idx];

    // Check if slot is consumed (ready == 0)
    if (slot.header.ready.load(std::memory_order_acquire) != 0) {
        return false;  // buffer full
    }

    // Write payload
    if (payload && size > 0) {
        std::memcpy(slot.data, payload, size);
    }
    slot.header.size = size;
    slot.header.msg_type = msg_type;

    // Publish: ready = 1 with release ordering.
    // All payload writes are visible before the ready flag.
    slot.header.ready.store(1, std::memory_order_release);

    // Advance write head
    header()->write_head.fetch_add(1, std::memory_order_relaxed);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// try_pop — consumer side, SPSC
// ═══════════════════════════════════════════════════════════════════════════

bool RingBuffer::try_pop(uint8_t* out_msg_type,
                          void* out_payload,
                          uint32_t* out_size) noexcept {
    if (!shm_ptr_) return false;

    uint64_t rh = header()->read_head.load(std::memory_order_relaxed);
    uint64_t idx = rh % RB_CAPACITY;
    Slot& slot = slots()[idx];

    // Check if slot is ready (ready == 1)
    if (slot.header.ready.load(std::memory_order_acquire) == 0) {
        return false;  // buffer empty
    }

    // Read payload
    if (out_msg_type) *out_msg_type = slot.header.msg_type;
    if (out_size) *out_size = slot.header.size;
    if (out_payload && slot.header.size > 0) {
        std::memcpy(out_payload, slot.data, slot.header.size);
    }

    // Mark consumed: ready = 0 with release ordering.
    slot.header.ready.store(0, std::memory_order_release);

    // Advance read head
    header()->read_head.fetch_add(1, std::memory_order_relaxed);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// is_full / is_empty
// ═══════════════════════════════════════════════════════════════════════════

bool RingBuffer::is_full() const noexcept {
    if (!shm_ptr_) return true;
    uint64_t wh = header()->write_head.load(std::memory_order_relaxed);
    uint64_t idx = wh % RB_CAPACITY;
    return slots()[idx].header.ready.load(std::memory_order_acquire) != 0;
}

bool RingBuffer::is_empty() const noexcept {
    if (!shm_ptr_) return true;
    uint64_t rh = header()->read_head.load(std::memory_order_relaxed);
    uint64_t idx = rh % RB_CAPACITY;
    return slots()[idx].header.ready.load(std::memory_order_acquire) == 0;
}

} // namespace transport
} // namespace memopt
