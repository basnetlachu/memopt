// nvme_async.cpp — Async NVMe I/O via io_uring with sync fallback.
//
// io_uring path: submit read/write to kernel, poll for completion.
// Sync fallback: pread()/write() when io_uring not available.
//
// All syscalls happen without the GIL.
// Error handling via callbacks — never throws.

#include "nvme_async.h"

#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <new>

#ifdef MEMOPT_IO_URING_AVAILABLE
#include <liburing.h>
#endif

// macOS doesn't have fdatasync — use fcntl(F_FULLFSYNC) instead
#ifdef __APPLE__
static int portable_fdatasync(int fd) {
    return ::fcntl(fd, F_FULLFSYNC);
}
#else
static int portable_fdatasync(int fd) {
    return ::fdatasync(fd);
}
#endif

namespace memopt {

// ── AsyncNVMeReader ──────────────────────────────────────────────────────

AsyncNVMeReader::AsyncNVMeReader(bool force_sync)
    : _use_uring(false)
{
    if (force_sync) {
        return;
    }

#ifdef MEMOPT_IO_URING_AVAILABLE
    // Attempt to initialize io_uring
    // This fails on kernels < 5.1 or when
    // RLIMIT_MEMLOCK is too small
    int ret = io_uring_queue_init(QUEUE_DEPTH, &_ring, 0);
    if (ret == 0) {
        _ring_initialized = true;
        _use_uring = true;
    }
    // If init fails: silent fallback to pread()
#endif
}

AsyncNVMeReader::~AsyncNVMeReader() {
    drain();
#ifdef MEMOPT_IO_URING_AVAILABLE
    if (_ring_initialized) {
        io_uring_queue_exit(&_ring);
    }
#endif
}

bool AsyncNVMeReader::read_async(
    const std::string& path,
    void* buf,
    size_t offset,
    size_t size,
    AsyncIOCallback callback) noexcept
{
    if (!_use_uring) {
        _sync_read(path, buf, offset, size, callback);
        return true;
    }

#ifdef MEMOPT_IO_URING_AVAILABLE
    int fd = open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (fd < 0) {
        // O_DIRECT may fail on some filesystems
        // Retry without O_DIRECT
        fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            callback(false, 0);
            _stats.reads_failed++;
            return true;
        }
    }

    struct io_uring_sqe* sqe = io_uring_get_sqe(&_ring);
    if (!sqe) {
        // Queue full
        close(fd);
        return false;
    }

    // Heap-allocate PendingRead
    // Freed in _handle_completion
    auto* pending = new(std::nothrow) PendingRead{
        buf, size, fd, std::move(callback)};
    if (!pending) {
        close(fd);
        callback(false, 0);
        return true;
    }

    io_uring_prep_read(sqe, fd,
        buf, (unsigned)size, (uint64_t)offset);
    io_uring_sqe_set_data(sqe, pending);

    int ret = io_uring_submit(&_ring);
    if (ret < 0) {
        close(fd);
        auto cb = std::move(pending->callback);
        delete pending;
        _sync_read(path, buf, offset, size, cb);
        _stats.sync_fallbacks++;
        return true;
    }

    _stats.reads_submitted++;
    return true;
#else
    _sync_read(path, buf, offset, size, callback);
    return true;
#endif
}

int AsyncNVMeReader::poll() noexcept {
#ifdef MEMOPT_IO_URING_AVAILABLE
    if (!_use_uring) return 0;

    struct io_uring_cqe* cqe;
    int count = 0;

    while (io_uring_peek_cqe(&_ring, &cqe) == 0) {
        _handle_completion(cqe);
        io_uring_cqe_seen(&_ring, cqe);
        count++;
    }
    return count;
#else
    return 0;
#endif
}

int AsyncNVMeReader::wait(int timeout_ms) noexcept {
#ifdef MEMOPT_IO_URING_AVAILABLE
    if (!_use_uring) return 0;

    struct __kernel_timespec ts;
    ts.tv_sec  = timeout_ms / 1000;
    ts.tv_nsec = (timeout_ms % 1000) * 1000000LL;

    struct io_uring_cqe* cqe;
    int ret = io_uring_wait_cqe_timeout(&_ring, &cqe, &ts);

    if (ret == 0) {
        _handle_completion(cqe);
        io_uring_cqe_seen(&_ring, cqe);
        // Drain any additional completions
        return 1 + poll();
    }
    return 0;
#else
    return 0;
#endif
}

void AsyncNVMeReader::drain() noexcept {
#ifdef MEMOPT_IO_URING_AVAILABLE
    if (!_use_uring) return;
    // Keep waiting until no more completions arrive
    while (true) {
        int n = wait(10);
        if (n == 0) break;
    }
#endif
}

AsyncNVMeReader::Stats AsyncNVMeReader::stats() const noexcept {
    return _stats;
}

#ifdef MEMOPT_IO_URING_AVAILABLE
void AsyncNVMeReader::_handle_completion(
    struct io_uring_cqe* cqe) noexcept
{
    auto* pending = static_cast<PendingRead*>(
        io_uring_cqe_get_data(cqe));
    if (!pending) return;

    bool success = (cqe->res > 0);
    size_t bytes = success ? (size_t)cqe->res : 0;

    close(pending->fd);
    pending->callback(success, bytes);

    if (success) {
        _stats.reads_completed++;
        _stats.bytes_read += bytes;
    } else {
        _stats.reads_failed++;
    }

    delete pending;
}
#endif

void AsyncNVMeReader::_sync_read(
    const std::string& path,
    void* buf,
    size_t offset,
    size_t size,
    AsyncIOCallback callback) noexcept
{
    int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        callback(false, 0);
        _stats.reads_failed++;
        return;
    }

    ssize_t n = pread(fd, buf, size, (off_t)offset);
    close(fd);

    bool ok = (n == (ssize_t)size);
    callback(ok, ok ? (size_t)n : 0);

    if (ok) {
        _stats.reads_completed++;
        _stats.bytes_read += (size_t)n;
    } else {
        _stats.reads_failed++;
    }
    _stats.sync_fallbacks++;
}

// ── AsyncNVMeWriter ──────────────────────────────────────────────────────
// Write path: open .tmp -> write -> fdatasync -> rename
// fdatasync submitted async via io_uring when available
// rename is always synchronous — kernel limitation
//
// Current implementation: sync-only for crash safety.
// Async write with chained SQEs (IOSQE_IO_LINK) requires kernel >= 5.3
// and is the next step after validating async reads work correctly.

AsyncNVMeWriter::AsyncNVMeWriter(bool force_sync)
    : _use_uring(false)
{
    if (force_sync) return;

#ifdef MEMOPT_IO_URING_AVAILABLE
    int ret = io_uring_queue_init(QUEUE_DEPTH, &_ring, 0);
    if (ret == 0) {
        _ring_initialized = true;
        _use_uring = true;
    }
#endif
}

AsyncNVMeWriter::~AsyncNVMeWriter() {
    drain();
#ifdef MEMOPT_IO_URING_AVAILABLE
    if (_ring_initialized) {
        io_uring_queue_exit(&_ring);
    }
#endif
}

bool AsyncNVMeWriter::write_async(
    const std::string& path,
    const void* data,
    size_t size,
    AsyncIOCallback callback) noexcept
{
    // Always fall through to sync write.
    // Async write with crash safety requires
    // chaining: write -> fdatasync -> rename
    // io_uring linked SQEs (IOSQE_IO_LINK)
    // support this on kernel >= 5.3.
    //
    // For now: sync write is correct and safe.
    // Async write is the next step after
    // validating async reads work correctly.
    _sync_write(path, data, size, callback);
    return true;
}

void AsyncNVMeWriter::_sync_write(
    const std::string& path,
    const void* data,
    size_t size,
    AsyncIOCallback callback) noexcept
{
    std::string tmp_path = path + ".tmp";

    int fd = open(tmp_path.c_str(),
        O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0) {
        callback(false, 0);
        return;
    }

    // Write in chunks (handles large blocks correctly)
    const auto* ptr = static_cast<const char*>(data);
    size_t remaining = size;
    while (remaining > 0) {
        ssize_t n = ::write(fd, ptr, remaining);
        if (n < 0) {
            if (errno == EINTR) continue;
            ::close(fd);
            ::unlink(tmp_path.c_str());
            callback(false, 0);
            return;
        }
        ptr += n;
        remaining -= static_cast<size_t>(n);
    }

    if (portable_fdatasync(fd) != 0) {
        ::close(fd);
        ::unlink(tmp_path.c_str());
        callback(false, 0);
        return;
    }
    ::close(fd);

    if (::rename(tmp_path.c_str(), path.c_str()) != 0) {
        ::unlink(tmp_path.c_str());
        callback(false, 0);
        return;
    }

    callback(true, size);
}

void AsyncNVMeWriter::drain() noexcept {
    // No-op for sync-only writes
}

int AsyncNVMeWriter::poll() noexcept {
    return 0;
}

} // namespace memopt
