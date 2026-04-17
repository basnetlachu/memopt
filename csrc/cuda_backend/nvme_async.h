// nvme_async.h — Async NVMe I/O via io_uring.
//
// Replaces synchronous pread()/write() with io_uring submissions.
// GPU stream no longer blocks on NVMe I/O — latency hidden behind compute.
//
// Degrades gracefully:
//   io_uring available (Linux 5.1+, liburing) → true async
//   io_uring unavailable                      → synchronous pread() fallback
//
// Thread safety: one io_uring per instance. Not shared across threads.
// GIL: never held — pure POSIX / io_uring syscalls.
#pragma once

#ifdef MEMOPT_IO_URING_AVAILABLE
#include <liburing.h>
#endif

#include <string>
#include <functional>
#include <cstddef>
#include <cstdint>

namespace memopt {

// Callback type: called on I/O completion
// Args: success (bool), bytes_transferred (size_t)
using AsyncIOCallback = std::function<void(bool, size_t)>;

/**
 * AsyncNVMeReader
 *
 * Submits NVMe reads via io_uring.
 * Non-blocking: read() returns immediately.
 * Caller polls complete() or blocks on wait().
 *
 * Thread-safe: one io_uring per instance.
 * Not shared across threads.
 *
 * Degrades to synchronous pread() when
 * io_uring is not available.
 *
 * REQUIRES: Linux 5.1+, liburing
 * FALLBACK: synchronous pread() on older kernels
 */
class AsyncNVMeReader {
public:
    static constexpr int QUEUE_DEPTH = 64;

    explicit AsyncNVMeReader(bool force_sync = false);
    ~AsyncNVMeReader();

    // Non-copyable
    AsyncNVMeReader(const AsyncNVMeReader&) = delete;
    AsyncNVMeReader& operator=(const AsyncNVMeReader&) = delete;

    /**
     * Submit async read.
     * Returns immediately.
     * callback fires on completion.
     *
     * path:      file to read
     * buf:       destination buffer
     *            caller owns, must stay valid
     *            until callback fires
     * offset:    byte offset in file
     * size:      bytes to read
     * callback:  fires on completion
     *
     * Returns false if queue is full.
     * Never throws.
     */
    bool read_async(
        const std::string& path,
        void* buf,
        size_t offset,
        size_t size,
        AsyncIOCallback callback) noexcept;

    /**
     * Poll for completed I/Os.
     * Non-blocking.
     * Returns number of completions processed.
     * Call regularly from a poller thread.
     */
    int poll() noexcept;

    /**
     * Block until at least one I/O completes.
     * Timeout in milliseconds.
     * Returns number of completions.
     */
    int wait(int timeout_ms = 100) noexcept;

    /**
     * Drain all pending I/Os.
     * Called at shutdown.
     */
    void drain() noexcept;

    bool is_async() const noexcept { return _use_uring; }

    struct Stats {
        uint64_t reads_submitted;
        uint64_t reads_completed;
        uint64_t reads_failed;
        uint64_t bytes_read;
        uint64_t sync_fallbacks;
    };
    Stats stats() const noexcept;

private:
    bool  _use_uring{false};
    Stats _stats{};

#ifdef MEMOPT_IO_URING_AVAILABLE
    struct io_uring _ring{};
    bool            _ring_initialized{false};

    struct PendingRead {
        void*           buf;
        size_t          size;
        int             fd;
        AsyncIOCallback callback;
    };

    // Map sqe user_data -> PendingRead
    // user_data is a pointer cast to uint64_t
    // We own the heap allocation
    void _handle_completion(struct io_uring_cqe* cqe) noexcept;
#endif

    // Synchronous fallback
    void _sync_read(
        const std::string& path,
        void* buf,
        size_t offset,
        size_t size,
        AsyncIOCallback callback) noexcept;
};

/**
 * AsyncNVMeWriter
 *
 * Crash-safe async writes via io_uring.
 * Write path: write to .tmp -> fdatasync -> rename
 * All three steps are async when io_uring available.
 *
 * The rename is always synchronous --
 * POSIX rename() is atomic but not async-safe
 * via io_uring on all kernels.
 * We use a background thread for rename.
 */
class AsyncNVMeWriter {
public:
    static constexpr int QUEUE_DEPTH = 64;

    explicit AsyncNVMeWriter(bool force_sync = false);
    ~AsyncNVMeWriter();

    AsyncNVMeWriter(const AsyncNVMeWriter&) = delete;
    AsyncNVMeWriter& operator=(const AsyncNVMeWriter&) = delete;

    /**
     * Async crash-safe write.
     * Writes data to path atomically.
     * callback fires when rename completes.
     *
     * Never throws. Returns false if
     * internal queue is full.
     */
    bool write_async(
        const std::string& path,
        const void* data,
        size_t size,
        AsyncIOCallback callback) noexcept;

    int poll() noexcept;
    void drain() noexcept;
    bool is_async() const noexcept { return _use_uring; }

private:
    bool _use_uring{false};

#ifdef MEMOPT_IO_URING_AVAILABLE
    struct io_uring _ring{};
    bool            _ring_initialized{false};
#endif

    void _sync_write(
        const std::string& path,
        const void* data,
        size_t size,
        AsyncIOCallback callback) noexcept;
};

} // namespace memopt
