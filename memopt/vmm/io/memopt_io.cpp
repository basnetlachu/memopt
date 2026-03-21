/*
 * memopt_io.cpp
 *
 * Async NVMe block reader using io_uring (Linux 5.1+).
 * Falls back to pread(2) on kernels without io_uring or when
 * io_uring_queue_init fails (RLIMIT_MEMLOCK, permissions, etc).
 *
 * Exposes a plain C API consumed by Python via ctypes:
 *
 *   int memopt_read_block(const char* path,
 *                         void*       dst,
 *                         size_t      size);
 *   Returns 0 on success, -errno on failure.
 *
 *   int memopt_uring_available(void);
 *   Returns 1 if io_uring is compiled in AND initialises
 *   successfully on this kernel, 0 otherwise.
 *
 * Build:
 *   If liburing present:
 *     g++ -O3 -std=c++17 -shared -fPIC -luring \
 *         -o libmemopt_io.so memopt_io.cpp
 *   Fallback (pread only):
 *     g++ -O3 -std=c++17 -shared -fPIC \
 *         -o libmemopt_io.so memopt_io.cpp
 *
 * The build script (build.sh) detects liburing automatically.
 * The library is identical in API regardless of which path compiled.
 */

#include <cstring>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef __has_include
#  if __has_include(<liburing.h>)
#    include <liburing.h>
#    define MEMOPT_HAVE_URING 1
#  endif
#endif

extern "C" {

int memopt_uring_available(void) {
#ifdef MEMOPT_HAVE_URING
    struct io_uring ring;
    /*
     * Use queue depth 1 for the availability probe.
     * IORING_SETUP_SQPOLL is not used here — it requires CAP_SYS_NICE
     * and is unnecessary for our access pattern.
     */
    int ret = io_uring_queue_init(1, &ring, 0);
    if (ret == 0) {
        io_uring_queue_exit(&ring);
        return 1;
    }
#endif
    return 0;
}

int memopt_read_block(const char* path, void* dst, size_t size) {
    if (!path || !dst || size == 0) return -EINVAL;

    int fd = open(path, O_RDONLY);
    if (fd < 0) return -errno;

#ifdef MEMOPT_HAVE_URING
    {
        struct io_uring ring;
        /*
         * Queue depth 1: one outstanding read at a time.
         * For batch reads (future), increase depth and loop.
         * No SQPOLL — we are not latency-critical enough to burn a CPU.
         */
        int ret = io_uring_queue_init(1, &ring, 0);
        if (ret == 0) {
            struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
            if (sqe) {
                io_uring_prep_read(sqe, fd, dst, (unsigned)size, 0);
                ret = io_uring_submit(&ring);
                if (ret == 1) {
                    struct io_uring_cqe* cqe;
                    ret = io_uring_wait_cqe(&ring, &cqe);
                    int result = (ret == 0) ? cqe->res : ret;
                    io_uring_cqe_seen(&ring, cqe);
                    io_uring_queue_exit(&ring);
                    close(fd);
                    /*
                     * cqe->res is the number of bytes read on success,
                     * or a negative errno on failure.
                     */
                    if (result == (int)size) return 0;
                    if (result < 0) return result;
                    /* Partial read — treat as EIO */
                    return -EIO;
                }
            }
            io_uring_queue_exit(&ring);
        }
        /* io_uring path failed — fall through to pread */
    }
#endif

    /* pread fallback — correct on all kernels */
    ssize_t n = pread(fd, dst, size, 0);
    int saved = errno;
    close(fd);
    if (n == (ssize_t)size) return 0;
    if (n < 0) return -saved;
    return -EIO; /* partial read */
}

} /* extern "C" */
