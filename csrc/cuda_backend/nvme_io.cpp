// nvme_io.cpp — Crash-safe NVMe I/O implementation.
//
// All syscalls happen without the GIL.
// Error handling via errno — never throws.

#include "nvme_io.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

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
namespace cuda_backend {

// ═══════════════════════════════════════════════════════════════════════════
// write_block_atomic
// ═══════════════════════════════════════════════════════════════════════════

NVMeWriteResult write_block_atomic(const std::string& path,
                                    const void* data,
                                    size_t size) noexcept {
    NVMeWriteResult result{};
    std::string tmp_path = path + ".tmp";

    // 1. Open temp file
    int fd = ::open(tmp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) {
        result.errno_val = errno;
        return result;
    }

    // 2. Write in chunks (handles large blocks correctly)
    const auto* ptr = static_cast<const char*>(data);
    size_t remaining = size;
    while (remaining > 0) {
        ssize_t n = ::write(fd, ptr, remaining);
        if (n < 0) {
            if (errno == EINTR) continue;  // interrupted — retry
            result.errno_val = errno;
            ::close(fd);
            ::unlink(tmp_path.c_str());
            return result;
        }
        ptr += n;
        remaining -= static_cast<size_t>(n);
    }
    result.bytes_written = static_cast<int64_t>(size);

    // 3. fdatasync — data only, faster than fsync (no metadata)
    if (portable_fdatasync(fd) != 0) {
        result.errno_val = errno;
        ::close(fd);
        ::unlink(tmp_path.c_str());
        return result;
    }

    // 4. Close
    ::close(fd);

    // 5. Atomic rename
    if (::rename(tmp_path.c_str(), path.c_str()) != 0) {
        result.errno_val = errno;
        ::unlink(tmp_path.c_str());
        return result;
    }

    result.success = true;
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// read_block
// ═══════════════════════════════════════════════════════════════════════════

NVMeReadResult read_block(const std::string& path,
                           void* out_buf,
                           size_t buf_size) noexcept {
    NVMeReadResult result{};

    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        result.errno_val = errno;
        return result;
    }

    // Get actual file size
    struct stat st{};
    if (::fstat(fd, &st) != 0) {
        result.errno_val = errno;
        ::close(fd);
        return result;
    }

    size_t to_read = static_cast<size_t>(st.st_size);
    if (to_read > buf_size) to_read = buf_size;

    auto* ptr = static_cast<char*>(out_buf);
    size_t total_read = 0;
    while (total_read < to_read) {
        ssize_t n = ::read(fd, ptr + total_read, to_read - total_read);
        if (n < 0) {
            if (errno == EINTR) continue;
            result.errno_val = errno;
            ::close(fd);
            return result;
        }
        if (n == 0) break;  // EOF
        total_read += static_cast<size_t>(n);
    }

    ::close(fd);
    result.success = true;
    result.bytes_read = static_cast<int64_t>(total_read);
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// recover_nvme_dir
// ═══════════════════════════════════════════════════════════════════════════

int recover_nvme_dir(const std::string& dir_path) noexcept {
    int removed = 0;

    DIR* dir = ::opendir(dir_path.c_str());
    if (!dir) return 0;  // directory doesn't exist — nothing to recover

    struct dirent* entry;
    while ((entry = ::readdir(dir)) != nullptr) {
        const char* name = entry->d_name;
        size_t len = std::strlen(name);

        // Match *.vmm_block.tmp
        static const char suffix[] = ".vmm_block.tmp";
        static const size_t suffix_len = sizeof(suffix) - 1;

        if (len > suffix_len &&
            std::strcmp(name + len - suffix_len, suffix) == 0) {
            std::string full_path = dir_path + "/" + name;
            if (::unlink(full_path.c_str()) == 0) {
                ++removed;
            }
        }
    }

    ::closedir(dir);
    return removed;
}

} // namespace cuda_backend
} // namespace memopt
