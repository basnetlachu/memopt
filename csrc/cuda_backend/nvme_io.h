// nvme_io.h — Crash-safe NVMe block I/O.
//
// Replaces Python's fsync+rename pattern with identical crash-safety
// semantics, but without holding the GIL during syscalls.
//
// All functions are noexcept. Errors reported via result structs.
// Thread safety: all functions are safe for concurrent use (no shared state).
// GIL: never held — pure POSIX syscalls.
#pragma once

#include <cstdint>
#include <string>

namespace memopt {
namespace cuda_backend {

// ═══════════════════════════════════════════════════════════════════════════
// Result types — no exceptions across pybind11 boundary
// ═══════════════════════════════════════════════════════════════════════════

struct NVMeWriteResult {
    bool    success     = false;
    int     errno_val   = 0;
    int64_t bytes_written = 0;
};

struct NVMeReadResult {
    bool    success     = false;
    int     errno_val   = 0;
    int64_t bytes_read  = 0;
};

// ═══════════════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════════════

/// Atomic write: data → path.tmp → fdatasync → rename(path.tmp, path).
/// Crash-safe: partial writes leave only .tmp files (cleaned by recover).
/// GIL: NOT held — caller must release before calling.
NVMeWriteResult write_block_atomic(const std::string& path,
                                    const void* data,
                                    size_t size) noexcept;

/// Read block from path into caller-provided buffer.
/// Uses read() in a loop (handles partial reads).
/// GIL: NOT held.
NVMeReadResult read_block(const std::string& path,
                           void* out_buf,
                           size_t buf_size) noexcept;

/// Remove incomplete .vmm_block.tmp files from a directory.
/// Called at process startup — same semantics as Python _recover_nvme_dir().
/// GIL: NOT held.
int recover_nvme_dir(const std::string& dir_path) noexcept;

} // namespace cuda_backend
} // namespace memopt
