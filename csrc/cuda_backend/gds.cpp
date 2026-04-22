// gds.cpp — cuFile GDS implementation with transparent fallback.
//
// Compilation:
//   With GDS:     -DMEMOPT_GDS_AVAILABLE + link -lcufile
//   Without GDS:  stubs that fall back to standard I/O
//
// cuFileReadAsync/cuFileWriteAsync were added in CUDA 11.8.
// On older CUDA: falls back to synchronous cuFileRead/cuFileWrite.
// On no GDS: falls back to mmap + cudaMemcpyAsync.

#include "gds.h"
#include "nvme_io.h"

#include <cstdio>
#include <fcntl.h>
#include <mutex>
#include <sys/mman.h>
#include <unistd.h>

#ifdef MEMOPT_GDS_AVAILABLE
#include <cufile.h>
#endif

namespace memopt {
namespace cuda_backend {

#ifdef MEMOPT_CUDA_AVAILABLE
// These functions are declared noexcept and return void, so CUDA failures
// cannot be propagated to the caller. Log to stderr so silent data
// corruption is at least visible in logs.
#define MEMOPT_CUDA_LOG(expr)                                               \
    do {                                                                    \
        cudaError_t _e = (expr);                                            \
        if (_e != cudaSuccess) {                                            \
            std::fprintf(stderr,                                            \
                "memopt gds: %s failed: %s\n",                              \
                #expr, cudaGetErrorString(_e));                             \
        }                                                                   \
    } while (0)
#endif

// ═══════════════════════════════════════════════════════════════════════════
// gds_is_available
// ═══════════════════════════════════════════════════════════════════════════

bool gds_is_available() noexcept {
#ifdef MEMOPT_GDS_AVAILABLE
    CUfileError_t err = cuFileDriverOpen();
    if (err.err != CU_FILE_SUCCESS) return false;
    cuFileDriverClose();
    return true;
#else
    return false;
#endif
}

// ═══════════════════════════════════════════════════════════════════════════
// gds_read_async
// ═══════════════════════════════════════════════════════════════════════════

void gds_read_async(const std::string& path,
                    void* gpu_ptr,
                    size_t size,
                    size_t file_offset,
                    cudaStream_t stream) noexcept {
#ifdef MEMOPT_GDS_AVAILABLE
    int fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) goto fallback;

    {
        CUfileDescr_t desc{};
        desc.handle.fd = fd;
        desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

        CUfileHandle_t fh{};
        CUfileError_t err = cuFileHandleRegister(&fh, &desc);
        if (err.err != CU_FILE_SUCCESS) {
            ::close(fd);
            goto fallback;
        }

        // ── Async path: cuFileReadAsync (CUDA >= 11.8) ──────────────
        // Returns immediately. Completion signaled through the stream.
        // The fd and fh must remain valid until the stream completes.
        // Since we deregister/close below, this requires stream sync
        // or keeping handles alive — for correctness we sync here.
        //
        // ── Sync path: cuFileRead (CUDA < 11.8) ─────────────────────
        // Blocks calling thread until read completes.
        // Stream parameter ignored.

#if defined(CUFILE_VERSION) && CUFILE_VERSION >= 11080
        // cuFileReadAsync available — truly async via stream
        ssize_t ret = cuFileReadAsync(fh, gpu_ptr, &size,
                                       file_offset, 0,
                                       nullptr, nullptr, stream);
        // cuFileReadAsync returns 0 on success, negative on error
        if (ret < 0) {
            cuFileHandleDeregister(fh);
            ::close(fd);
            goto fallback;
        }
        // Must synchronize before closing the file handle —
        // cuFileReadAsync uses the fd internally via DMA.
        if (stream) {
            MEMOPT_CUDA_LOG(cudaStreamSynchronize(stream));
        }
#else
        // cuFileReadAsync not available — synchronous fallback.
        // Log once so operators know async is not active.
        static std::once_flag sync_warn;
        std::call_once(sync_warn, []() {
            std::fprintf(stderr,
                "memopt: cuFileReadAsync unavailable "
                "(CUDA < 11.8), using synchronous cuFileRead\n");
        });
        ssize_t ret = cuFileRead(fh, gpu_ptr, size, file_offset, 0);
#endif

        cuFileHandleDeregister(fh);
        ::close(fd);

        if (ret >= 0 && static_cast<size_t>(ret) == size) {
            return;  // Success — GDS path
        }
        // GDS read returned wrong size — fall through to mmap
    }

fallback:
#endif

    // Fallback: mmap + cudaMemcpyAsync (or synchronous copy)
    // Used when: GDS not compiled, driver not loaded,
    // O_DIRECT open fails, cuFile registration fails,
    // or cuFileRead/Async returns error.

#ifdef MEMOPT_CUDA_AVAILABLE
    int fd2 = ::open(path.c_str(), O_RDONLY);
    if (fd2 < 0) return;

    void* mapped = ::mmap(nullptr, size, PROT_READ,
                          MAP_PRIVATE | MAP_POPULATE, fd2, file_offset);
    if (mapped == MAP_FAILED) {
        ::close(fd2);
        return;
    }

    if (stream) {
        MEMOPT_CUDA_LOG(cudaMemcpyAsync(gpu_ptr, mapped, size,
                        cudaMemcpyHostToDevice, stream));
        // mmap is about to be unmapped; ensure the DMA has completed first.
        MEMOPT_CUDA_LOG(cudaStreamSynchronize(stream));
    } else {
        MEMOPT_CUDA_LOG(cudaMemcpy(gpu_ptr, mapped, size, cudaMemcpyHostToDevice));
    }

    ::munmap(mapped, size);
    ::close(fd2);
#else
    (void)stream;
    read_block(path, gpu_ptr, size);
#endif
}

// ═══════════════════════════════════════════════════════════════════════════
// gds_write_async
// ═══════════════════════════════════════════════════════════════════════════

void gds_write_async(const std::string& path,
                     const void* gpu_ptr,
                     size_t size,
                     size_t file_offset,
                     cudaStream_t stream) noexcept {
#ifdef MEMOPT_GDS_AVAILABLE
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_DIRECT, 0600);
    if (fd < 0) goto write_fallback;

    {
        CUfileDescr_t desc{};
        desc.handle.fd = fd;
        desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

        CUfileHandle_t fh{};
        CUfileError_t err = cuFileHandleRegister(&fh, &desc);
        if (err.err != CU_FILE_SUCCESS) {
            ::close(fd);
            goto write_fallback;
        }

#if defined(CUFILE_VERSION) && CUFILE_VERSION >= 11080
        ssize_t ret = cuFileWriteAsync(fh, gpu_ptr, &size,
                                        file_offset, 0,
                                        nullptr, nullptr, stream);
        if (ret < 0) {
            cuFileHandleDeregister(fh);
            ::close(fd);
            goto write_fallback;
        }
        if (stream) {
            MEMOPT_CUDA_LOG(cudaStreamSynchronize(stream));
        }
#else
        static std::once_flag sync_warn_w;
        std::call_once(sync_warn_w, []() {
            std::fprintf(stderr,
                "memopt: cuFileWriteAsync unavailable "
                "(CUDA < 11.8), using synchronous cuFileWrite\n");
        });
        ssize_t ret = cuFileWrite(fh, gpu_ptr, size, file_offset, 0);
#endif

        cuFileHandleDeregister(fh);
        ::close(fd);

        if (ret >= 0 && static_cast<size_t>(ret) == size) {
            return;
        }
    }

write_fallback:
#endif

    // Fallback: synchronous copy through CPU
#ifdef MEMOPT_CUDA_AVAILABLE
    auto* cpu_buf = static_cast<char*>(::malloc(size));
    if (!cpu_buf) return;

    if (stream) {
        MEMOPT_CUDA_LOG(cudaMemcpyAsync(cpu_buf, gpu_ptr, size,
                        cudaMemcpyDeviceToHost, stream));
        MEMOPT_CUDA_LOG(cudaStreamSynchronize(stream));
    } else {
        MEMOPT_CUDA_LOG(cudaMemcpy(cpu_buf, gpu_ptr, size, cudaMemcpyDeviceToHost));
    }

    write_block_atomic(path, cpu_buf, size);
    ::free(cpu_buf);
#else
    (void)stream;
    write_block_atomic(path, gpu_ptr, size);
#endif
}

} // namespace cuda_backend
} // namespace memopt
