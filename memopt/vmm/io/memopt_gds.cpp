/*
 * memopt_gds.cpp
 *
 * GPUDirect Storage (GDS) block reader.
 * NVMe → GPU HBM with zero CPU copies via NVIDIA cuFile API.
 *
 * This file compiles in two modes:
 *
 * WITH GDS (cuFile.h present + nvidia-fs kernel module loaded):
 *   Exports memopt_gds_available()=1 and a working memopt_gds_read().
 *   Requires: CUDA toolkit, nvidia-fs, A100/H100/H200 class GPU.
 *
 * WITHOUT GDS (default on all other hardware):
 *   Exports memopt_gds_available()=0 and a stub memopt_gds_read()
 *   that returns -ENOTSUP. Python caller uses io_uring instead.
 *   No CUDA dependency. Compiles with plain g++.
 *
 * Build (auto-detected by build.sh):
 *   WITH:    nvcc -O3 -shared -Xcompiler -fPIC -lcufile \
 *                 -o libmemopt_gds.so memopt_gds.cpp
 *   WITHOUT: g++ -O3 -std=c++17 -shared -fPIC \
 *                 -o libmemopt_gds.so memopt_gds.cpp
 *
 * The Python caller never needs to know which mode is active.
 * memopt_gds_available() is the single source of truth.
 */

#include <cerrno>
#include <cstddef>

#ifdef __has_include
#  if __has_include(<cufile.h>)
#    include <cufile.h>
#    include <cuda_runtime.h>
#    define MEMOPT_HAVE_GDS 1
#  endif
#endif

#include <fcntl.h>
#include <unistd.h>

extern "C" {

int memopt_gds_available(void) {
#ifdef MEMOPT_HAVE_GDS
    return 1;
#else
    return 0;
#endif
}

/*
 * memopt_gds_read
 *
 * Reads `size` bytes from the file at `path` directly into the
 * CUDA device buffer pointed to by `device_ptr`.
 *
 * `device_ptr` must be a cudaMalloc'd pointer.
 * Returns 0 on success, -errno or -EIO on failure, -ENOTSUP if
 * GDS is not compiled in.
 */
int memopt_gds_read(const char* path, void* device_ptr, size_t size) {
#ifdef MEMOPT_HAVE_GDS
    if (!path || !device_ptr || size == 0) return -EINVAL;

    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) return -errno;

    CUfileDescr_t descr = {};
    descr.handle.fd     = fd;
    descr.type          = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    CUfileHandle_t handle;
    CUfileError_t  status = cuFileHandleRegister(&handle, &descr);
    if (status.err != CU_FILE_SUCCESS) {
        close(fd);
        return -EIO;
    }

    /* Register the GPU buffer with the cuFile driver */
    status = cuFileBufRegister(device_ptr, size, 0);
    if (status.err != CU_FILE_SUCCESS) {
        cuFileHandleDeregister(handle);
        close(fd);
        return -EIO;
    }

    ssize_t n = cuFileRead(handle, device_ptr, size, 0, 0);

    cuFileBufDeregister(device_ptr);
    cuFileHandleDeregister(handle);
    close(fd);

    if (n == (ssize_t)size) return 0;
    return -EIO;

#else
    /* GDS not compiled in */
    (void)path; (void)device_ptr; (void)size;
    return -ENOTSUP;
#endif
}

} /* extern "C" */
