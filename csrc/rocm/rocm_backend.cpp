// memopt AMD ROCm / HIP backend.
//
// Two compile modes, chosen by `-DMEMOPT_ROCM_AVAILABLE`:
//
//   1. Real build (HIP + ROCm installed): full pybind11 module that
//      wraps hipMalloc / hipFree / hipMemcpyAsync / hipDeviceProperties.
//      Targets: gfx90a (MI210/MI250), gfx940/941/942 (MI300X).
//
//   2. Stub build (CUDA-only host or no ROCm): compiles the same
//      module name with a single `ROCM_AVAILABLE = False` attribute
//      so `import memopt._memopt_rocm` never fails on NVIDIA / macOS
//      build machines.
//
// REQUIRES AMD HARDWARE to exercise the real path. The stub path is
// what actually runs in CI on non-AMD hosts.

#ifdef MEMOPT_ROCM_AVAILABLE

#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// ─────────────────────────────────────────────
// Error checking
// ─────────────────────────────────────────────

#define HIP_CHECK(call)                                              \
    do {                                                             \
        hipError_t err = (call);                                     \
        if (err != hipSuccess) {                                     \
            throw std::runtime_error(                                \
                std::string("HIP error: ")                           \
                + hipGetErrorString(err)                             \
                + " at " __FILE__ ":"                                \
                + std::to_string(__LINE__));                         \
        }                                                            \
    } while (0)

// ─────────────────────────────────────────────
// Stream pool (reused across calls)
// ─────────────────────────────────────────────

class ROCmStreamPool {
public:
    explicit ROCmStreamPool(int n = 4) : streams_(n) {
        for (auto& s : streams_) {
            HIP_CHECK(hipStreamCreate(&s));
        }
    }

    ~ROCmStreamPool() {
        for (auto& s : streams_) {
            hipStreamDestroy(s);
        }
    }

    hipStream_t get(int idx) const {
        return streams_[static_cast<size_t>(idx)
                        % streams_.size()];
    }

    int size() const {
        return static_cast<int>(streams_.size());
    }

private:
    std::vector<hipStream_t> streams_;
};

static ROCmStreamPool g_stream_pool(4);

// ─────────────────────────────────────────────
// Device info
// ─────────────────────────────────────────────

struct ROCmDeviceInfo {
    int         index;
    std::string name;
    size_t      total_memory;
    size_t      free_memory;
    std::string gcn_arch;
    bool        has_unified_memory;
};

static bool is_unified_arch(const std::string& arch) {
    // MI300X = gfx940 / gfx941 / gfx942 — unified CPU+GPU memory.
    // MI250X (gfx90a) is discrete HBM; MI210 likewise.
    return arch.find("gfx94") != std::string::npos;
}

static ROCmDeviceInfo get_device_info(int device_id) {
    HIP_CHECK(hipSetDevice(device_id));

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device_id));

    size_t free_mem = 0, total_mem = 0;
    HIP_CHECK(hipMemGetInfo(&free_mem, &total_mem));

    ROCmDeviceInfo info;
    info.index              = device_id;
    info.name               = props.name;
    info.total_memory       = total_mem;
    info.free_memory        = free_mem;
    info.gcn_arch           = props.gcnArchName;
    info.has_unified_memory =
        is_unified_arch(info.gcn_arch);
    return info;
}

static bool is_unified_memory_device(int device_id) {
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, device_id) != hipSuccess) {
        return false;
    }
    return is_unified_arch(props.gcnArchName);
}

// ─────────────────────────────────────────────
// Memory operations
// ─────────────────────────────────────────────

static py::bytes hip_alloc(size_t size_bytes) {
    void* ptr = nullptr;
    HIP_CHECK(hipMalloc(&ptr, size_bytes));
    return py::bytes(reinterpret_cast<char*>(&ptr),
                     sizeof(void*));
}

static void hip_free(const py::bytes& ptr_bytes) {
    void* ptr = nullptr;
    std::memcpy(&ptr,
                ptr_bytes.cast<std::string>().data(),
                sizeof(void*));
    HIP_CHECK(hipFree(ptr));
}

static py::bytes hip_alloc_host(size_t size_bytes) {
    void* ptr = nullptr;
    HIP_CHECK(hipHostMalloc(&ptr, size_bytes,
                            hipHostMallocDefault));
    return py::bytes(reinterpret_cast<char*>(&ptr),
                     sizeof(void*));
}

static void hip_free_host(const py::bytes& ptr_bytes) {
    void* ptr = nullptr;
    std::memcpy(&ptr,
                ptr_bytes.cast<std::string>().data(),
                sizeof(void*));
    HIP_CHECK(hipHostFree(ptr));
}

// ─────────────────────────────────────────────
// Async copies
// ─────────────────────────────────────────────

static void hip_memcpy_h2d_async(
    const py::bytes& dst_bytes,
    const py::bytes& src_bytes,
    size_t           size_bytes,
    int              stream_idx)
{
    // On MI300X unified memory this is effectively a no-op at
    // the DMA level — HIP recognizes unified pages and returns
    // immediately. On discrete GPUs (MI250X, RX 7900) this is a
    // real PCIe DMA. Either way we route through a pooled stream.
    void* dst = nullptr;
    void* src = nullptr;
    std::memcpy(&dst,
                dst_bytes.cast<std::string>().data(),
                sizeof(void*));
    std::memcpy(&src,
                src_bytes.cast<std::string>().data(),
                sizeof(void*));

    HIP_CHECK(hipMemcpyAsync(dst, src, size_bytes,
                             hipMemcpyHostToDevice,
                             g_stream_pool.get(stream_idx)));
}

static void hip_memcpy_d2h_async(
    const py::bytes& dst_bytes,
    const py::bytes& src_bytes,
    size_t           size_bytes,
    int              stream_idx)
{
    void* dst = nullptr;
    void* src = nullptr;
    std::memcpy(&dst,
                dst_bytes.cast<std::string>().data(),
                sizeof(void*));
    std::memcpy(&src,
                src_bytes.cast<std::string>().data(),
                sizeof(void*));

    HIP_CHECK(hipMemcpyAsync(dst, src, size_bytes,
                             hipMemcpyDeviceToHost,
                             g_stream_pool.get(stream_idx)));
}

// ─────────────────────────────────────────────
// Synchronization
// ─────────────────────────────────────────────

static void hip_device_synchronize() {
    HIP_CHECK(hipDeviceSynchronize());
}

static void hip_stream_synchronize(int stream_idx) {
    HIP_CHECK(hipStreamSynchronize(
        g_stream_pool.get(stream_idx)));
}

// ─────────────────────────────────────────────
// Bindings
// ─────────────────────────────────────────────

PYBIND11_MODULE(_memopt_rocm, m) {
    m.doc() = "memopt AMD ROCm backend";

    m.attr("ROCM_AVAILABLE") = true;

    py::class_<ROCmDeviceInfo>(m, "ROCmDeviceInfo")
        .def_readonly("index",         &ROCmDeviceInfo::index)
        .def_readonly("name",          &ROCmDeviceInfo::name)
        .def_readonly("total_memory",  &ROCmDeviceInfo::total_memory)
        .def_readonly("free_memory",   &ROCmDeviceInfo::free_memory)
        .def_readonly("gcn_arch",      &ROCmDeviceInfo::gcn_arch)
        .def_readonly("has_unified_memory",
                      &ROCmDeviceInfo::has_unified_memory);

    m.def("get_device_info", &get_device_info,
          "Get device info for a ROCm GPU");

    m.def("hip_alloc",          &hip_alloc,
          "Allocate GPU memory");
    m.def("hip_free",           &hip_free,
          "Free GPU memory");
    m.def("hip_alloc_host",     &hip_alloc_host,
          "Allocate pinned host memory");
    m.def("hip_free_host",      &hip_free_host,
          "Free pinned host memory");
    m.def("hip_memcpy_h2d_async", &hip_memcpy_h2d_async,
          "Async host-to-device copy (stream-ordered)");
    m.def("hip_memcpy_d2h_async", &hip_memcpy_d2h_async,
          "Async device-to-host copy (stream-ordered)");
    m.def("hip_device_synchronize", &hip_device_synchronize,
          "Synchronize device");
    m.def("hip_stream_synchronize", &hip_stream_synchronize,
          "Synchronize one pooled stream");
    m.def("is_unified_memory_device", &is_unified_memory_device,
          "True for MI300X-class unified memory");

    m.def("rocm_version", []() -> std::string {
        return std::to_string(HIP_VERSION_MAJOR)
               + "." + std::to_string(HIP_VERSION_MINOR);
    });
}

#else
// ─────────────────────────────────────────────
// Stub module — no ROCm at build time
// ─────────────────────────────────────────────
//
// Compiled on CUDA-only hosts, macOS, etc. The import is guaranteed
// to succeed; Python callers should check `ROCM_AVAILABLE`.

#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(_memopt_rocm, m) {
    m.doc() = "memopt ROCm backend (stub — "
              "ROCm not available at build time)";
    m.attr("ROCM_AVAILABLE") = false;
}
#endif  // MEMOPT_ROCM_AVAILABLE
