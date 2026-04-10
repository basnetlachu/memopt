// bindings.cpp — pybind11 surface for _memopt_cuda.
//
// Exposes: stream pool, NVMe I/O, GDS availability, HBM stats.
//
// GIL discipline:
//   - write_block_atomic: RELEASE (POSIX syscalls)
//   - read_block:         RELEASE (POSIX syscalls)
//   - recover_nvme_dir:   RELEASE (directory walk)
//   - sync_all:           RELEASE (CUDA sync can block)
//   - gds_is_available:   no GIL needed (fast check)
//   - hbm_used_bytes:     no GIL needed (CUDA API)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gds.h"
#include "nvme_io.h"
#include "stream_pool.h"

#include <vector>

namespace py = pybind11;
using namespace memopt::cuda_backend;

PYBIND11_MODULE(_memopt_cuda, m) {
    m.doc() = "memopt CUDA backend — stream pool, NVMe I/O, GDS";

    // ── Stream pool ──────────────────────────────────────────────────
    py::class_<CUDAStreamPool>(m, "CUDAStreamPool")
        .def(py::init<int>(), py::arg("device_id") = 0)
        .def("is_available", &CUDAStreamPool::is_available)
        .def("sync_all", [](CUDAStreamPool& self) {
            py::gil_scoped_release release;
            self.sync_all();
        })
        .def("device_id", &CUDAStreamPool::device_id)
        .def("stats", [](const CUDAStreamPool& self) {
            py::dict d;
            d["is_available"] = self.is_available();
            d["device_id"] = self.device_id();
            d["num_streams"] = CUDAStreamPool::NUM_STREAMS;
            py::list counts;
            for (auto c : self.use_counts()) counts.append(c);
            d["use_counts"] = counts;
            return d;
        });

    m.def("init_stream_pool", &init_stream_pool,
          py::arg("device_id") = 0,
          "Initialize the global stream pool singleton.");

    m.def("shutdown_stream_pool", &shutdown_stream_pool,
          "Destroy the global stream pool.");

    // ── NVMe I/O ─────────────────────────────────────────────────────
    m.def("write_block_atomic",
        [](const std::string& path, py::bytes data) -> py::dict {
            std::string buf = data;  // copy bytes to std::string
            NVMeWriteResult result;
            {
                py::gil_scoped_release release;
                result = write_block_atomic(path, buf.data(), buf.size());
            }
            py::dict d;
            d["success"] = result.success;
            d["errno_val"] = result.errno_val;
            d["bytes_written"] = result.bytes_written;
            return d;
        },
        py::arg("path"), py::arg("data"),
        "Atomic write: data → .tmp → fdatasync → rename. GIL released.");

    m.def("read_block",
        [](const std::string& path, int64_t size) -> py::object {
            std::vector<char> buf(size);
            NVMeReadResult result;
            {
                py::gil_scoped_release release;
                result = read_block(path, buf.data(),
                                     static_cast<size_t>(size));
            }
            if (!result.success) return py::none();
            return py::bytes(buf.data(), result.bytes_read);
        },
        py::arg("path"), py::arg("size"),
        "Read block from file. Returns bytes or None on error.");

    m.def("recover_nvme_dir",
        [](const std::string& dir) -> int {
            py::gil_scoped_release release;
            return recover_nvme_dir(dir);
        },
        py::arg("dir_path"),
        "Remove .vmm_block.tmp files. Returns count removed.");

    // ── GDS ──────────────────────────────────────────────────────────
    m.def("gds_is_available", &gds_is_available,
          "True if cuFile GDS driver is loaded and SDK available.");

    m.def("gds_compiled", []() -> bool {
        #ifdef MEMOPT_GDS_AVAILABLE
        return true;
        #else
        return false;
        #endif
    }, "True if compiled with cuFile GDS support.");

    // ── Capability flags ─────────────────────────────────────────────
    m.def("cuda_is_available", []() -> bool {
        #ifdef MEMOPT_CUDA_AVAILABLE
        return true;
        #else
        return false;
        #endif
    }, "True if compiled with CUDA support.");

    m.def("hbm_used_bytes", []() -> int64_t {
        #ifdef MEMOPT_CUDA_AVAILABLE
        size_t free_mem = 0, total_mem = 0;
        if (cudaMemGetInfo(&free_mem, &total_mem) != cudaSuccess)
            return -1;
        return static_cast<int64_t>(total_mem - free_mem);
        #else
        return 0;
        #endif
    }, "HBM bytes currently in use. 0 on CPU-only, -1 on CUDA error.");
}
