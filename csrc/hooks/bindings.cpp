// bindings.cpp — pybind11 surface for _memopt_hooks.
//
// Exposes the C++ dispatch table, FNV-1a key generation, and atomic
// counters to Python. The hot-path hook functions (apply_rope, etc.)
// live in the Python shim which calls into these C++ primitives for
// the performance-critical inner operations.
//
// GIL discipline:
//   - notify():           no GIL needed (pure atomic)
//   - make_cache_key():   no GIL needed (pure arithmetic)
//   - register_kernel():  GIL held by caller (PyObject* refcount)
//   - invalidate_kernel():GIL held by caller (PyObject* refcount)
//   - hook_stats():       GIL released for C++ read, re-acquired for dict

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dispatch_table.h"
#include "key_hash.h"

namespace py = pybind11;
using namespace memopt::hooks;

// Python callback for synthesis signaling.
// Set by AutoOptimizer at startup via set_synthesis_callback().
static py::object synthesis_callback;

// Python references for cache and optimizer (kept alive for stats).
static py::object py_kernel_cache;
static py::object py_auto_optimizer;

PYBIND11_MODULE(_memopt_hooks, m) {
    m.doc() = "memopt kernel hooks — C++ dispatch table with "
              "FNV-1a keys and lock-free atomic counters";

    // ── Initialization ───────────────────────────────────────────────
    m.def("init_hooks",
        [](int arch_id, py::object cache, py::object optimizer) {
            current_arch = static_cast<ArchId>(arch_id);
            py_kernel_cache = cache;
            py_auto_optimizer = optimizer;
            init_dispatch_table(current_arch);
        },
        py::arg("arch_id"),
        py::arg("cache"),
        py::arg("optimizer"),
        "Initialize dispatch table with architecture ID and Python objects.");

    // ── Synthesis signal callback ────────────────────────────────────
    m.def("set_synthesis_callback",
        [](py::object cb) { synthesis_callback = std::move(cb); },
        py::arg("callback"),
        "Set callback invoked when synthesis threshold is crossed.");

    // ── Atomic notify — replaces AutoOptimizer.notify() hot path ─────
    m.def("notify",
        [](int op_id) -> bool {
            if (op_id < 0 || op_id >= static_cast<int>(OpId::NUM_OPS))
                return false;
            bool should_fire = notify_cpp(static_cast<uint8_t>(op_id));
            if (should_fire && synthesis_callback) {
                // Signal Python synthesis thread
                synthesis_callback(op_id);
            }
            return should_fire;
        },
        py::arg("op_id"),
        "Atomic counter increment. Returns True if synthesis should fire.");

    // ── FNV-1a cache key generation ──────────────────────────────────
    m.def("make_cache_key",
        [](int op, int arch, int batch, int seq, int heads, int dim)
        -> uint64_t {
            return make_cache_key(
                static_cast<OpId>(op),
                static_cast<ArchId>(arch),
                static_cast<uint16_t>(batch),
                static_cast<uint16_t>(seq),
                static_cast<uint16_t>(heads),
                static_cast<uint16_t>(dim));
        },
        py::arg("op"), py::arg("arch"),
        py::arg("batch"), py::arg("seq"),
        py::arg("heads"), py::arg("dim"),
        "Generate FNV-1a cache key — zero allocation, ~3 CPU instructions.");

    // ── Kernel registration (called by KernelCache.put) ──────────────
    m.def("register_kernel",
        [](int op_id, uint64_t shape_key,
           py::object callable, uint64_t version) {
            if (op_id < 0 || op_id >= static_cast<int>(OpId::NUM_OPS))
                return;
            // Increment refcount — we're storing a borrowed reference.
            PyObject* ptr = callable.ptr();
            Py_INCREF(ptr);
            register_kernel(static_cast<uint8_t>(op_id),
                            shape_key, ptr, version);
        },
        py::arg("op_id"), py::arg("shape_key"),
        py::arg("callable"), py::arg("version") = 0,
        "Register a synthesised kernel. GIL must be held.");

    m.def("invalidate_kernel",
        [](int op_id, uint64_t shape_key) {
            if (op_id < 0 || op_id >= static_cast<int>(OpId::NUM_OPS))
                return;
            // Decrement refcount of old callable before clearing.
            void* old = nullptr;
            if (shape_key == 0) {
                old = base_entries[op_id].callable.load(
                    std::memory_order_acquire);
            }
            invalidate_kernel(static_cast<uint8_t>(op_id), shape_key);
            if (old) Py_DECREF(reinterpret_cast<PyObject*>(old));
        },
        py::arg("op_id"), py::arg("shape_key") = 0,
        "Invalidate a registered kernel. GIL must be held.");

    // ── Stats ────────────────────────────────────────────────────────
    m.def("hook_stats",
        []() -> py::dict {
            DispatchStats s;
            {
                py::gil_scoped_release release;
                s = get_dispatch_stats();
            }
            py::dict result;

            py::dict call_counts;
            py::dict fallback_counts;
            py::dict error_counts;
            const char* op_names[] = {
                "rope", "layer_norm_residual", "scaled_softmax"
            };
            for (int i = 0; i < 3; ++i) {
                call_counts[op_names[i]] = s.call_counts[i];
                fallback_counts[op_names[i]] = s.fallback_counts[i];
                error_counts[op_names[i]] = s.error_counts[i];
            }

            result["call_counts"] = call_counts;
            result["fallback_counts"] = fallback_counts;
            result["error_counts"] = error_counts;
            result["shape_cache_size"] = s.shape_cache_size;
            result["hooks_initialised"] = true;

            // Include Python cache/optimizer stats if available
            if (!py_kernel_cache.is_none()) {
                try {
                    result["cache"] = py_kernel_cache.attr("stats")();
                } catch (...) {}
            }
            if (!py_auto_optimizer.is_none()) {
                try {
                    result["optimizer"] = py_auto_optimizer.attr("stats")();
                } catch (...) {}
            }

            return result;
        },
        "Return dispatch table stats for Grafana/monitoring.");

    m.def("reset_counters",
        []() {
            py::gil_scoped_release release;
            reset_all_counters();
        },
        "Reset all counters and entries. For testing only.");

    // ── Architecture detection ───────────────────────────────────────
    m.def("detect_arch_id",
        []() -> int {
            // Query torch.cuda for compute capability
            try {
                py::module_ torch = py::module_::import("torch");
                py::object cuda = torch.attr("cuda");
                if (cuda.attr("is_available")().cast<bool>()) {
                    py::tuple cap = cuda.attr("get_device_capability")(0);
                    int major = cap[0].cast<int>();
                    int minor = cap[1].cast<int>();
                    if (major >= 10)
                        return static_cast<int>(ArchId::BLACKWELL);
                    if (major == 9)
                        return static_cast<int>(ArchId::HOPPER);
                    if (major == 8 && minor == 9)
                        return static_cast<int>(ArchId::ADA);
                    if (major >= 8)
                        return static_cast<int>(ArchId::AMPERE);
                }
            } catch (...) {}
            return static_cast<int>(ArchId::UNKNOWN);
        },
        "Detect GPU architecture from torch.cuda. "
        "Returns ArchId int (0=ampere, 1=hopper, 2=ada, 3=blackwell, "
        "4=unknown).");

    // ── Op/Arch enums exposed for Python interop ─────────────────────
    py::enum_<OpId>(m, "OpId")
        .value("ROPE", OpId::ROPE)
        .value("LAYER_NORM_RESIDUAL", OpId::LAYER_NORM_RESIDUAL)
        .value("SCALED_SOFTMAX", OpId::SCALED_SOFTMAX)
        .export_values();

    py::enum_<ArchId>(m, "ArchId")
        .value("AMPERE", ArchId::AMPERE)
        .value("HOPPER", ArchId::HOPPER)
        .value("ADA", ArchId::ADA)
        .value("BLACKWELL", ArchId::BLACKWELL)
        .value("UNKNOWN", ArchId::UNKNOWN)
        .export_values();
}
