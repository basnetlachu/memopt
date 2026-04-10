// bindings.cpp — pybind11 surface for _memopt_simd.
//
// Exposes find_lcp() with two interfaces:
//   1. find_lcp(list, list) — converts Python lists to int32_t vectors
//   2. find_lcp_buffer(ndarray, ndarray) — zero-copy for numpy int32 arrays
//
// GIL: released during all SIMD computation (pure C, no Python objects).

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "prefix_match.h"

#include <cstdint>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace memopt::simd;

// ═══════════════════════════════════════════════════════════════════════════
// Helper: convert Python list[int] to vector<int32_t>
// ═══════════════════════════════════════════════════════════════════════════
static std::vector<int32_t> to_int32_vec(py::object obj) {
    // Try buffer protocol first (numpy arrays)
    try {
        py::buffer buf = obj.cast<py::buffer>();
        py::buffer_info info = buf.request();
        if (info.ndim == 1 &&
            info.format == py::format_descriptor<int32_t>::format()) {
            const auto* ptr = static_cast<const int32_t*>(info.ptr);
            return std::vector<int32_t>(ptr, ptr + info.size);
        }
    } catch (...) {}

    // Fall back to Python list iteration
    py::list lst = obj.cast<py::list>();
    std::vector<int32_t> v;
    v.reserve(py::len(lst));
    for (auto item : lst) {
        v.push_back(item.cast<int32_t>());
    }
    return v;
}

// ═══════════════════════════════════════════════════════════════════════════
// Module definition
// ═══════════════════════════════════════════════════════════════════════════

PYBIND11_MODULE(_memopt_simd, m) {
    m.doc() = "memopt SIMD — AVX-512/AVX2/scalar prefix matching";

    // ── Primary interface: accepts list[int] or numpy int32 ──────────
    m.def("find_lcp",
        [](py::object a, py::object b) -> size_t {
            std::vector<int32_t> va = to_int32_vec(a);
            std::vector<int32_t> vb = to_int32_vec(b);
            size_t len = std::min(va.size(), vb.size());
            if (len == 0) return 0;
            py::gil_scoped_release no_gil;
            return find_lcp(va.data(), vb.data(), len);
        },
        py::arg("a"), py::arg("b"),
        "Find length of longest common prefix of two token sequences. "
        "Accepts list[int] or numpy int32 arrays. "
        "GIL released during computation.");

    // ── Buffer protocol: zero-copy for numpy arrays ──────────────────
    m.def("find_lcp_buffer",
        [](py::buffer a, py::buffer b) -> size_t {
            py::buffer_info ai = a.request();
            py::buffer_info bi = b.request();

            if (ai.format != py::format_descriptor<int32_t>::format() ||
                bi.format != py::format_descriptor<int32_t>::format()) {
                throw py::type_error(
                    "find_lcp_buffer requires int32 arrays");
            }
            if (ai.ndim != 1 || bi.ndim != 1) {
                throw py::value_error(
                    "find_lcp_buffer requires 1D arrays");
            }

            size_t len = std::min(
                static_cast<size_t>(ai.size),
                static_cast<size_t>(bi.size));
            if (len == 0) return 0;

            const auto* pa = static_cast<const int32_t*>(ai.ptr);
            const auto* pb = static_cast<const int32_t*>(bi.ptr);

            py::gil_scoped_release no_gil;
            return find_lcp(pa, pb, len);
        },
        py::arg("a"), py::arg("b"),
        "Zero-copy version for numpy int32 arrays.");

    // ── ISA detection ────────────────────────────────────────────────
    m.def("detected_isa", []() -> std::string {
        switch (detect_best_isa()) {
            case ISA::AVX512: return "avx512";
            case ISA::AVX2:   return "avx2";
            default:          return "scalar";
        }
    }, "Return name of best SIMD ISA detected on this CPU.");

    m.def("has_avx512", []() -> bool {
        return detect_best_isa() == ISA::AVX512;
    }, "True if AVX-512 is available on this CPU.");

    m.def("has_avx2", []() -> bool {
        ISA isa = detect_best_isa();
        return isa == ISA::AVX2 || isa == ISA::AVX512;
    }, "True if AVX2 (or better) is available on this CPU.");
}
