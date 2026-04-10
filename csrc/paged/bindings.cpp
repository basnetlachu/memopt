// bindings.cpp — pybind11 surface for _memopt_paged.
//
// Architecture: C++ manages block allocation and sequence metadata.
// Python manages the GPU tensor storage (k_blocks, v_blocks) as PyTorch
// tensors. The bindings coordinate: C++ decides block_id/offset,
// Python writes/reads the tensor at that position.
//
// This split exists because:
// 1. PyTorch tensors are Python objects — moving them to C++ requires
//    libtorch linkage, which complicates the build for CPU-only environments.
// 2. The performance bottleneck is the lock (replaced by C++ block pool),
//    not the tensor indexing (already optimized by PyTorch's C++ backend).
// 3. Tests run without GPU — tensor storage must work on CPU tensors.
//
// GIL discipline:
//   - allocate_sequence/free_sequence: GIL released (pure C++)
//   - store(): GIL released for block lookup, re-acquired for tensor write
//   - fetch(): GIL released for block info, re-acquired for tensor read

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "block_pool.h"
#include "gather_kernel.cuh"
#include "paged_kv_cache.h"

namespace py = pybind11;
using namespace memopt::paged;

// ═══════════════════════════════════════════════════════════════════════════
// PyPagedKVCache — Python-facing wrapper that owns both C++ metadata
// and Python tensor storage
// ═══════════════════════════════════════════════════════════════════════════

class PyPagedKVCache {
public:
    PyPagedKVCache(int32_t num_blocks,
                    int32_t num_layers,
                    int32_t num_heads,
                    int32_t head_dim,
                    py::object dtype_obj,
                    py::str device_str)
        : cache_(num_blocks, /*block_size=*/16, num_heads, head_dim, num_layers),
          num_blocks_(num_blocks),
          block_size_(16),
          num_heads_(num_heads),
          head_dim_(head_dim),
          num_layers_(num_layers)
    {
        // Import torch and allocate storage tensors
        py::module_ torch = py::module_::import("torch");
        py::object zeros = torch.attr("zeros");

        // Default dtype: torch.float16
        py::object dtype = dtype_obj;
        if (dtype.is_none()) {
            dtype = torch.attr("float16");
        }

        std::string device = device_str.cast<std::string>();

        // Allocate: (num_blocks, BLOCK_SIZE, num_heads, head_dim)
        py::tuple shape = py::make_tuple(
            num_blocks, block_size_, num_heads, head_dim);

        k_blocks_ = zeros(shape,
            py::arg("dtype") = dtype,
            py::arg("device") = device);
        v_blocks_ = zeros(shape,
            py::arg("dtype") = dtype,
            py::arg("device") = device);
    }

    // ── Python API matching paged_attention.py exactly ────────────────

    py::object allocate_sequence(const std::string& seq_id) {
        {
            py::gil_scoped_release release;
            cache_.allocate_sequence(seq_id);
        }
        // Return a SequenceState-like object for API compatibility.
        // The Python code does: seq = cache.allocate_sequence("req_001")
        // then passes `seq` to store/fetch (but store/fetch only use seq_id).
        // We return a simple namespace with seq_id and properties.
        py::module_ types = py::module_::import("types");
        py::object ns = types.attr("SimpleNamespace")(
            py::arg("seq_id") = seq_id,
            py::arg("block_ids") = py::list(),
            py::arg("current_pos") = 0
        );
        return ns;
    }

    void free_sequence(py::object seq_or_id) {
        std::string seq_id;
        if (py::isinstance<py::str>(seq_or_id)) {
            seq_id = seq_or_id.cast<std::string>();
        } else {
            seq_id = seq_or_id.attr("seq_id").cast<std::string>();
        }
        py::gil_scoped_release release;
        cache_.free_sequence(seq_id);
    }

    void store(py::object seq_or_id,
               int32_t layer_idx,
               int32_t token_pos,
               py::object k,
               py::object v) {
        std::string seq_id;
        if (py::isinstance<py::str>(seq_or_id)) {
            seq_id = seq_or_id.cast<std::string>();
        } else {
            seq_id = seq_or_id.attr("seq_id").cast<std::string>();
        }

        // Get or allocate block (GIL released during C++ work)
        PagedKVCache::BlockLocation loc;
        {
            py::gil_scoped_release release;
            loc = cache_.get_or_allocate_block(seq_id, token_pos);
            cache_.update_pos(seq_id, token_pos);
        }

        // Write tensor data (GIL held — PyTorch tensor indexing)
        k_blocks_.attr("__setitem__")(
            py::make_tuple(loc.block_id, loc.block_offset), k);
        v_blocks_.attr("__setitem__")(
            py::make_tuple(loc.block_id, loc.block_offset), v);
    }

    py::tuple fetch(py::object seq_or_id, int32_t layer_idx) {
        std::string seq_id;
        if (py::isinstance<py::str>(seq_or_id)) {
            seq_id = seq_or_id.cast<std::string>();
        } else {
            seq_id = seq_or_id.attr("seq_id").cast<std::string>();
        }

        // Get block info (GIL released)
        PagedKVCache::FetchInfo info;
        {
            py::gil_scoped_release release;
            info = cache_.get_fetch_info(seq_id);
        }

        py::module_ torch = py::module_::import("torch");

        // Empty sequence
        if (info.current_pos == 0) {
            py::object empty = torch.attr("zeros")(
                py::make_tuple(0, num_heads_, head_dim_),
                py::arg("device") = k_blocks_.attr("device"),
                py::arg("dtype") = k_blocks_.attr("dtype"));
            return py::make_tuple(empty, empty);
        }

        // ── CUDA gather kernel path ──────────────────────────────────
        // Conditions: gather kernel compiled + tensors on CUDA device.
        // Single kernel launch replaces N torch.cat() calls.
        bool use_gather = false;
        if (has_cuda_gather()) {
            py::object dev = k_blocks_.attr("device");
            py::object dev_type = dev.attr("type");
            use_gather = (dev_type.cast<std::string>() == "cuda");
        }

        if (use_gather) {
            try {
                int32_t num_blocks_needed =
                    static_cast<int32_t>(info.block_ids.size());
                int32_t max_seq = num_blocks_needed * block_size_;

                // Build block_ids tensor on GPU: shape [1, num_blocks_needed]
                // Padded with -1 for unused slots (none here — all used).
                py::object ids_cpu = torch.attr("tensor")(
                    py::cast(info.block_ids),
                    py::arg("dtype") = torch.attr("int32"));
                py::object ids_gpu = ids_cpu.attr("unsqueeze")(0).attr("to")(
                    k_blocks_.attr("device"));

                // Allocate output tensors: [1, current_pos, num_heads, head_dim]
                py::object k_out = torch.attr("empty")(
                    py::make_tuple(1, info.current_pos, num_heads_, head_dim_),
                    py::arg("dtype") = k_blocks_.attr("dtype"),
                    py::arg("device") = k_blocks_.attr("device"));
                py::object v_out = torch.attr("empty")(
                    py::make_tuple(1, info.current_pos, num_heads_, head_dim_),
                    py::arg("dtype") = k_blocks_.attr("dtype"),
                    py::arg("device") = k_blocks_.attr("device"));

                // Get raw data pointers via .data_ptr()
                auto k_storage_ptr = k_blocks_.attr("data_ptr")().cast<uintptr_t>();
                auto v_storage_ptr = v_blocks_.attr("data_ptr")().cast<uintptr_t>();
                auto ids_ptr = ids_gpu.attr("data_ptr")().cast<uintptr_t>();
                auto k_out_ptr = k_out.attr("data_ptr")().cast<uintptr_t>();
                auto v_out_ptr = v_out.attr("data_ptr")().cast<uintptr_t>();

                // Detect dtype for template dispatch
                std::string dtype_str =
                    py::str(k_blocks_.attr("dtype")).cast<std::string>();

                // Get current CUDA stream (0 = default stream)
                void* stream = nullptr;

                // Release GIL during kernel launch
                {
                    py::gil_scoped_release release;

                    auto ids_raw = reinterpret_cast<const int32_t*>(ids_ptr);

#ifdef MEMOPT_CUDA_AVAILABLE
                    if (dtype_str.find("float16") != std::string::npos) {
                        using T = __half;
                        launch_gather_kv_blocks<T>(
                            reinterpret_cast<const T*>(k_storage_ptr),
                            ids_raw, reinterpret_cast<T*>(k_out_ptr),
                            1, info.current_pos, num_heads_, head_dim_,
                            block_size_, num_blocks_needed, stream);
                        launch_gather_kv_blocks<T>(
                            reinterpret_cast<const T*>(v_storage_ptr),
                            ids_raw, reinterpret_cast<T*>(v_out_ptr),
                            1, info.current_pos, num_heads_, head_dim_,
                            block_size_, num_blocks_needed, stream);
                    } else if (dtype_str.find("bfloat16") != std::string::npos) {
                        using T = __nv_bfloat16;
                        launch_gather_kv_blocks<T>(
                            reinterpret_cast<const T*>(k_storage_ptr),
                            ids_raw, reinterpret_cast<T*>(k_out_ptr),
                            1, info.current_pos, num_heads_, head_dim_,
                            block_size_, num_blocks_needed, stream);
                        launch_gather_kv_blocks<T>(
                            reinterpret_cast<const T*>(v_storage_ptr),
                            ids_raw, reinterpret_cast<T*>(v_out_ptr),
                            1, info.current_pos, num_heads_, head_dim_,
                            block_size_, num_blocks_needed, stream);
                    } else {
                        launch_gather_kv_blocks<float>(
                            reinterpret_cast<const float*>(k_storage_ptr),
                            ids_raw, reinterpret_cast<float*>(k_out_ptr),
                            1, info.current_pos, num_heads_, head_dim_,
                            block_size_, num_blocks_needed, stream);
                        launch_gather_kv_blocks<float>(
                            reinterpret_cast<const float*>(v_storage_ptr),
                            ids_raw, reinterpret_cast<float*>(v_out_ptr),
                            1, info.current_pos, num_heads_, head_dim_,
                            block_size_, num_blocks_needed, stream);
                    }
#else
                    // CPU stub — launch_gather_kv_blocks is a no-op.
                    // Should never reach here because use_gather is false
                    // when CUDA is not available. Defensive fallback.
                    (void)ids_raw;
                    (void)k_storage_ptr; (void)v_storage_ptr;
                    (void)k_out_ptr; (void)v_out_ptr;
#endif
                }

                // Squeeze batch dim: [1, current_pos, heads, dim] → [current_pos, heads, dim]
                return py::make_tuple(
                    k_out.attr("squeeze")(0),
                    v_out.attr("squeeze")(0));

            } catch (py::error_already_set& e) {
                // Gather kernel failed — fall through to torch.cat
                // Log for debugging but do not crash inference
                py::module_::import("logging").attr("getLogger")(
                    "memopt.paged").attr("debug")(
                    "gather kernel failed, falling back to torch.cat: " +
                    std::string(e.what()));
            } catch (...) {
                // Unknown error — fall through to torch.cat
            }
        }

        // ── torch.cat fallback ───────────────────────────────────────
        // Used when: CUDA not available, tensors on CPU, or gather failed.
        py::list k_parts, v_parts;
        int32_t remaining = info.current_pos;

        for (int32_t block_id : info.block_ids) {
            int32_t tokens = std::min(block_size_, remaining);
            py::object k_slice = k_blocks_.attr("__getitem__")(
                py::make_tuple(block_id,
                               py::slice(py::none(), tokens, py::none())));
            py::object v_slice = v_blocks_.attr("__getitem__")(
                py::make_tuple(block_id,
                               py::slice(py::none(), tokens, py::none())));
            k_parts.append(k_slice);
            v_parts.append(v_slice);
            remaining -= tokens;
            if (remaining <= 0) break;
        }

        py::object cat = torch.attr("cat");
        return py::make_tuple(
            cat(k_parts, py::arg("dim") = 0),
            cat(v_parts, py::arg("dim") = 0));
    }

    int32_t free_blocks_count() {
        return cache_.num_free_blocks();
    }

    double utilization() {
        int32_t used = num_blocks_ - cache_.num_free_blocks();
        return static_cast<double>(used) / num_blocks_;
    }

    py::dict stats() {
        py::dict d;
        d["num_blocks"] = num_blocks_;
        d["block_size"] = block_size_;
        d["num_heads"] = num_heads_;
        d["head_dim"] = head_dim_;
        d["num_layers"] = num_layers_;
        d["free_blocks"] = cache_.num_free_blocks();
        d["used_blocks"] = num_blocks_ - cache_.num_free_blocks();
        d["sequences"] = cache_.num_sequences();
        d["utilization"] = utilization();
        return d;
    }

    // Expose storage tensors for direct access (used by build_for_model)
    py::object get_k_blocks() { return k_blocks_; }
    py::object get_v_blocks() { return v_blocks_; }

private:
    PagedKVCache cache_;
    int32_t num_blocks_, block_size_, num_heads_, head_dim_, num_layers_;
    py::object k_blocks_;
    py::object v_blocks_;
};

// ═══════════════════════════════════════════════════════════════════════════
// PYBIND11 MODULE
// ═══════════════════════════════════════════════════════════════════════════

PYBIND11_MODULE(_memopt_paged, m) {
    m.doc() = "memopt paged KV cache — C++ block manager with "
              "C++ block pool (lock-free on x86-64, mutex on other platforms)";

    // ── PagedKVCache ─────────────────────────────────────────────────
    py::class_<PyPagedKVCache>(m, "PagedKVCache")
        .def(py::init<int32_t, int32_t, int32_t, int32_t,
                       py::object, py::str>(),
             py::arg("num_blocks"),
             py::arg("num_layers"),
             py::arg("num_heads"),
             py::arg("head_dim"),
             py::arg("dtype") = py::none(),
             py::arg("device") = "cpu")
        .def("allocate_sequence", &PyPagedKVCache::allocate_sequence,
             py::arg("seq_id"))
        .def("free_sequence", &PyPagedKVCache::free_sequence,
             py::arg("seq_id"))
        .def("store", &PyPagedKVCache::store,
             py::arg("seq_id"), py::arg("layer_idx"),
             py::arg("token_pos"), py::arg("k"), py::arg("v"))
        .def("fetch", &PyPagedKVCache::fetch,
             py::arg("seq_id"), py::arg("layer_idx"))
        .def("free_blocks_count", &PyPagedKVCache::free_blocks_count)
        .def("utilization", &PyPagedKVCache::utilization)
        .def("stats", &PyPagedKVCache::stats)
        .def_property_readonly("k_blocks", &PyPagedKVCache::get_k_blocks)
        .def_property_readonly("v_blocks", &PyPagedKVCache::get_v_blocks)
        .def_property_readonly("num_blocks",
             [](PyPagedKVCache& self) { return self.stats()["num_blocks"]; })
        .def_property_readonly("num_heads",
             [](PyPagedKVCache& self) { return self.stats()["num_heads"]; })
        .def_property_readonly("head_dim",
             [](PyPagedKVCache& self) { return self.stats()["head_dim"]; })
        .def_property_readonly("block_size",
             [](PyPagedKVCache& self) { return self.stats()["block_size"]; });

    // ── BlockPool — exposed for testing ──────────────────────────────
    py::class_<BlockPool>(m, "BlockPool")
        .def(py::init<int32_t>(), py::arg("num_blocks"))
        .def("allocate", &BlockPool::allocate)
        .def("free", &BlockPool::free, py::arg("block_id"))
        .def("num_free_blocks", &BlockPool::num_free_blocks)
        .def("capacity", &BlockPool::capacity);

    // ── CUDA gather availability ─────────────────────────────────────
    m.def("has_cuda_gather", &has_cuda_gather,
          "Returns True if CUDA gather kernel is compiled in.");
}
