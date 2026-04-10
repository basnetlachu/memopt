// bindings.cpp — pybind11 surface for _memopt_core.
//
// Exposes PageTable and MemoryOracle to Python with the exact same API
// as page_table.py and oracle.py. Existing tests run unchanged.
//
// GIL discipline:
//   - All public methods RELEASE the GIL before entering C++ code.
//   - Python objects (handles) are converted to/from void* across the boundary.
//   - stats() returns py::dict so it must hold the GIL while building the dict.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "block_directory.h"
#include "oracle.h"
#include "page_table.h"

namespace py = pybind11;
using namespace memopt;

// ═══════════════════════════════════════════════════════════════════════════
// Helper: PageTableEntry → Python-visible wrapper
// ═══════════════════════════════════════════════════════════════════════════
//
// The Python code accesses entry.sequence_id, entry.block_index, entry.tier,
// entry.handle, entry.size_bytes, entry.last_accessed, entry.pin_count.
// We wrap the C++ struct in a Python class that exposes these as properties.
// The handle is stored as a py::object in a side table.

// Side table: maps PageTableEntry* → py::object handle.
// This keeps the Python handle alive as long as the entry exists.
static std::unordered_map<PageTableEntry*, py::object> handle_table;
static std::mutex handle_table_mu;

static void store_handle(PageTableEntry* entry, py::object handle) {
    std::lock_guard<std::mutex> lock(handle_table_mu);
    handle_table[entry] = std::move(handle);
}

static py::object get_handle(PageTableEntry* entry) {
    std::lock_guard<std::mutex> lock(handle_table_mu);
    auto it = handle_table.find(entry);
    if (it != handle_table.end()) return it->second;
    return py::none();
}

static void remove_handle(PageTableEntry* entry) {
    std::lock_guard<std::mutex> lock(handle_table_mu);
    handle_table.erase(entry);
}

// ═══════════════════════════════════════════════════════════════════════════
// Python-compatible PageTableEntry wrapper
// ═══════════════════════════════════════════════════════════════════════════

struct PyPageTableEntry {
    PageTableEntry* ptr;  // non-owning, PageTable owns the memory

    // Properties matching Python dataclass fields
    std::string sequence_id() const { return ptr->sequence_id; }
    int32_t block_index() const { return ptr->block_index; }
    std::string tier() const { return ptr->tier; }
    py::object handle() const { return get_handle(ptr); }
    int64_t size_bytes() const { return ptr->size_bytes; }
    double last_accessed() const { return ptr->last_accessed; }
    int32_t pin_count() const { return ptr->pin_count; }
};

// ═══════════════════════════════════════════════════════════════════════════
// PageTable Python wrapper
// ═══════════════════════════════════════════════════════════════════════════

class PyPageTable {
public:
    PyPageTable() : pt_() {}

    PyPageTableEntry insert(const std::string& sequence_id,
                             int32_t block_index,
                             const std::string& tier,
                             py::object handle,
                             int64_t size_bytes) {
        PageTableEntry* entry;
        {
            py::gil_scoped_release release;
            entry = pt_.insert(sequence_id, block_index, tier,
                               nullptr, size_bytes);
        }
        store_handle(entry, std::move(handle));
        return PyPageTableEntry{entry};
    }

    py::object lookup(const std::string& sequence_id,
                       int32_t block_index) {
        PageTableEntry* entry;
        {
            py::gil_scoped_release release;
            entry = pt_.lookup(sequence_id, block_index);
        }
        if (!entry) return py::none();
        return py::cast(PyPageTableEntry{entry});
    }

    void update_tier(const std::string& sequence_id,
                      int32_t block_index,
                      const std::string& new_tier,
                      py::object new_handle) {
        {
            py::gil_scoped_release release;
            pt_.update_tier(sequence_id, block_index, new_tier, nullptr);
        }
        // Update handle in side table
        BlockKey key{sequence_id, block_index};
        auto& sh = pt_.shards_[fnv1a_str(sequence_id) % PageTable::NUM_SHARDS];
        std::shared_lock<std::shared_mutex> lock(sh.mu);
        auto it = sh.table.find(key);
        if (it != sh.table.end()) {
            store_handle(it->second, std::move(new_handle));
        }
    }

    py::object remove(const std::string& sequence_id,
                       int32_t block_index) {
        PageTableEntry* entry;
        {
            py::gil_scoped_release release;
            entry = pt_.remove(sequence_id, block_index);
        }
        if (!entry) return py::none();
        auto result = py::cast(PyPageTableEntry{entry});
        remove_handle(entry);
        return result;
    }

    py::list remove_sequence(const std::string& sequence_id) {
        std::vector<PageTableEntry*> entries;
        {
            py::gil_scoped_release release;
            entries = pt_.remove_sequence(sequence_id);
        }
        py::list result;
        for (auto* e : entries) {
            result.append(PyPageTableEntry{e});
            remove_handle(e);
        }
        return result;
    }

    void pin(const std::string& sequence_id, int32_t block_index) {
        py::gil_scoped_release release;
        pt_.pin(sequence_id, block_index);
    }

    void unpin(const std::string& sequence_id, int32_t block_index) {
        py::gil_scoped_release release;
        pt_.unpin(sequence_id, block_index);
    }

    py::list lru_candidates(const std::string& tier, int count) {
        std::vector<PageTableEntry*> candidates;
        {
            py::gil_scoped_release release;
            candidates = pt_.lru_candidates(tier, count);
        }
        py::list result;
        for (auto* e : candidates) {
            result.append(PyPageTableEntry{e});
        }
        return result;
    }

    py::dict stats() {
        PageTable::Stats s;
        {
            py::gil_scoped_release release;
            s = pt_.stats();
        }
        py::dict blocks_per_tier;
        for (const auto& [k, v] : s.blocks_per_tier) {
            blocks_per_tier[py::cast(k)] = py::cast(v);
        }
        py::dict bytes_per_tier;
        for (const auto& [k, v] : s.bytes_per_tier) {
            bytes_per_tier[py::cast(k)] = py::cast(v);
        }
        py::dict result;
        result["total_blocks"] = s.total_blocks;
        result["blocks_per_tier"] = blocks_per_tier;
        result["bytes_per_tier"] = bytes_per_tier;
        return result;
    }

    void clear() {
        py::gil_scoped_release release;
        pt_.clear();
    }

    // Expose internal PageTable for update_tier's shard access.
    // This is not ideal but necessary for the handle side table.
    PageTable pt_;
};

// ═══════════════════════════════════════════════════════════════════════════
// MemoryOracle Python wrapper
// ═══════════════════════════════════════════════════════════════════════════

class PyMemoryOracle {
public:
    PyMemoryOracle(int horizon = 50,
                    int max_transitions = 100000,
                    double min_confidence = 0.3)
        : oracle_(horizon, max_transitions, min_confidence) {}

    void observe(const std::string& sequence_id,
                  int32_t block_index,
                  py::object step_obj = py::none()) {
        int32_t step = -1;
        if (!step_obj.is_none()) {
            step = step_obj.cast<int32_t>();
        }
        py::gil_scoped_release release;
        oracle_.observe(sequence_id, block_index, step);
    }

    py::list predict(const std::string& sequence_id,
                      int32_t current_block,
                      int top_k = 10) {
        std::vector<BlockPrediction> preds;
        {
            py::gil_scoped_release release;
            preds = oracle_.predict(sequence_id, current_block, top_k);
        }
        // Return actual BlockPrediction dataclass instances so
        // isinstance(p, BlockPrediction) passes in tests.
        py::module_ oracle_mod = py::module_::import(
            "memopt.vmm._oracle_py");
        py::object BPCls = oracle_mod.attr("BlockPrediction");

        py::list result;
        for (const auto& p : preds) {
            result.append(BPCls(
                py::cast(p.sequence_id),
                py::cast(p.block_index),
                py::cast(p.confidence),
                py::cast(p.predicted_at_step),
                py::cast(p.source)
            ));
        }
        return result;
    }

    void record_outcome(const std::string& sequence_id,
                         int32_t block_index) {
        py::gil_scoped_release release;
        oracle_.record_outcome(sequence_id, block_index);
    }

    py::object stats() {
        OracleStats s;
        {
            py::gil_scoped_release release;
            s = oracle_.stats();
        }
        // Return an OracleStats-like object.
        // Tests access s.total_predictions_made, s.accuracy_pct, etc.
        py::module_ oracle_mod = py::module_::import(
            "memopt.vmm._oracle_py");
        py::object OracleStatsCls = oracle_mod.attr("OracleStats");
        return OracleStatsCls(
            s.total_predictions_made,
            s.total_predictions_correct,
            s.accuracy_pct,
            s.transitions_learned,
            s.sequences_tracked,
            s.horizon,
            s.uptime_seconds
        );
    }

    void reset(py::object sequence_id_obj = py::none()) {
        std::string seq_id;
        if (!sequence_id_obj.is_none()) {
            seq_id = sequence_id_obj.cast<std::string>();
        }
        py::gil_scoped_release release;
        oracle_.reset(seq_id);
    }

    int warm_from_log(const std::string& log_path,
                       int max_events = 50000) {
        py::gil_scoped_release release;
        return oracle_.warm_from_log(log_path, max_events);
    }

    // ── Properties for test compatibility ────────────────────────────
    // Tests access oracle._transitions, oracle._horizon, etc.

    int get_horizon() const { return oracle_.horizon(); }
    double get_min_confidence() const { return oracle_.min_confidence(); }
    int get_max_transitions() const { return oracle_.max_transitions(); }

    /// Return transitions as a Python dict[int, Counter] for test compat.
    /// Tests do: assert 1 in oracle._transitions[0]
    py::dict get_transitions() {
        py::dict result;
        const auto& trans = oracle_.transitions();
        for (const auto& [from, tos] : trans) {
            py::dict inner;
            for (const auto& [to, count] : tos) {
                inner[py::cast(to)] = py::cast(count);
            }
            result[py::cast(from)] = inner;
        }
        return result;
    }

private:
    MemoryOracle oracle_;
};

// ═══════════════════════════════════════════════════════════════════════════
// PYBIND11 MODULE
// ═══════════════════════════════════════════════════════════════════════════

PYBIND11_MODULE(_memopt_core, m) {
    m.doc() = "C++ accelerated page table and memory oracle for memopt VMM";

    // ── PyPageTableEntry ─────────────────────────────────────────────
    py::class_<PyPageTableEntry>(m, "PageTableEntry")
        .def_property_readonly("sequence_id", &PyPageTableEntry::sequence_id)
        .def_property_readonly("block_index", &PyPageTableEntry::block_index)
        .def_property_readonly("tier", &PyPageTableEntry::tier)
        .def_property_readonly("handle", &PyPageTableEntry::handle)
        .def_property_readonly("size_bytes", &PyPageTableEntry::size_bytes)
        .def_property_readonly("last_accessed",
                                &PyPageTableEntry::last_accessed)
        .def_property_readonly("pin_count", &PyPageTableEntry::pin_count);

    // ── PageTable ────────────────────────────────────────────────────
    py::class_<PyPageTable>(m, "PageTable")
        .def(py::init<>())
        .def("insert", &PyPageTable::insert,
             py::arg("sequence_id"), py::arg("block_index"),
             py::arg("tier"), py::arg("handle"), py::arg("size_bytes"))
        .def("lookup", &PyPageTable::lookup,
             py::arg("sequence_id"), py::arg("block_index"))
        .def("update_tier", &PyPageTable::update_tier,
             py::arg("sequence_id"), py::arg("block_index"),
             py::arg("new_tier"), py::arg("new_handle"))
        .def("remove", &PyPageTable::remove,
             py::arg("sequence_id"), py::arg("block_index"))
        .def("remove_sequence", &PyPageTable::remove_sequence,
             py::arg("sequence_id"))
        .def("pin", &PyPageTable::pin,
             py::arg("sequence_id"), py::arg("block_index"))
        .def("unpin", &PyPageTable::unpin,
             py::arg("sequence_id"), py::arg("block_index"))
        .def("lru_candidates", &PyPageTable::lru_candidates,
             py::arg("tier"), py::arg("count"))
        .def("stats", &PyPageTable::stats)
        .def("clear", &PyPageTable::clear);

    // ── MemoryOracle ─────────────────────────────────────────────────
    py::class_<PyMemoryOracle>(m, "MemoryOracle")
        .def(py::init<int, int, double>(),
             py::arg("horizon") = 50,
             py::arg("max_transitions") = 100000,
             py::arg("min_confidence") = 0.3)
        .def("observe", &PyMemoryOracle::observe,
             py::arg("sequence_id"), py::arg("block_index"),
             py::arg("step") = py::none())
        .def("predict", &PyMemoryOracle::predict,
             py::arg("sequence_id"), py::arg("current_block"),
             py::arg("top_k") = 10)
        .def("record_outcome", &PyMemoryOracle::record_outcome,
             py::arg("sequence_id"), py::arg("block_index"))
        .def("stats", &PyMemoryOracle::stats)
        .def("reset", &PyMemoryOracle::reset,
             py::arg("sequence_id") = py::none())
        .def("warm_from_log", &PyMemoryOracle::warm_from_log,
             py::arg("log_path"), py::arg("max_events") = 50000)
        // Properties for test compatibility
        .def_property_readonly("_horizon", &PyMemoryOracle::get_horizon)
        .def_property_readonly("_min_confidence",
                                &PyMemoryOracle::get_min_confidence)
        .def_property_readonly("_max_transitions",
                                &PyMemoryOracle::get_max_transitions)
        .def_property_readonly("_transitions",
                                &PyMemoryOracle::get_transitions);

    // ── BlockDirectoryCpp ────────────────────────────────────────────
    py::class_<BlockDirectoryCpp>(m, "BlockDirectoryCpp")
        .def(py::init<std::string, double>(),
             py::arg("node_id"),
             py::arg("default_ttl_s") = 3600.0)
        .def("register_block",
             [](BlockDirectoryCpp& self, py::dict entry) {
                 BlockEntryC e{};
                 auto copy_str = [](const std::string& s,
                                    char* dst, size_t n) {
                     std::strncpy(dst, s.c_str(), n - 1);
                     dst[n - 1] = '\0';
                 };
                 copy_str(entry["content_hash"].cast<std::string>(),
                          e.content_hash, sizeof(e.content_hash));
                 copy_str(entry["node_id"].cast<std::string>(),
                          e.node_id, sizeof(e.node_id));
                 copy_str(entry["tier"].cast<std::string>(),
                          e.tier, sizeof(e.tier));
                 copy_str(entry["path"].cast<std::string>(),
                          e.path, sizeof(e.path));
                 e.size_bytes    = entry["size_bytes"].cast<int64_t>();
                 e.registered_at = entry["registered_at"].cast<double>();
                 e.lease_count   = 0;
                 e.ttl_s         = 0;
                 py::gil_scoped_release no_gil;
                 self.register_block(e);
             },
             py::arg("entry"))
        .def("lookup",
             [](BlockDirectoryCpp& self,
                const std::string& hash) -> py::object {
                 BlockEntryC e{};
                 bool found;
                 {
                     py::gil_scoped_release no_gil;
                     found = self.lookup(hash, &e);
                 }
                 if (!found) return py::none();
                 py::dict d;
                 d["content_hash"]  = std::string(e.content_hash);
                 d["node_id"]       = std::string(e.node_id);
                 d["tier"]          = std::string(e.tier);
                 d["path"]          = std::string(e.path);
                 d["size_bytes"]    = e.size_bytes;
                 d["registered_at"] = e.registered_at;
                 d["lease_count"]   = e.lease_count;
                 return d;
             },
             py::arg("content_hash"))
        .def("acquire_lease",
             [](BlockDirectoryCpp& self,
                const std::string& h,
                const std::string& node) -> bool {
                 py::gil_scoped_release no_gil;
                 return self.acquire_lease(h, node);
             },
             py::arg("content_hash"),
             py::arg("requesting_node"))
        .def("release_lease",
             [](BlockDirectoryCpp& self,
                const std::string& h,
                const std::string& node) -> bool {
                 py::gil_scoped_release no_gil;
                 return self.release_lease(h, node);
             },
             py::arg("content_hash"),
             py::arg("requesting_node"))
        .def("can_evict",
             [](BlockDirectoryCpp& self,
                const std::string& h) -> bool {
                 py::gil_scoped_release no_gil;
                 return self.can_evict(h);
             },
             py::arg("content_hash"))
        .def("deregister",
             [](BlockDirectoryCpp& self,
                const std::string& h) {
                 py::gil_scoped_release no_gil;
                 self.deregister(h);
             },
             py::arg("content_hash"))
        .def("size", &BlockDirectoryCpp::size)
        .def("expired_count", &BlockDirectoryCpp::expired_count);
}
