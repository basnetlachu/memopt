# memopt Substrate (Layer 1) — Design

| Field    | Value                                                       |
| -------- | ----------------------------------------------------------- |
| Status   | Phase 3 complete. Awaiting review before Phase 4.           |
| Scope    | Memory management substrate underneath existing memopt code |
| Started  | 2026-04-29                                                  |
| Repo     | /Users/lachumanbasnet/Personal/Sophisticates/MEMOPT/memopt  |
| Baseline | 682 tests passing on `main` at commit `6f0678d`             |

Sections:

  1. Discovery (Phase 1 — this section, complete)
  2. Design (Phase 2 — pending)
  3. Test plan (Phase 3 — pending)
  4. Implementation prompt (Phase 4 — pending)

The discovery section is read-only: it records what exists today.
No design decisions appear here — those go in Phase 2 after the
user reviews this map.

---

## Phase 1 — Discovery

### 1.1 Python memory layer surface

The discovery walked nine Python files. Each subsection lists every
public class with its public methods, plus whether `tenant_id` is
a parameter, whether streams are tracked, and whether any code path
emits events.

A "public" symbol here means: defined at module level OR named in
`__all__`, AND not prefixed with `_`. Methods named `_x` are private
to the class but listed when behaviour-relevant.

#### memopt/vmm/__init__.py (174 lines)

```
class VMM:
    def __init__(self, gkd: Optional["GKDStore"] = None) -> None
    def allocate(
        self,
        sequence_id: str,
        block_index: int,
        size_bytes: int,
        tenant_id: str = "_default",
    ) -> PageTableEntry
    def fetch(
        self,
        sequence_id: str,
        block_index: int,
        tenant_id: str = "_default",
    ) -> PageTableEntry
    def free_sequence(
        self,
        sequence_id: str,
        tenant_id: str = "_default",
    ) -> None
    def stats(self) -> dict

# PEP 562 lazy module attrs:
__getattr__("backend") | __getattr__("tiers") | __getattr__("tier_names")
__all__ = ["VMM", "WeightManager", "backend", "tiers", "tier_names",
           "PageTableEntry"]
```

- `tenant_id` flows through every call. First `allocate()` for a
  `sequence_id` sets the owning tenant in `self._sequence_owners:
  dict[str, str]`. Subsequent calls by a different tenant raise
  `PermissionError`.
- Streams are NOT tracked by VMM. No per-handle stream registry,
  no stream-locked free.
- Events are NOT emitted by VMM. No callbacks, no ring buffer.

#### memopt/vmm/cuda_vmm.py (192 lines)

```
class CUDAVMMAllocator:
    def __init__(
        self,
        device: int = 0,
        pool_gb: float = 16.0,
        evict_threshold: float = 0.80,
    )
    def close(self) -> None
    def malloc(self, size_bytes: int, tag: str = "python") -> int
    def free(self, va: int) -> None
    def quiesce(self) -> None
    def evict_n_pages(self, n: int) -> int
    def evict_to_target(self, target_fraction: float) -> int
    def promote(self, va: int) -> int
    def stats(self) -> dict
```

- ctypes wrapper around `libmemopt_vmm.so`. Returns the CUDA
  virtual address as a Python `int`.
- `tenant_id` is NOT a parameter anywhere. The `tag` argument is a
  debug label, not a routing key.
- Streams are NOT tracked. The C library uses `cuStreamSynchronize`
  on quiesce.
- No events.

#### memopt/vmm/torch_allocator.py (223 lines)

```
class MemoptTorchAllocator:
    def __init__(
        self,
        pool_gb: float = 60.0,
        evict_threshold: float = 0.80,
        device: int = 0,
        verbose: bool = False,
    )
    def install(self) -> bool
    def step_boundary(self) -> dict
    def stats(self) -> dict
    def is_installed(self) -> bool

def install_memopt_allocator(
    pool_gb: float = 60.0,
    evict_threshold: float = 0.80,
    device: int = 0,
    verbose: bool = True,
) -> MemoptTorchAllocator
```

- Wraps `torch.cuda.memory.CUDAPluggableAllocator` to install
  memopt as PyTorch's allocator (CUDA only, torch >= 2.1).
- Patches `torch.cuda.memory_reserved` and `memory_allocated` to
  swallow `RuntimeError` from the pluggable backend (workaround
  for `accelerate`/`transformers` calling these during model load).
- `tenant_id`: NOT exposed. PyTorch's allocator API has no
  per-call metadata channel.
- Streams are NOT tracked. `step_boundary()` is the only quiesce
  hook — caller drives it from the decode loop.
- No events.

#### memopt/vmm/hal.py (448 lines)

```
class HardwareBackend(enum.Enum):
    NVIDIA_CUDA = "nvidia_cuda"
    AMD_ROCM    = "amd_rocm"
    CPU_ONLY    = "cpu_only"
    UNKNOWN     = "unknown"

class GPUInfo:
    index: int
    name: str
    total_bytes: int
    free_bytes: int
    compute_cap: str
    backend: HardwareBackend
    def to_dict(self) -> dict

class HAL:
    def __init__(self)
    @property backend: HardwareBackend
    @property gpu_count: int
    @property total_hbm_bytes: int
    @property free_hbm_bytes: int
    @property gpus: List[GPUInfo]
    @property tier_names: List[str]
    def is_gpu_available(self) -> bool
    def stats(self) -> dict

def get_hal() -> HAL          # singleton
def reset_hal() -> None       # tests only
def get_backend()             # legacy contract

# Lazy module-level singletons (PEP 562):
backend, tiers, tier_names
```

- Detection order: pynvml → torch CUDA → torch HIP → CPU_ONLY.
  Caches the result in a module-level singleton; `reset_hal()`
  exists for tests.
- `tenant_id`: not relevant — HAL is per-process hardware metadata.
- Streams: not tracked.
- No events.

#### memopt/vmm/backends/cuda_backend.py (70 lines)

A shim that picks between `_cuda_backend_py.CUDABackend` and a
hypothetical C++-accelerated subclass when `memopt._memopt_cuda`
is importable. Re-exports `CUDABackend` and `MemoryTier`. The
class surface is fully defined in `_cuda_backend_py.py` (below).

#### memopt/vmm/backends/_cuda_backend_py.py (376 lines)

```
@dataclass
class MemoryTier:
    name: str
    capacity_bytes: int
    latency_us: float
    bandwidth_gbps: float

class CUDABackend:
    def __init__(self) -> None
        # Creates two fixed torch.cuda.Stream():
        #   self._copy_stream  (HBM <-> DRAM)
        #   self._nvme_stream  (NVMe staging hops)
    def detect_tiers(self) -> list[MemoryTier]
        # Returns [hbm, dram, nvme]
    def allocate(
        self,
        size_bytes: int,
        tier: str,                  # "hbm" | "dram" | "nvme"
        tenant_id: str = "_default",
        sequence_id: str = "",
        block_index: int = 0,
    ) -> Union["torch.Tensor", str]
    def free(
        self,
        handle: Union["torch.Tensor", str],
        tier: str,
    ) -> None
    @staticmethod _gds_available() -> bool
    def async_copy(
        self,
        src: Union["torch.Tensor", str],
        dst: Union["torch.Tensor", str],
        prefer_nvme_stream: bool = False,
    ) -> object   # returns the chosen torch.cuda.Stream
    def record_event(
        self, stream: "torch.cuda.Stream",
    ) -> "torch.cuda.Event"
    def current_hbm_used_bytes(self) -> int
    def synchronize(self) -> None
    def _load_to_tensor(self, handle) -> "torch.Tensor"

class AsyncNVMeManager:
    POLL_INTERVAL_MS: int    # default 1, env MEMOPT_NVME_POLL_MS
    def __init__(self) -> None
    def read_block(self, path: str, size: int) -> Optional[bytes]
    def write_block(self, path: str, data: bytes) -> bool
    def stats(self) -> dict
    def stop(self) -> None

def get_async_nvme_manager() -> AsyncNVMeManager
```

- `allocate()` returns either a `torch.Tensor` (for `hbm`/`dram`)
  or a filesystem path `str` (for `nvme`). The handle type is
  tier-dependent and the caller must know which it received.
- `tenant_id` is a parameter on `allocate()`. It is used ONLY to
  build the NVMe path under `_nvme_block_path()` — see 1.3 below.
- Two fixed CUDA streams are created at construction and reused
  forever. Per-allocation stream tracking does NOT exist.
- `record_event(stream)` records a `torch.cuda.Event` on a stream
  but the backend does not associate that event with any handle.
- No events emitted.

#### memopt/vmm/backends/unified_backend.py (242 lines)

```
class UnifiedBackend:
    def __init__(self) -> None
        # Calls _recover_nvme_dir(tempfile.gettempdir()) — clears
        # *.vmm_block.tmp files left by previous crash.
    def detect_tiers(self) -> list[MemoryTier]
        # Returns [dram, nvme]
    def allocate(
        self,
        size_bytes: int,
        tier: str,                  # "hbm" | "dram" | "nvme"
        tenant_id: str = "_default",
        sequence_id: str = "",
        block_index: int = 0,
    ) -> Union["torch.Tensor", str]
        # tier="hbm" is silently remapped to "dram"
    def free(self, handle, tier: str) -> None
    def async_copy(self, src, dst, stream=None) -> threading.Event
        # stream parameter is ignored (kept for interface compat)
    def current_hbm_used_bytes(self) -> int
        # Returns process RSS (no VRAM on this hardware)
    def synchronize(self) -> None  # no-op

# Module-level helpers:
def _sanitize_id(kind: str, value: str) -> str
def _nvme_block_path(nvme_dir, tenant_id, sequence_id, block_index)
def _write_nvme_block(path: str, data: bytes) -> None
def _remove_nvme_block(path: str) -> None
def _recover_nvme_dir(nvme_dir: str) -> int
```

- For Apple Silicon and CPU-only Linux. HBM requests degrade to
  DRAM tensors (`torch.empty(...)` on CPU; bytearray fallback if
  torch is missing).
- `async_copy()` runs on a Python `threading.Thread`, not a CUDA
  stream — the `stream=` parameter is unused.
- NVMe write protocol: write `path.tmp`, fsync, atomic rename to
  `path`. Crash-safe.
- `tenant_id` is sanitized via `_ID_RE = ^[A-Za-z0-9_-]{1,128}$`
  and forms the parent directory for the NVMe block file.

#### memopt/vmm/backends/rocm_backend.py (86 lines)

```
class ROCmBackend:
    BACKEND_NAME = "AMD ROCm (stub)"
    IS_STUB = True
    def __init__(self, *args, **kwargs)
        # raises NotImplementedError
    def allocate(self, size_bytes: int, tag: str = "")
        # raises
    def free(self, handle) -> None
        # raises
    def hbm_used_bytes(self) -> int
        # raises
    def hbm_total_bytes(self) -> int
        # raises
    def detect_tiers(self)
        # raises
    @staticmethod is_available() -> bool
        # True iff torch.version.hip is set and torch.cuda is up
```

- This is a hard stub. Constructor raises so production code
  cannot accidentally use it.
- A more substantive contributor reference exists at
  `memopt/vmm/backends/_rocm_backend_py.py` (allocate_hbm,
  allocate_host, copy_h2d_async, copy_d2h_async). It is not
  wired up; the public ROCm path is the stub above.

#### memopt/cluster/block_directory.py (130 lines)

```
class LocalBlockDirectory:
    def __init__(self, node_id: str = "",
                 default_ttl_s: float = _BLOCK_TTL_S)
    def register(self, entry: BlockEntry) -> None
    def lookup(self, content_hash: str) -> Optional[BlockEntry]
    def acquire_lease(self, content_hash: str,
                      requesting_node: str) -> bool
    def release_lease(self, content_hash: str,
                      requesting_node: str) -> None
    def can_evict(self, content_hash: str) -> bool
    def deregister(self, content_hash: str) -> None
    def list_node_blocks(self, node_id: str) -> list
    def stats(self) -> dict

# Re-exported from _block_directory_py:
class BlockEntry: ...   # fields: content_hash, node_id, tier,
                        # path, size_bytes, registered_at,
                        # lease_count
class RedisBlockDirectory: ...

def make_directory(node_id: str = "")
```

- C++-accelerated when `memopt._memopt_core.BlockDirectoryCpp` is
  importable (16-shard concurrent map). Falls back to pure Python.
- `tenant_id`: NOT a field on `BlockEntry`. Block addressing is
  by `content_hash`, which is computed from the block's tokens
  (post-isolation salt for GKD if `MEMOPT_GKD_TENANT_ISOLATION=1`).
- Streams: not relevant — this is a metadata index.
- No events.

#### memopt/cluster/gkd_store.py (1218 lines)

Public surface (the GKDStore class is the consumer entry point;
`LocalGKDBackend` and `RedisGKDBackend` are private to it):

```
@dataclass
class GKDEntry: ...     # tokens, block_ref, ts, hit_count, ...
@dataclass
class GKDHit: ...        # block_ref, prefix_match_tokens, ...

class TenantGKDStats:
    def record_hit(
        self, tenant_id: str, tokens_saved: int,
        is_partial: bool, ...
    ) -> None
    def record_miss(self, tenant_id: str) -> None
    def get_tenant_stats(self, tenant_id: str) -> dict
    def get_all_tenants(self) -> Dict[str, dict]

class HitRateWindow:
    def record(self, is_hit: bool) -> None
    def hit_rate_pct(self) -> float
    def stats(self) -> dict

class GKDStore:
    def __init__(self, backend="local", redis_url=None,
                 redis_host=None, node_id="", ...)
    def tenant_stats(self, tenant_id: str = "") -> dict
    def lookup(...)
    def register(...)
    def register_workflow_block(...)
    def lookup_workflow(...)
    def workflow_stats(self, workflow_id: str) -> dict
    def get_output(self, block_ref: str) -> Optional[dict]
    def register_output(self, block_ref: str, output: dict) -> None
    def invalidate(self, token_ids: List[int],
                   sequence_length: int)
    def stats(self) -> Dict[str, Any]
    def reset_stats(self) -> None

def make_gkd_backend(...)
```

- `tenant_id` flows through `lookup()` and `register()`. When
  `MEMOPT_GKD_TENANT_ISOLATION=1`, the content hash is salted
  with the tenant id so different tenants produce different
  hashes (cross-tenant reuse is disabled).
- Stream/event surface: not relevant — this is a cache, not an
  allocator.

#### memopt/serving/server.py (1127 lines, alloc paths only)

The serving server allocates VMM and GKDStore once during
`_build_engine()`:

```
# Around line 987:
from memopt.vmm import VMM
_vmm_instance = VMM()
log.info("VMM initialized for serving engine")

# Around line 1003:
from memopt.cluster.gkd_store import GKDStore
_gkd_store = GKDStore(
    redis_url=redis_url, node_id=_node_id   # or backend="local"
)
```

- The serving server uses the `VMM` facade, not `CUDAVMMAllocator`
  directly.
- `tenant_id` for serving is derived from `X-Memopt-Api-Key` and
  defaults to `"_default"`. It is passed through to VMM via the
  per-request authentication layer.
- Streams: serving uses `torch.cuda.Stream` indirectly via the
  CUDABackend. No per-request stream pinning.

### 1.2 C++ allocator surface

Five subdirectories under `csrc/`:

```
csrc/
├── core/          page table, oracle, block directory (pybind: _memopt_core)
├── cuda_backend/  NVMe I/O, GDS, stream pool      (pybind: _memopt_cuda)
├── cuda_vmm/      C VMM allocator                 (.so: libmemopt_vmm)
├── hooks/         kernel hook dispatch            (pybind: _memopt_hooks)
├── paged/         paged KV cache                  (pybind: _memopt_paged)
├── rocm/          ROCm/HIP shim                   (pybind: _memopt_rocm)
├── simd/          LCP / prefix matching           (pybind: _memopt_simd)
└── transport/     RDMA + TCP fallback (no pybind, daemon binary)
```

#### csrc/cuda_vmm/vmm_allocator.h (153 lines, C ABI)

```c
#define MEMOPT_PAGE_SIZE_BYTES (2 * 1024 * 1024ULL)   // 2 MiB
#define MEMOPT_MAX_PAGES        (8192)                // 16 GiB cap

#define TIER_HBM   0
#define TIER_DRAM  1
#define TIER_NVME  2     // declared but not used in M1

typedef struct MemoptAllocator MemoptAllocator;

MemoptAllocator* memopt_allocator_create(
    int device_idx, size_t pool_size_bytes);
void memopt_allocator_destroy(MemoptAllocator* alloc);

CUdeviceptr memopt_malloc(
    MemoptAllocator* alloc, size_t size_bytes, const char* tag);
void memopt_free(MemoptAllocator* alloc, CUdeviceptr va);

void memopt_quiesce(MemoptAllocator* alloc);
int  memopt_evict_n_pages(MemoptAllocator* alloc, int n_pages);
int  memopt_promote(MemoptAllocator* alloc, CUdeviceptr va);
size_t memopt_evict_to_target(
    MemoptAllocator* alloc, float target_fraction);

typedef struct {
    int      pages_total;
    int      pages_hbm;
    int      pages_dram;
    size_t   bytes_hbm;
    size_t   bytes_dram;
    uint64_t eviction_count;
    uint64_t promotion_count;
    size_t   bytes_evicted_total;
    float    hbm_pressure;
} MemoptStats;

MemoptStats memopt_stats(MemoptAllocator* alloc);
void memopt_dump_pages(MemoptAllocator* alloc);
```

#### csrc/cuda_vmm/vmm_allocator.cpp (955 lines)

CUDA driver APIs called (verified by grep, lines noted):

| API                              | Where (line)         |
| -------------------------------- | -------------------- |
| `cuMemGetAllocationGranularity`  | 219                  |
| `cuMemAddressReserve`            | 230                  |
| `cuMemAddressFree`               | 785                  |
| `cuMemCreate`                    | 376, 651             |
| `cuMemMap`                       | 391, 659             |
| `cuMemSetAccess`                 | 401, 671             |
| `cuMemUnmap`                     | 403, 460, 524, 756, 767 |
| `cuMemRelease`                   | 393, 404, 461, 531, 661, 757, 768 |

- Eviction: `evict_page_locked()` at line 504 unmaps the HBM
  handle, copies the page to a pinned-DRAM mirror (slab-allocated
  via a single `cudaHostAlloc` at allocator construction), and
  releases the HBM handle. HBM ↔ DRAM only.
- Promotion: `memopt_promote()` at line 631 does the reverse —
  `cuMemCreate` a fresh HBM handle, `cuMemMap` it to the same VA,
  `cuMemSetAccess`, copy from DRAM mirror, free the mirror slab
  slot.
- TIER_NVME is declared in the header but no path in the C
  allocator writes to NVMe. The NVMe tier lives in the Python
  backend (`UnifiedBackend.allocate(tier="nvme")` and
  `CUDABackend.allocate(tier="nvme")`) which creates files under
  `tempfile.gettempdir()`.
- VA-reuse fast path: `memopt_free()` keeps an exact-size reclaim
  list of `(VA, handle)` pairs (line 80 comment, line 327 hot
  path). A subsequent `memopt_malloc()` of the same size grabs a
  reclaimed entry and skips `cuMemCreate`/`Map`/`SetAccess`.
- PyTorch shim (lines 800-945, prefixed `memopt_torch_*`):
  `memopt_torch_malloc`, `memopt_torch_free`,
  `memopt_torch_step_boundary`, `memopt_torch_get_stats`,
  `memopt_torch_get_allocator`. Uses a process-global allocator
  `g_torch_allocator`.
- Fabric handles: `cuMemExportToShareableHandle` is NOT called.
  No code path requests
  `CU_MEM_HANDLE_TYPE_FABRIC` /
  `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR`. Cross-node sharing
  uses RDMA (csrc/transport/), not VMM peer mapping.
- Tenant: `memopt_malloc` takes `(size, tag)` only. The C
  allocator has no tenant concept.
- Streams: `memopt_quiesce()` calls `cuStreamSynchronize` on the
  default stream. There is no per-handle stream tracking. The
  PyTorch shim's `memopt_torch_free` documents (line 895
  comment) that immediate `cuMemUnmap` after free is safe only
  when no kernels are in flight on the freeing stream — but the
  code does not enforce this; the comment is informational.

#### csrc/cuda_vmm/test_vmm_allocator.cpp (221 lines)

A standalone CMake test binary that exercises the C ABI directly
(allocator_create, malloc, free, quiesce, evict, promote, stats).
Not a pytest test. Not part of the 682.

#### csrc/CMakeLists.txt (524 lines)

Top-level build. Conditionally builds:

```
_memopt_core      pybind11, csrc/core/*
_memopt_simd      pybind11, csrc/simd/*
_memopt_paged     pybind11, csrc/paged/*
_memopt_hooks     pybind11, csrc/hooks/*
_memopt_cuda      pybind11, csrc/cuda_backend/*  (CUDA only)
_memopt_rocm      pybind11, csrc/rocm/*           (HIP only)
libmemopt_vmm.so  shared lib, csrc/cuda_vmm/*    (CUDA only)
memopt_transport  binary, csrc/transport/*
```

#### csrc/core/page_table.h (233 lines)

```cpp
namespace memopt {

struct BlockKey {
    std::string sequence_id;
    int32_t     block_index;
};

struct PageTableEntry {
    std::string sequence_id;
    int32_t     block_index;
    std::string tier;            // "hbm" | "dram" | "nvme"
    int64_t     size_bytes;
    void*       handle_ptr;      // borrowed py::object
    double      last_accessed;
    int32_t     pin_count;
    PageTableEntry* lru_prev;
    PageTableEntry* lru_next;
};

class PageTable {
public:
    static constexpr int NUM_SHARDS = 64;

    explicit PageTable(int max_blocks = 0);
    ~PageTable();

    PageTableEntry* insert(seq_id, block_index, tier,
                           handle, size_bytes);
    PageTableEntry* lookup(seq_id, block_index);
    void update_tier(seq_id, block_index, new_tier, new_handle);
    PageTableEntry* remove(seq_id, block_index);
    std::vector<PageTableEntry*> remove_sequence(seq_id);
    void pin(seq_id, block_index);
    void unpin(seq_id, block_index);
    std::vector<PageTableEntry*> lru_candidates(tier, count);
    Stats stats() const;
    void clear();
};

} // namespace memopt
```

- 64 shards, each `std::shared_mutex` (RW lock). Reads concurrent,
  writes serialized per shard.
- Intrusive doubly-linked LRU list per shard for O(1) move-to-front.
- `lru_candidates(tier, n)` walks the tail of the LRU list — O(n),
  not O(total_entries).
- **`PageTableEntry` has NO tenant_id field.** The page table is
  tenant-blind. Tenant enforcement happens in Python at the VMM
  facade.
- No event hooks.

#### csrc/core/page_table.cpp (326 lines)

Implementation of the above. GIL is released on every public
method (per the comment at line 15).

#### Other csrc modules (relevant subset)

- `csrc/cuda_backend/nvme_async.cpp`: `AsyncNVMeReader`,
  `AsyncNVMeWriter`. Uses `liburing` (`io_uring_queue_init`,
  `io_uring_get_sqe`, `io_uring_prep_read`, `io_uring_submit`,
  `io_uring_peek_cqe`). Falls back to `pread()`/`write()` if
  io_uring init fails.
- `csrc/cuda_backend/gds.cpp`: `gds_is_available()` (cuFile).
  Code paths gated on `MEMOPT_GDS_AVAILABLE` build flag.
- `csrc/cuda_backend/stream_pool.cpp`: 8-stream pool for copy
  operations (referenced by header comment in `cuda_backend.py`).
- `csrc/transport/`: RDMA engine, ring buffer, TCP fallback. The
  `memopt_transport` binary runs as a daemon. NOT a pybind module.
  Bus to Python is via socket protocol defined in `protocol.h`.

#### Pybind11 surface (full enumeration)

```
_memopt_core:
    PageTable, PageTableEntry, MemoryOracle, BlockDirectoryCpp

_memopt_simd:
    find_lcp, find_lcp_buffer, detected_isa,
    has_avx512, has_avx2

_memopt_paged:
    PagedKVCache, BlockPool, has_cuda_gather

_memopt_hooks:
    init_hooks, set_synthesis_callback, notify,
    make_cache_key, register_kernel, invalidate_kernel,
    hook_stats, reset_counters, detect_arch_id

_memopt_rocm:
    ROCmDeviceInfo, get_device_info,
    hip_alloc, hip_free, hip_alloc_host, hip_free_host,
    hip_memcpy_h2d_async, hip_memcpy_d2h_async,
    hip_device_synchronize, hip_stream_synchronize,
    is_unified_memory_device, rocm_version

_memopt_cuda:
    stream pool, NVMe I/O, GDS availability, HBM stats
    (full list in csrc/cuda_backend/bindings.cpp)
```

### 1.3 Tenant isolation today

Where `tenant_id` lives, by layer:

```
LAYER                       tenant_id present?    Enforcement
────────────────────────────────────────────────────────────────────
Python:
  VMM (facade)              YES                   PermissionError
  TierManager               YES (passed through)  none of its own
  CUDABackend.allocate      YES                   used in NVMe path only
  UnifiedBackend.allocate   YES                   used in NVMe path only
  ROCmBackend               raises (stub)         n/a
  CUDAVMMAllocator          NO                    n/a
  MemoptTorchAllocator      NO                    n/a
  GKDStore                  YES                   hash salt (opt-in)
  LocalBlockDirectory       NO (not on entry)     n/a

C++:
  libmemopt_vmm.so          NO                    n/a
  PageTable / PageTableEntry NO                   n/a
  BlockDirectoryCpp         NO                    n/a
  AsyncNVMeReader/Writer    NO                    n/a
  RDMA transport            NO                    n/a
```

NVMe paths are tenant-namespaced in the Python backends:

```
<nvme_dir>/<tenant_id>/<sequence_id>_<block_index>.vmm_block
```

The `tenant_id` and `sequence_id` are sanitized by
`_sanitize_id()` against the regex `^[A-Za-z0-9_\-]{1,128}$`
before being concatenated into a path. This blocks `..`, `/`,
NUL, and overlong values. Both `_cuda_backend_py.py` and
`unified_backend.py` use this rule.

Specific enforcement sites:

- `memopt/vmm/__init__.py:104-115` (`VMM.allocate`),
  `:128-134` (`VMM.fetch`), `:156-163` (`VMM.free_sequence`).
  All take `_owner_lock`, look up `_sequence_owners[sequence_id]`,
  raise `PermissionError` if mismatch.

- `memopt/vmm/backends/_cuda_backend_py.py:29-42` and
  `unified_backend.py:39-52` define `_nvme_block_path()` with
  the `_sanitize_id()` whitelist.

- `memopt/cluster/gkd_store.py:587-598` reads the
  `MEMOPT_GKD_TENANT_ISOLATION` env var. When set, the content
  hash is salted with `tenant_id` so different tenants produce
  different hashes (no cross-tenant reuse).

TODO/FIXME items mentioning tenant: none found in the read set.

### 1.4 Stream awareness today

```
COMPONENT                       Stream tracking?
────────────────────────────────────────────────────────────
VMM facade                      no
TierManager                     no
CUDABackend                     two FIXED streams (_copy_stream, _nvme_stream)
                                created at __init__, reused forever.
                                Not associated with any handle.
async_copy(prefer_nvme_stream)  selects between the two fixed streams.
record_event(stream)            records but does not bind to a handle.
UnifiedBackend.async_copy       stream= parameter ignored.
CUDAVMMAllocator                no — quiesces all streams.
MemoptTorchAllocator            no — relies on caller-driven step_boundary().
libmemopt_vmm.so                cuStreamSynchronize on quiesce only.
PageTable                       no.
GKDStore                        no.
```

There is no PyTorch CCA-style "stream-locked free": no allocator
records that handle X is in flight on stream Y and refuses to
recycle X until Y has caught up. The serving path relies on
`step_boundary()` to bracket eviction between decode iterations,
which is correct only when the caller is disciplined. This is a
gap the substrate must fill.

### 1.5 Test surface and regression net

Total test functions found by `grep -r "def test_"`: 721.
Last full-suite run on `main` at `6f0678d`:
  682 collected (with `-k "not gpu and not cuda"`)
   19 skipped
   11 deselected
    0 failed
The grep count exceeds the collected count because some tests
are inside class bodies, parametrised, or in files that fail
collection on macOS without a built `libmemopt_vmm.so`.

Pre-existing collection failures on macOS / no-CUDA hosts:
  memopt/vmm/tests/test_cuda_vmm_smoke.py
  memopt/vmm/tests/test_torch_allocator_sanity.py
Both fail with `FileNotFoundError: libmemopt_vmm.so` because the
shared library is built only on hosts with CUDA. They are NOT
counted in 682.

Test files most likely to regress when the substrate ships:

```
memopt/vmm/tests/
    test_hal.py                 HAL detection / tier lookup
    test_vmm_smoke.py           VMM end-to-end happy path
    test_vmm_hardening.py       error paths, NVMe recovery
    test_vmm_benchmark.py       perf sanity (no thresholds asserted)
    test_layer3.py              tier_manager allocate/fetch/evict
    test_universal_profile.py   hw profile detection
    test_oracle.py              memory oracle
    test_global_oracle.py       cross-process oracle
    test_access_log.py          access tracking
    test_discovery.py           backend discovery
    test_pod_controller.py      pod-level orchestration
    test_rocm_backend.py        stub raises NotImplementedError

memopt/cluster/tests/
    test_gkd.py                 GKD lookup/register
    test_gum_hardening.py       block directory + remote fetch
    test_pillar5_gum.py         GUM end-to-end
    test_pillar2_twonode.py     2-node coherence
    test_pod_gkd.py             pod-level GKD

tests/
    test_pillar2_enhanced.py    PagedKVCache, tenant GKD stats
    test_hardware_support.py    cross-platform import smoke
    test_async_nvme.py          AsyncNVMeManager io_uring
    test_integration_gaps.py    cross-pillar wiring
    test_golden_image.py        end-to-end golden
```

These tests form the regression net. The substrate must keep
all of them passing through Phase A (parallel addition; no edits
to existing modules). Phase B will gate substrate-backed paths
behind `MEMOPT_VMM_USE_SUBSTRATE=1` and re-run the same tests
against the new path.

### 1.6 Gaps the substrate must fill

Each gap is a thing the existing code does NOT have. They drive
the Phase 2 design.

**G1. No unified handle type.**
`VMM.allocate()` returns `PageTableEntry`. `CUDAVMMAllocator.malloc()`
returns `int` (CUdeviceptr). `CUDABackend.allocate()` returns
`torch.Tensor` for HBM/DRAM and `str` for NVMe — the caller must
already know which. There is no single `MemoryHandle` type that
encapsulates `(va, size, tenant, tag, placement, stream_owner,
backend)`.

**G2. Tenant isolation lives only in Python.**
The C++ allocator and page table have no tenant field. A consumer
who calls into `libmemopt_vmm.so` directly (or who has raw
`cuMemMap` access) bypasses tenant enforcement. The substrate
should keep tenant in the handle metadata at all layers and
namespace per-tenant state explicitly.

**G3. No stream registry.**
No allocator tracks which stream a handle was last used on. Free
is fire-and-forget. There is no PyTorch CCA-style guard that
prevents reusing a handle on stream B before stream A has caught
up. The substrate must add this.

**G4. No event surface.**
No code path emits `alloc` / `free` / `evict` / `promote` /
`migrate` events to subscribers. Higher layers (tier orchestrator,
prefetch oracle, policy engine) cannot observe the allocator
without polling stats. The substrate must add a lock-free ring
for this.

**G5. No fabric-handle export.**
`cuMemExportToShareableHandle` is never called. Cross-node sharing
goes through the RDMA transport in `csrc/transport/`, not through
VMM peer mapping. The substrate should expose
`backend.export_fabric_handle(ph)` even if the v1 implementation
returns `None` on systems without IMEX.

**G6. No CXL / NUMA backend.**
No code path allocates from a NUMA node or recognises a CXL
memory device. CXL appears as a NUMA node with memory but no
CPUs; nothing parses `/sys/devices/system/node/`.

**G7. No Intel Level Zero backend.**
No detection or allocation path for Intel GPUs.

**G8. ROCm is a stub.**
`ROCmBackend` raises on construction. The reference impl in
`_rocm_backend_py.py` is not wired into the public path.

**G9. No placement DSL.**
`tier` is one of the strings `"hbm"`/`"dram"`/`"nvme"` understood
per-backend. There is no `"auto"` / `"hot"` / `"warm"` / `"cold"`
abstraction that maps to per-backend tiers.

**G10. No TTL on allocations.**
Nothing expires a handle. The block directory has a default TTL
(`MEMOPT_BLOCK_DIR_TTL_S=3600s`) for cluster-wide content
indexing, but individual VA allocations are not time-bounded.

**G11. No tag-based routing.**
`CUDAVMMAllocator.malloc(size, tag)` accepts a tag string; the
allocator stores it in the page metadata for debug only and does
not consult it for placement.

**G12. 16 GiB hard cap on the C allocator.**
`MEMOPT_MAX_PAGES = 8192` × 2 MiB = 16 GiB. H100 80 GB needs
40,960 pages; MI300X 192 GB needs 98,304 pages. The substrate
must allow the page table to grow with the device pool size.

**G13. No 64 KiB pages.**
The native granularity on Hopper/Ampere is 2 MiB. vAttention's
patched UVM driver is required for 64 KiB pages. memopt does not
ship a custom driver. The substrate must document this and
default to 2 MiB; expose a feature flag for users with the
patched driver.

These thirteen gaps are the design surface for Phase 2.

---

## Phase 2 — Design

The substrate is added as a new package `memopt/substrate/`. No
existing module is edited in Phase A. Every decision below is
chosen with a 20-year horizon: future maintainers should never
have to undo a shortcut. Where a behaviour matches an established
production pattern (PyTorch CCA, CUDA VMM driver semantics), the
substrate copies it exactly rather than inventing a near-twin.

### 2.0 Constraints carried forward from Phase 1

  C0  Existing public APIs (VMM, GKDStore, serving endpoints,
      kernels) keep their current signatures. The substrate is
      added underneath; migration is opt-in via flag.
  C1  Tenant isolation is API-level. Hardware side channels and
      driver-level vulnerabilities are out of scope. (See 2.6.)
  C2  Stream semantics match PyTorch CCA exactly. Reference:
      `c10/cuda/CUDACachingAllocator.cpp` in the PyTorch repo.
      (See 2.5 for the edge-case audit. — DECISION 2.)
  C3  Bridge test `tests/test_substrate_legacy_parity.py` is the
      trust anchor for Phase B. Byte-equivalent results required.
      (See 2.9. — DECISION 3.)
  C4  Numbers in this doc are TARGETS, not measurements. Section
      2.10 cites the research from which targets are drawn; real
      numbers come from benchmarks during Phase A.

### 2.1 Public API surface (Python)

The substrate's Python entry point is `memopt.substrate`. To keep
the import path short for users, `memopt/__init__.py` re-exports
the four primary symbols (`alloc`, `free`, `context`, `stats`,
`observe`). The substrate module is the implementation; `memopt.x`
is a thin alias.

#### memopt.alloc

```python
def alloc(
    size_bytes: int,
    *,
    tenant: str = "_default",
    tag: str = "default",
    placement: str = "auto",
    stream: Optional["torch.cuda.Stream"] = None,
    ttl_seconds: Optional[float] = None,
    hint: Optional[dict] = None,
) -> MemoryHandle: ...
```

Pre-conditions:
  - `size_bytes > 0`. Zero is rejected with `ValueError`. (CUDA
    runtime accepts size=0 but PyTorch CCA does too — we match
    the cuda runtime semantic of `cudaMalloc(0) -> nullptr` by
    raising; never returning a sentinel handle that callers must
    check.)
  - `tenant` matches `^[A-Za-z0-9_\-]{1,128}$`. Same regex as the
    existing `_sanitize_id()` (Phase 1 §1.3) so no migration risk.
  - `tag` matches `^[A-Za-z0-9_\-]{1,64}$`. Tags are routing keys
    for arena size classes and observers.
  - `placement` is one of the values listed in §2.2.
  - `stream`, when provided, must belong to the device that the
    chosen backend allocates from.
  - `ttl_seconds`, when provided, must be > 0.
  - `hint` is an opaque dict reserved for future use (Layer 3+).
    The substrate ignores its contents in v1; it is preserved on
    the handle so observers can read it. Callers should NOT
    depend on any key.

Post-conditions:
  - Returned `MemoryHandle` is owned by `tenant` and tagged `tag`.
  - The handle's physical memory is backed by the chosen backend.
    `handle.placement` reports the actual tier (which may differ
    from the requested `placement` when `placement="auto"`).
  - One `Event(kind="alloc", ...)` is emitted before the call
    returns.
  - If `stream` is provided, the handle is bound to that stream
    in the StreamRegistry (§2.5). Subsequent `free()` is
    stream-locked: the handle's physical backing cannot be reused
    on another stream until the binding stream catches up to a
    recorded event.

Raises:
  - `ValueError` for invalid arguments.
  - `MemoryError` when no backend can satisfy the request and
    eviction has already been attempted.
  - `PermissionError` when the calling thread holds a context
    (`memopt.context(tenant=...)`) for a tenant other than the
    one passed.
  - `RuntimeError` when the chosen backend reports a fatal
    driver error (e.g. CUDA driver fault). The error message
    includes the backend name and the underlying status.

Thread safety: SAFE. Multiple threads may call `alloc()`
concurrently for the same or different tenants. Per-tenant arena
locks serialize tenants; cross-tenant calls do not contend.

Stream safety: SAFE. The substrate records the binding stream
under its own lock; it does not synchronize the caller's stream.
The caller may issue device work on `stream` immediately after
`alloc()` returns.

CUDA callback safety: NOT SAFE. `alloc()` may take locks and
make CUDA driver calls; both are forbidden inside
`cuStreamAddCallback` per CUDA driver contract. Document this
explicitly in the docstring.

#### memopt.free

```python
def free(handle: MemoryHandle) -> None: ...
```

Pre-conditions:
  - `handle` is a `MemoryHandle` returned by `memopt.alloc()` or
    by `MemoryHandle.__enter__`.
  - The handle has not been freed before. Double-free is
    idempotent and silent (matching PyTorch CCA, which treats
    double-free as a no-op rather than raising); a debug
    counter `double_free_count` is incremented and visible in
    `memopt.stats()`.
  - The caller's tenant context (if any) matches `handle.tenant`.

Post-conditions:
  - The handle is marked freed in the registry. Future
    operations on the handle (read/write/as_tensor) raise
    `RuntimeError("handle is freed")`.
  - If the handle had a stream binding, the substrate records a
    `cudaEvent` on that stream and queues the physical block for
    reclaim only after the event completes.
  - One `Event(kind="free", ...)` is emitted before the call
    returns.

Raises:
  - `PermissionError` on tenant mismatch (same rule as alloc).
  - `RuntimeError` only on irrecoverable backend errors during
    physical release. Logical free is best-effort and never
    raises.

Thread/stream safety: SAFE. Same model as alloc.

#### memopt.context

```python
def context(
    tenant: Optional[str] = None,
    tag: Optional[str] = None,
    placement: Optional[str] = None,
) -> AbstractContextManager: ...
```

A `with`-block that scopes default values for subsequent
`memopt.alloc()` calls on the calling thread. Implemented as a
`contextvars.ContextVar` so the binding propagates to threads
spawned by `concurrent.futures.ThreadPoolExecutor.submit()` and
to coroutines via `asyncio`.

```
with memopt.context(tenant="alice", tag="kv_cache"):
    h1 = memopt.alloc(1 << 20)            # tenant=alice, tag=kv_cache
    h2 = memopt.alloc(1 << 20, tag="weight")  # tenant=alice, tag=weight
```

Pre-conditions: tenant/tag/placement, when given, satisfy the
same regex/whitelist as `alloc()`.

Post-conditions: on exit, the previous context is restored. The
context manager is reentrant.

Raises: `ValueError` for invalid arguments at `__enter__` time.

Thread safety: SAFE. `ContextVar` is thread-local with explicit
async propagation rules.

CUDA callback safety: callable but not useful inside callbacks
(no allocations should happen there).

#### memopt.stats

```python
def stats(
    tenant: Optional[str] = None,
) -> dict: ...
```

Returns a snapshot dict. When `tenant` is None and the calling
process has the admin token (`MEMOPT_ADMIN_TOKEN` env var
configured AND the calling thread holds a matching context),
returns aggregate stats across all tenants. Otherwise returns
stats for the calling tenant only (same enforcement as VMM
today).

Snapshot shape:

```python
{
    "tenant": str,                        # echo of input or "_aggregate"
    "live_handles": int,
    "live_bytes": int,
    "high_water_bytes": int,
    "tag_breakdown": {str: {              # keyed by tag
        "live_handles": int,
        "live_bytes": int,
    }},
    "placement_breakdown": {str: int},    # bytes per current placement
    "events_dropped": int,                # ring overflow counter (§2.8)
    "double_free_count": int,
    "stream_locked_pending": int,
    "ttl_expired_count": int,
    "alloc_count": int,                   # cumulative
    "free_count": int,                    # cumulative
    "evict_count": int,                   # cumulative
    "promote_count": int,                 # cumulative
    "migrate_count": int,                 # cumulative
    "backend": {                          # per-backend health
        "cuda": {
            "available": bool,
            "device_count": int,
            "granularity_bytes": int,
        },
        "hip":  {...},
        "cxl":  {...},
        "cpu":  {...},
    },
}
```

Pre-conditions: none.
Post-conditions: snapshot is internally consistent (taken under
a single lock pass per tenant; cross-tenant aggregation is best-
effort consistent — sums may not perfectly match a simultaneous
free).
Raises: `PermissionError` when `tenant=None` is requested without
the admin token.
Thread safety: SAFE. Stream safety: SAFE.

#### memopt.observe

```python
def observe(
    event: Literal["alloc", "free", "evict",
                   "promote", "migrate"],
    callback: Callable[[Event], None],
) -> SubscriptionHandle: ...
```

Subscribes `callback` to events of the given kind. Returns a
`SubscriptionHandle` which can be passed to `memopt.unobserve()`
or used as a context manager (the subscription is dropped on
`__exit__`).

Delivery contract — see §2.8.

Pre-conditions: `event` is one of the five literals; `callback`
is callable.
Post-conditions: callbacks are invoked from a dedicated
dispatcher thread, never the allocating thread.
Raises: `ValueError` for unknown event kinds.
Thread safety: SAFE.
CUDA callback safety: subscribers run on a Python thread, NOT
inside `cuStreamAddCallback`. Subscribers may call CUDA APIs.

#### MemoryHandle

```python
@dataclass
class MemoryHandle:
    handle_id: int
    size_bytes: int
    tenant: str
    tag: str
    placement: str                          # current physical tier
    stream: Optional["torch.cuda.Stream"]   # binding stream, or None
    backend_name: str                       # "cuda" | "hip" | ...
    created_at: float                        # time.monotonic()
    ttl_seconds: Optional[float]
    hint: Optional[dict]                    # caller-supplied, opaque

    def read(self, offset: int = 0,
             size: Optional[int] = None) -> bytes: ...
    def write(self, data: bytes,
              offset: int = 0) -> None: ...
    def as_tensor(self, dtype, shape) -> "torch.Tensor": ...
    def as_numpy(self, dtype, shape) -> "np.ndarray": ...
    def stats(self) -> dict: ...
    def free(self) -> None: ...

    # Context-manager protocol (auto-free on exit)
    def __enter__(self) -> "MemoryHandle": ...
    def __exit__(self, exc_type, exc, tb) -> None: ...
```

Per-method:

  read(offset=0, size=None)
    Pre: handle is live; 0 <= offset; offset + size <= size_bytes
    Post: returns `size` bytes (or `size_bytes - offset` if size
          is None) as a Python `bytes` object. For HBM/DRAM
          handles the read goes through a host staging buffer.
          For NVMe handles it is a file read.
    Raises: RuntimeError(handle is freed); ValueError on bounds.
    Thread/stream safety: SAFE; performs an implicit
          synchronisation against the binding stream when handle
          is bound. Callers wanting stream-ordered reads should
          use `as_tensor()` and `tensor.copy_()`.

  write(data, offset=0)
    Symmetric to read. Same safety.

  as_tensor(dtype, shape)
    Pre: handle is live; backend supports torch interop (cuda,
         hip, cpu — NOT nvme/cxl raw paths in v1). dtype/shape
         are torch-compatible; product(shape) * sizeof(dtype) ==
         size_bytes.
    Post: returns a `torch.Tensor` view over the handle's
          memory. The handle's lifetime extends until both the
          handle and the tensor are released (the tensor holds a
          back-reference).
    Raises: NotImplementedError when backend is nvme/cxl in v1
          (use read/write); RuntimeError(handle is freed).
    Thread safety: the returned tensor is owned by the calling
          thread per torch convention.
    Stream safety: the tensor inherits the handle's binding
          stream. Subsequent kernel launches on a different
          stream require `tensor.record_stream(other_stream)`
          (the standard PyTorch idiom).

  as_numpy(dtype, shape)
    Pre: same as as_tensor but for CPU-resident handles only
         (cpu, dram, nvme via mmap, cxl via mmap). HBM handles
         raise NotImplementedError — copy via as_tensor().cpu()
         first.
    Post: returns a numpy array view; lifetime is shared with
          the handle.

  stats()
    Per-handle stats: bytes, age, last_access, evict count for
    THIS handle, current placement, ttl remaining.

  free()
    Calls `memopt.free(self)`. Idempotent (counter only).

  __enter__ / __exit__
    `with memopt.alloc(N) as h: ...` auto-frees on exit. Auto-
    free runs through `memopt.free` so all stream-locked rules
    apply.

CUDA callback safety: NONE of MemoryHandle's methods are safe
to call inside `cuStreamAddCallback`. Documented per-method.

### 2.2 Placement values

Eight placement strings are accepted. Each maps to a per-backend
strategy. When a backend cannot satisfy an EXPLICIT placement, it
raises `MemoryError`; when `placement="auto"` (or one of the
soft preference values), the substrate falls back to the next
best tier.

```
placement   meaning           cuda          hip           cxl       cpu       LZ stub
─────────── ───────────────── ───────────── ───────────── ───────── ───────── ────────
auto        let memopt decide hbm or dram   hbm or dram   dram      ram       raises
hot         strong pref HBM   hbm           hbm           dram      ram       raises
warm        strong pref DRAM  dram (pinned) dram (pinned) cxl       ram       raises
cold        strong pref NVMe  nvme          nvme          cxl       nvme      raises
hbm         explicit HBM      hbm           hbm           ERR       ERR       raises
dram        explicit DRAM     dram (pinned) dram (pinned) dram      ram       raises
cxl         explicit CXL node ERR if no CXL ERR if no CXL cxl       ERR       raises
nvme        explicit NVMe     nvme          nvme          nvme      nvme      raises
```

Notes:
  - "ERR" means `MemoryError("backend X cannot satisfy
    placement Y")`. The error message names the backend, the
    placement, and the reason.
  - On the CUDA backend with no CXL hardware, `placement="cxl"`
    raises rather than silently falling through to DRAM, because
    explicit placement is a contract. `placement="warm"` falls
    through.
  - "dram (pinned)" means `cudaHostAlloc(cudaHostAllocPortable |
    cudaHostAllocMapped)` so the allocation is GPU-addressable.
    "dram" without GPU context (cpu/cxl/LZ-stub backends) is
    plain `malloc`/`mmap`.
  - The Level Zero stub raises `NotImplementedError` from
    `is_available()` returning False; the substrate skips to
    the next backend.

### 2.3 Backend strategy interface

Seven primitives, chosen because they are the union of what the
following systems expose:

  - vAttention (USENIX ATC 2024): cuMemAddressReserve / Create /
    Map / SetAccess / Unmap / Release / AddressFree.
  - GMLake (ASPLOS 2024): same seven CUDA primitives.
  - PyTorch expandable_segments (since 2.1): same seven, wrapped
    in `c10::cuda::CUDAExpandableSegment`. (TODO-VERIFY: confirm
    file path is `c10/cuda/CUDAAllocatorConfig.cpp` — the API
    has moved between PyTorch versions.)
  - AMD HIP VMM API: hipMemAddressReserve / Create / Map /
    SetAccess / Unmap / Release / AddressFree. (TODO-VERIFY:
    confirm exact symbol names against ROCm 6.x docs at
    https://rocm.docs.amd.com/projects/HIP/en/latest/.)
  - Intel Level Zero: zeMemAllocShared / Device / Host. The
    primitives DO NOT line up; LZ uses a unified-address model.
    See §2.4 for why LZ is a stub in v1.
  - libnuma + cuMemHostRegister for CXL: numa_alloc_onnode +
    register-as-pinned. Different shape but expressible as the
    seven primitives where reserve/free are no-ops on the host
    address.

```python
class PhysLoc(enum.Enum):
    HBM    = "hbm"
    DRAM   = "dram"
    CXL    = "cxl"
    NVME   = "nvme"

@dataclass
class PhysHandle:
    backend_name: str
    location:     PhysLoc
    raw:          object       # backend-private; cuMemHandle, void*, fd, ...

class BackendStrategy(abc.ABC):
    BACKEND_NAME: ClassVar[str]
    IS_REAL:      ClassVar[bool]   # False for honest stubs

    @abc.abstractmethod
    def is_available(self) -> bool: ...
        # Cheap. Side-effect-free. Does NOT initialise CUDA / HIP.
        # Suitable to call at every alloc.

    @abc.abstractmethod
    def granularity_bytes(self) -> int: ...
        # Smallest mappable physical page on this backend.
        # CUDA: cuMemGetAllocationGranularity(...,
        #        CU_MEM_ALLOC_GRANULARITY_RECOMMENDED).
        # HIP: hipMemGetAllocationGranularity (TODO-VERIFY).
        # CPU/CXL: getpagesize() (typically 4 KiB or huge page).

    @abc.abstractmethod
    def reserve_va(self, size: int) -> int: ...
        # Reserve a virtual address range. Returns the VA as an
        # integer. CUDA: cuMemAddressReserve. CPU: mmap(MAP_NORESERVE).

    @abc.abstractmethod
    def create_physical(
        self, size: int, location: PhysLoc
    ) -> PhysHandle: ...
        # Allocate physical backing. Does NOT map.

    @abc.abstractmethod
    def map(self, va: int, ph: PhysHandle, offset: int = 0) -> None: ...

    @abc.abstractmethod
    def set_access(self, va: int, size: int,
                   devices: list[int]) -> None: ...
        # Make `va` readable/writable from each device id in
        # `devices`. CPU backend: no-op.

    @abc.abstractmethod
    def unmap(self, va: int, size: int) -> None: ...

    @abc.abstractmethod
    def release_physical(self, ph: PhysHandle) -> None: ...

    @abc.abstractmethod
    def free_va(self, va: int, size: int) -> None: ...

    @abc.abstractmethod
    def export_fabric_handle(
        self, ph: PhysHandle
    ) -> Optional[bytes]: ...
        # Returns CU_MEM_HANDLE_TYPE_FABRIC bytes when supported
        # (CUDA 12.4+, IMEX daemon present). None otherwise. Never
        # raises — absence of fabric support is normal.
```

The substrate's allocation manager (§2.5) calls into this
interface and never directly touches `torch.cuda`, `cudart`, or
`hipMemMap`.

### 2.4 Concrete backends

Five concrete classes plus the abstract base above.

#### CUDABackend (real, v1.0)

Maps to:
  reserve_va        cuMemAddressReserve
  create_physical   cuMemCreate (CU_MEM_ALLOCATION_TYPE_PINNED,
                    location HBM or DRAM-pinned)
  map               cuMemMap
  set_access        cuMemSetAccess
  unmap             cuMemUnmap
  release_physical  cuMemRelease
  free_va           cuMemAddressFree
  granularity       cuMemGetAllocationGranularity(
                    CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)
  export_fabric     cuMemExportToShareableHandle(
                    CU_MEM_HANDLE_TYPE_FABRIC), CUDA 12.4+ only.
                    Returns None on older drivers and on systems
                    without the IMEX daemon — both are normal.

Honest limit (G13): native granularity on Hopper, Ada, Ampere is
2 MiB. 64 KiB pages require vAttention's patched UVM driver,
which memopt does NOT ship. v1 defaults to 2 MiB. Users with the
patched driver may set `MEMOPT_ENABLE_VATTENTION_DRIVER=1`; the
substrate then probes the smaller granularity at startup. If the
probe fails, the flag is logged as ignored and we fall back to
2 MiB. Document this in `docs/substrate_v1_design.md` and in
the CUDABackend docstring.

The CUDABackend wraps `libmemopt_vmm.so` in Phase A by
extracting the seven primitives from `vmm_allocator.cpp` into
a thin C ABI. The existing public C API (`memopt_malloc` /
`memopt_free` etc.) stays. Phase B can then route the legacy
allocator at `memopt_malloc` through the substrate; Phase C
flips the default. (See 2.9 and 2.12.)

#### HIPBackend (real, day one — gated on driver capability)

Maps to:
  reserve_va        hipMemAddressReserve
  create_physical   hipMemCreate
  map               hipMemMap
  set_access        hipMemSetAccess
  unmap             hipMemUnmap
  release_physical  hipMemRelease
  free_va           hipMemAddressFree
  granularity       hipMemGetAllocationGranularity
                    (TODO-VERIFY: confirm signature against ROCm
                    6.x. As of ROCm 5.7, the symbol exists but
                    the docs warn the API is "Beta".)
  export_fabric     none in ROCm 6 (TODO-VERIFY); returns None.

Capability probe at `is_available()`:
```
hipDeviceGetAttribute(
    &supported,
    hipDeviceAttributeVirtualMemoryManagementSupported,
    device_id)
```
  (TODO-VERIFY: attribute name in ROCm 6.x. If the attribute is
  not present, fall through to is_available()=False so the
  substrate skips this backend rather than crashing. Document
  the version we tested against in the implementation PR.)

Status notes (carried into the doc honestly):
  - ROCm VMM API is marked Beta in ROCm 6.x docs.
  - Linux supported; Windows under development.
  - Tested hardware: TODO-VERIFY against MI300X. If the test
    rig in CI does not have an MI300X, the day-one impl ships
    with `IS_REAL=True` but flag tests `@gpu_amd` so they are
    skipped without hardware. The substrate must NOT claim AMD
    works in any external doc until the @gpu_amd tests pass on
    real hardware.

This replaces the existing `ROCmBackend` stub (G8). The legacy
stub stays in `memopt/vmm/backends/rocm_backend.py` to preserve
import compatibility; new code uses the substrate path.

#### LevelZeroBackend (honest stub, v1.0)

Why a stub: Level Zero's allocation model is
`zeMemAllocShared/Device/Host` plus `zeContextCreate`, NOT a
reserve+create+map split. Re-shaping the strategy to fit LZ
without losing the seven-primitive abstraction is a non-trivial
design change that should not be rushed for v1.

v1 behaviour:
  - `is_available()` returns True iff `libze_loader.so` is on
    the dynamic linker path AND `zeInit(0)` succeeds.
  - Every other method raises `NotImplementedError` with a
    clear message: "Intel Level Zero backend not implemented in
    v1.0. See docs/substrate_v1_design.md §2.4. Track via
    issue #TBD."
  - The allocation manager treats LZ as unavailable when
    is_available() returns False, so on Intel Arc / Ponte
    Vecchio hosts without the patch, the substrate falls
    through to CPUFallbackBackend without raising.

v1.1+: real implementation in a follow-up doc.

#### CXLBackend (real, where available)

Detection (the heuristic from the prompt, hardened):
  - Walk `/sys/devices/system/node/`.
  - For each `nodeN`: if `cpulist` is empty AND `meminfo`
    reports MemTotal > 0, treat node N as a CXL/host-backed
    NUMA node.
  - TODO-VERIFY: this heuristic catches many CXL deployments
    but is NOT authoritative. Some BIOS configurations expose
    CXL as a normal NUMA node with assigned CPUs. The
    implementation must document the exact kernel and
    BIOS combinations it has been verified against. If the
    deployment environment uses a vendor-specific path
    (`/sys/bus/cxl/`), prefer that over the cpulist heuristic
    when both are present.

Allocation:
  - Userspace: `numa_alloc_onnode(size, node)` (libnuma) OR
    `set_mempolicy(MPOL_BIND, &node_mask, ...) + mmap(...)`
    when libnuma is unavailable.
  - GPU access: when the CUDA backend is also active and the
    handle is mapped for GPU access, call
    `cuMemHostRegister(ptr, size, CU_MEMHOSTREGISTER_PORTABLE
    | CU_MEMHOSTREGISTER_DEVICEMAP)`. The CUDA driver returns
    a device pointer via `cuMemHostGetDevicePointer`.

Honest performance bounds (cited as TARGETS, not measurements):
  - Latency: ~200-500 ns vs ~10 ns local DRAM (per CXL 2.0 spec
    band; vendor-specific within that range).
  - Bandwidth: lower than HBM, higher than NVMe; varies by CXL
    generation (1.x ~32 GB/s, 2.x ~64 GB/s, 3.x ~128 GB/s per
    x16 port — these are theoretical link bandwidths, not
    measured).
  - The placement table (§2.2) reflects this.

Reference doc: §2.10 cites the CXL spec sections we relied on.

#### CPUFallbackBackend (real, v1.0)

For CI, dev machines without GPU, and as the universal safety
net.
  reserve_va        mmap(NULL, size, PROT_NONE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
  create_physical   anonymous mmap of `size` (with appropriate
                    protection)
  map               mremap or copy — see implementation note
  set_access        no-op (single host, single device)
  unmap             munmap
  release_physical  munmap
  free_va           munmap
  granularity       sysconf(_SC_PAGESIZE), promoted to a huge
                    page (2 MiB) when `MAP_HUGETLB` succeeds.
  export_fabric     None.

Implementation note: a strict map/unmap-on-physical model on
plain mmap requires `mremap(MREMAP_FIXED)` plus a tracker.
v1 may take the simpler approach of "create_physical does the
real allocation, map is a no-op when va == ph.raw, unmap +
release_physical both call munmap once". Document this divergence
clearly so future GPU-backend work understands why.

Used when no GPU backend's `is_available()` returns True. Tests
that don't need real GPU run on this. The substrate's behaviour
here is authoritative for CI.

### 2.5 The Allocation Manager (the MMU)

Three core data structures plus the event ring.

```
                     ┌────────────────────────────┐
                     │    AllocationManager        │
                     │      (process singleton)    │
                     └──────────────┬──────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
   ┌────────▼────────┐    ┌─────────▼─────────┐    ┌────────▼────────┐
   │ HandleRegistry  │    │  TenantArenas     │    │ StreamRegistry  │
   │   (process)     │    │   {tenant: arena} │    │   (process)     │
   └────────┬────────┘    └─────────┬─────────┘    └────────┬────────┘
            │                       │                       │
   ┌────────▼────────┐    ┌─────────▼─────────┐    ┌────────▼────────┐
   │   EventRing     │    │  Backends list    │    │  TTL sweeper    │
   │  (lock-free)    │    │ [CUDA, HIP, CXL,  │    │ (background)    │
   │                 │    │  CPU, LZ stub]    │    │                 │
   └─────────────────┘    └───────────────────┘    └─────────────────┘
```

#### HandleRegistry

Process-wide map of `handle_id -> HandleRecord`. handle_id is a
monotonically increasing 64-bit integer.

```python
@dataclass
class HandleRecord:
    handle_id:     int
    tenant:        str
    tag:           str
    size_bytes:    int
    va:            int
    physical:      PhysHandle
    placement:     str               # current tier
    stream_owner:  Optional["torch.cuda.Stream"]
    created_at:    float
    ttl_seconds:   Optional[float]
    ref_count:     int               # incremented by as_tensor()
                                     # bridges
    backend:       BackendStrategy
    state:         Literal["live", "freed_pending_stream",
                            "freed_pending_event",
                            "released"]
    free_event:    Optional["torch.cuda.Event"]   # set on free
                                                   # when stream-bound
```

States transition: `live -> freed_pending_stream ->
freed_pending_event -> released`.

  - `live`: allocated, may be in use.
  - `freed_pending_stream`: free() was called; we recorded an
    event on the binding stream and the handle is queued. The
    physical block is NOT yet returned to the arena.
  - `freed_pending_event`: in the queue; the physical block is
    eligible for reuse only after `cudaEventQuery(free_event)`
    returns `CUDA_SUCCESS`.
  - `released`: physical block is in the arena freelist or has
    been returned to the backend.

Concurrency: `HandleRegistry` uses a `ReadWriteLock`. Lookups
are read-locked; insert/state-change is write-locked. A single
contended write does not block lookups.

#### TenantArena

One arena per tenant. The arena holds size-class freelists in
the jemalloc style.

```python
class TenantArena:
    tenant: str
    free_blocks:   dict[int, deque[HandleRecord]]
        # size_class_bytes -> queue of returned blocks
    committed_bytes:    int
    high_water_bytes:   int
    tag_stats:     dict[str, TagStats]
    _lock:         threading.Lock        # per-tenant
```

Size classes follow PyTorch CCA: 512 B, 1 KiB, 2 KiB, 4 KiB,
... doubling up to 2 MiB; then 2 MiB, 4 MiB, 6 MiB, ... in
2 MiB steps up to 1 GiB; then 1 GiB, 2 GiB, ... up to the
device pool. (Same scheme as `c10::cuda::CUDACachingAllocator`'s
`get_allocation_size` table.) (TODO-VERIFY: confirm exact sizes
against current PyTorch; quote the source path in the impl PR.)

Per-tenant lock means cross-tenant calls do NOT contend. Within
a tenant, `alloc()` and `free()` serialize on the arena lock.
The lock is released BEFORE any backend driver call to avoid
holding a Python lock during a CUDA driver call.

#### StreamRegistry — DECISION 2: PyTorch CCA semantic exactly

This is the single most important design decision below. The
substrate copies PyTorch CUDACachingAllocator's stream-locked
semantic byte-for-byte. Reference:
`pytorch/c10/cuda/CUDACachingAllocator.cpp` in the upstream
PyTorch repo. (TODO-VERIFY: line numbers move between PyTorch
releases. Cite by symbol name, not line number, in this doc.)

Core data:

```python
class StreamRegistry:
    # handle_id -> set of streams that have used it
    stream_uses: dict[int, set["torch.cuda.Stream"]]

    # per-stream FIFO of (event, handle_record) pending reclaim
    pending: dict["torch.cuda.Stream",
                  deque[tuple["torch.cuda.Event",
                              HandleRecord]]]

    # cached events ready to reuse (PyTorch CCA does this)
    event_pool: deque["torch.cuda.Event"]

    _lock: threading.Lock
```

Algorithm (matching PyTorch CCA):

  on alloc(stream=S):
    record_stream(handle, S)            # adds S to stream_uses

  on tensor.record_stream(other_S):     # caller-driven idiom
    record_stream(handle, other_S)

  on free(handle):
    streams = stream_uses.pop(handle_id, set())
    if not streams:
        # Never used by any stream; safe to recycle now.
        return_to_arena(handle)
    else:
        for S in streams:
            event = event_pool.pop() or torch.cuda.Event()
            event.record(stream=S)
            pending[S].append((event, handle))
            handle.state = "freed_pending_event"

  on every alloc() (process-incoming events):
    # Drain ready pending entries (cheap: check head only)
    for S in list(pending.keys()):
        q = pending[S]
        while q and q[0][0].query():       # ready
            event, h = q.popleft()
            event_pool.append(event)
            return_to_arena(h)

This block-on-event-not-stream pattern is the heart of CCA: a
free is observed by the allocator as "the block becomes
re-allocatable when the recorded event has fired", not "when
the binding stream is synchronised". That distinction is what
lets two streams share an allocator without synchronising each
other.

##### Edge cases the substrate must handle (PyTorch CCA parity)

Audit the upstream behaviour, then the substrate's match:

  E1. CUDA Graphs (capture/replay).
      PyTorch CCA: when a stream is in CUDA Graph capture mode,
      allocations are routed to a private mempool tied to the
      capture. Frees are deferred until capture ends. On graph
      replay, the same VAs are reused.
      Substrate v1: detect capture mode via
      `cudaStreamIsCapturing(stream, &mode)`. When capture is
      active, route the alloc to a per-capture private mempool
      (a sub-arena under the tenant). Frees during capture do
      NOT return to the arena; they're queued and released
      when capture ends. On replay, the substrate does NOT
      re-allocate — the captured graph holds raw VAs that are
      already valid. (TODO-VERIFY: confirm `cudaStreamIsCapturing`
      and `cudaStreamCaptureStatus` API names against CUDA
      Runtime API docs at developer.nvidia.com.)
      No divergence from PyTorch CCA.

  E2. Peer access (multi-GPU).
      PyTorch CCA: on first cross-device touch,
      `cudaDeviceEnablePeerAccess` is called; subsequent
      allocations on the source device are mapped for access
      from the peer device.
      Substrate: `set_access(va, size, devices)` is the
      primitive. `as_tensor()` records the device the tensor
      runs on; if the tensor is consumed on a peer device, the
      caller passes that device to a future `set_access` call.
      The substrate does NOT auto-enable peer access in v1
      (PyTorch CCA's auto-enable has caused subtle bugs around
      per-process resource limits; we make it explicit). This
      IS a divergence from PyTorch CCA — documented and
      intentional.

  E3. IPC tensors.
      PyTorch CCA: `cudaIpcGetMemHandle` /
      `cudaIpcOpenMemHandle`. CCA tracks reference counts so an
      allocation backing an IPC handle is not freed while
      another process holds it.
      Substrate: not exposed in the v1 public API
      (`memopt.alloc` does not return IPC handles). Cross-
      process sharing in memopt today goes through the RDMA
      transport (Phase 1 §1.2). The substrate's
      `export_fabric_handle()` returns an opaque `bytes` for
      CUDA 12.4+ FABRIC handles; the consumer's API for
      importing it in another process is OUT OF SCOPE for v1.
      Documented as a v1.1 follow-up.
      Divergence from PyTorch CCA: yes, intentional.

  E4. Cross-device.
      PyTorch CCA: each device has its own allocator instance.
      Cross-device tensors require explicit `tensor.to(device)`.
      Substrate: same. The substrate keeps one
      `AllocationManager` per process, but the manager indexes
      backends by device id. `memopt.alloc(...,
      placement="hbm")` on a process with multiple CUDA devices
      uses the current device per `cuCtxGetCurrent` (matching
      PyTorch's `torch.cuda.current_device()`). Explicit
      multi-device requires extending the API in v1.1; v1 is
      single-device per call.
      Divergence: same model as PyTorch CCA, but the substrate
      surface for explicit device selection is deferred to v1.1.

  E5. Cross-stream (same device).
      PyTorch CCA: this is the canonical case the stream_uses
      set covers. Substrate matches exactly per the algorithm
      above.
      No divergence.

  E6. Out-of-memory retry.
      PyTorch CCA: on alloc failure, frees the cache, retries,
      and if still failing, raises OOM with diagnostic.
      Substrate: yes — on failure, the AllocationManager
      requests eviction (calls into the tier orchestrator at
      Layer 2 if registered, or directly into the backend's
      legacy `evict_to_target` for the CUDA path). After one
      retry, MemoryError is raised with the snapshot dict from
      `stats()` embedded in the message.
      No divergence.

  E7. Expandable segments (PyTorch ≥ 2.1).
      PyTorch CCA: `expandable_segments=True` switches CCA to
      a VMM-backed model where one large VA reservation is
      mapped/unmapped piece by piece, identical to memopt's
      existing strategy.
      Substrate: this IS our default model — we always use
      VMM. The substrate is what `expandable_segments` would
      look like if it were the only path. No flag needed.

  E8. Caching across allocations of the same size.
      PyTorch CCA: hits the size-class freelist before falling
      to the system. Substrate: same — TenantArena freelists.
      No divergence.

  E9. Block splitting and merging.
      PyTorch CCA: a 2 MiB block can be split for sub-2 MiB
      allocations; adjacent freed sub-blocks coalesce.
      Substrate: same. Sub-2 MiB allocations from the same
      tenant share a 2 MiB physical page when they fit.
      Coalescing runs inside the arena lock on free.
      No divergence.

The upshot: in 2.5 the substrate matches PyTorch CCA on E1,
E5–E9 and intentionally diverges on E2 (no auto peer-access),
E3 (no IPC in v1), E4 (no multi-device alloc in v1). The
divergences are documented in the API surface and in user-facing
docs.

#### EventRing

Lock-free single-producer-multiple-consumer ring buffer.
Capacity 4096 entries (configurable via
`MEMOPT_EVENT_RING_CAPACITY`). Producer is the alloc/free/evict
path (any allocating thread). Consumers are the dispatcher
threads owned by `memopt.observe()` subscribers.

Implementation: a `multiprocessing.Array(ctypes.c_uint64)` is
overkill; a `collections.deque(maxlen=N)` is too slow. Use a
fixed-size `array.array` of struct-packed event records plus
two `threading.atomic` indices (`head`, `tail`). On a full
ring, the producer increments an `events_dropped` counter and
overwrites the oldest record. (TODO-VERIFY: `threading.atomic`
is not in stdlib until 3.13; on older Python use
`multiprocessing.Value` or a `_thread.RLock` — the impl PR
will pin the chosen mechanism.)

Target: < 200 ns per emission on the producer side. Bench in
§2.10.

### 2.6 Tenant isolation

#### Guarantees (provided by the substrate)

  G1  Two tenants cannot read each other's handles via the
      memopt API. `MemoryHandle.read/write/as_tensor/as_numpy`
      validate `handle.tenant` against the calling tenant
      context (or against a tenant explicitly passed in a
      future override) and raise `PermissionError` on mismatch.
      Implementation: the check is one dict lookup + one string
      compare; target overhead < 100 ns on the read/write path
      (§2.10).

  G2  `memopt.stats(tenant=X)` only returns stats for tenant X.
      Cross-tenant aggregate stats require an explicit admin
      token configured via `MEMOPT_ADMIN_TOKEN` AND the calling
      thread holds a context that resolves to that token. Both
      conditions necessary. The token is compared in constant
      time (`hmac.compare_digest`).

  G3  NVMe paths are namespaced per tenant. Spilled pages live
      under `<MEMOPT_NVME_DIR>/<tenant_hash>/...`. Tenant_hash
      is `hashlib.sha256(tenant.encode()).hexdigest()[:32]`.
      The directory is created with mode `0700`. The
      `_sanitize_id` regex from Phase 1 (`^[A-Za-z0-9_-]
      {1,128}$`) is applied BEFORE hashing as defence in depth.
      The path is constructed with `os.path.join` then verified
      with `os.path.realpath(p).startswith(realpath(root))` to
      catch symlink escapes.

  G4  Tenant tags are NOT a security boundary. Two tenants may
      use the same tag string. Tags are routing/observability
      keys only. This is documented at the top of the
      `memopt.alloc` docstring.

#### Non-guarantees (out of scope, made explicit)

  N1  Hardware side channels. Cache timing, power analysis,
      Rowhammer-class attacks, GPU memory reuse residue. For
      adversarial multi-tenant deployments, use NVIDIA
      Confidential Computing (CC mode on H100/H200/B200) or
      Multi-Instance GPU (MIG). The substrate cooperates with
      MIG (one substrate per MIG slice) but does not provide
      MIG-equivalent isolation on its own.

  N2  Direct driver bypass. A tenant with `cudaMalloc` /
      `cuMemCreate` access can allocate outside memopt
      entirely. memopt is opt-in. To force opt-in, install
      memopt as PyTorch's allocator (existing `torch_allocator`
      path) and run inside a process namespace where the
      allocator cannot be replaced.

  N3  Driver-level vulnerabilities. Recent disclosures include
      CVE-2025-23266 (NVIDIA Container Toolkit container
      escape), CVE-2025-33220 (NVIDIA driver issue),
      CVE-2025-23352 (NVIDIA driver) — TODO-VERIFY: these CVE
      identifiers are taken from research notes; before this
      doc ships externally, each must be re-verified against
      the NVD entry to confirm the description matches what we
      claim. If unsure, omit the specific identifier and link
      to the vendor advisory page instead. The substrate is
      not a substitute for keeping drivers patched.

  N4  Bus-level peer access. The substrate controls
      `cuMemSetAccess` for handles it manages. A process with
      direct CUDA driver access can call `cuMemSetAccess` on
      its own allocations independently. Same caveat as N2.

### 2.7 Page size and granularity policy

  - NVIDIA: native granularity is 2 MiB on Hopper, Ada, Ampere
    (verified by Phase 1: `cuMemGetAllocationGranularity` is
    called at line 219 of vmm_allocator.cpp; the value returned
    on H100/A100 is 2 MiB).
  - Sub-2 MiB allocations are coalesced in the TenantArena. One
    2 MiB physical page can back many small handles for the
    same tenant when the size class matches. This is the
    standard CCA-style sub-allocator pattern.
  - 64 KiB pages are NOT supported in v1.0. They require
    vAttention's patched UVM driver
    (https://github.com/microsoft/vattention/tree/main/uvm).
    memopt does NOT ship a custom driver. Users with the patched
    driver may set `MEMOPT_ENABLE_VATTENTION_DRIVER=1`; the
    substrate then probes for the smaller granularity at startup.
  - On AMD: `hipMemGetAllocationGranularity`. Current ROCm 6 docs
    suggest 2 MiB is typical. TODO-VERIFY against MI300X test rig.
  - On CPU/CXL: `sysconf(_SC_PAGESIZE)` (typically 4 KiB) by
    default; promoted to 2 MiB huge pages when `MAP_HUGETLB`
    succeeds. The choice is made at backend startup and reported
    via `granularity_bytes()`.
  - Internal fragmentation policy: each TenantArena tracks
    committed vs in-use bytes per size class. When arena waste
    exceeds `MEMOPT_FRAG_THRESHOLD` (default 25 %), a background
    reclamation thread coalesces and releases unused 2 MiB pages
    back to the backend. The thread runs at most once per
    `MEMOPT_FRAG_INTERVAL_MS` (default 1000 ms) and is woken
    explicitly by free() when the threshold is exceeded.

### 2.8 Event surface

```python
@dataclass(frozen=True)
class Event:
    kind:           Literal["alloc", "free", "evict",
                            "promote", "migrate"]
    timestamp_ns:   int                  # time.monotonic_ns()
    handle_id:      int
    tenant:         str
    tag:            str
    size_bytes:     int
    from_placement: Optional[str]        # for evict/promote/migrate
    to_placement:   Optional[str]
    reason:         Optional[str]        # short string, e.g.
                                         # "ttl_expired",
                                         # "manager_pressure",
                                         # "tier_orchestrator"
```

Delivery contract:

  D1  Best-effort. Subscribers can be slow without back-pressuring
      the allocator. A slow subscriber affects only its own
      delivery thread.
  D2  Drops are accounted in `memopt.stats()["events_dropped"]`.
      The counter is a process-wide atomic. Producers always
      publish; on a full ring the oldest entry is overwritten.
  D3  Subscribers run on a dedicated dispatcher thread (one per
      `memopt.observe()` call), NOT on the allocating thread.
      Allocating threads are NOT blocked by subscriber work.
  D4  Subscriber exceptions are caught and logged at WARNING
      level via `logging.getLogger("memopt.substrate.events")`.
      The exception is NOT re-raised. The subscription is NOT
      auto-cancelled — repeated exceptions log every time.
  D5  Event ordering: events from a single allocating thread
      are observed in the order they were emitted. Events from
      different threads may interleave.
  D6  No persistence. Events are in-memory only. A consumer
      that wants persistence (e.g. for auditing) wraps the
      callback in their own writer.

### 2.9 Migration plan

Phase A — substrate ships in parallel.
  - New module `memopt/substrate/`. No existing module is
    edited.
  - New tests: `memopt/substrate/tests/test_*.py` (see Phase 3
    for the file list).
  - Existing 682 tests untouched. Substrate tests add to the
    count.
  - PyTorch CCA semantic in StreamRegistry implemented behind
    the new public API. The legacy `MemoptTorchAllocator` is
    NOT changed in Phase A.
  - One bridge test, `tests/test_substrate_legacy_parity.py`,
    written in Phase A. See "Bridge test" below.
  - Approval gate: Lachuman approves Phase A by approving this
    design doc.

Phase B — opt-in flag.
  - Add an internal adapter: `VMM.allocate()` may route to
    `memopt.alloc()` when `MEMOPT_VMM_USE_SUBSTRATE=1`. The
    flag defaults to OFF.
  - Run the 682 existing VMM tests with the flag ON. They must
    pass identically to the OFF path.
  - Run the bridge test in both modes. Byte-equivalent results
    required.
  - Approval gate: separate review pre-merge, citing the
    bridge test results.

Phase C — flip the default.
  - `MEMOPT_VMM_USE_SUBSTRATE` default goes ON. The legacy
    code path stays as a side-by-side fallback — set the flag
    to OFF to go back.
  - One release with the flag default ON. No code is removed.
  - Approval gate: separate review.

Phase D — remove the legacy path.
  - After at least two stable releases on Phase C, delete the
    legacy code path. The flag becomes a no-op (logs a
    deprecation warning).
  - Approval gate: separate review.

#### Bridge test (DECISION 3)

```
Path:  tests/test_substrate_legacy_parity.py
Marks: not @gpu (runs on CPUFallbackBackend); a second test
       file `tests/test_substrate_legacy_parity_gpu.py`
       carries the @gpu variant.

Workload:
  3 tenants:                  alice, bob, carol
  Allocations per tenant:     67 (alice gets one extra so total = 200)
  Size mix per tenant:
     - 50 % at 64 KiB         (sub-page; exercises arena split)
     - 30 % at 2 MiB          (single physical page)
     - 15 % at 8 MiB          (multi-page)
     -  5 % at 64 MiB         (large; forces eviction on
                                small CPU pool)
  Pool size:                  256 MiB total — chosen so the
                              workload triggers eviction.
  Stream binding:             half the allocations are bound to
                              one of two streams (stream A,
                              stream B); the other half are
                              unbound. Both streams emit
                              dummy work before any free.
  Lifecycle:                  alloc all 200; access half via
                              read/write to drive promotion;
                              free 60 to force eviction of
                              cold blocks; alloc 60 more to
                              hit the freelist; free
                              everything.

Recorded metrics:
  - stats() snapshot after each phase (alloc / access / free60 /
    realloc60 / final).
  - eviction_count (cumulative)
  - promotion_count (cumulative)
  - peak HBM bytes (peak `bytes_hbm` from stats)
  - per-tenant live_bytes at each phase

Phase 1 baseline: run the workload through the existing VMM
path (`memopt.vmm.VMM`), record the metrics.

Phase 2 candidate: same workload through `memopt.substrate`
(in Phase A this runs as a standalone smoke; in Phase B it runs
through the flag-on VMM adapter).

Assertions:
  - eviction_count: EXACT MATCH (legacy and substrate evict the
    same number of pages under the same workload).
  - promotion_count: EXACT MATCH.
  - byte counts (live_bytes per tenant, bytes_hbm, bytes_dram):
    within ±0.1 %. The tolerance accommodates accounting
    differences in how partially-coalesced arenas report
    committed-vs-in-use bytes; it does NOT permit different
    behaviour.
  - per-tenant cross-isolation: at every phase, no tenant's
    handles are accessible from another tenant. Verified by
    attempting handle.read() with a foreign tenant context and
    asserting PermissionError.
  - No event drops in either path (events_dropped == 0).
```

The bridge test is the trust anchor for Phase B. When the flag
flips, byte-equivalence is verified by this test. No hidden
behavioural drift.

### 2.10 Performance targets and how we measure

Targets (GOALS, not measurements):

```
Operation                        Goal       Reference
─────────────────────────────── ─────────── ─────────────────────────
alloc(2 MiB) cold (full path)    < 30 µs    vAttention (USENIX ATC 2024)
alloc(2 MiB) warm (arena hit)    <  1 µs    jemalloc fast paths (~150 ns
                                            on hot loops; we allow 6×
                                            overhead for Python+lock)
alloc(small, sub-allocated)      <  500 ns  arena split, no driver
free (warm, return to arena)     <  500 ns  symmetric to small alloc
free (cold, full unmap)          <  50 µs   cuMemUnmap + cuMemRelease
                                            measured on H100 in
                                            vAttention paper
event emit                       <  200 ns  ring write, atomic increment
cross-tenant access denial       <  100 ns  one dict + one strcmp
```

Each goal has a benchmark in `memopt/substrate/tests/
test_perf_microbench.py`. The benchmark prints measured times
and does NOT assert specific values (hardware varies). It DOES
assert ordering: `cold > warm > sub-allocated`, `free_cold >
free_warm`, `event_emit < alloc_warm`. Ordering invariants
catch regressions; absolute thresholds are noise.

Hardware baseline for our published targets: H100 SXM5 80 GB,
PCIe Gen 5, x86-64 host. Other hardware: targets scale with
device characteristics; the benchmark records actual numbers
into a JSON file under `/tmp/memopt-bench/<git-sha>.json` for
historical tracking. Numbers from this file may NOT appear in
the README or any external doc unless explicitly labelled with
the hardware on which they were measured.

### 2.11 Out of scope (explicit boundaries)

The substrate is foundational. Each of the following is a
separate layer with its own design doc.

  - Prefetching, Markov chain modelling, ML-based access
    prediction (Layer 3 — prefetch oracle).
  - Policy DSL (Layer 4 — placement policy).
  - Cross-node fabric pooling. The substrate exposes
    `export_fabric_handle()` returning bytes; the consumer's
    process for importing those bytes into another process is
    Layer 2 (tier orchestrator) or a separate "fabric"
    package, TBD.
  - vLLM integration. Lives in a separate package
    (`memopt-vllm` or upstream contribution). The substrate
    does NOT ship vLLM-specific shims.
  - Custom UVM driver (vAttention's 64 KiB pages). Out of
    scope; opt-in via `MEMOPT_ENABLE_VATTENTION_DRIVER=1`.
  - Confidential computing attestation. The substrate
    cooperates with NVIDIA CC mode (does not break it) but
    does not provide attestation tokens.
  - Cost / FinOps integration. Lives in `memopt-trust`
    (Pillar 7). Substrate emits events; finops consumes them
    out-of-process.

### 2.12 G12 fix specification (DECISION 1)

The 16 GiB hard cap from `MEMOPT_MAX_PAGES = 8192` is removed
during Phase A. The fix is in C/C++ only; no Python signature
changes.

#### Header — csrc/cuda_vmm/vmm_allocator.h

  Line 26 (current):
    `#define MEMOPT_MAX_PAGES (8192)`

  Phase A change:
    Replace the `#define` with a documented note that the cap
    is now dynamic and lives in the allocator instance. The
    `MEMOPT_PAGE_SIZE_BYTES` macro at line 25 stays.

  Add (after the existing macros) — informational only:
    `// Capacity is now dynamic, sized at allocator_create from`
    `// pool_size_bytes / MEMOPT_PAGE_SIZE_BYTES, plus 25%`
    `// headroom for sub-2-MiB sub-allocation churn.`

#### Implementation — csrc/cuda_vmm/vmm_allocator.cpp

  Line 55 (current):
    `MemoptPage   pages[MEMOPT_MAX_PAGES];`
  Phase A change:
    Replace with:
      `MemoptPage*  pages;          // heap-allocated`
      `int          n_pages_cap;    // capacity, set in create`
    The `n_pages` field on line 56 stays (length of populated
    slots).

  Line 245 (current):
    `a->va_free_cap = MEMOPT_MAX_PAGES * 8;`
  Phase A change:
    Use the new `n_pages_cap` directly:
      `a->va_free_cap = a->n_pages_cap * 8;`

  Line 320 (current):
    `if (a->n_pages >= MEMOPT_MAX_PAGES) {`
  Phase A change:
      `if (a->n_pages >= a->n_pages_cap) {`

  Line 584 (current):
    `int safety = MEMOPT_MAX_PAGES + 1;`
  Phase A change:
      `int safety = a->n_pages_cap + 1;`

  Inside `memopt_allocator_create()` (after the existing
  `cuMemAddressReserve` block, around line 235): compute
  `n_pages_cap` and allocate the pages array:

      `// Dynamic capacity: pool / page + 25% sub-alloc headroom.`
      `// Even if the pool is 2 MiB, we always allow at least 8`
      `// pages for sub-allocation churn.`
      `size_t pages_target = pool_size_bytes / MEMOPT_PAGE_SIZE_BYTES;`
      `pages_target = pages_target + (pages_target / 4);`
      `if (pages_target < 8) pages_target = 8;`
      `if (pages_target > INT_MAX) pages_target = INT_MAX;`
      `a->n_pages_cap = (int)pages_target;`
      `a->pages = (MemoptPage*)calloc(a->n_pages_cap,`
      `                                sizeof(MemoptPage));`
      `if (!a->pages) { /* clean up VA, return NULL */ }`

  Inside `memopt_allocator_destroy()` (around line 744): free
  the pages array:
      `free(a->pages);`
      `a->pages = NULL;`

  Every reference to `a->pages[i]` already takes the array via
  the struct member; the field type changes from inline array
  to pointer, but the indexing syntax is identical, so lines
  157, 158, 159, 161, 411, 437, 480, 505, 559, 599, 612, 636,
  638, 639, 701, 703, 705, 708 require NO source change.

  Loop bounds that read `MEMOPT_MAX_PAGES` outside the
  manager state: search the file with grep at impl time. The
  Phase 1 grep shows lines 245, 320, 584 only — all addressed
  above. The fix is mechanical.

#### New test

```
Path:  csrc/cuda_vmm/test_vmm_allocator.cpp (extends existing)
Test:  test_dynamic_capacity_above_legacy_cap

What it does:
  - allocator_create(device=0, pool_size_bytes = 32 GiB)
    (chosen above the legacy 16 GiB cap; runs only on hosts
    with >= 32 GiB free HBM — gate via
    cuda::mem_get_info()).
  - Allocate 16385 pages of 2 MiB each (one more than the old
    cap). Each malloc must succeed.
  - Assert that memopt_stats().pages_total == 16385.
  - Free everything; assert pages_total drops back to 0.
  - allocator_destroy.

Skip condition:
  - Skip with reason "requires 32 GiB free HBM" when
    cuda::mem_get_info().free < 34 GiB.

Hardware coverage in CI:
  - Skipped on Mac (no CUDA).
  - Skipped on T4 (16 GB) and L4 (24 GB) runners.
  - Runs on H100 / H200 / A100-80G / B200 runners.

Phase A definition of done:
  - The test passes on at least one >= 80 GiB runner.
  - The legacy `memopt_evict_n_pages`, `memopt_evict_to_target`
    tests still pass (no regression).
  - The libmemopt_vmm.so ABI is unchanged: existing C consumers
    (cuda_vmm.py ctypes bindings) compile and run without
    edit.
```

### 2.13 Traceability — gaps to design sections

```
Gap                                             Addressed in
─────────────────────────────────────────────── ─────────────
G1.  No unified handle type                     §2.1, §2.5 (HandleRegistry)
G2.  Tenant isolation only in Python            §2.6 (G1–G4 guarantees)
G3.  No stream registry                         §2.5 (StreamRegistry; DECISION 2)
G4.  No event surface                           §2.5 (EventRing), §2.8
G5.  No fabric-handle export                    §2.3, §2.4 (CUDABackend export)
G6.  No CXL / NUMA backend                      §2.4 (CXLBackend)
G7.  No Intel Level Zero backend                §2.4 (LZ stub, v1.1+ for real)
G8.  ROCm is a stub                             §2.4 (HIPBackend day-one)
G9.  No placement DSL                           §2.2 (placement table)
G10. No TTL on allocations                      §2.1 (alloc ttl_seconds), §2.5
                                                 (TTL sweeper)
G11. No tag-based routing                       §2.1 (tag), §2.5 (TenantArena
                                                 size class + observers)
G12. 16 GiB hard cap                            §2.12 (DECISION 1)
G13. No 64 KiB pages                            §2.7 (policy + opt-in flag)
```

Every gap has a numbered design section. Phase 3 (test plan)
will pair each section with at least one new test.

---

---

## Phase 3 — Test Plan

The test plan covers three things: (a) a Step Zero verification
gate for every TODO-VERIFY in Phase 2, before any implementation
code is written; (b) the per-section unit-test surface; (c) the
commit-by-commit implementation order. Plus the divergence-test
spec, the resolved bridge-test order question, and the
regression strategy.

Long-term frame: every test below protects against a future bug
that would have to be debugged without context. Prefer one extra
test today over a 3am page next year.

### 3.0 Step Zero — TODO-VERIFY checklist (must pass before Phase A)

Phase A (implementation) does not begin until every line below
is checked off. Step Zero produces a single artefact at
`docs/substrate_v1_step_zero_report.md` that records the result
of each item. Step Zero failures may force Phase 2 doc updates;
those updates re-trigger Phase 3 review.

#### S0.1  CUDA fabric handle (G5)

  Check against     CUDA Driver API headers and reference docs.
  Sources           - cuda.h on the test rig with CUDA 12.4 or
                      newer
                    - https://docs.nvidia.com/cuda/cuda-driver-api/
                      group__CUDA__VA.html
  Method            1. `grep -nE "CU_MEM_HANDLE_TYPE_FABRIC|
                       cuMemExportToShareableHandle"
                       $CUDA_HOME/include/cuda.h`
                    2. Build a probe (~30 lines) that calls
                       cuMemCreate + cuMemExportToShareableHandle
                       with CU_MEM_HANDLE_TYPE_FABRIC and prints
                       the CUresult.
                    3. Run the probe on at least one
                       IMEX-enabled host AND one host without
                       IMEX (so we observe both the SUCCESS
                       and the NOT_SUPPORTED branches).
  Verified when     a) The enumerator `CU_MEM_HANDLE_TYPE_FABRIC`
                       is present and has the value documented
                       by NVIDIA; AND
                    b) On the IMEX host the probe returns
                       CUDA_SUCCESS and a non-empty handle blob;
                       AND
                    c) On the non-IMEX host the probe returns
                       CUDA_ERROR_NOT_SUPPORTED (or the
                       documented "no fabric" status), exposed
                       via the substrate as None per §2.3.
  Fallback          - CUDA <12.4 only on test rigs: keep the
                      `export_fabric_handle()` method but it
                      returns None unconditionally. No design
                      change.
                    - Enumerator renamed/removed in newer CUDA:
                      update §2.3, §2.4 to cite the current
                      symbol and re-run §2.13 traceability for
                      G5.
                    - FABRIC fully retired in CUDA: drop fabric
                      export from v1, document in §2.11 as
                      "deferred to v1.1 with replacement
                      mechanism TBD". Substrate ships without
                      G5; G5 stays in the gap list.

#### S0.2  HIP VMM capability and granularity (G8)

  Check against     ROCm 6.x HIP runtime headers and docs.
  Sources           - hip/hip_runtime_api.h on the test rig
                      with ROCm >= 6.0
                    - https://rocm.docs.amd.com/projects/HIP/
                      en/latest/
  Method            1. `grep -nE "hipMemGetAllocationGranularity|
                       hipDeviceAttributeVirtualMemory"
                       /opt/rocm/include/hip/*.h`
                    2. Build a HIP probe that calls
                       `hipDeviceGetAttribute(&v,
                       hipDeviceAttributeVirtualMemoryManagement
                       Supported, dev_id)` and reports the value.
                    3. Probe runs on the AMD CI rig (TODO: name
                       the rig in the Step Zero report).
  Verified when     a) Both symbols are present, with signatures
                      matching what §2.4 HIPBackend assumes; AND
                    b) `hipDeviceGetAttribute` returns 1
                      (supported) on at least the test rig; AND
                    c) `hipMemGetAllocationGranularity` returns
                      a power-of-two byte count >= 64 KiB.
  Fallback          - Attribute renamed (e.g.
                      hipDeviceAttributeVirtualMemoryManagement
                      without "Supported"): update §2.4
                      HIPBackend to cite current name; rerun
                      probe.
                    - Attribute returns 0 on every available
                      MI300X tested: HIPBackend ships as the
                      same-shape stub as Level Zero (raises
                      NotImplementedError on construction).
                      Mark G8 as "addressed only on capability
                      probe failure"; future ROCm releases may
                      flip the bit. Phase 4 prompt then
                      generates the stub instead of the real
                      backend.
                    - Symbols absent entirely: same as above
                      (stub).

#### S0.3  PyTorch CCA semantics (G3, DECISION 2)

  Check against     PyTorch upstream `c10/cuda/CUDACachingAllocator
                    .cpp` at the version used on the substrate
                    test rigs.
  Sources           - https://github.com/pytorch/pytorch/blob/
                      <pinned-commit>/c10/cuda/CUDACachingAllocator
                      .cpp
                    - The PyTorch version installed on the rig
                      (`torch.__version__` exact)
  Method            1. Pin a specific upstream commit hash for
                       Step Zero (record in the report).
                    2. From that commit, search for the symbols
                       referenced in §2.5: `record_stream`,
                       `stream_uses`, `pending_events`,
                       `event_pool`, `get_allocation_size`,
                       `BlockPool`, the size-class table.
                    3. Compare the size-class scheme to what
                       §2.5 documents; record any deviation.
                    4. Re-read the cross-device, IPC, and CUDA
                       Graphs handling and compare against the
                       E1–E9 audit in §2.5. Record any new edge
                       case CCA handles that we missed.
  Verified when     a) Every symbol referenced in §2.5 exists in
                      the pinned commit, possibly under a renamed
                      name (rename is acceptable; semantic match
                      is required); AND
                    b) The size-class scheme in §2.5 matches the
                      `get_allocation_size` table within the
                      tolerance of "we round more conservatively"
                      (we never split where CCA wouldn't); AND
                    c) The E1–E9 audit covers every observable
                      CCA behaviour. Anything new is added as
                      E10+ to §2.5.
  Fallback          - Symbol rename only: update §2.5 to cite
                      current symbol; no design change.
                    - Size-class table changed materially:
                      update §2.5 to track the new scheme. The
                      bridge test (§2.9) will re-validate parity.
                    - New edge case discovered (E10+): add it
                      to §2.5 with a substrate behaviour and a
                      divergence note (if any). May require
                      Phase 2 doc update before Phase A.

#### S0.4  CUDA Graphs capture APIs (E1 in §2.5)

  Check against     CUDA Runtime API headers.
  Sources           - cuda_runtime_api.h
                    - https://docs.nvidia.com/cuda/cuda-runtime-
                      api/group__CUDART__STREAM.html
  Method            1. `grep -nE "cudaStreamIsCapturing|
                       cudaStreamCaptureStatus"
                       $CUDA_HOME/include/cuda_runtime_api.h`
                    2. Confirm the enum members
                       cudaStreamCaptureStatusNone,
                       cudaStreamCaptureStatusActive,
                       cudaStreamCaptureStatusInvalidated.
  Verified when     Both function symbols and all three enum
                    values are present with the documented
                    signatures.
  Fallback          - APIs renamed: update §2.5 E1 to cite
                      current symbols; no behaviour change.
                    - APIs removed (unlikely): redesign E1 to
                      use `cudaGraphInstantiate` introspection
                      or to disable capture-aware behaviour
                      entirely (allocations during capture
                      would route through normal arenas; CUDA
                      Graphs that allocate in capture would not
                      work — same as PyTorch CCA pre-2.1).
                      Document the regression and update §2.11.

#### S0.5  CXL detection heuristic (G6)

  Check against     A real CXL-equipped machine OR vendor docs.
  Sources           - https://www.kernel.org/doc/Documentation/
                      ABI/stable/sysfs-devices-system-node
                    - https://www.kernel.org/doc/Documentation/
                      ABI/testing/sysfs-bus-cxl
                    - At least one CXL deployment for empirical
                      check (cloud or on-prem; record in the
                      report)
  Method            1. On the CXL host:
                       `for n in /sys/devices/system/node/node*;
                        do echo "$(basename $n)
                                 cpulist=$(cat $n/cpulist)
                                 mem=$(grep MemTotal $n/meminfo)";
                        done`
                    2. Verify that at least one node has empty
                       cpulist AND non-zero MemTotal AND that
                       the same node corresponds to the CXL
                       device under `/sys/bus/cxl/`.
                    3. On a non-CXL host, run the same and
                       confirm the heuristic produces no
                       matches (no false positives on plain
                       NUMA boxes).
  Verified when     a) The heuristic correctly identifies the
                      CXL node on the test deployment; AND
                    b) `/sys/bus/cxl/` is present AND aligns
                      with the heuristic on Linux >= 6.0; AND
                    c) Zero false positives on the no-CXL
                      reference host.
  Fallback          - /sys/bus/cxl/ is universally available on
                      target deployments: prefer it; the cpulist
                      heuristic becomes a fallback for older
                      kernels. Update §2.4 CXLBackend.
                    - The cpulist heuristic produces false
                      positives on some Intel/AMD configs:
                      require explicit MEMOPT_CXL_NODES env var
                      and disable auto-detection. Document in
                      §2.4. Substrate still ships with CXL
                      support; just opt-in.
                    - No CXL deployment available for empirical
                      check: ship CXLBackend with auto-detection
                      disabled and gate it behind
                      MEMOPT_CXL_NODES. Mark G6 as "partial v1;
                      auto-detect deferred to v1.1".

#### S0.6  Event ring atomicity primitive (§2.5 EventRing)

  Check against     The Python versions in the project's support
                    matrix (currently 3.10–3.13 per pyproject).
  Sources           - https://docs.python.org/3.13/library/
                      threading.html
                    - https://docs.python.org/3.13/library/
                      ctypes.html
                    - PEP 703 (free-threaded CPython, info only)
  Method            1. `python3.13 -c "import threading;
                       print(hasattr(threading, 'atomic'))"`
                    2. If absent, evaluate three alternatives:
                       (a) `ctypes.c_uint64` with platform
                            atomic intrinsics via `os` (3.13+
                            only, NOT generally available);
                       (b) `threading.Lock` around two integers
                            for head/tail (always works, costs
                            ~50 ns per emission);
                       (c) Single-producer-single-consumer ring
                            using a `_thread`-level fence and
                            module-global `int` (relies on
                            CPython's bytecode atomicity for
                            int store on x86; portable to
                            free-threaded CPython is uncertain).
                    3. Microbenchmark each alternative in a
                       small probe; pick the one whose ratio
                       (mean+p99) is closest to the < 200 ns
                       target from §2.10.
  Verified when     The chosen primitive:
                    a) Is available on every supported Python
                       version (3.10+); AND
                    b) Microbenchmarks at < 200 ns mean on the
                       test rig; AND
                    c) Has documented thread-safety semantics
                       (no relying on CPython implementation
                       details that PEP 703 may break).
  Fallback          - All three alternatives miss the < 200 ns
                      target: relax the §2.10 target for event
                      emission to "< 1 µs" and document the
                      reason. The substrate still works, just
                      with higher event overhead. Bench
                      ordering invariant (event_emit <
                      alloc_warm) is preserved if alloc_warm
                      target is also relaxed.
                    - PEP 703 makes implementation-defined
                      atomicity unsafe: switch to
                      `threading.Lock` even if slower; relax
                      target accordingly.

  HONEST NOTE       Phase 2 wrote `TODO-VERIFY: threading.atomic
                    is not in stdlib until 3.13`. As of this
                    writing the author is not certain
                    `threading.atomic` exists in 3.13 either —
                    it may be an artefact of free-threaded
                    CPython work that hasn't landed. Step Zero
                    must confirm or rule out before
                    implementation. Default expectation:
                    fall back to `threading.Lock` (option b).

#### S0.7  CVE identifiers (§2.6 N3)

  Check against     The National Vulnerability Database.
  Sources           - https://nvd.nist.gov/vuln/search
                    - The NVIDIA security bulletin pages
  Method            1. For each candidate identifier:
                       - CVE-2025-23266
                       - CVE-2025-33220
                       - CVE-2025-23352
                       Look up at https://nvd.nist.gov/vuln/
                       detail/<CVE-ID>.
                    2. Confirm:
                       - The identifier resolves to a published
                         entry; AND
                       - The CVE description matches what §2.6
                         N3 claims (i.e. the CVE is in fact
                         driver-level / container escape /
                         relevant to GPU memory).
                    3. If any identifier doesn't resolve OR the
                       description doesn't match, treat as
                       failed and replace with a verified one
                       OR remove the specific identifier.
  Verified when     Every CVE identifier in §2.6 N3 has been
                    confirmed against NVD; OR every unverified
                    identifier has been removed.
  Fallback          - All three identifiers fail: rewrite §2.6
                      N3 to omit specific CVE numbers and link
                      to the NVIDIA Security Bulletin index
                      (https://www.nvidia.com/en-us/security/)
                      instead. The threat model section keeps
                      its claim ("driver-level vulnerabilities
                      are out of scope") without leaning on
                      unverified identifiers.

#### S0.8  MI300X granularity (§2.7)

  Check against     A real AMD MI300X (or MI250X / MI200 as
                    fallback for the probe — the H100 of the
                    AMD line we have access to).
  Sources           - The hardware itself
                    - ROCm release notes
  Method            1. On MI300X, run the HIP probe from S0.2
                       and capture
                       hipMemGetAllocationGranularity for both
                       MINIMUM and RECOMMENDED.
                    2. Record exact values in the Step Zero
                       report.
  Verified when     The substrate's documented assumption (2 MiB
                    typical) matches the runtime probe within
                    a power-of-two factor. Either way, the
                    HIPBackend impl uses the runtime value, so
                    "verification" here is for the doc claim,
                    not for code behaviour.
  Fallback          - Granularity differs materially (e.g.
                      4 MiB or 64 KiB): update §2.7 to record
                      the actual value with hardware tag.
                      Update the bridge test (§2.9) workload
                      sizes if necessary so the workload still
                      forces eviction on AMD.
                    - No MI300X available for the probe: run
                      probe on whatever AMD card we have
                      (MI200 / MI100 / RX 7900 XT) and tag the
                      result with the hardware. Mark §2.7 with
                      a TODO-NEXT-RIG note for when MI300X
                      access is obtained. Substrate still ships;
                      bridge test runs on whatever HIP rig is
                      available.

#### Step Zero summary table

| ID    | Item                          | Doc section | Block A on fail? |
| ----- | ----------------------------- | ----------- | ---------------- |
| S0.1  | CUDA fabric handle            | §2.3, 2.4   | No (degrades)    |
| S0.2  | HIP VMM capability            | §2.4        | No (stub-down)   |
| S0.3  | PyTorch CCA symbols           | §2.5        | Yes (semantic)   |
| S0.4  | CUDA Graphs APIs              | §2.5 E1     | No (degrades)    |
| S0.5  | CXL heuristic                 | §2.4        | No (opt-in)      |
| S0.6  | Event ring primitive          | §2.5        | No (target)      |
| S0.7  | CVE identifiers               | §2.6 N3     | No (omit)        |
| S0.8  | MI300X granularity            | §2.7        | No (doc only)    |

"Block A on fail" means: a failure in this item must redesign
the affected section before Phase A starts. S0.3 is the only
hard block — every other item has a documented graceful fallback.
S0.3 is hard because the substrate's stream-locked semantic IS
PyTorch CCA's; if we cannot match it we must redesign §2.5.

### 3.1 Unit test files (per Phase 2 section)

For each Phase 2 design section, the test file(s) that cover it.
Each test file lists its tests with a one-line description and
a marker tag (no marker = CPU/general; @gpu = NVIDIA required;
@gpu_amd = AMD required; @cxl = CXL required; @perf = excluded
from default `pytest -k "not perf"` runs).

#### test_handle.py — covers §2.1 MemoryHandle

```
test_alloc_returns_handle_with_metadata
    - Checks handle.size_bytes, tenant, tag, placement,
      backend_name, created_at, ttl_seconds, hint are all set
      from the alloc() call.

test_handle_state_lifecycle
    - alloc -> live, free -> released (no stream), and on a
      stream-bound handle: free -> freed_pending_event ->
      released after event.query() == True.

test_double_free_idempotent
    - calling handle.free() twice does not raise; the second
      call increments stats()["double_free_count"].

test_handle_freed_blocks_read
    - after free, handle.read() raises RuntimeError("handle is
      freed"). Same for write, as_tensor, as_numpy.

test_context_manager_auto_free
    - `with memopt.alloc(N) as h: pass` leaves the handle in
      released state on exit. Confirm via stats() free_count.

test_handle_ref_count_with_as_tensor
    - as_tensor() bumps ref_count; tensor goes out of scope ->
      ref_count drops; handle.free() blocks reclaim until
      ref_count is 0. (CPU backend variant; @gpu variant is
      under test_backend_cuda.)

test_invalid_size_zero_raises
    - memopt.alloc(0) raises ValueError. memopt.alloc(-1) raises.

test_invalid_tenant_regex_raises
    - tenant="alice/bob" raises. tenant="" raises.
    - tag="x"*65 raises (max 64).

test_handle_dataclass_is_frozen
    - direct field write raises (handle metadata is immutable
      after creation).
```

#### test_arena.py — covers §2.5 TenantArena, §2.7 fragmentation

```
test_arena_returns_block_to_freelist
    - alloc 2 MiB, free, alloc 2 MiB again. Second call hits
      freelist (no driver call). Verify via observed_alloc_count
      from a backend mock.

test_arena_size_class_routing
    - alloc 600 B, 1100 B, 2000 B. Each lands in a distinct
      size class; freed blocks of class N are reused only by
      class-N allocations.

test_arena_fragmentation_threshold_triggers_reclaim
    - alloc many small handles, free 80 % of them. After
      MEMOPT_FRAG_INTERVAL_MS, the reclamation thread runs and
      released >= one 2-MiB physical page. Verify via stats().

test_arena_high_water_tracking
    - alloc 100 MiB, free, alloc 50 MiB. high_water_bytes ==
      100 MiB. Cumulative metric, never decreases.

test_arena_per_tenant_isolation_no_cross_freelist
    - alice frees a 2 MiB handle. bob's alloc(2 MiB) does NOT
      receive alice's freed block. Verify via PhysHandle id.
```

#### test_tenant_isolation.py — covers §2.6 G1–G4

```
test_g1_cross_tenant_read_denied
    - alice allocates h. bob calls h.read() inside
      memopt.context(tenant="bob"). Raises PermissionError.

test_g1_cross_tenant_write_denied
    - same shape, write() instead of read().

test_g1_cross_tenant_as_tensor_denied
    - same shape, as_tensor() instead of read().

test_g2_stats_namespacing
    - memopt.stats(tenant="alice") returns alice's data only.
    - memopt.stats(tenant=None) without admin token raises
      PermissionError.
    - memopt.stats(tenant=None) WITH admin token returns
      aggregate.

test_g3_nvme_path_namespaced
    - On a backend that spills to NVMe, alice and bob each
      allocate and force a spill. Verify the on-disk paths
      live under different tenant_hash directories with mode
      0700, and that os.path.realpath does not escape root.

test_g3_nvme_path_traversal_blocked
    - Construct a tenant id "../../etc" — the regex rejects
      it before any path is built. ValueError.
    - Construct a sequence-id with embedded NUL — same.

test_g4_tag_is_not_a_security_boundary
    - alice allocates with tag="kv". bob allocates with same
      tag. Both succeed; no tag-collision error.

test_n2_direct_cudaMalloc_bypasses_memopt
    - Documents the limitation by example: cudaMalloc'd
      pointer is invisible to memopt.stats(). The test
      passes when memopt.stats()["live_bytes"] does not
      include the bypass allocation. Marked @gpu and
      conditionally skipped when torch.cuda is absent.
```

#### test_stream_registry.py — covers §2.5 StreamRegistry, E5

```
test_alloc_with_stream_records_stream_use
    - alloc(stream=S) -> registry.stream_uses[h] contains S.

test_record_stream_idiom_adds_to_uses
    - alloc(stream=S1); tensor.record_stream(S2); free(h).
    - Both S1 and S2 receive recorded events; reclaim is
      blocked until BOTH events query() == True.

test_unbound_handle_recycles_immediately
    - alloc(stream=None); free; next alloc of same size hits
      arena freelist on the same call (no event wait).

test_stream_a_free_does_not_block_stream_b
    - alloc(stream=A); free; while A's event is unfinished,
      alloc(stream=B) of unrelated size proceeds without
      blocking.

test_event_pool_reused
    - 100 alloc/free cycles on a stream. event_pool grows to
      a steady state that is < 100 (events are recycled).

test_pending_drained_on_alloc
    - Free h on stream A; sleep until A is idle; next alloc
      runs the pending drain, returning h's block to the
      arena. Verify via observed handle_id reuse.
```

(All @gpu-marked. CPU variant covered by smoke tests.)

#### test_events.py — covers §2.5 EventRing, §2.8

```
test_alloc_emits_alloc_event
    - subscribe to "alloc"; alloc; observe Event with kind,
      handle_id, tenant, tag, size_bytes set.

test_free_emits_free_event
    - same shape with kind="free", from_placement set.

test_evict_emits_evict_event
    - force eviction (via small pool); observe Event with
      kind="evict", from_placement="hbm", to_placement="dram".

test_promote_emits_promote_event
    - touch an evicted handle; observe Event with
      kind="promote", from="dram", to="hbm".

test_subscriber_runs_on_dispatcher_thread_not_caller
    - subscribe with a callback that records its threading
      ident. Allocate. Assert the callback's thread ident
      != the allocating thread's ident.

test_subscriber_exception_does_not_kill_dispatcher
    - subscribe with a callback that raises every time. Allocate
      10 times. Assert the dispatcher is still alive (next
      well-behaved subscriber receives all 10 events).

test_full_ring_drops_oldest_and_increments_counter
    - subscribe but block in callback. Allocate >> ring
      capacity. Resume. Assert events_dropped > 0 and matches
      the overflow count.

test_event_ordering_within_thread
    - Single thread allocates A then frees A. Subscriber
      observes alloc(A) before free(A).

test_unsubscribe_drops_callback
    - subscribe -> handle.unsubscribe() -> next allocation
      does NOT invoke the callback.

test_event_emit_under_200ns_target
    - @perf. Microbench. Asserts ordering only (alloc_warm <
      alloc_cold; event_emit < alloc_warm). Does NOT assert
      absolute thresholds.
```

#### test_backend_cpu.py — covers §2.4 CPUFallbackBackend

```
test_cpu_backend_is_available_when_no_gpu
    - on a host with torch.cuda.is_available() == False AND
      no HIP, the CPUFallbackBackend.is_available() == True.

test_cpu_backend_alloc_free_smoke
    - reserve_va, create_physical, map, unmap, release,
      free_va — each succeeds in order; no leaks reported by
      tracemalloc.

test_cpu_backend_granularity_is_pagesize_or_huge
    - returns sysconf(_SC_PAGESIZE) by default. With env
      MEMOPT_FORCE_HUGE=1, attempts MAP_HUGETLB; if the
      kernel rejects, falls back without raising.

test_cpu_backend_export_fabric_returns_none
    - export_fabric_handle() always returns None.

test_cpu_backend_set_access_is_noop
    - set_access(va, size, [0]) returns; backend.stats()
      shows no driver call.
```

#### test_backend_cuda.py — covers §2.4 CUDABackend  @gpu

```
test_cuda_backend_seven_primitives_smoke
    - one round trip through the seven primitives. Confirms
      no error from any cuMem* call.

test_cuda_backend_granularity_returns_2mib
    - cuMemGetAllocationGranularity returns 2 MiB on the
      tested hardware.

test_cuda_backend_fabric_handle_export
    - On CUDA 12.4+ AND IMEX present: returns non-empty bytes.
      On older CUDA OR no IMEX: returns None. Skips with
      reason when neither environment is reachable.

test_cuda_backend_set_access_for_peer_is_explicit
    - set_access(va, size, [0,1]) succeeds when peer access is
      available. WITHOUT calling set_access, attempting to
      access from device 1 raises (this is one of the
      divergence behaviours from PyTorch CCA — see
      test_cca_divergences.py).

test_cuda_backend_evict_promote_round_trip
    - Allocate filling 95 % of a small pool. evict_to_target
      drops below 80 %. Promote a evicted page; verify it's
      readable again.
```

#### test_backend_hip.py — covers §2.4 HIPBackend  @gpu_amd

Mirror of test_backend_cuda.py with `hip*` symbols. Skip
gracefully when `is_available() == False`.

#### test_backend_l0_stub.py — covers §2.4 LevelZeroBackend stub

```
test_l0_is_available_only_when_libze_loader_present
    - mock the dlopen to return found / not found. Both cases.

test_l0_calls_raise_clear_message
    - reserve_va() raises NotImplementedError with text that
      mentions "v1.0" and the design doc path.

test_l0_does_not_register_with_manager
    - on a system with libze_loader, the AllocationManager
      excludes Level Zero from its backend list because
      is_available() returns False from a method other than
      construction. The substrate falls through to CPU.
```

#### test_backend_cxl.py — covers §2.4 CXLBackend  @cxl

Marked `@cxl` and excluded from default runs unless
MEMOPT_CXL_PRESENT=1.

```
test_cxl_detection_via_sys_devices
    - mock /sys/devices/system/node/ contents; verify the
      heuristic identifies the CXL node.

test_cxl_alloc_via_libnuma
    - happy path; require libnuma (skips otherwise).

test_cxl_gpu_addressable_via_host_register
    - on a host with CUDA + CXL: numa_alloc + cuMemHostRegister
      yields a device pointer with cuMemHostGetDevicePointer.

test_cxl_unavailable_explicit_placement_raises
    - with no CXL detected, alloc(placement="cxl") raises
      MemoryError (per §2.2 explicit-placement rule).
```

#### test_placement.py — covers §2.2

```
test_placement_auto_picks_hot_when_available
test_placement_hot_falls_through_to_dram_when_hbm_full
test_placement_explicit_hbm_raises_on_cpu_backend
test_placement_explicit_cxl_raises_on_no_cxl_backend
test_placement_warm_uses_pinned_dram_on_cuda
test_placement_cold_uses_nvme_when_available
test_placement_invalid_string_raises_value_error
```

#### test_fabric_handle.py — covers §2.3 export_fabric_handle  @gpu

```
test_fabric_export_returns_bytes_on_cuda_124_with_imex
test_fabric_export_returns_none_on_older_cuda
test_fabric_export_does_not_raise
```

#### test_perf_microbench.py — covers §2.10  @gpu @perf

```
bench_alloc_2mib_cold
bench_alloc_2mib_warm_arena
bench_alloc_small_sub_allocated
bench_free_warm
bench_free_cold
bench_event_emit
bench_cross_tenant_access_denial

# Asserts ORDERING only:
test_ordering_invariants
    - alloc_cold > alloc_warm; alloc_warm > sub_allocated.
    - free_cold > free_warm.
    - event_emit < alloc_warm.

# Writes /tmp/memopt-bench/<git-sha>.json with measured
# numbers for historical tracking. NEVER asserts absolute
# values.
```

#### test_substrate_smoke.py — end-to-end smoke

```
test_alloc_use_free_round_trip_cpu
    - simple round trip on CPU backend; observes alloc and
      free events.

test_alloc_use_free_round_trip_cuda  @gpu
test_alloc_use_free_round_trip_hip   @gpu_amd

test_observe_unobserve_lifecycle
test_context_propagation_into_thread_pool
    - memopt.context(tenant="alice") captures into a future
      submitted to ThreadPoolExecutor; the worker's alloc
      sees tenant="alice" without explicit pass.
```

#### test_backend_protocol.py — covers §2.3 ABC contract

```
test_all_backends_implement_seven_primitives
    - introspect each registered backend; each abstractmethod
      is overridden by a concrete implementation.

test_is_available_is_pure
    - call is_available() 100 times; assert no CUDA / HIP
      context is initialised by the calls (probe via
      torch.cuda.is_initialized()).
```

### 3.2 PyTorch CCA divergence tests (ADDITION 2)

Three tests in `memopt/substrate/tests/test_cca_divergences.py`.
Each exercises one of the documented divergences from §2.5,
asserts the substrate's behaviour, and pins the design date so
future maintainers can revisit if the rationale changes.

```python
"""
Tests that document where memopt substrate intentionally diverges
from PyTorch CUDACachingAllocator (CCA) semantics.

Each test references the PyTorch CCA behaviour, asserts memopt's
divergent behaviour, and stamps the date the design decision was
made. If the rationale changes, the date stamp helps locate the
rolling-back decision.
"""
import pytest

# Design decision date for ALL three divergences below: 2026-04-29
# Source: docs/substrate_v1_design.md §2.5, edge cases E2 / E3 / E4.
DESIGN_DATE = "2026-04-29"


def test_no_auto_peer_access_enable():
    """
    PyTorch CCA: on first cross-device touch, auto-calls
    `cudaDeviceEnablePeerAccess`. memopt: requires explicit
    `set_access([peer_dev])` before cross-device access.

    Why we diverge:
      Auto-enable has caused subtle bugs around per-process
      resource limits (max enabled peers per device). Explicit
      is auditable. See §2.5 E2.

    Decision date: 2026-04-29. Revisit if a future CUDA driver
    removes the per-process peer-access limit.
    """
    pytest.importorskip("torch")
    import torch
    if torch.cuda.device_count() < 2:
        pytest.skip("requires >= 2 CUDA devices")
    import memopt

    with memopt.context(tenant="div1"):
        h = memopt.alloc(1 << 20, placement="hbm")
        # h is on current device (device 0). Without set_access,
        # accessing from device 1 raises:
        with torch.cuda.device(1):
            with pytest.raises((RuntimeError, MemoryError)):
                _ = h.as_tensor(torch.uint8, (1 << 20,))
        # After explicit set_access, access from device 1 works.
        h._backend.set_access(h._va, h.size_bytes, devices=[0, 1])
        with torch.cuda.device(1):
            t = h.as_tensor(torch.uint8, (1 << 20,))
            assert t.numel() == (1 << 20)


def test_no_ipc_handle_export_in_v1():
    """
    PyTorch CCA: supports `cudaIpcGetMemHandle` /
    `cudaIpcOpenMemHandle` for cross-process tensor sharing.
    memopt v1: NO IPC API. Cross-process sharing in memopt
    today goes through the RDMA transport (csrc/transport/);
    cross-process VMM peer mapping is deferred to v1.1.

    Why we diverge:
      IPC handles tie reference counts to a kernel object that
      doesn't fit our backend strategy abstraction cleanly.
      Solving it well requires v1.1 design work. See §2.5 E3.

    Decision date: 2026-04-29. Revisit when v1.1 design is
    drafted (target: post-Q3-2026).
    """
    import memopt

    with memopt.context(tenant="div2"):
        h = memopt.alloc(1 << 20)
        # No to_ipc_handle method exists on MemoryHandle.
        assert not hasattr(h, "to_ipc_handle"), (
            "v1 must not expose IPC API. "
            f"Decision date: {DESIGN_DATE}"
        )
        # export_fabric_handle is the only cross-process channel.
        # On CUDA without 12.4+/IMEX OR on the CPU backend, it
        # returns None — no error, no panic.
        result = h._backend.export_fabric_handle(h._physical)
        assert result is None or isinstance(result, bytes)


def test_single_device_per_alloc_call_in_v1():
    """
    PyTorch CCA: each device has its own allocator; explicit
    `tensor.to(device)` for cross-device. memopt v1: alloc()
    has NO `device=` parameter — uses the current CUDA device
    via `cuCtxGetCurrent`.

    Why we diverge:
      Multi-device allocation in one call adds a parameter that
      90 % of users won't need; it complicates the public surface
      and is easy to forget. Explicit `with torch.cuda.device(N):`
      around the call is the existing PyTorch idiom. See §2.5 E4.

    Decision date: 2026-04-29. Revisit if user feedback shows
    `device=` would prevent real bugs.
    """
    import inspect
    import memopt

    sig = inspect.signature(memopt.alloc)
    assert "device" not in sig.parameters, (
        "v1 alloc() must not have device parameter. "
        f"Decision date: {DESIGN_DATE}"
    )

    pytest.importorskip("torch")
    import torch
    if torch.cuda.device_count() < 2:
        pytest.skip("requires >= 2 CUDA devices to verify")

    with memopt.context(tenant="div3"):
        with torch.cuda.device(0):
            h0 = memopt.alloc(1 << 20, placement="hbm")
        with torch.cuda.device(1):
            h1 = memopt.alloc(1 << 20, placement="hbm")
        # The two handles live on the device that was current at
        # alloc time. Verify via backend introspection:
        assert h0._backend_device_index == 0
        assert h1._backend_device_index == 1
```

These three tests ship in commit 10 of the implementation
sequence (§3.5).

### 3.3 Bridge test order sensitivity (ADDITION 3) — OPTION Y

After Phase 1 evidence review, **OPTION Y is selected**: the
bridge test does NOT assert allocation order; it asserts byte
counts (±0.1 %), exact eviction/promotion counts, per-tenant
isolation, and zero event drops.

#### Phase 1 evidence (§1.5 regression net audit)

A grep of the regression net for tests that depend on allocator
output order returned no positives. Specifically:

  - `assert == sorted(...)` matches in
    test_universal_profile.py:84 (HAL tier order, not alloc),
    test_oracle.py:77 (oracle confidence ranking, not alloc),
    test_global_oracle.py:237 (sorted by count, not alloc),
    test_pod_controller.py:312 (sorted by count, not alloc).
    None of these assert allocation order.

  - `assert == [...]` matches in test_layer3.py:436 (peer
    list), test_rocm_backend.py:128 (tier names list ["hbm",
    "dram", "nvme"] which is a HAL property, not alloc order),
    test_hal.py:176, 180 (CSV parser empty cases),
    test_discovery.py:104 (peers list empty), various other
    files for empty error/event lists. None assert alloc order.

  - VMM tests (test_vmm_smoke.py, test_vmm_hardening.py,
    test_layer3.py): every `vmm.allocate(...)` is followed by
    lookups keyed on (sequence_id, block_index), never by
    iteration order. PageTable's API surface (Phase 1 §1.2)
    returns entries via explicit keys; iteration is via
    `lru_candidates(tier, count)` which is sorted by access
    pattern (not alloc order).

  - `list_node_blocks(node_id)` test
    (test_pillar5_gum.py:143): asserts `len == 2` and
    "all entries match this node id". Does NOT assert order.

  - No test reads handle pointers, VAs, or block indices and
    compares them as ordered sequences.

Conclusion: no existing test depends on allocator output order.
OPTION Y (no order assertion in the bridge test) preserves the
substrate's design freedom — for example, a future
size-class-aware allocator may legitimately allocate small
blocks before large ones within a single batch.

#### What the bridge test asserts (formalised)

```
For each phase {alloc, access, free60, realloc60, final}:
    (1) eviction_count_legacy == eviction_count_substrate         # exact
    (2) promotion_count_legacy == promotion_count_substrate       # exact
    (3) for each tenant t in {alice, bob, carol}:
            abs(live_bytes_legacy[t] - live_bytes_substrate[t])
                <= 0.001 * live_bytes_legacy[t]                   # ±0.1%
    (4) bytes_hbm_legacy and bytes_hbm_substrate within ±0.1%
    (5) bytes_dram_legacy and bytes_dram_substrate within ±0.1%
    (6) PermissionError on every cross-tenant access attempt     # G1
    (7) events_dropped == 0 in both runs                          # §2.8 D2
```

The bridge test does NOT assert:
  - The order in which handles are allocated.
  - The exact VA values (different runs may pick different VAs).
  - The exact eviction order within a phase.
  - Microbenchmark timings.

#### Future-proofing comment in the bridge test source

```python
# Bridge test assertions follow OPTION Y from
# docs/substrate_v1_design.md §3.3 (decided 2026-04-29).
# Order of allocations is NOT asserted because no existing
# test in the regression net depends on it (audit recorded
# in §3.3 of the design doc). A future change that makes
# alloc order observable to consumers should re-run that
# audit before relaxing this test's strictness.
```

### 3.4 Regression strategy

  R1  Substrate is added as `memopt/substrate/` with no edits
      to existing modules. The 682 baseline tests continue to
      run unchanged.

  R2  CI runs the full suite at the end of every commit
      (every commit in §3.5). Any regression halts the
      sequence.

  R3  The bridge test (§2.9, §3.3) runs in two modes during
      Phase B: with `MEMOPT_VMM_USE_SUBSTRATE=0` and `=1`. Both
      modes must produce byte-equivalent metrics per §3.3.

  R4  The substrate's own tests (`memopt/substrate/tests/*`)
      are written with no environmental dependencies on
      `memopt.vmm.*` so a substrate test failure cannot mask a
      legacy regression and vice versa.

  R5  Pre-existing collection failures
      (`test_cuda_vmm_smoke.py`,
      `test_torch_allocator_sanity.py`) stay skipped on macOS
      / no-CUDA hosts as they are today. Their behaviour is
      not affected by the substrate work.

### 3.5 Implementation commit plan (ADDITION 4)

Phase A is delivered as 16 commits. Each commit is one Claude
Code invocation, reviewed by the user before the next is
launched. No bulk commits; each is independently revertible.

| #   | Commit                                                  | Touches                                                       | New tests                                                                    | Regression run |
| --- | ------------------------------------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------------------- | -------------- |
| 0   | Step Zero verification report                           | `docs/substrate_v1_step_zero_report.md`                       | (none — verification only)                                                   | full           |
| 1   | csrc G12 fix (dynamic page cap)                         | `csrc/cuda_vmm/vmm_allocator.{h,cpp}`                         | `test_dynamic_capacity_above_legacy_cap`                                     | full + GPU CI  |
| 2   | substrate shell + handle dataclass                      | `memopt/substrate/__init__.py`, `handle.py`                   | `memopt/substrate/tests/test_handle.py`                                      | full           |
| 3   | backend protocol + ABC                                  | `memopt/substrate/backends/base.py`                           | `test_backend_protocol.py`                                                   | full           |
| 4   | CPU fallback backend                                    | `memopt/substrate/backends/cpu_fallback.py`                   | `test_backend_cpu.py`                                                        | full           |
| 5   | CUDA VMM backend (wraps libmemopt_vmm)                  | `memopt/substrate/backends/cuda_vmm.py`                       | `test_backend_cuda.py` (@gpu)                                                | full + GPU CI  |
| 6   | ROCm/HIP backend (real or stub per S0.2)                | `memopt/substrate/backends/rocm_hip.py`                       | `test_backend_hip.py` (@gpu_amd)                                             | full + AMD CI  |
| 7   | Level Zero stub                                         | `memopt/substrate/backends/level_zero.py`                     | `test_backend_l0_stub.py`                                                    | full           |
| 8   | CXL/NUMA backend                                        | `memopt/substrate/backends/cxl_numa.py`                       | `test_backend_cxl.py` (@cxl)                                                 | full           |
| 9   | TenantArena + tenant isolation                          | `memopt/substrate/arena.py`, `tenant.py`                      | `test_arena.py`, `test_tenant_isolation.py`                                  | full           |
| 10  | StreamRegistry + CCA divergence tests                   | `memopt/substrate/stream_registry.py`                         | `test_stream_registry.py`, `test_cca_divergences.py`                         | full + GPU CI  |
| 11  | EventRing + observer dispatch                           | `memopt/substrate/events.py`                                  | `test_events.py`                                                             | full           |
| 12  | AllocationManager assembly + smoke + perf bench         | `memopt/substrate/manager.py`                                 | `test_substrate_smoke.py`, `test_perf_microbench.py` (@gpu @perf)            | full           |
| 13  | Bridge test (legacy parity)                             | `tests/test_substrate_legacy_parity.py`                       | (the bridge test itself; runs against substrate built in 12 + legacy VMM)    | full           |
| 14  | Documentation: threat model, user guide, README updates | `docs/substrate_v1_threat_model.md`, `docs/substrate_v1_user_guide.md`, `README.md` | (none — doc only)                                                  | full           |
| 15  | Final regression run + version bump                     | `pyproject.toml`, `memopt/__init__.py` (version), CHANGELOG   | (none — release pass)                                                        | full           |

#### Per-commit acceptance criteria

  AC1  The committed code compiles / parses with no new
       warnings.
  AC2  Every test introduced in the commit passes locally.
  AC3  The full 682-test regression suite passes (with the
       same 19 skipped, 11 deselected as the baseline).
  AC4  No existing module has been edited, EXCEPT in the
       commits explicitly listed as touching them (commit 1:
       csrc only; commit 14: docs only; commit 15: pyproject
       and CHANGELOG only).
  AC5  The commit message references the design doc section
       it implements (e.g. "Implements §2.5 EventRing per
       docs/substrate_v1_design.md").
  AC6  The commit is independently revertible: reverting it
       does not break any earlier commit's tests.

#### Commit ordering rationale

  - 0 first: nothing else may proceed if a TODO-VERIFY fails
    in a way that requires a Phase 2 redesign.
  - 1 (G12) before 5 (CUDA VMM backend): the CUDA backend
    depends on libmemopt_vmm.so without the legacy cap.
  - 3 (ABC) before any concrete backend (4–8): dependency.
  - 9 (arena/tenant) before 10 (stream registry): the
    StreamRegistry references arena freelists for reclaim.
  - 10 (stream) before 11 (events): the EventRing's emit path
    is hot inside StreamRegistry's free path.
  - 12 (manager) after every backend and core data structure:
    the manager assembles them.
  - 13 (bridge) after 12 (manager): the bridge test calls
    memopt.alloc which requires the manager.
  - 14 (docs) after every code commit: docs reference real
    implementation paths.
  - 15 last: version bump only after everything else is green.

#### When a commit fails

  - AC1 / AC2 fail: the commit is fixed in the same
    invocation. No new commit.
  - AC3 fails: the commit is reverted. Either the regression
    is real (substrate has a bug — fix in same invocation) or
    the test was already flaky (record in flake-tracking
    issue; run again).
  - AC4 fails: the commit is rejected. Restart the
    invocation with corrected scope.
  - AC5 / AC6 fail: amend message / split commits.

### 3.6 Phase 3 stop

Phase 3 test plan complete. Awaiting review before Phase 4
(implementation prompt).

The Phase 4 prompt will be a separate, self-contained doc at
`docs/substrate_v1_implementation_prompt.md`. It will cite this
file as authoritative and walk through the 16 commits above in
prompt form, one prompt per commit. It will include the
guardrail "do not edit existing files in `memopt/vmm/`,
`memopt/cluster/`, or `memopt/serving/` outside the listed
commits". It will require all 682 baseline tests to pass after
every commit and the substrate's own tests to pass per §3.1.

---

Phase 3 test plan complete. Awaiting review before Phase 4
(implementation prompt).
