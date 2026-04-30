# memopt Substrate v1 — User Guide

The substrate is memopt's Layer 1 memory management surface. It exposes
five primary functions plus the `MemoryHandle` dataclass.

| Function | Purpose |
| --- | --- |
| `memopt.alloc(size_bytes, ...)` | Allocate a managed handle |
| `memopt.free(handle)` | Release a handle (idempotent) |
| `memopt.context(tenant=..., tag=..., placement=...)` | Scope defaults |
| `memopt.stats(tenant=...)` | Snapshot stats |
| `memopt.observe(kind, callback)` | Subscribe to events |

> Authoritative spec: `docs/substrate_v1_design.md` §2.1.
> Threat model: `docs/substrate_v1_threat_model.md`.

## Quickstart

```python
import memopt

with memopt.context(tenant="alice", placement="cpu"):
    h = memopt.alloc(1 << 20)        # 1 MiB
    h.write(b"\\x42" * 1024)
    sample = h.read(0, 16)
    memopt.free(h)
```

## `memopt.alloc`

```python
h = memopt.alloc(
    size_bytes,
    tenant="_default",
    tag="default",
    placement="auto",
    stream=None,
    ttl_seconds=None,
    hint=None,
)
```

`size_bytes` must be a positive int. `tenant` matches
`^[A-Za-z0-9_-]{1,128}$`; `tag` matches `^[A-Za-z0-9_-]{1,64}$`.
`placement` is one of `auto`, `hot`, `warm`, `cold`, `hbm`, `dram`,
`cxl`, `nvme`, `cpu`. `stream`, when provided, must be the binding
stream (PyTorch CCA semantics — see §2.5 of the design doc).

Returns a `MemoryHandle`.

```python
h = memopt.alloc(2 * 1024 * 1024, tenant="alice", tag="kv_cache",
                 placement="hbm")
```

## `memopt.free`

```python
memopt.free(h)
```

Idempotent. Double-free increments a debug counter visible in
`memopt.stats()["double_free_count"]`. If the handle had a stream
binding, the substrate records a `cudaEvent` and queues the physical
block for reclaim only after the event completes.

## `memopt.context`

```python
with memopt.context(tenant="alice", tag="kv_cache", placement="hbm"):
    h1 = memopt.alloc(1 << 20)
    h2 = memopt.alloc(1 << 20, tag="weight")  # tenant=alice, placement=hbm
```

`contextvars`-based, so the binding propagates to threads spawned by
`concurrent.futures.ThreadPoolExecutor.submit()` and to coroutines.

## `memopt.stats`

```python
snap = memopt.stats(tenant="alice")    # alice's stats
agg = memopt.stats(tenant=None)        # admin-token-gated aggregate
```

The aggregate form requires `MEMOPT_ADMIN_TOKEN`. See the threat model
G2 entry.

## `memopt.observe`

```python
def on_alloc(event):
    print(f"alloc: tenant={event.tenant} size={event.size_bytes}")

with memopt.observe("alloc", on_alloc):
    memopt.alloc(1024)
```

Subscribers run on a dedicated dispatcher thread per subscription
(D3, §2.8). A slow subscriber affects only its own thread; the
allocating thread is never blocked.

## `MemoryHandle`

```python
@dataclass(frozen=True)
class MemoryHandle:
    handle_id: int
    size_bytes: int
    tenant: str
    tag: str
    placement: str                     # current physical tier
    stream: Optional[torch.cuda.Stream]
    backend_name: str                  # "cuda" | "hip" | "cpu" | ...
    created_at: float                   # time.monotonic()
    ttl_seconds: Optional[float]
    hint: Optional[dict]
```

Methods:

  - `read(offset=0, size=None) -> bytes`
  - `write(data, offset=0)`
  - `as_tensor(dtype, shape) -> torch.Tensor`
  - `as_numpy(dtype, shape) -> np.ndarray`
  - `stats() -> dict`
  - `free()`
  - context-manager protocol (`with memopt.alloc(N) as h:`)

## Performance

These are **TARGETS**, not measurements. Per design §2.10:

| Operation | Target | Reference |
| --- | --- | --- |
| `alloc()` cold (cuMemCreate path) | <50 µs | typical CUDA driver call |
| `alloc()` warm (arena freelist hit) | <500 ns | jemalloc-style sub-allocator |
| `alloc()` sub-allocated (<1 MiB) | <300 ns | shared 2 MiB physical page |
| `free()` warm | <300 ns | freelist append |
| `free()` cold (drives unmap+release) | <30 µs | CUDA driver call |
| Event emit | <1 µs | per Step Zero S0.6 fallback (relaxed from <200 ns) |
| Cross-tenant access denial (G1) | <100 ns | dict lookup + string compare |

These numbers are **GOALS**, not contracts. The
`memopt/substrate/tests/test_perf_microbench.py` harness records the
actual measurements per commit at `/tmp/memopt-bench/<git-sha>.json`
and asserts only ordering invariants.

## Honest limits

  - **G13** (granularity): native CUDA granularity on Hopper / Ada /
    Ampere is 2 MiB. v1 does not ship the patched UVM driver required
    for 64 KiB pages.
  - **S0.1 fabric handle support**: DEGRADED on the rig that built this
    artifact — `export_fabric_handle()` returns `None` until re-verified
    on a CUDA 12.4+ rig with IMEX.
  - **S0.2 ROCm/HIP**: DEGRADED — ships as honest stub. AMD support
    requires an MI200/MI300X rig pre-Commit 6.
  - **S0.5 CXL detection**: DEGRADED — auto-detection disabled; the
    backend is gated on `MEMOPT_CXL_NODES`.
  - **S0.6 event ring atomic**: uses `threading.Lock` per documented
    fallback. <1 µs target instead of the <200 ns aspiration.

See `docs/substrate_v1_step_zero_report.md` for the complete
verification record.

## Phase B / C / D

What v1 does NOT deliver: the VMM adapter that routes to the substrate
under `MEMOPT_VMM_USE_SUBSTRATE=1` (Phase B), flipping the default
(Phase C), and removing the legacy code path (Phase D). Each is a
separate engagement.
