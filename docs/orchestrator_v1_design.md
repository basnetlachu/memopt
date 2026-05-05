# memopt Tier Orchestrator (Layer 2) — Design

| Field    | Value                                                           |
| -------- | --------------------------------------------------------------- |
| Status   | Phase 1 complete. Awaiting review before Phase 2.               |
| Scope    | Tier orchestration on top of substrate v1.1                     |
| Started  | 2026-05-01                                                      |
| Repo     | `/Users/lachumanbasnet/Personal/Sophisticates/MEMOPT/memopt`    |
| Layer 1  | Substrate v1.1 at `13cd56d` (`Substrate v1 release — Phase A complete`) |
| Baseline | 752 passed, 22 skipped, 15 deselected on `main` at `13cd56d` (Mac, Python 3.12, no CUDA) |

Sections:

  1. Discovery (Phase 1 — this section)
  2. Design (Phase 2 — pending)
  3. Test plan (Phase 3 — pending)
  4. Implementation prompt (Phase 4 — pending)

---

## Phase 1 — Discovery

Discovery only. No design decisions appear in this section. Every claim
about existing code is cited at `file:line`. Where the citation is to an
absent thing, the section says so explicitly.

### 1.1 Substrate event surface (what Layer 2 receives)

#### 1.1.1 The `Event` record

Defined at `memopt/substrate/events.py:36-46`:

```text
@dataclass(frozen=True) Event:
  kind:            str            # one of "alloc","free","evict","promote","migrate"
  timestamp_ns:    int            # time.monotonic_ns() at emit time
  handle_id:       int            # AllocationManager-assigned, monotonically increasing
  tenant:          str
  tag:             str
  size_bytes:      int
  from_placement:  Optional[str]
  to_placement:    Optional[str]
  reason:          Optional[str]
```

The five legal values for `kind` are pinned at
`memopt/substrate/events.py:33`:
`("alloc", "free", "evict", "promote", "migrate")`.

#### 1.1.2 Subscription surface

The user-facing entry point is `memopt.observe(event, callback)` —
re-exported at `memopt/__init__.py:59` and at
`memopt/substrate/__init__.py:61-66`. It delegates to
`AllocationManager.observe()` at `memopt/substrate/manager.py:350-355`,
which calls `Dispatcher.subscribe(kind, callback)` at
`memopt/substrate/events.py:150-163`.

Subscription returns a `SubscriptionHandle`
(`memopt/substrate/events.py:108-125`) that is a context manager and
exposes `unsubscribe()`.

#### 1.1.3 Delivery contract

Documented in the module docstring at `memopt/substrate/events.py:1-15`
and implemented as follows:

  - **Per-subscriber thread.** `_SubscriberThread`
    (`memopt/substrate/events.py:179-227`) runs the callback off the
    allocating thread. The thread name is `memopt-dispatch-<kind>`
    (`:193`).
  - **Filter by kind.** `notify()` returns immediately if
    `event.kind != self._kind` (`:206-207`). A subscriber today can
    only listen to one of the five kinds; observing all five requires
    five `subscribe()` calls.
  - **Best-effort delivery (D1).** Subscriber exceptions are caught
    and logged at WARNING (`:222-227`); they never propagate to the
    producer.
  - **Drops are accounted.** When the ring exceeds capacity (default
    4096, `MEMOPT_EVENT_RING_CAPACITY`, see `:30`), the oldest record
    is overwritten and `events_dropped` is incremented
    (`memopt/substrate/events.py:82-91`). The counter is exposed at
    `:73-75` and bubbled into stats via
    `memopt/substrate/manager.py:325`.
  - **In-memory only.** No persistence (D6). The ring is a
    `List[Optional[Event]]` of size `capacity`
    (`memopt/substrate/events.py:60`).

#### 1.1.4 Where each event kind is emitted

  - `alloc` — `memopt/substrate/manager.py:232-242`. `reason` is
    `"cold"` or `"freelist"`.
  - `free`  — `memopt/substrate/manager.py:269-278`. `reason` is
    `"stream_locked"` or `"immediate"`.
  - `evict`, `promote`, `migrate` — **never emitted by Layer 1.** No
    `_dispatcher.emit(Event(kind="evict",...))` call exists in any
    file under `memopt/substrate/`. The five kinds are validated at
    subscription time (`memopt/substrate/events.py:155-156`), but
    only `alloc` and `free` are produced. `evict`/`promote`/`migrate`
    are part of the contract for Layer 2 to fill.

#### 1.1.5 Allocation request fields visible at the boundary

`AllocationManager.alloc()` at `memopt/substrate/manager.py:154-243`
takes `(size_bytes, tenant, tag, placement, stream, ttl_seconds, hint)`.
Of these:

  - `tenant`, `tag`, `placement`, `size_bytes` are echoed into every
    event.
  - `hint` is stored on the handle (`memopt/substrate/handle.py:75`)
    but is never inspected by the manager and never reaches a
    subscriber. Layer 2 will need its own channel for this if it
    wants to take hints into account.
  - `ttl_seconds` is also stored on the handle but never enforced —
    the only counter is `_ttl_expired_count` (`manager.py:78`), which
    is initialized to 0 and never incremented. Confirmed: no
    `_ttl_expired_count += 1` in the substrate.

### 1.2 Existing eviction/promotion logic (what Layer 2 replaces or coexists with)

Three independent eviction/promotion implementations exist in the
repo today. Layer 1 does **not** call any of them.

#### 1.2.1 C++ allocator (`csrc/cuda_vmm/vmm_allocator.cpp`)

LRU eviction over a fixed page pool, plus promotion back to HBM.

  - **`evict_page_locked`** at `csrc/cuda_vmm/vmm_allocator.cpp:524-561`.
    Copies HBM → DRAM via a pre-pinned host slab (`host_mirror_acquire`,
    `:530`); on success unmaps the VA, releases the HBM physical handle,
    sets `p->tier = TIER_DRAM`, and bumps `eviction_count` /
    `bytes_evicted_total`.
  - **`memopt_evict_n_pages`** at `:567-593`. Picks the LRU page (lowest
    `last_access` tick) per round and evicts up to `n` pages.
  - **`memopt_evict_to_target`** at `:599-645`. Repeatedly evicts the
    LRU page until `pool_pressure_locked(a) <= target_fraction`. The
    eviction policy is unconditional LRU; no tag/tenant priority.
  - **`memopt_promote`** at `:651-708`. DRAM → HBM. Allocates a new
    HBM physical handle (`cuMemCreate`, `:671-672`), maps it into the
    existing VA (`cuMemMap`, `:679`), restores access
    (`cuMemSetAccess`, `:691`), copies bytes back from the host
    mirror (`cudaMemcpy`, `:694-695`), returns the mirror to the slab
    (`host_mirror_release`, `:698`), updates `last_access`, increments
    `promotion_count`.
  - **Caller surface.** Eviction is triggered from two places only:
    - On allocation failure: `memopt_torch_alloc` at
      `csrc/cuda_vmm/vmm_allocator.cpp:893-905` calls
      `memopt_evict_to_target(0.70f)` after a failed `memopt_malloc`.
    - On step boundary: `memopt_torch_step_boundary` at `:960-972`
      checks `s.hbm_pressure > threshold`, then calls
      `memopt_evict_to_target(threshold - 0.10f)`.

  This allocator does not communicate with the Layer 1 substrate at
  all. It owns its own `MemoptAllocator*` (`g_torch_allocator`) and is
  reachable only through the PyTorch pluggable-allocator install path.

#### 1.2.2 Python `TierManager` (`memopt/vmm/tier_manager.py`)

A Python tier manager built on top of `PageTable` (sharded C++ table)
and `hal.backend`.

  - **`allocate`** at `memopt/vmm/tier_manager.py:92-106`. Always
    allocates in the hot tier (`_hal.tier_names[0]`); calls
    `_ensure_capacity(hot, size_bytes)` first.
  - **`fetch`** at `:108-157`. The page-fault handler. If the entry's
    tier is not the hot tier, calls `_promote(entry, hot)` (`:122-124`).
    On lookup miss, attempts `_try_gum_fetch` (`:128-152`) before
    raising `KeyError`.
  - **`_promote`** at `:183-193`. Allocates fresh in the target tier,
    `_sync_copy` from old to new, calls `page_table.update_tier`,
    frees the old handle.
  - **`_evict_one`** at `:195-221`. Walks `lru_candidates(from_tier,
    count=1)` (`:202`), allocates in the next-colder tier, copies,
    updates page table, frees old handle. If evicting to NVMe,
    registers in the GUM block directory (`:218-219`).
  - **`_ensure_capacity`** at `:322-339`. Watermark loop:
    `MEMOPT_EVICT_HIGH=0.90` (`:50`) triggers, drains down to
    `MEMOPT_EVICT_LOW=0.75` (`:51`). Defaults at `:50-51`; env
    overrides at `:72-75`; validation at `:77-86`.

  The Python `TierManager` is invoked only through `memopt.vmm.VMM`
  (`memopt/vmm/__init__.py:81-87`). It does **not** subscribe to any
  substrate event and does not emit events. The substrate cannot
  observe its activity, and it cannot observe the substrate's.

#### 1.2.3 Backend-level allocate/free (`memopt/vmm/backends/`)

  - **`CUDABackend`** at `memopt/vmm/backends/_cuda_backend_py.py`:
    `allocate(size, tier, tenant_id, sequence_id, block_index)`
    (`:81-119`) returns a `torch.Tensor` for `hbm`/`dram`, a file path
    for `nvme`. `free(handle, tier)` (`:121-126`) deletes/unlinks.
    These have no awareness of "evict" or "promote" — the tier choice
    is supplied by the caller.
  - **`UnifiedBackend`** at
    `memopt/vmm/backends/unified_backend.py:139-184`. Two-tier
    (DRAM + NVMe) with `hbm` silently remapped to `dram` (`:152-157`).
    Same shape as `CUDABackend`. Used on Apple Silicon and CPU-only
    hosts (the current Mac dev box).

#### 1.2.4 Page-table LRU helper (`csrc/core/page_table.{h,cpp}`)

  - **`lru_candidates(tier, count)`** at
    `csrc/core/page_table.h:192-193` and implemented at
    `csrc/core/page_table.cpp:240-286`. Walks the tail of the per-shard
    intrusive LRU list, collects entries matching `tier` with
    `pin_count == 0`, partial-sorts by `last_accessed` ascending and
    returns the oldest `count`.
  - **`pin` / `unpin`** at `csrc/core/page_table.h:182-187`. Used by
    Layer 2 candidates to mark "do not evict."
  - **`lookup` / `update_tier`** at `:154-170` and
    `csrc/core/page_table.cpp:71-175`. The page table holds
    `(sequence_id, block_index, tier, handle, size, last_accessed,
    pin_count)`. There is no `tenant` field on `PageTableEntry`
    (`page_table.h:79-103`).

  This page-table is the closest thing in the repo today to a
  "cross-component access record." The Layer 1 substrate has no
  equivalent — `TenantArena` (`memopt/substrate/arena.py:63-173`)
  tracks size-class freelists and per-tag byte counts only, not
  per-handle access timestamps.

#### 1.2.5 Files matching the Phase 1 globs

  - `*oracle*.py`: `memopt/vmm/oracle.py` (shim,
    `:14-37`), `memopt/vmm/_oracle_py.py` (Markov + sequential +
    recency oracle, see §1.3.1), `memopt/vmm/global_oracle.py`
    (control-plane aggregator), `memopt/vmm/oracle_trainer.py`
    (background warm-from-log, `:25-40`),
    `memopt/vmm/oracle_data_cleaner.py`,
    `memopt/serving/tests/test_oracle_endpoints.py`,
    `memopt/vmm/tests/test_oracle.py`,
    `memopt/vmm/tests/test_global_oracle.py`. There is also a C++
    oracle at `csrc/core/oracle.{h,cpp}` referenced by
    `memopt/vmm/oracle.py:16` (`from memopt._memopt_core import
    MemoryOracle`), used as the optional fast path.
  - `*prefetch*.py`: `memopt/vmm/prefetch_engine.py` (see §1.3.2).
  - `*access*.py` / `*access_log*.py`: `memopt/vmm/access_log.py`
    (see §1.3.3) and `memopt/profiler/access_analyzer.py`
    (kernel-level GPU access pattern analyzer, unrelated to tier
    orchestration; verified at `memopt/profiler/access_analyzer.py:1-40`).
  - `*tier*.py`: `memopt/vmm/tier_manager.py` only.

### 1.3 Existing access tracking / oracle / prefetch code

#### 1.3.1 `MemoryOracle` (Python fallback)

Pure-Python first-order Markov oracle at
`memopt/vmm/_oracle_py.py:40-251`. Public surface:

  - `observe(sequence_id, block_index, step=None)` at `:77-113`.
    Records a transition `prev_block → block_index` in
    `self._transitions` (`Counter` per source block, `:49-50`), bumps
    a per-sequence step counter, refreshes recency.
  - `predict(sequence_id, current_block, top_k=10)` at `:115-180`.
    Combines four sources, ordered: transition (`:122-132`),
    sequential (`+1`/`+2`, `:134-141`), recency (most-recent five for
    the sequence, `:143-155`), fallback (`+1` at confidence 0.3,
    `:157-163`). Returns up to `top_k` `BlockPrediction` records,
    sorted by confidence descending (`:165-169`).
  - `record_outcome(sequence_id, block_index)` at `:182-188`. If the
    block was in `_pending_predictions`, increments
    `_stats.total_predictions_correct`.
  - `warm_from_log(log_path, max_events=50_000)` at `:229-251`. Reads
    a JSONL access log and calls `observe()` per record.

A C++ implementation backs the same name when
`memopt._memopt_core` builds (`memopt/vmm/oracle.py:14-26`). On Mac
without that build, the Python class is used (verified by
`memopt/vmm/oracle.py:27-37`).

The oracle is keyed on `(sequence_id, block_index)`. There is **no
tenant dimension** on transitions: `_transitions[prev_block]` maps
`int → Counter[int]`, so two tenants who use overlapping
`block_index` values will pollute each other's transition tables.
This is a Layer-2 concern — Layer 1 events do carry `tenant`.

#### 1.3.2 `PrefetchEngine`

`memopt/vmm/prefetch_engine.py:141-411`. Wraps `TierManager` and an
optional oracle.

  - `record_access(sequence_id, block_index, **kwargs)` at `:176-237`.
    Updates `_transitions` (its own private Markov chain, `:184-185`),
    EWMA-smooths inter-access gap into `_gap_estimate[seq]` (`:187-192`),
    fires `_accuracy_tracker.record_access` (`:200-202`), writes the
    `BlockAccessEvent` to `AccessLog` if attached (`:204-215`), then
    schedules predicted prefetches (oracle path `:218-232`,
    fallback path `:234-237`).
  - `_schedule_prefetch(key, window_s)` at `:387-411`. Spawns a
    daemon thread per prefetch, sleeps `window_s`, then calls
    `tier_manager.fetch(seq_id, block_idx)`. Window defaults to
    `_DEFAULT_PREFETCH_WINDOW_S = 0.0005s` × `_PREFETCH_FRACTION = 0.80`
    (`:25-27`).
  - `PrefetchAccuracyTracker` at `:30-138`. Tracks accurate/wasted/
    missed prefetches with a default 5 s window
    (`MEMOPT_PREFETCH_WINDOW_S`, `:168`).

  The engine is constructed by `VMM.__init__`
  (`memopt/vmm/__init__.py:87`). It does not subscribe to substrate
  events. Its prefetch thread calls into `TierManager.fetch` directly.

#### 1.3.3 `AccessLog`

Append-only JSONL logger at `memopt/vmm/access_log.py:37-141`.

  - `record(BlockAccessEvent)` is non-blocking
    (`queue.Queue(maxsize=10_000)`, `:47`); drops are counted in
    `_events_dropped` (`:71-73`).
  - The `BlockAccessEvent` schema at
    `memopt/vmm/access_log.py:25-35` carries
    `(sequence_id, block_index, token_position, attention_layer,
    timestamp, tier_at_access, promotion_latency_ms, tenant_id)`.
    The `tenant_id` field IS present here, unlike on
    `PageTableEntry`.
  - File path: `~/.memopt/access_logs/access_log_<YYYYMMDD>.jsonl`
    by default (`:53-56`).

  This is the only persisted access record in the repo. It is fed
  only by `PrefetchEngine.record_access` (`prefetch_engine.py:204-215`)
  and read by `MemoryOracle.warm_from_log` and `OracleTrainer`.

#### 1.3.4 Other oracle infrastructure (out-of-process)

`memopt/vmm/global_oracle.py` and `memopt/vmm/pod_controller.py`
implement a three-tier oracle hierarchy
(`memopt/vmm/global_oracle.py:1-20` describes it). These run as
control-plane daemons that pull transitions from peer oracles and
push back high-confidence transitions. They are out-of-process and
do not share state with the substrate; they consume the same
`MemoryOracle` API.

### 1.4 Layer 2 gaps (what does NOT exist today)

Each gap is something Layer 2 will need that the codebase does not
have. Format follows `docs/substrate_v1_design.md §1.6`.

**O1. No subscription to substrate events.**
`grep -r "memopt.observe\|substrate.observe\|AllocationManager.observe"
memopt/ tests/` returns no callers. The substrate emits `alloc`/`free`
to the dispatcher (`memopt/substrate/manager.py:232,269`) but no
process subscribes. The `evict`/`promote`/`migrate` kinds are
declared in the kind-tuple (`memopt/substrate/events.py:33`) but
never emitted by Layer 1 (verified §1.1.4). Layer 2 must (a)
subscribe and (b) be the producer of the three unused kinds.

**O2. No tenant-aware access tracker.**
The C++ `PageTable` holds per-block `last_accessed` and `pin_count`
(`csrc/core/page_table.h:79-103`) but no `tenant`. The substrate
`TenantArena` (`memopt/substrate/arena.py:63-173`) holds per-tag
byte counts but no per-handle access timestamps. The substrate
`MemoryHandle` carries `tenant` and `tag` but not access metadata
(`memopt/substrate/handle.py:64-83`). Layer 2 needs a structure
keyed `(tenant, tag, handle_id) → access_history` that does not
exist anywhere today.

**O3. No prefetch-ready data structure for substrate handles.**
The existing Markov chain in `_oracle_py.py` is keyed on
`(sequence_id, block_index)`, not on substrate handles or tags. The
"oracle" abstraction is tied to KV-cache block indexing
(`memopt/vmm/_oracle_py.py:40-50`), which has no substrate analogue.
A new tag-or-pattern-keyed predictor will be needed; whether to
reuse `_oracle_py.MemoryOracle` by re-keying it, or build a new one,
is a Phase 2 decision.

**O4. No policy DSL.**
Eviction policy in `TierManager._ensure_capacity`
(`memopt/vmm/tier_manager.py:322-339`) is hard-wired LRU + two
watermarks (`MEMOPT_EVICT_HIGH`, `MEMOPT_EVICT_LOW`,
`tier_manager.py:50-51`). The C++ allocator's policy is hard-wired
LRU + a single target ratio (`vmm_allocator.cpp:567-645`). There is
no place where a user can declare "evict tag=draft before tag=prod"
or "never evict tenant=X below 4 GiB." Adding such a hook in
Layer 1 is explicitly out of scope per substrate design §2.11
(verified `docs/substrate_v1_design.md:2119+`); it belongs in
Layer 2.

**O5. No coordination between Layer 1 `placement="auto"` and a
smarter "auto."**
`AllocationManager._pick_backend` at
`memopt/substrate/manager.py:129-152` resolves `"auto"` to
`PhysLoc.HBM` via the static map at `:39-49` (the entry
`"auto": PhysLoc.HBM`) and prefers a CUDA backend when available.
There is no callback hook by which Layer 2 could intercept and
override the placement. `alloc()` accepts a `hint` argument
(`manager.py:163`) that is stored on the handle (`handle.py:75`)
but never consulted.

**O6. No backpressure mechanism.**
`alloc()` either succeeds or raises `MemoryError`
(`manager.py:135,152`). There is no advisory channel by which Layer 2
could veto a placement, request a wait, or redirect a class of
allocations to a different tier. `_dispatcher.emit` is one-way
(`events.py:144-148`); subscribers cannot return a verdict.

**O7. No cross-tenant fairness logic.**
`TenantArena` is per-tenant (`memopt/substrate/manager.py:121-127`)
and tenants do not share blocks (test
`test_arena_per_tenant_isolation_no_cross_freelist` referenced in
`memopt/substrate/arena.py:5-7`). But there is no quota, weight, or
priority tracked per tenant; nothing prevents tenant A from
exhausting the device in line with tenant B. The C++ allocator
has no tenant concept at all (verified §1.2.1; `MemoptPage`
struct in `vmm_allocator.cpp` does not contain a tenant field).

**O8. No telemetry / observability surface for orchestrator
decisions.**
`AllocationManager.stats()` (`manager.py:303-335`) reports per-tenant
arena stats, `events_dropped`, `double_free_count`,
`ttl_expired_count`, and per-backend availability. There is no slot
for "decisions Layer 2 made," "predictions hit/missed," "policy
overrides applied." `MemoryOracle.stats` and
`PrefetchAccuracyTracker.stats` exist
(`_oracle_py.py:190-211`, `prefetch_engine.py:112-138`) but for the
VMM path only. Layer 2 will need to either extend
`AllocationManager.stats()` or expose its own.

**O9. No bridge between substrate events and the existing oracle/
access-log infrastructure.**
`AccessLog.record()` is fed only by `PrefetchEngine.record_access`
(`prefetch_engine.py:204-215`). `MemoryOracle.observe` is fed only
by `PrefetchEngine.record_access` (`prefetch_engine.py:232`). Neither
sees substrate `alloc`/`free` events. If Layer 2 wants to reuse
the existing oracle, it must either fan out substrate events to
`oracle.observe()` (with key remapping per O3) or duplicate the
oracle.

**O10. No place where placement is decided as a function of
predicted future use.**
The closest existing decision-maker is `ElasticAllocator.decide(...)`
in `memopt/vmm/elastic_allocator.py:7-13` (peeked at the docstring),
which selects a target tier from a confidence score and node
memory state. It is invoked from `prefetch_engine.py:227-228` only
when `_allocator` is wired in by the caller. It is not connected to
the substrate; it does not know about `MemoryHandle`,
`PhysLoc`, or `BackendStrategy`. Whether to absorb it, replace it,
or ignore it is a Phase 2 decision.

**O11. No async / background coordinator process.**
The substrate's only background work is per-subscriber dispatcher
threads (`events.py:179-227`) and the optional reclamation pass
inside `TenantArena.reclaim` (`arena.py:142-158`), which is invoked
synchronously by `manager.free()`. There is no long-running
"orchestrator" loop today. Layer 2 will need to decide whether
decisions run inline on the subscriber thread or on a dedicated
worker.

### 1.5 Layer 1 / Layer 2 boundary

This is the single most load-bearing Phase 1 deliverable. Phase 2
is not allowed to violate it.

#### 1.5.1 Layer 1 owns

  - **The seven primitives.** `reserve_va`, `create_physical`, `map`,
    `set_access`, `unmap`, `release_physical`, `free_va` — defined as
    `BackendStrategy` abstract methods at
    `memopt/substrate/backends/base.py:46-72` and enumerated as
    `SEVEN_PRIMITIVES` at `:82-91`.
  - **Handle lifecycle.** `MemoryHandle` allocation, the state
    machine `live → freed_pending_event → released`
    (`memopt/substrate/handle.py:85-103`), `free()` semantics
    (`:150-165`), the double-free counter
    (`:153-156`), the ref-count for `as_tensor` weakrefs (`:122-129`,
    `:174-179`).
  - **Stream registry semantics.** PyTorch CCA-equivalent
    `StreamRegistry` (`memopt/substrate/stream_registry.py:32-132`):
    `record_stream`, `free_stream_locked`, `drain_pending`, the
    per-stream `pending` FIFO of `(event, handle_record)` tuples,
    and the pooled event-record/event-query path.
  - **Per-tenant arenas.** `TenantArena`
    (`memopt/substrate/arena.py:63-173`): the size-class table
    (`:30-45`), per-tag byte/handle counts (`_TagStats`, `:57-61`),
    fragmentation tracking (`_should_reclaim_locked`, `:131-140`),
    bounded `reclaim()` (`:142-158`).
  - **Tenant identity and namespacing.** `tenant_context`,
    `assert_tenant_match`, `tenant_hash`, `nvme_block_path`
    (`memopt/substrate/tenant.py:38-122`).
  - **The event ring + dispatcher** as a transport
    (`memopt/substrate/events.py:49-227`). Layer 1 produces `alloc`
    and `free`; the ring, the per-kind subscriber threads, and the
    drop-accounting are L1 forever.
  - **The validated public API surface.**
    `alloc / free / context / stats / observe / MemoryHandle / Event /
    SubscriptionHandle` re-exported at
    `memopt/substrate/__init__.py:69-78` and at
    `memopt/__init__.py:59`. Argument validation
    (`memopt/substrate/handle.py:32-61`) including the closed
    `_VALID_PLACEMENTS` set (`:26-29`).

#### 1.5.2 Layer 2 owns

  - **Decisions about *where*.** The mapping from `placement="auto"`
    (and any new tag-driven placement) to a concrete `PhysLoc` /
    backend, beyond the static `_PLACEMENT_TO_LOC` table at
    `memopt/substrate/manager.py:39-49`.
  - **Decisions about *when*.** Eviction triggers (whether to
    pre-empt before allocation, whether to hold a handle past TTL,
    whether to migrate idle handles to colder tiers).
  - **Decisions about *what*.** Which handle to evict, which to
    promote, which to prefetch — i.e. the policy currently
    hard-coded as LRU in `TierManager._evict_one`
    (`memopt/vmm/tier_manager.py:195-221`) and in
    `memopt_evict_to_target` (`vmm_allocator.cpp:599-645`).
  - **Emission of `evict`, `promote`, `migrate` events.** These three
    `kind` values exist in the substrate's contract
    (`memopt/substrate/events.py:33`) but are emitted only by Layer 2.
  - **Cross-tenant fairness, quotas, priorities.** None of these
    exist today (gap O7).
  - **Telemetry for Layer 2 decisions** (gap O8).

#### 1.5.3 The interface between them

Layer 2 talks to Layer 1 across exactly four surfaces:

  1. **Events going up: `Dispatcher.subscribe(kind, callback)`**
     (`memopt/substrate/events.py:150-163`). Each callback runs on a
     per-kind dispatcher thread, never on the allocator thread. This
     is L2's read-only firehose.
  2. **Allocation requests going down: `memopt.alloc(...)` /
     `memopt.free(...)`** (`memopt/__init__.py:59`,
     `memopt/substrate/__init__.py:17-41`). L2 may call these on
     behalf of itself (e.g. to materialize a promotion target) but
     the existing `alloc()` does not accept a "do not place" verdict
     from L2 — backpressure is gap O6.
  3. **Hint payload: `alloc(..., hint=dict)`**
     (`memopt/substrate/manager.py:163`,
     `memopt/substrate/handle.py:75`). The substrate stores `hint` on
     the handle but does not interpret it. L2 may read it from
     `handle.hint` after subscribing to the corresponding `alloc`
     event, but the event itself does NOT carry `hint` today (the
     `Event` dataclass has no `hint` field — verified
     `memopt/substrate/events.py:36-46`). This is something Phase 2
     must reconcile.
  4. **Stats integration:** L2 may expose its decision metrics
     either by extending `AllocationManager.stats()`
     (`memopt/substrate/manager.py:303-335`) or by exposing its own
     `memopt.orchestrator.stats()`. The substrate today aggregates
     only its own counters (verified §1.4 / O8).

#### 1.5.4 What `placement="auto"` means today vs. with L2

  - **Today:** `_PLACEMENT_TO_LOC["auto"] = PhysLoc.HBM`
    (`manager.py:40`). `_pick_backend("auto")` walks the backend
    list, prefers CUDA when available, falls back to CPU
    (`manager.py:129-152`).
  - **With L2:** Layer 2 should be the resolver of `"auto"` —
    inspecting tenant pressure, tag patterns, predicted lifetime,
    and current tier occupancy before answering "HBM" or "DRAM" or
    "NVMe." The exact handoff (callback registered on the manager?
    pre-alloc subscriber that sets `hint`? subclassing
    `AllocationManager`?) is a Phase 2 decision.
  - **What stays in L1 forever:** the explicit placements
    (`"hot"`, `"warm"`, `"cold"`, `"hbm"`, `"dram"`, `"nvme"`,
    `"cxl"`, `"cpu"`) listed in `_VALID_PLACEMENTS`
    (`memopt/substrate/handle.py:26-29`). When a caller writes
    `placement="hbm"`, that is a contract that Layer 1 honours
    directly; Layer 2 does not get to second-guess explicit
    placement.

#### 1.5.5 Non-negotiables for Phase 2

  - Phase A of L2 must not edit any file under `memopt/substrate/`
    other than to subscribe to its public API. No reaching into
    `_HandleRecord`, `_BackendAdapter`, or the dispatcher internals.
  - L2 must not re-implement the seven primitives or the stream
    registry. If a need arises, it goes through `memopt.alloc()`.
  - The event delivery contract D1–D6
    (`memopt/substrate/events.py:8-15`) is unchanged. L2 callbacks
    must not assume in-order delivery across threads, must not
    raise to the producer, and must tolerate drops.
  - Phase A must not regress the 752-pass baseline (§1.6).

### 1.6 Test surface and regression net

#### 1.6.1 Baseline

Run on Mac (Darwin 25.4.0, Python 3.12.x, no CUDA) at HEAD
`13cd56d`:

```text
PYTHONPATH=. python3 -m pytest tests/ memopt/ -q -p no:cacheprovider \
  -k 'not gpu and not cuda' \
  --ignore=memopt/vmm/tests/test_cuda_vmm_smoke.py \
  --ignore=memopt/vmm/tests/test_torch_allocator_sanity.py
```

Result: **752 passed, 22 skipped, 15 deselected, 9 warnings in 47.51 s**.

Environmental notes:
  - The two ignored files require a built `_memopt_cuda` extension
    or a working CUDA device (verified by file paths).
  - The 15 deselected are filtered by the `-k 'not gpu and not cuda'`
    expression.
  - The 22 skipped include marker-gated tests (`pytest.mark.gpu`
    appears in `memopt/substrate/tests/test_substrate_smoke.py:50`)
    plus environment-dependent skips.
  - One unhandled-thread warning fires from
    `memopt/serving/auto_optimizer.py:136` via
    `memopt/kernels/jit_generator.py:414` — `MockGen` lacks
    `_portability`. Pre-existing, unrelated to this work; not a
    test failure.

This baseline is the regression net for L2 Phase A. The same
command, same machine, same skips/deselects — Phase A must
produce 752+ passed (Phase A is additive; new tests can only
add, not subtract).

#### 1.6.2 Tests that gate eviction/promotion behaviour today

Found via `grep -rln "evict\|promote\|tier\|access_log\|oracle"
tests/ memopt/ --include="test_*.py"`. The substantive ones:

  - **`memopt/vmm/tests/test_vmm_smoke.py`** — 13 tests covering the
    full `VMM.allocate / fetch / free_sequence` path including
    cross-tenant isolation
    (`test_cross_tenant_allocate_raises`, `test_cross_tenant_fetch_raises`,
    `test_cross_tenant_free_raises`). These exercise the Python
    `TierManager` end-to-end. L2 must not break them if it touches
    `VMM`.
  - **`memopt/vmm/tests/test_vmm_hardening.py`** — 8 tests including
    `test_nvme_size_limit_triggers_eviction` (asserts NVMe eviction
    fires when `MEMOPT_NVME_MAX_GB` is exceeded),
    `test_evict_thresholds_from_env`,
    `test_evict_thresholds_invalid_uses_defaults`,
    `test_evict_thresholds_defaults_without_env` (assert the env-var
    plumbing for `MEMOPT_EVICT_HIGH`/`_LOW` at
    `tier_manager.py:72-86`).
  - **`memopt/vmm/tests/test_oracle.py`** — 23 tests covering the
    Markov oracle: `observe` / `predict` / `record_outcome` /
    `warm_from_log` / `OracleTrainer` / `OracleDataCleaner`. These
    pin the public oracle behaviour that L2 may depend on.
  - **`memopt/vmm/tests/test_access_log.py`** — 8 tests covering
    JSONL append, queue overflow accounting, shutdown-flush, thread
    safety, and event field persistence.
  - **`memopt/vmm/tests/test_layer3.py`** — 25 tests covering
    federation, `ElasticAllocator`, and `MemoryGovernor`. Includes
    `test_high_urgency_selects_local_hbm`,
    `test_medium_urgency_selects_local_dram`,
    `test_no_space_falls_back_to_nvme`, and
    `test_pressure_critical_when_high_utilization`. The decision
    logic in these tests is the closest existing analogue to what
    L2 will produce.
  - **`memopt/vmm/tests/test_global_oracle.py`** — 12 tests covering
    the control-plane global oracle (out-of-process aggregator).
  - **`memopt/vmm/tests/test_pod_controller.py`** — 18 tests covering
    the pod-level oracle aggregator.
  - **`tests/test_async_nvme.py`** — 16 tests covering
    `AsyncNVMeManager` (NVMe-tier I/O) and `PrefetchAccuracyTracker`
    (accurate / wasted / missed counters).
  - **`tests/test_integration_gaps.py`** — 12 tests including
    `test_prefetch_tier_passed`, `test_tier_manager_accepts_remote_client`,
    `test_nvme_async_in_tier_manager_stats`. Surface-level integration
    coverage.
  - **`memopt/serving/tests/test_oracle_endpoints.py`** — 11 tests
    covering the HTTP oracle endpoints exposed by the serving stack.
  - **`memopt/substrate/tests/test_events.py`** — 11 tests pinning
    event-emission behaviour for `alloc` / `free` / `evict` /
    `promote`, subscriber threading, exception isolation, ring
    overflow accounting, ordering, unsubscribe, and the
    sub-200 ns / sub-1 µs emission target. Note in particular
    `test_evict_emits_evict_event` and `test_promote_emits_promote_event`
    — these tests *call into the dispatcher directly* (Layer 1 does
    not emit these kinds itself, §1.1.4), so they pin the contract
    that Layer 2 inherits.
  - **`memopt/substrate/tests/test_backend_cuda.py`** — includes
    `test_cuda_backend_evict_promote_round_trip`. CUDA-only, skipped
    on Mac.
  - **`memopt/cluster/tests/test_pillar5_gum.py`,
    `test_pod_gkd.py`, `test_cluster.py`,
    `test_pillar2_twonode.py`, `test_gkd.py`,
    `test_gum_hardening.py`** — cluster-wide block sharing /
    federation tests, included by the grep but not directly relevant
    to Layer 2 placement decisions.
  - **`memopt/control_plane/tests/test_database.py`,
    `test_pods.py`** — control-plane DB tests, peripheral.
  - **`memopt/operator/tests/test_integration.py`,
    `test_models.py`** — k8s operator tests, peripheral.

The combined eviction/promotion regression net is dominated by
`test_vmm_smoke.py` + `test_vmm_hardening.py` + `test_oracle.py` +
`test_access_log.py` + `test_layer3.py` + `test_events.py`. These
are the files Phase 2 must keep green.

### 1.7 Open questions for Phase 2

The following questions are flagged for Phase 2; Phase 1 takes no
position on any of them.

1. **Reuse vs. greenfield decider.** Should Layer 2 absorb
   `memopt/vmm/tier_manager.py:54-455`,
   `memopt/vmm/elastic_allocator.py`,
   `memopt/vmm/memory_governor.py`,
   `memopt/vmm/prefetch_engine.py`, and the C++
   `memopt_evict_to_target` path
   (`csrc/cuda_vmm/vmm_allocator.cpp:599-645`) under one orchestrator,
   or stand up a new package alongside them? The two existing
   eviction loops (Python watermarks vs. C++ pool-pressure target)
   make different policy assumptions; reconciling them is non-trivial.

2. **Hint propagation across the event boundary.** The substrate
   stores `hint` on the handle (`handle.py:75`) but does not put it
   in the `Event` (`events.py:36-46`). For Layer 2 to decide based
   on hints, either (a) the `Event` schema must grow a `hint` field,
   (b) Layer 2 must look up the handle via `handle_id` after
   receiving the event (no such lookup API exists today —
   `AllocationManager._handles` at `manager.py:74` is private), or
   (c) hints must be expressed via a pre-alloc callback that
   bypasses events. Each option has a different blast radius on
   Layer 1.

3. **Backpressure shape.** If Layer 2 wants to veto an allocation
   (gap O6), the contract change must be defined: does `alloc()`
   call out to a Layer 2 advisor synchronously? Does it ask for a
   tier hint and then pick? Does it wait? The current `alloc()` has
   no advisory seam (`manager.py:154-243`).

4. **Tenant key on access tracking.** The existing
   `MemoryOracle._transitions` is keyed on `int → Counter[int]`
   (`_oracle_py.py:49-50`), which does not preserve tenant identity.
   If Layer 2 reuses this oracle (vs. building a new one keyed on
   `(tenant, tag, ...)`), how does it prevent cross-tenant
   transition pollution while still benefiting from learned
   patterns?

5. **Where does Layer 2 run?** Inline on the per-kind subscriber
   threads (`events.py:179-227`)? On a dedicated coordinator
   thread? In a separate process (the existing
   `global_oracle.py` / `pod_controller.py` model)? The choice
   constrains latency budget and synchronization design.

6. **Coexistence with `TierManager`.** While Layer 2 is being
   developed, `memopt.vmm.VMM`'s page-fault path
   (`memopt/vmm/__init__.py:117-146`) continues to drive
   `TierManager.fetch` independently of any Layer 2 events. Does
   Layer 2 silently observe and learn? Does it eventually replace
   `TierManager._ensure_capacity` (`tier_manager.py:322-339`)?
   Does the C++ `memopt_torch_step_boundary`
   (`vmm_allocator.cpp:960-972`) pathway surrender its hard-coded
   eviction policy to Layer 2?

7. **Event semantics of `migrate`.** `kind="migrate"` is in the
   contract (`events.py:33`) but Phase 1 found no producer or
   documented semantics. Is "migrate" different from a
   `free` followed by an `alloc`? Different from a `promote` /
   `evict`? Phase 2 must define this before any code emits it.

8. **Telemetry namespace.** `AllocationManager.stats()`
   (`manager.py:303-335`) is the only documented stats surface today.
   Whether Layer 2 hangs its counters off
   `stats()["orchestrator"]`, off a separate
   `memopt.orchestrator.stats()`, or wires into the existing
   `MemoryOracle.stats` / `PrefetchAccuracyTracker.stats` is open.

---

*End of Phase 1.*

---

## Phase 2 — Design

The orchestrator is added as a new package `memopt/orchestrator/`. No
file under `memopt/substrate/` is edited. Every decision below is
chosen with a 20-year horizon: future maintainers should never have
to undo a shortcut. Where a behaviour matches an established
production pattern (PyTorch CCA stream-locked semantic, jemalloc
arenas, cgroup `memory.high`/`memory.max` watermarks, Linux NUMA
balancing daemon, Linux kswapd), the orchestrator copies it exactly
rather than inventing a near-twin.

### 2.0 Constraints carried from Phase 1

  C0  The §1.5 Layer 1 / Layer 2 boundary is fixed. Layer 1 owns
      handle lifecycle, the seven primitives, the stream registry,
      per-tenant arenas, the event ring (as transport), and the
      validated public API surface (re-cited from §1.5.1). Layer 2
      may not touch any of these from inside.

  C1  The substrate's public API is unchanged. `memopt.alloc`,
      `memopt.free`, `memopt.context`, `memopt.stats`,
      `memopt.observe`, `MemoryHandle`, `Event`,
      `SubscriptionHandle` retain identical signatures and
      semantics (§1.5.1, `memopt/substrate/__init__.py:69-78`).

  C2  The 752-pass Mac baseline (§1.6.1) must remain green through
      Phase A. New tests may add to it; none may flip from pass to
      fail.

  C3  The five legal `Event.kind` values are pinned at
      `memopt/substrate/events.py:33` (§1.1.1). Layer 2 produces
      `evict`/`promote`/`migrate`; introducing a sixth kind requires
      a substrate change and a new design doc.

  C4  Subscriber callbacks run on per-kind dispatcher threads, never
      on the allocator thread; they must not raise to the producer;
      they must tolerate drops (§1.1.3, D1–D6 from
      `memopt/substrate/events.py:8-15`).

  C5  Phase A is purely additive. Eleven gaps O1–O11 (§1.4) are
      addressed by new code under `memopt/orchestrator/`. The legacy
      paths in §1.2 stay frozen during Phase A.

### 2.1 DECISIONS 1–8 (resolves §1.7 open questions)

Decisions are ordered by load-bearingness: Q1, Q3, Q5, Q7 are
critical; Q2, Q4, Q6, Q8 follow.

#### DECISION 1 — Resolves §1.7 Q1 (reuse vs. greenfield)

**Greenfield. Layer 2 ships as `memopt/orchestrator/`, a new
package alongside (not absorbing) the legacy paths in §1.2.**

Rationale: §1.2 documents three independent eviction
implementations — the C++ `memopt_evict_to_target`
(`vmm_allocator.cpp:599-645`), the Python `TierManager._evict_one`
(`tier_manager.py:195-221`), and the watermark loop
`_ensure_capacity` (`tier_manager.py:322-339`). They make
incompatible policy assumptions (pool-pressure-target vs.
high/low-watermark) and serve different consumer surfaces (PyTorch
allocator hook vs. KV-cache page table vs. block file system).
Refactoring all three under one umbrella in v1.0 would couple the
orchestrator's release schedule to three legacy code paths whose
test coverage (§1.6.2) is itself the regression net Phase A must
preserve. A greenfield package can ship without touching any of
them.

The orchestrator subscribes to substrate events (gap O1). It does
not reach into `TierManager`, `ElasticAllocator`, `MemoryGovernor`,
or `PrefetchEngine`. Migration of the legacy paths is the §2.7
Phase D work, not v1.0.

Trade-off: code duplication during Phase A–C. A second LRU
candidate-selection routine, a second access-recency tracker, a
second prefetch policy. We accept this duplication so the legacy
test surface is untouched. §2.7 Phase D collapses the duplication
once the orchestrator is the default.

Revisit if: the maintenance cost of running two parallel decision
systems exceeds two engineer-weeks per quarter, OR a substrate
change forces a coordinated edit across both stacks.

#### DECISION 2 — Resolves §1.7 Q3 (backpressure shape)

**Layer 2 advises before allocation via a registered pre-alloc
callback. The callback returns a placement hint or `None` (no
opinion). It cannot veto the allocation outright. v1.0 does not
add a "wait" verdict.**

Rationale: §1.5.3 lists four interface surfaces between L1 and L2.
A "vetoing" advisor would force `alloc()` to either raise or block
— both are observable behaviour changes that violate C1 (substrate
API stability). An *advisory* hook can be installed entirely from
Layer 2 with one new optional callback registration on
`AllocationManager`; it does not change `alloc()`'s success/failure
contract. The substrate's existing pattern of "fail to MemoryError
when no backend can satisfy" (`manager.py:135,152`) is preserved.

The pattern matches Linux NUMA `set_mempolicy` advisory hints
(MPOL_PREFERRED): the kernel asks for a node, but if memory is
exhausted there it falls back rather than failing. cgroup
`memory.high` (soft limit) versus `memory.max` (hard limit) is the
same dichotomy — Layer 2 sets the soft limit, Layer 1 keeps owning
the hard limit.

Trade-off: Layer 2 cannot delay an allocation request to perform a
coordinated eviction first. Phase A workloads that need this
behaviour must over-provision and let the orchestrator catch up
asynchronously via §2.3.4 decision pump. We accept this because
adding a synchronous "wait" verdict makes Layer 1 dependent on
Layer 2's liveness — a regression on the substrate's
"orchestrator-optional" guarantee (C0).

Revisit if: a real-world workload (named, with a measured
allocation-failure rate) demonstrates that advisory hints are
insufficient and a wait-or-fail verdict is required. At that
point, the substrate ships a new optional `placement_resolver`
callback that may return `WAIT(timeout_ms)`.

#### DECISION 3 — Resolves §1.7 Q5 (where Layer 2 runs)

**In-process, single dedicated coordinator thread per
`AllocationManager` instance. Subscriber callbacks fan out to a
bounded queue; the coordinator drains it. Out-of-process operation
is a Layer 5 concern.**

Rationale: §1.1.3 already pins per-kind subscriber threads
(`memopt/substrate/events.py:179-227`). Running policy on those
threads conflates event delivery with decision-making — a slow
policy evaluation would back up the per-kind queue and risk drops
(§1.1.3 D2). A separate coordinator thread isolates the two:
subscribers do bounded work (parse event, enqueue update),
coordinator does unbounded work (predict, decide, emit).

This matches Linux kswapd: per-CPU page-reclaim daemons separate
from the allocation fast path, woken by water-mark signals,
bounded by `vm.watermark_scale_factor`. It also matches jemalloc
background threads (one per arena, opt-in via
`background_thread:true`). One thread per `AllocationManager`
instance keeps the model singleton-aligned with
`AllocationManager._instance` (`manager.py:64-86`); a future
multi-process model would run one coordinator per process.

Trade-off: a coordinator thread is a long-lived resource. It must
shut down cleanly on `AllocationManager.reset` (`manager.py:88-91`)
and on test fixtures. It cannot be reused across tests that reset
the manager singleton without restarting the thread. §2.3.4
specifies the lifecycle.

Revisit if: profiling shows the coordinator thread saturates a
core under expected load (>50% CPU on a steady allocation
workload), in which case a coordinator pool keyed by tenant is the
next step. Or if multi-process orchestration becomes a v2.0
requirement.

#### DECISION 4 — Resolves §1.7 Q7 (`migrate` semantics)

**`migrate` is a tier change that preserves the handle's `va`
(virtual address) but moves the physical backing between
`PhysLoc`s. It is distinct from `free`+`alloc` (which would
invalidate the handle) and from `evict`/`promote` (which name the
direction).**

Rationale: the substrate's `MemoryHandle` carries `_va`
(`memopt/substrate/handle.py:77`) and `_physical`
(`handle.py:78`). The seven primitives at
`memopt/substrate/backends/base.py:46-72` separate VA reservation
(`reserve_va` / `free_va`) from physical backing
(`create_physical` / `release_physical`) from the `map`/`unmap`
binding between them — the same model that the C++ allocator
implements when it unmaps an HBM physical handle but leaves the VA
reserved (`vmm_allocator.cpp:482-491`). The substrate is built for
remap-without-VA-change. `migrate` names that operation in the
event surface.

Definitions, frozen for v1.0:

  - `evict`     — `from_placement` is hotter than `to_placement`,
                  initiated by pressure or policy. The handle's
                  `placement` field updates; `va` unchanged.
  - `promote`   — `from_placement` is colder than `to_placement`,
                  initiated by predicted access. The handle's
                  `placement` field updates; `va` unchanged.
  - `migrate`   — neither hotter nor colder; same-tier or
                  lateral motion (e.g. between two NUMA nodes,
                  between two HBM partitions on a future MIG-aware
                  build, or between two CXL devices). The handle's
                  `placement` field may or may not update;
                  `to_placement` is set explicitly. `va` unchanged.

`free`+`alloc` is NOT a migrate: it produces a new `handle_id`.
`migrate` reuses the same `handle_id` and the same `va`.

Trade-off: requires future backends to implement an in-place
remap path (call `unmap`, `release_physical`, `create_physical`,
`map` while holding the handle live). Backends that cannot do this
(e.g. CPU mmap with `MAP_FIXED` already in use) must refuse the
migrate and Layer 2 must downgrade to free+alloc, surfacing a
different `handle_id` to consumers. The contract preserves the
distinction.

Revisit if: a backend ships that cannot guarantee VA preservation
under any circumstances — at which point `migrate` becomes
backend-conditional and the event needs a `va_preserved: bool`
field.

#### DECISION 5 — Resolves §1.7 Q2 (hint propagation)

**The `Event` schema gains no new fields in v1.0. Layer 2 looks up
the handle's `hint` via a new substrate read-only API:
`memopt.peek_handle(handle_id) -> Optional[MemoryHandle]`. The
peek is tenant-scoped and validated identically to the existing
tenant-context check (G1, §1.5.1).**

Rationale: extending the `Event` dataclass
(`memopt/substrate/events.py:36-46`) is a wire-format change. Once
shipped, future events must carry the field; removing it later
breaks consumers. Per gap O8 we expect L2 to grow more state than
fits in an `Event`. Threading state through the event is the wrong
abstraction.

A `peek` API is the smallest possible substrate addition: it is
read-only, returns an immutable `MemoryHandle` (frozen dataclass
per `handle.py:64`), and reuses the existing
`AllocationManager._handles` dict (`manager.py:74`). The cost is
one new method on `AllocationManager` and the public re-export.
Tenant scoping uses the existing
`tenant.assert_tenant_match` helper (`tenant.py:72-82`).

Trade-off: Layer 2 pays a dict lookup per event it cares about,
versus Layer 1 paying a dataclass field copy on every event. Per
§2.6 we target a 5 µs event-ingest budget — a dict lookup fits.
The downside is that `peek_handle` becomes a stable API the
substrate must keep working forever.

Revisit if: more than three Layer 2 features need a per-event
context that is not on `Event`, at which point a single new
`Event.context: Optional[dict]` is cheaper than three peek calls.

#### DECISION 6 — Resolves §1.7 Q4 (predictor key)

**Build a new tenant-keyed predictor in `memopt.orchestrator.predict`.
Do not reuse `MemoryOracle` directly. The predictor's transition
table is keyed `(tenant, tag) → Counter[(tenant, tag)]`.**

Rationale: §1.3.1 documents that
`MemoryOracle._transitions` is keyed `int → Counter[int]`
(`_oracle_py.py:49-50`), with no tenant dimension. Reusing that
table would require either (a) namespacing by adding tenant_id
into the key as a tuple — an API change that breaks the existing
13 tests in `test_oracle.py` (§1.6.2), or (b) maintaining one
oracle per tenant — which would deny cross-tenant pattern learning
forever. Both options pin a long-term design choice on a transient
expediency.

A new predictor is small (the algorithm is well-defined: Markov
chain with sequential and recency fallback per `_oracle_py.py:115-180`)
and lets the key shape be tenant-aware from day one. The four
prediction sources (transition / sequential / recency / fallback)
are reproduced; the keyspace is `(tenant, tag)` tuples instead of
`int`. Layer 2 can later import the C++ accelerated oracle when
its key shape converges (out of scope for v1.0).

Trade-off: code duplication of the Markov logic. We accept this
because the existing oracle's key shape is load-bearing for the
VMM tests (§1.6.2 pins `test_observe_updates_transitions`,
`test_predict_returns_block_predictions`, etc.) and changing it
would risk regressions in code Phase A is not allowed to touch
(C5).

Revisit if: a backport-compatible key extension to `MemoryOracle`
(adding an optional `namespace` argument that defaults to a
single bucket) becomes practical, OR if the Layer 2 predictor
needs the C++ acceleration that
`memopt._memopt_core.MemoryOracle` provides.

#### DECISION 7 — Resolves §1.7 Q6 (coexistence with TierManager)

**For v1.0, Layer 2 silently observes substrate events and never
calls into `TierManager`, `ElasticAllocator`, or
`memopt_torch_step_boundary`. Eviction and promotion of substrate
handles happen through new orchestrator code paths only. The
legacy paths continue to manage VMM page-table blocks and
PyTorch-allocator pages independently.**

Rationale: §1.2.2 documents that `TierManager.fetch`
(`tier_manager.py:108-157`) operates on `(sequence_id, block_index)`
keys, not substrate `handle_id`s. There is no shared state between
the two systems today. Driving `TierManager` from Layer 2 would
require either a key-translation layer or a substrate-vs-VMM
adapter; both are non-trivial new abstractions.

The C++ `memopt_torch_step_boundary` path
(`vmm_allocator.cpp:960-972`) is even more decoupled: it operates
on the PyTorch pluggable-allocator pool, which the substrate does
not own. Surrendering its eviction policy to Layer 2 requires a
new C ABI surface; that work belongs in §2.7 Phase D.

For v1.0, Layer 2 is the orchestrator *for substrate handles
only*. VMM blocks remain `TierManager`'s responsibility. Both
systems can run in the same process.

Trade-off: a workload that allocates through both paths (substrate
for general data, VMM for KV cache) gets two uncoordinated
eviction loops. We accept this because each path's pressure
signal is local to its pool — the substrate owns its arenas
(`manager.py:121-127`), the VMM owns the KV blocks. Cross-pool
pressure-balancing is a Layer 5 concern.

Revisit if: a measured workload shows that uncoordinated eviction
between substrate and VMM causes thrashing (e.g. substrate evicts
to DRAM while VMM is promoting from DRAM). Phase D §2.7 collapses
both onto the orchestrator decision pump.

#### DECISION 8 — Resolves §1.7 Q8 (telemetry namespace)

**Layer 2 exposes its own `memopt.orchestrator.stats()` returning
a dict. It does NOT extend `AllocationManager.stats()`. The
existing `MemoryOracle.stats` and `PrefetchAccuracyTracker.stats`
are left untouched.**

Rationale: extending `AllocationManager.stats()` couples the
substrate's stats schema to the orchestrator's release lifecycle —
a Phase B opt-in (§2.7) would either change the substrate's stats
output conditionally on a flag (ugly) or change it unconditionally
(violates C1). A separate top-level entry keeps the substrate's
output stable forever and makes the orchestrator opt-in
observability cleanly side-by-side.

This matches Prometheus's `/metrics` per-component scoping (each
exporter emits its own namespace) and Linux `/proc/<pid>/`
per-subsystem files (memory under `status`, scheduling under
`sched`, namespaces under `ns/`). Each subsystem owns its own
schema.

Trade-off: callers who want a global view do two reads instead of
one. We accept this. The schema for `memopt.orchestrator.stats()`
is specified in §2.3.5.

Revisit if: a top-level `memopt.stats(scope="all")` becomes a
documented requirement, in which case it federates the per-layer
stats by calling each.

### 2.2 Public API surface

Layer 2's public API is on top of, not replacing, the substrate's
(C1). Five new public symbols.

#### 2.2.1 `memopt.orchestrator.start(*, config=None) -> OrchestratorHandle`

Pre-conditions: the substrate's `AllocationManager` may or may
not exist yet — `start()` does not force-construct it. If called
before any `memopt.alloc()`, the orchestrator subscribes lazily
when the manager is first instantiated.

Post-conditions: the coordinator thread is running (DECISION 3);
five subscriptions are registered (one per `Event.kind`); the
returned `OrchestratorHandle` exposes `stop()` and `is_alive()`.

Thread safety: idempotent — calling `start()` twice returns the
same handle. Process-singleton, mirroring `AllocationManager.get`
(`manager.py:81-86`).

CUDA callback safety: never called from a CUDA callback. The
coordinator thread does not enter the CUDA driver directly; all
allocations it triggers go through `memopt.alloc()`, which is
already CUDA-callback-unsafe per the substrate's contract.

`config` is an optional `OrchestratorConfig` dataclass (§2.3.6).
`None` selects defaults.

#### 2.2.2 `OrchestratorHandle.stop() -> None`

Pre-conditions: the orchestrator is running.

Post-conditions: the coordinator thread has joined; subscriptions
are dropped; the access-tracker state is dropped. Calling
`memopt.alloc()` after `stop()` works exactly as if the
orchestrator had never been started.

Thread safety: idempotent; safe to call from any thread.

CUDA callback safety: not callable from a CUDA callback.

#### 2.2.3 `memopt.orchestrator.stats() -> dict`

Pre-conditions: none. Returns an empty dict if the orchestrator
is not running.

Post-conditions: the dict shape is documented at §2.3.5. No
state is mutated.

Thread safety: snapshot under a lock; consistent within itself,
not synchronized with the substrate's `stats()`.

Returns: `{"running": bool, "subscriptions": int,
"events_ingested": {"alloc": int, "free": int, ...},
"decisions": {"evict": int, "promote": int, "migrate": int},
"predictor": {...}, "policy": {...}, "coordinator": {...}}`.

#### 2.2.4 `memopt.orchestrator.register_policy(policy) -> None`

Pre-conditions: `policy` implements the `Policy` protocol (§2.3.3
— `evaluate(snapshot) -> List[Decision]`).

Post-conditions: the policy is added to an ordered list. On the
next coordinator cycle, all registered policies are evaluated;
their decisions are merged via the conflict-resolution rule in
§2.3.3. v1.0 ships exactly one built-in policy (LRU + watermark);
external callers may register their own.

Thread safety: lock-protected mutation of the policy list.

CUDA callback safety: never called from a CUDA callback.

#### 2.2.5 `memopt.peek_handle(handle_id, *, tenant=None) -> Optional[MemoryHandle]`

This is the one substrate-side addition required by DECISION 5.
It lives in `memopt.substrate.manager.AllocationManager` and is
re-exported as `memopt.peek_handle`.

Pre-conditions: `handle_id` is an integer. `tenant`, if provided,
must match the current tenant context (`tenant.assert_tenant_match`,
`tenant.py:72-82`); if not provided, the current context is used.

Post-conditions: returns a reference to the existing
`MemoryHandle` if it is live and the caller is permitted by G1
(§1.5.1); returns `None` if the handle does not exist or is not
visible to the caller. No state mutation.

Thread safety: dict lookup under
`AllocationManager._handles_lock` (`manager.py:75`).

CUDA callback safety: safe to call from a CUDA callback (no
allocation, no driver call).

#### 2.2.6 Event-emission contract (Layer 2 as producer)

Per DECISION 4 and gap O1, Layer 2 emits `evict`, `promote`,
`migrate` via the substrate's existing `Dispatcher`. The
substrate gains one new internal API (not public):

  `AllocationManager._emit_orchestrator_event(event: Event) -> None`

It validates `event.kind in ("evict", "promote", "migrate")`,
asserts `from_placement` and `to_placement` per §2.1 DECISION 4,
and forwards to `self._dispatcher.emit(event)`. Layer 2 is the
sole caller. This keeps emission centralised on the manager (so
the existing ring overflow accounting at
`memopt/substrate/events.py:82-91` covers Layer 2 events
identically) without exposing `_dispatcher.emit` publicly.

#### 2.2.7 Backwards compatibility

The substrate API is unchanged (C1). The substrate gains exactly
two new methods (`peek_handle`, `_emit_orchestrator_event`) and
zero modified ones. All existing pre/post-conditions on
`alloc`/`free`/`context`/`stats`/`observe` hold. No existing
optional kwarg is repurposed.

### 2.3 Internal architecture

```
                    ┌─────────────────────────────┐
                    │   memopt.orchestrator.start │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼─────────────────────┐
                    │      OrchestratorCoordinator        │
                    │   (1 thread per AllocationManager)  │
                    └──┬──────────┬──────────┬────────────┘
                       │          │          │
              ┌────────▼─┐  ┌─────▼────┐  ┌──▼─────────┐
              │ Access   │  │Predictor │  │Policy      │
              │ Tracker  │  │          │  │Engine      │
              └────────┬─┘  └─────┬────┘  └──┬─────────┘
                       │          │          │
                       └──────────┼──────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │   Telemetry Collector      │
                    └────────────┬───────────────┘
                                 │
                                 ▼
                       memopt.orchestrator.stats()


  ▲ events going up                      ▼ allocations / hints going down
  │                                      │
  │ Dispatcher.subscribe (S1)            │ memopt.alloc / memopt.free (S2)
  │ events.py:150-163                    │ alloc(..., hint=) (S3)
  │                                      │ AllocationManager.stats (S4)
  └──────────────────────────┬───────────┘
                             │
                             ▼
                ┌──────────────────────┐
                │   Layer 1 substrate  │
                │  (memopt/substrate/) │
                └──────────────────────┘

      The four numbered surfaces S1–S4 are the §1.5.3 boundary.
      Layer 2 talks to Layer 1 only through them.
```

#### 2.3.1 Access tracker (gap O2)

  Class: `memopt.orchestrator.access.AccessTracker`

  Public surface:
    - `record(event: Event) -> None`
    - `last_seen(tenant: str, tag: str, handle_id: int) -> Optional[float]`
    - `lru_candidates(tenant: str, placement: str, count: int)
                                            -> List[int]`
    - `forget_handle(handle_id: int) -> None`
    - `forget_tenant(tenant: str) -> None`
    - `snapshot() -> dict`

  Internal state: a per-tenant dict
  `{tenant: {handle_id: AccessRecord}}` where `AccessRecord` is a
  frozen dataclass `(tag, placement, last_seen_ns, hit_count,
  size_bytes)`. Plus a per-(tenant, placement) intrusive doubly
  linked LRU list, mirroring the structure at
  `csrc/core/page_table.h:108-138` so the candidate-walk path
  remains O(count) not O(total).

  Threading model: single writer (the coordinator thread, which
  drains the subscriber queue); reads from any thread under a
  per-tenant `threading.Lock`. The LRU list updates and the dict
  updates happen under the same lock to keep them consistent.

  Concurrency primitives: one `threading.Lock` per tenant. No
  global lock (avoids cross-tenant contention).

  Consumes substrate events: `alloc` (insert), `free` (remove),
  `evict`/`promote`/`migrate` from Layer 2 itself (update
  `placement`). Does NOT subscribe to handle reads — the substrate
  does not emit a "read" event (verified §1.1.4) and instrumenting
  reads is out of scope (would require a substrate change).

  Emits orchestrator events: none. Pure consumer + state.

  Failure modes: an event for an unknown `handle_id` is
  silently dropped (best-effort delivery, C4). A `forget_tenant`
  during an in-flight `record` is safe under the per-tenant lock.

#### 2.3.2 Predictor (gap O3, DECISION 6)

  Class: `memopt.orchestrator.predict.Predictor`

  Public surface:
    - `observe(tenant: str, tag: str) -> None`
    - `predict(tenant: str, tag: str, top_k: int = 10) -> List[Prediction]`
    - `forget_tenant(tenant: str) -> None`
    - `stats() -> dict`

  `Prediction` is `(tenant, tag, confidence, source)` frozen,
  where `source ∈ {"transition", "sequential", "recency",
  "fallback"}` mirroring `_oracle_py.py:120-163`.

  Internal state: `_transitions: Dict[(tenant, tag), Counter[(tenant, tag)]]`,
  `_recency: OrderedDict[(tenant, tag), int]`, per-tenant step
  counter `_steps: Dict[tenant, int]`. Bounded: max 100 000
  transitions globally; LRU eviction when full (same default as
  `_oracle_py.py:46`).

  Threading model: a single `threading.RLock` protects
  `_transitions` and `_recency`. Read-heavy workloads can later
  shard the lock per tenant; v1.0 keeps it simple.

  Consumes substrate events: indirectly. The coordinator calls
  `observe(tenant, tag)` on every `alloc`/`promote` event after
  `AccessTracker.record`. The predictor never subscribes itself.

  Emits orchestrator events: none.

  Failure modes: an `observe` on a never-seen tenant initialises
  fresh state. `predict` on a cold tenant returns the
  `"sequential"` and `"fallback"` sources only.

#### 2.3.3 Policy engine (gap O4)

  Class: `memopt.orchestrator.policy.PolicyEngine`

  `Policy` protocol:
    `def evaluate(snapshot: PolicySnapshot) -> List[Decision]`

  `PolicySnapshot` is a frozen dataclass:
    `(per_tenant_pressure: Dict[str, float],
      per_placement_used_bytes: Dict[str, int],
      lru_candidates: Dict[(str, str), List[int]],
      predictor: PredictorSnapshot,
      ts_ns: int)`

  `Decision` is a frozen dataclass:
    `(kind: Literal["evict","promote","migrate"],
      handle_id: int, target_placement: str, reason: str,
      priority: int)`

  Built-in v1.0 policy:
    `LRUWatermarkPolicy(per_tenant_high=0.90, per_tenant_low=0.75)`
    — mirrors the existing Python defaults
    (`memopt/vmm/tier_manager.py:50-51`) and the env vars
    `MEMOPT_EVICT_HIGH`/`MEMOPT_EVICT_LOW` are honoured for parity
    with §1.6.2 (`test_evict_thresholds_from_env`).

  Conflict resolution: when two policies emit conflicting decisions
  for the same `handle_id`, the higher `priority` wins; ties
  break on registration order. Decisions are evaluated, not
  applied — the coordinator applies them in priority order until
  the snapshot's pressure signal is below `low`.

  Threading model: `evaluate()` is called only from the
  coordinator thread; policies need not be reentrant. Adding a
  policy via `register_policy` (2.2.4) takes a brief lock.

  Consumes substrate events: none directly. The snapshot is
  built by the coordinator from `AccessTracker` and `Predictor`.

  Emits orchestrator events: indirectly — its `Decision` outputs
  are converted to events by the coordinator via
  `_emit_orchestrator_event` (§2.2.6).

  Failure modes: a buggy user policy raising in `evaluate` is
  caught, logged at WARNING, and the policy is skipped for that
  cycle. After three consecutive raises, the policy is auto-
  unregistered. Mirrors `events.py:222-227`.

#### 2.3.4 Coordinator / decision pump (DECISION 3)

  Class: `memopt.orchestrator.coordinator.OrchestratorCoordinator`

  Public surface:
    - `start() -> None`
    - `stop(timeout: float = 2.0) -> None`
    - `is_alive() -> bool`
    - `tick() -> None`  (test-visible single-cycle drive)

  Internal state:
    - `_event_queue: queue.Queue[Event]` (bounded, default
      capacity 16384, configurable; oldest-drop policy under
      pressure with a counter)
    - `_subs: List[SubscriptionHandle]`
    - `_thread: threading.Thread`
    - `_stopping: threading.Event`
    - `_cycle_period_ms: int` (default 50; calibrated against
      Linux kswapd's typical 100ms wake interval — half that
      because we have a smaller working set per tenant)

  Threading model: one daemon thread named
  `memopt-orchestrator-coordinator`. The five subscriber threads
  (one per `Event.kind`, supplied by the substrate dispatcher per
  §1.1.3) push events into `_event_queue`. The coordinator drains
  the queue, updates `AccessTracker` and `Predictor`, then on
  every `_cycle_period_ms` boundary builds a `PolicySnapshot`,
  calls `PolicyEngine.evaluate`, and applies decisions via
  `_emit_orchestrator_event` + (for `evict`/`promote`/`migrate`
  with backend support) the corresponding substrate operation.

  Consumes substrate events: all five kinds, via five
  `Dispatcher.subscribe` calls. The subscriber callbacks push
  into `_event_queue` and return immediately — bounded work,
  satisfying C4.

  Emits orchestrator events: yes, all three Layer-2 kinds, via
  `_emit_orchestrator_event`.

  Failure modes:
    - Queue full: oldest event is dropped; `events_dropped`
      counter increments (mirrors substrate ring overflow,
      `events.py:88-90`).
    - Coordinator thread crashes: `is_alive()` returns False;
      `stop()` is a no-op; users can call `start()` again.
    - `_emit_orchestrator_event` raises: caught and logged; the
      coordinator continues. A backend that refuses migrate
      (per DECISION 4 trade-off) returns a non-fatal error, the
      decision is downgraded to free+alloc by Layer 2, and a new
      `migrate` event is NOT emitted for that handle.

#### 2.3.5 Telemetry collector (DECISION 8)

  Class: `memopt.orchestrator.telemetry.TelemetryCollector`

  Public surface:
    - `snapshot() -> dict`

  Internal state: counters incremented by the coordinator
  (`events_ingested`, `decisions_emitted`, `policy_raises`,
  `queue_drops`, `coordinator_cycles`). Read by
  `memopt.orchestrator.stats()` (2.2.3).

  Threading model: counters use `itertools.count`-backed atomic
  increments where possible; aggregate read takes a brief lock.

  Consumes substrate events: indirectly via the coordinator.

  Emits orchestrator events: none.

  Failure modes: a counter overflow at 2^63 is theoretically
  possible; not a practical concern in v1.0.

#### 2.3.6 Configuration

  Frozen dataclass `memopt.orchestrator.OrchestratorConfig`:
    - `cycle_period_ms: int = 50`
    - `event_queue_capacity: int = 16384`
    - `predictor_max_transitions: int = 100_000`
    - `predictor_min_confidence: float = 0.3`
    - `built_in_policy: bool = True`  (registers
      `LRUWatermarkPolicy` automatically)
    - `lru_high_watermark: float = 0.90`
    - `lru_low_watermark: float = 0.75`

  Env-var overrides (read at `start()` time, not per-cycle, to
  match substrate convention):
    `MEMOPT_ORCH_CYCLE_MS`, `MEMOPT_ORCH_QUEUE_CAP`,
    `MEMOPT_ORCH_MAX_TRANSITIONS`, `MEMOPT_EVICT_HIGH`,
    `MEMOPT_EVICT_LOW`. The last two are shared with
    `TierManager` (`tier_manager.py:50-51`) intentionally —
    Phase B parity.

### 2.4 Coexistence with legacy paths

§1.2 enumerates three legacy implementations. Each gets the same
treatment: frozen during Phase A; observed (not driven) by Layer 2
behind the opt-in flag in Phase B; replaced in Phase D.

#### 2.4.1 C++ allocator (`csrc/cuda_vmm/vmm_allocator.cpp`)

  - Phase A (v1.0): frozen. Layer 2 does NOT subscribe to its
    eviction events (the C++ allocator emits no substrate events,
    §1.2.1). The PyTorch pluggable-allocator install path is
    unchanged.
  - Phase B (v1.1): if `MEMOPT_USE_ORCHESTRATOR=1` AND the
    process registers the substrate as PyTorch's allocator, the
    C++ allocator's `memopt_torch_step_boundary`
    (`vmm_allocator.cpp:960-972`) defers eviction policy to a
    Python callback that asks Layer 2 instead of running
    `memopt_evict_to_target` directly. Behind a flag.
  - Phase D: the hardcoded LRU loop in `evict_page_locked`
    (`vmm_allocator.cpp:524-561`) is removed; the C ABI exposes a
    `memopt_set_evict_callback(fn)` hook that Layer 2 fills.

  Tests gated: `tests/test_async_nvme.py` (16 tests) is unaffected
  through Phase B. Phase D regresses `test_evict_thresholds_*`
  (`test_vmm_hardening.py`, §1.6.2); those tests will be
  re-pointed at the orchestrator's `LRUWatermarkPolicy` env vars.

#### 2.4.2 Python `TierManager` (`memopt/vmm/tier_manager.py`)

  - Phase A: frozen. Layer 2 does not call `TierManager.fetch`,
    `TierManager._evict_one`, or `TierManager._ensure_capacity`.
  - Phase B: behind `MEMOPT_USE_ORCHESTRATOR=1`, `VMM.allocate`
    consults Layer 2 for placement (the substrate's
    `placement="auto"` resolution per §1.5.4 and gap O5). The
    eviction loop in `_ensure_capacity` is unchanged.
  - Phase D: `_ensure_capacity` is replaced by a thin shim that
    enqueues a Layer-2 decision; the watermark loop is deleted.

  Tests gated: `test_vmm_smoke.py` (13 tests) and
  `test_vmm_hardening.py` (8 tests) must remain green through
  Phase B (per §1.6.2 they are the regression net for the legacy
  path). Phase D pivots them to assert the equivalent
  orchestrator-side behaviour.

#### 2.4.3 Backend `allocate`/`free` (`memopt/vmm/backends/`)

  - Phase A: frozen. The CUDABackend's `allocate(size, tier, ...)`
    contract (`memopt/vmm/backends/_cuda_backend_py.py:81-119`) is
    unchanged.
  - Phase B: backends gain an optional `migrate(handle, src_tier,
    dst_tier)` method behind a feature-detection probe. Layer 2
    calls it only when the backend reports support.
  - Phase D: backends become substrate-only consumers; the VMM
    backend layer collapses into the substrate backend layer.

  Tests gated: backend tests in
  `memopt/substrate/tests/test_backend_cpu.py` and
  `test_backend_cuda.py` (§1.6.2) — only the CUDA round-trip
  test (`test_cuda_backend_evict_promote_round_trip`) is affected;
  the new `migrate` method is purely additive.

#### 2.4.4 Page-table LRU helper (`csrc/core/page_table.{h,cpp}`)

  - All phases: Layer 2 does not call `lru_candidates`,
    `pin`, or `unpin` on the C++ page table. Those are the VMM's
    private helpers. Layer 2 maintains its own LRU per §2.3.1.

#### 2.4.5 Cluster-wide oracle infrastructure

  - All phases: untouched.
  - `memopt/vmm/global_oracle.py`, `memopt/vmm/pod_controller.py`
    continue to operate on the VMM's `MemoryOracle` only. Layer 2
    has its own predictor (DECISION 6); cross-pollination is a
    Layer 5 concern.

  Tests gated: `test_global_oracle.py` (12 tests) and
  `test_pod_controller.py` (18 tests) must remain green forever.

### 2.5 Tenant isolation (G1–G4, N1–N4)

Following substrate v1 §2.6 shape.

#### Guarantees (provided by the orchestrator)

  G1  Two tenants cannot read each other's access histories via
      the orchestrator API. `AccessTracker.last_seen`,
      `lru_candidates`, and `snapshot()` are scoped by `tenant`
      and reject cross-tenant reads with `PermissionError` when
      a tenant context is set.

  G2  Per-tenant policies are enforced. A `LRUWatermarkPolicy`
      computes per-tenant pressure independently
      (per-tenant arena byte counts from
      `AllocationManager.stats(tenant=X)`, `manager.py:303-316`).
      Tenant A's pressure does not trigger eviction of tenant B's
      handles.

  G3  Predictor state is partitioned by tenant in the key tuple
      (DECISION 6). `Predictor.forget_tenant(tenant)` removes all
      transitions whose source key has that tenant. Used on
      tenant teardown.

  G4  Telemetry counters aggregate across tenants only; per-tenant
      decision counts are visible via
      `memopt.orchestrator.stats()["decisions_per_tenant"]` ONLY
      when the caller supplies the substrate's
      `MEMOPT_ADMIN_TOKEN` (mirroring the substrate's G2 at
      `manager.py:337-348`). Without the token, only aggregate
      counts are returned.

#### Non-guarantees (out of scope, made explicit)

  N1  Cross-tenant pattern leakage via shared storage. If a
      future Layer 5 oracle aggregates transitions across tenants
      (gossip across pods, like `pod_controller.py`), tenant A's
      access pattern statistics may inform tenant B's predictions.
      v1.0 does NOT do this; Layer 5 will require an opt-in
      consent gate per tenant.

  N2  Side channels via predictor-induced eviction timing. A
      tenant who can observe its own access latencies could in
      principle infer the orchestrator's decision cadence from
      another tenant's pressure. Mitigation: per-tenant pressure
      is computed in isolation (G2). The orchestrator does NOT
      promise constant-time decisions; for adversarial multi-
      tenant deployments use MIG (cf. substrate N1).

  N3  Orchestrator policy injection. A tenant who can call
      `memopt.orchestrator.register_policy` can register a policy
      that observes other tenants' snapshots. v1.0 does NOT
      restrict who can register policies; this matches the
      substrate's "opt-in installation" model (substrate N2).
      Production deployments must restrict process-level access
      to `register_policy` via the same mechanism that controls
      `MEMOPT_ADMIN_TOKEN`.

  N4  Coordinator-thread starvation. A tenant with very high
      allocation rate could fill `_event_queue` and starve other
      tenants' events. The orchestrator's queue is FIFO with
      oldest-drop, not per-tenant fair. Per-tenant fair queueing
      is a Phase C / Layer 4 concern.

### 2.6 Performance targets

Targets (TARGETS, not measurements):

```
Operation                          Target    Reference system
────────────────────────────────── ────────  ───────────────────────────
Event ingest                       <  5 µs   per-event path: queue.put on
(subscriber callback to            (P50)     a bounded Queue. Python's
AccessTracker.record updated)      < 50 µs   queue.Queue is ~2 µs per put
                                   (P99)     under no contention; we allow
                                             headroom for the AccessTracker
                                             dict + LRU update.

Predictor query                    < 20 µs   _oracle_py.py:115-180 is pure
(Predictor.predict, top_k=10,      (P50)     Python with one Counter scan
warm cache, single tenant)         < 200 µs  per source. Sub-100 µs measured
                                   (P99)     in test_oracle.py implicitly
                                             (tests assert ordering, not
                                             absolute time).

Policy evaluation                  < 1 ms    cgroup memory pressure stall
(LRUWatermarkPolicy.evaluate,      (P50)     info (PSI) is computed every
1000 handles, 16 tenants)          < 10 ms   2 s; we run 50 ms cycles, so
                                   (P99)     a 1 ms policy is comfortable.

Decision pump cycle time           50 ms     Linux kswapd: ~100 ms typical.
(coordinator wake-to-wake)         (default) Half because per-tenant arenas
                                             are smaller working sets than
                                             a Linux NUMA node.

Worst-case advisory turnaround     < 100 µs  jemalloc background_thread
for alloc("auto")                  (P99)     callbacks run in <50 µs;
                                             we double the budget for the
                                             optional pre-alloc callback
                                             path (DECISION 2).

Decision-to-emit                   < 200 µs  substrate event emit goal is
(Decision -> _emit_orchestrator_   (P50)     <200 ns; the 200 µs budget
event -> ring update)              < 1 ms    covers Python overhead +
                                   (P99)     ring write under contention.
```

Each target has a benchmark in
`memopt/orchestrator/tests/test_perf_microbench.py` that prints
measured times and asserts ordering only (substrate §2.10 model):
`event_ingest < predictor_query < policy_eval < cycle_time`.
Absolute thresholds are noise across hardware; ordering catches
regressions.

Hardware baseline for these targets: development machine is the
Mac (Darwin 25.4.0, Python 3.12, no CUDA). Production targets
should be re-measured on H100 SXM5 80 GB / A100 / MI300X under
realistic workload. TODO-VERIFY (Phase 3 Step Zero): the Python
`queue.Queue` overhead under contention with five producer
threads (the substrate's per-kind dispatchers) — measured number
on the Mac baseline before Phase A starts.

### 2.7 Migration plan

Mirrors substrate v1's Phase A/B/C/D model exactly.

#### Phase A — parallel addition (v1.0)

  Scope:
    - Ship `memopt/orchestrator/` as a new package.
    - Ship `memopt.peek_handle` and the internal
      `_emit_orchestrator_event` on `AllocationManager`.
    - Ship `memopt.orchestrator.start/stop/stats/register_policy`.
    - The orchestrator does NOT auto-start. Users opt in by
      calling `memopt.orchestrator.start()`.
    - All legacy paths (§1.2) remain in their current state.

  Gating criteria for releasing v1.0:
    - 752-pass Mac baseline (§1.6.1) preserved unchanged
      (additive only; new tests under
      `memopt/orchestrator/tests/` may add to the count).
    - Every TODO-VERIFY in §2 is resolved (the Step Zero
      checklist of Phase 3 covers this).
    - At least one end-to-end test demonstrates a substrate
      `alloc` event reaching the orchestrator and producing a
      `evict` event observed by a second subscriber.
    - `memopt.orchestrator.stop()` followed by 5 ms idle
      passes leak-detection (no zombie threads).

#### Phase B — opt-in flag (v1.1)

  Scope:
    - `MEMOPT_USE_ORCHESTRATOR=1` causes the orchestrator to
      auto-start at first `memopt.alloc()`.
    - With the flag set, `placement="auto"` consults the
      orchestrator (§1.5.4 / gap O5) before resolving to a
      `PhysLoc`.
    - `TierManager._ensure_capacity` (`tier_manager.py:322-339`)
      remains the eviction driver for VMM blocks; the
      orchestrator drives substrate-handle eviction only
      (DECISION 7).
    - C++ allocator's step-boundary callback (§2.4.1)
      conditionally defers to Layer 2 behind the same flag.

  Gating criteria for releasing v1.1:
    - All Phase A criteria still pass.
    - `MEMOPT_USE_ORCHESTRATOR=1` test suite (new, gated by
      env var) passes alongside the default (no-flag) suite.
    - `test_evict_thresholds_*` in `test_vmm_hardening.py` pass
      both with and without the flag.
    - Parity test: a workload that exhausts a tenant arena
      produces the same hit/miss ratio with the flag set as
      without it (within 5%).

#### Phase C — default ON (v2.0)

  Scope:
    - The flag flips: orchestrator is on by default.
      `MEMOPT_USE_ORCHESTRATOR=0` is the new opt-out.
    - All Phase B integrations become live.
    - Per-tenant fairness (N4) is implemented in the queue.

  Gating criteria for releasing v2.0:
    - Three calendar months running with the flag on at >= 5%
      of opt-in users (or, in the absence of opt-in telemetry,
      three months on the development team's own workloads).
    - Zero regressions reported against the legacy paths in
      that period.
    - Performance targets (§2.6) verified under at least one
      production-shape benchmark.

#### Phase D — remove legacy (v3.0)

  Scope:
    - Delete `TierManager._ensure_capacity` watermark loop;
      replace with orchestrator decision-pump call.
    - Delete `ElasticAllocator.decide` (`elastic_allocator.py`);
      its logic is absorbed into `LRUWatermarkPolicy`.
    - Delete `MemoryGovernor` pressure adjustments
      (`memory_governor.py`); orchestrator carries pressure
      signal.
    - Delete `memopt_torch_step_boundary` direct
      `memopt_evict_to_target` call; replace with
      `memopt_set_evict_callback` (§2.4.1).
    - Delete `PrefetchEngine`'s built-in Markov chain
      (`prefetch_engine.py:184-185`); engine consults
      orchestrator predictor instead.

  Gating criteria for releasing v3.0:
    - Two calendar quarters with no v2.0 user reporting a
      missing legacy-path feature.
    - Documentation migration: every reference to
      `MEMOPT_EVICT_HIGH`/`MEMOPT_EVICT_LOW` in
      `tier_manager.py:72-86` repointed to the orchestrator
      config (§2.3.6).
    - Tests in `test_vmm_smoke.py`, `test_vmm_hardening.py`,
      `test_layer3.py` pivoted to assert orchestrator-side
      behaviour and re-measured for parity.

### 2.8 Out of scope

The orchestrator is the second of a multi-layer stack. Each of
the following is a separate layer with its own design doc.

  - **Layer 3 — prefetch oracle (ML-driven prediction).**
    The Markov-chain predictor in §2.3.2 is intentionally
    classical. ML-based prediction (RNN / transformer over
    access traces) lives in a separate package
    `memopt-prefetch-oracle`.

  - **Layer 4 — policy DSL.** v1.0 ships one built-in
    `LRUWatermarkPolicy` plus the `Policy` protocol for users
    who write their own in Python. A declarative DSL ("if
    tag matches X, never evict below Y bytes") is Layer 4.
    Lives in `memopt-policy-dsl`.

  - **Layer 5 — federated orchestration across nodes.**
    Multi-node coordination — the analogue of
    `memopt/vmm/global_oracle.py` for substrate handles —
    is Layer 5. The orchestrator's predictor explicitly does
    not gossip across processes (DECISION 7 trade-off).

  - **vLLM integration.** The orchestrator does not ship vLLM-
    specific policies. `memopt-vllm` (separate package) wires
    the two.

  - **Custom CUDA kernels for predictor.** `_oracle_py.py`
    has a C++ acceleration path (`memopt._memopt_core`,
    `oracle.py:14-26`); Layer 2's predictor does not consume
    it in v1.0 (DECISION 6 trade-off). Phase D may revisit.

  - **Persistence of learned patterns across restarts.** The
    predictor starts cold every process. Persisting transitions
    to disk is `MemoryOracle.warm_from_log`'s job
    (`_oracle_py.py:229-251`); the orchestrator's predictor
    can reuse that file format in a future revision but does
    not ship persistence in v1.0.

  - **Cost / FinOps integration.** Lives in `memopt-trust`.
    Substrate emits events; orchestrator emits events;
    finops consumes both out-of-process. Same model as
    substrate §2.11.

  - **Confidential computing / attestation.** Out of scope.
    The orchestrator cooperates with NVIDIA CC mode (does not
    break it) but does not provide attestation.

### 2.9 Traceability — gaps to design sections

```
Gap                                              Addressed in
───────────────────────────────────────────────  ─────────────────────────
O1.  No subscription to substrate events         §2.3.4 (coordinator
                                                  subscribes to all 5
                                                  Event.kinds), §2.2.6
                                                  (Layer 2 emits 3 of 5)

O2.  No tenant-aware access tracker              §2.3.1 (AccessTracker
                                                  per-tenant LRU)

O3.  No prefetch-ready data structure for        §2.3.2 (Predictor with
     substrate handles                            (tenant, tag) keys),
                                                  DECISION 6

O4.  No policy DSL                               §2.3.3 (PolicyEngine +
                                                  Policy protocol;
                                                  declarative DSL is
                                                  Layer 4 per §2.8)

O5.  No coordination between L1 placement="auto" §2.4.2 (Phase B),
     and a smarter "auto"                         §1.5.4 (boundary spec),
                                                  DECISION 2

O6.  No backpressure mechanism                   DECISION 2 (advisory
                                                  pre-alloc callback;
                                                  no veto in v1.0)

O7.  No cross-tenant fairness                    §2.5 G2 (per-tenant
                                                  pressure isolation),
                                                  §2.5 N4 (fair queue
                                                  is Phase C)

O8.  No telemetry / observability for            §2.3.5 (TelemetryCollector),
     orchestrator decisions                       §2.2.3 (stats() shape),
                                                  DECISION 8

O9.  No bridge between substrate events and      §2.3.4 (coordinator),
     existing oracle / access-log infrastructure  §2.4.5 (legacy oracle
                                                  untouched in all phases),
                                                  DECISION 6

O10. No place where placement is decided as a    §2.3.3 (PolicyEngine),
     function of predicted future use             §2.3.2 (Predictor),
                                                  §2.4.2 Phase B
                                                  (placement="auto"
                                                  resolution)

O11. No async / background coordinator process   §2.3.4 (coordinator
                                                  thread, DECISION 3)
```

Every gap has at least one numbered design section. Phase 3 (test
plan) will pair each section with at least one new test.

#### TODO-VERIFY items (for Phase 3 Step Zero)

  TV1. Python `queue.Queue` enqueue/dequeue overhead under five
       concurrent producers (the substrate's per-kind dispatcher
       threads, §1.1.3) on the Mac baseline. The §2.6 ingest
       target of <5 µs P50 assumes ~2 µs put cost; verify
       empirically before Phase A.

  TV2. The substrate's `Dispatcher.subscribe` performance with
       five concurrent subscriptions (one per `Event.kind`).
       `events.py:150-163` documents one subscriber thread per
       call; verify that five threads do not cause head-of-line
       blocking on the dispatcher's `_lock` (`events.py:138`).

  TV3. `AllocationManager._handles` lookup cost
       (`manager.py:74`). The `peek_handle` API (DECISION 5)
       calls into this dict on every interesting event. Verify
       the lock contention with the allocator hot path is
       acceptable on the substrate's measured `alloc(2 MiB) warm`
       path (substrate §2.10 target: <1 µs).

  TV4. CUDA backend `migrate` feasibility. DECISION 4 defines
       `migrate` as a tier change preserving `va`. Verify on
       H100 / MI300X that `cuMemUnmap` + `cuMemRelease` +
       `cuMemCreate(new_loc)` + `cuMemMap` to the same VA is a
       supported sequence (the C++ allocator's promote path
       `vmm_allocator.cpp:651-708` uses an analogous sequence
       for DRAM → HBM, but inter-NUMA migrate is not exercised
       today).

  TV5. `MEMOPT_EVICT_HIGH`/`MEMOPT_EVICT_LOW` semantics overlap.
       Sharing env vars between `TierManager` and
       `LRUWatermarkPolicy` (§2.3.6) is intentional for parity;
       verify in a test that setting them both ways does not
       cause unexpected double-eviction during Phase B (when
       both run simultaneously).

  TV6. `OrchestratorCoordinator.stop` interaction with the
       substrate's `Dispatcher.shutdown`
       (`events.py:171-176`). The substrate joins subscriber
       threads with a 2.0 s timeout; the orchestrator's
       coordinator thread is independent. Verify that
       `AllocationManager.reset()` followed by
       `orchestrator.stop()` does not deadlock or leak threads.

  TV7. C++ allocator's `memopt_torch_step_boundary` callback
       hook (Phase B, §2.4.1). The function's current signature
       is `void memopt_torch_step_boundary(float threshold)`
       (`vmm_allocator.cpp:960-972`). Adding a Python callback
       requires either a global function pointer or a
       per-allocator field; verify which is feasible without
       changing the C ABI for existing consumers.

  Each TODO-VERIFY produces a Step Zero check in Phase 3 with a
  pass/fail criterion. Phase A does not begin until all TV1–TV7
  are resolved.

---

*End of Phase 2.*

---

## Phase 3 — Test Plan

The test plan covers (a) a Step Zero verification gate for every
TODO-VERIFY in Phase 2, before any implementation code is written;
(b) per-component unit tests; (c) a bridge test for legacy parity;
(d) coexistence regression strategy; (e) performance regression
strategy; (f) a commit-by-commit implementation plan; (g) a
Phase A acceptance gate.

Long-term frame: every test below protects against a future bug
that would have to be debugged without the design context.
Prefer one extra test today over a 3 am page next year.

Test buckets follow the substrate convention:
  - **REQUIRED-CI** — runs on Mac/Linux without CUDA; no marker.
  - **REQUIRED-LOCAL** — needs a GPU rig; `@gpu`.
  - **OPTIONAL** — specialised hardware: `@gpu_amd`, `@cxl`,
    `@perf`, `@imex`.

### 3.0 Step Zero verification (TV1–TV7)

Phase A (implementation) does not begin until every line below
is checked off. Step Zero produces a single artefact at
`docs/orchestrator_v1_step_zero_report.md` that records the
result of each item. Step Zero failures may force Phase 2 doc
updates; those updates re-trigger Phase 3 review.

Hard-block classification: TV3, TV4, and TV7 hard-block Phase A
on failure (each invalidates a §2.1 decision). TV1, TV2, TV5,
TV6 have documented graceful fallbacks (mark §2 as DEGRADED and
proceed).

#### S0.1  Python `queue.Queue` overhead under multi-producer load (TV1)

  Check against     Python 3.12 stdlib on the Mac baseline rig.
  Sources           - `Lib/queue.py` in CPython 3.12 source
                      tree
                    - `python -c "import queue,
                       inspect; print(inspect.getsourcefile(
                       queue))"`
                    - The Mac baseline (Darwin 25.4.0,
                      Python 3.12).
  Method            1. Write a 60-line micro-benchmark:
                       five producer threads, one consumer
                       thread, 100 000 events each, bounded
                       Queue(maxsize=16384). Measure P50/P99
                       per-event put time on the producer
                       side.
                    2. Run the benchmark on the Mac baseline
                       three times; report median of P50 and
                       P99.
                    3. Compare against the §2.6 ingest
                       budget (P50 < 5 µs, P99 < 50 µs).
  Verified when     a) P50 measured ≤ 5 µs on the Mac
                       baseline; AND
                    b) P99 measured ≤ 50 µs on the Mac
                       baseline; AND
                    c) Zero exceptions, zero queue.Full.
  Fallback          - P50 in [5, 25] µs: degraded — proceed
                      with §2.6 target relaxed to "P50 ≤ 25 µs"
                      and a note in the user guide.
                    - P50 > 25 µs: re-evaluate the data
                      structure. Either replace `queue.Queue`
                      with a `collections.deque` + a `Condition`
                      (skip the put-side validation cost) or a
                      lock-free `multiprocessing.Queue` shim.
                      §2.3.4 changes; rerun Phase 2 review.
                    - P99 > 50 µs and P50 ≤ 5 µs: tail-latency
                      issue. Add per-producer back-off; document
                      in §2.6 as "P99 worst-case under five
                      concurrent producers."

#### S0.2  Substrate `Dispatcher.subscribe` head-of-line (TV2)

  Check against     `memopt/substrate/events.py:128-176`
                    (Dispatcher implementation, §1.1.3).
  Sources           - `memopt/substrate/events.py:144-148`
                      (`emit` holds `self._lock` while iterating
                      subscribers).
                    - `memopt/substrate/tests/test_events.py`
                      (existing test
                      `test_subscriber_runs_on_dispatcher_thread_not_caller`
                      pins the threading model).
  Method            1. Write a probe: subscribe five callbacks
                       (one per `Event.kind`); each callback
                       sleeps 1 ms. Emit 1000 events of mixed
                       kinds. Measure the producer-side `emit`
                       time.
                    2. Compare to a control run with one
                       subscriber.
                    3. The five-subscriber emit time must
                       differ from one-subscriber by less than
                       50 µs in P50 (i.e., the hold-the-lock
                       cost of iterating the subscriber dict
                       must dominate, not subscriber latency).
  Verified when     a) Five-subscriber `emit` P50 ≤ 50 µs.
                    b) No callback raised; `events_dropped`
                       == 0.
                    c) Per-kind subscriber threads have
                       independent queues (verified by tail
                       latency on one kind not blocking
                       another).
  Fallback          - If `emit` blocks > 50 µs P50 with five
                      subscribers: §2.3.4 must use ONE
                      subscriber (kind="*"-equivalent via
                      five separate calls each pushing to the
                      same queue) instead of five. Rewrite
                      §2.3.4 coordinator subscribe loop.
                    - If subscriber threads share state
                      (head-of-line within one kind blocks
                      another): file substrate bug; degrade by
                      bounding callback work to < 100 ns of
                      pure enqueue.

#### S0.3  `AllocationManager._handles` lookup cost (TV3) — HARD BLOCK

  Check against     `memopt/substrate/manager.py:74,229,247,291`
                    (the `_handles` dict + its lock).
  Sources           - `manager.py:74` (dict declaration).
                    - `manager.py:75` (lock declaration).
                    - `manager.py:229-231` (insert under lock).
                    - `manager.py:246-247` (lookup under lock).
  Method            1. Write a probe: spawn 1 thread doing 10 000
                       `memopt.alloc(2 MiB)` followed by `free`,
                       and a second thread doing 10 000
                       `memopt.peek_handle(<id>)` calls (after
                       the new method ships in commit 1).
                    2. Measure: (a) the producer-thread
                       `alloc` warm path P50 — must remain
                       below the substrate's stated <1 µs
                       (substrate §2.10); (b) the consumer-
                       thread `peek_handle` P50 — must be
                       below 5 µs.
  Verified when     a) `alloc` warm-path P50 unchanged
                       (within 5 % of the pre-`peek_handle`
                       baseline measured in commit 0).
                    b) `peek_handle` P50 ≤ 5 µs.
                    c) No PermissionError on tenant-context
                       reads of in-tenant handles.
  Fallback          HARD BLOCK if (a) regresses by more than
                    20 %. The substrate's <1 µs warm-path
                    target is non-negotiable. Either:
                    - Switch `peek_handle` to a separate
                      lock-free read-only side-table (more
                      substrate change; redo Phase 2 §2.2.5);
                      OR
                    - Drop DECISION 5 in favour of "Layer 2
                      stores its own handle metadata snapshot
                      from the alloc event" (skip the dict
                      lookup; lose the live `MemoryHandle`
                      reference).

#### S0.4  CUDA `migrate` semantics (TV4) — HARD BLOCK on H100

  Check against     CUDA Driver API on H100 SXM5 80 GB.
  Sources           - `cuda.h` on the GPU rig.
                    - https://docs.nvidia.com/cuda/cuda-driver-
                      api/group__CUDA__VA.html
                    - The C++ allocator's promote sequence
                      (`csrc/cuda_vmm/vmm_allocator.cpp:651-708`)
                      as a known-good reference.
  Method            1. Build a probe (~50 lines): reserve VA
                       with `cuMemAddressReserve`; create
                       physical handle on device 0
                       (HBM); map; write a sentinel; unmap;
                       release; create a NEW physical handle
                       (still HBM) at the same VA via
                       `cuMemMap`; read sentinel back.
                    2. Repeat with a cross-tier sequence:
                       HBM → DRAM (via host pinned memory) →
                       HBM at the same VA. Verify content
                       round-trips.
                    3. Repeat the cross-tier round trip 1000
                       times to detect any leak in the VA
                       reservation pool.
  Verified when     a) Same-tier remap returns the sentinel.
                    b) Cross-tier round trip preserves bytes.
                    c) `cuMemAddressReserve` does not leak
                       1000 round trips in.
                    d) The probe runs in < 30 s on H100.
  Fallback          HARD BLOCK if (a) fails. DECISION 4 is
                    invalidated — `migrate` cannot preserve
                    `va`. Re-open Q7; choose either (1) drop
                    `migrate` from the v1.0 emission contract
                    (re-run §2.2.6) or (2) redefine `migrate`
                    as a free+alloc with a `from_handle_id`
                    cross-reference field (re-run §2.1
                    DECISION 4 and §2.2.6).

#### S0.5  Watermark env-var collision (TV5)

  Check against     `memopt/vmm/tier_manager.py:50-86`
                    (env-var read + validation) and the new
                    `LRUWatermarkPolicy` per §2.3.3 / §2.3.6.
  Sources           - `tier_manager.py:50-51` (defaults).
                    - `tier_manager.py:72-86` (env read +
                      validation).
                    - `test_evict_thresholds_from_env`,
                      `test_evict_thresholds_invalid_uses_defaults`,
                      `test_evict_thresholds_defaults_without_env`
                      in `test_vmm_hardening.py` (§1.6.2).
  Method            1. Construct a `VMM` and a Phase B-style
                       orchestrator with both reading the same
                       env vars (`MEMOPT_EVICT_HIGH=0.85`,
                       `MEMOPT_EVICT_LOW=0.65`).
                    2. Verify both report the same configured
                       values.
                    3. Drive a synthetic workload that
                       exhausts the VMM's hot tier; verify only
                       ONE eviction loop fires per pressure
                       event (the orchestrator observes; the
                       VMM acts in Phase A; per DECISION 7).
  Verified when     a) Both subsystems read the same value.
                    b) No double-eviction observed in Phase A
                       (orchestrator silent observer).
                    c) `test_evict_thresholds_*` still pass
                       unchanged.
  Fallback          - If double-eviction does fire: rename the
                      orchestrator's env vars to
                      `MEMOPT_ORCH_EVICT_HIGH`/`_LOW` (§2.3.6
                      change) and document Phase B alignment
                      separately.

#### S0.6  Coordinator/dispatcher shutdown interaction (TV6)

  Check against     The substrate's `Dispatcher.shutdown`
                    (`memopt/substrate/events.py:171-176`)
                    and `AllocationManager.reset`
                    (`memopt/substrate/manager.py:88-91`).
  Sources           - `events.py:171-176` (shutdown loop).
                    - `events.py:197-203` (subscriber stop;
                      2.0 s join timeout).
                    - `manager.py:88-91` (reset).
  Method            1. Start orchestrator, allocate 100
                       handles, free 50, call
                       `orchestrator.stop()`.
                    2. Repeat with a non-default
                       `OrchestratorConfig(cycle_period_ms=10)`.
                    3. Repeat with `AllocationManager.reset()`
                       called BEFORE `orchestrator.stop()`
                       (anti-pattern; must not deadlock).
                    4. Run under `pytest --threads-detect`
                       (or a manual `threading.enumerate()`
                       check) and assert no extra threads
                       remain after stop+reset.
  Verified when     a) Each scenario completes within 5 s.
                    b) `threading.enumerate()` shows no
                       `memopt-orchestrator-coordinator`
                       thread post-stop.
                    c) No deadlock under reset-before-stop.
  Fallback          - Deadlock under reset-before-stop:
                      orchestrator's `stop()` must drain its
                      queue without holding a lock that the
                      substrate's reset path also takes.
                      Document the ordering rule in §2.2.2
                      ("call orchestrator.stop() before
                      AllocationManager.reset()").
                    - Thread leak: increase the orchestrator
                      stop() join timeout from 2.0 s to 5.0 s
                      and log a warning if join fails.

#### S0.7  C++ `step_boundary` callback hook feasibility (TV7) — HARD BLOCK FOR PHASE B (not for Phase A)

  Check against     `csrc/cuda_vmm/vmm_allocator.cpp:960-972`
                    and the existing C ABI used by
                    `cuda_vmm.py` ctypes bindings.
  Sources           - `vmm_allocator.cpp:960-972` (current
                      function).
                    - `csrc/cuda_vmm/vmm_allocator.h` (ABI
                      header).
                    - `memopt/vmm/cuda_vmm.py` (current ctypes
                      consumer).
  Method            1. Read the existing call sites of
                       `memopt_torch_step_boundary` outside the
                       allocator; document them.
                    2. Sketch a new function pointer
                       `memopt_set_evict_callback(void
                       (*fn)(MemoptAllocator*, float))` with
                       a default of NULL ⇒ existing
                       behaviour. Verify it can be added
                       without changing the existing function's
                       signature.
                    3. Confirm the existing ctypes bindings
                       in `memopt/vmm/cuda_vmm.py` continue to
                       load `libmemopt_vmm.so` after the
                       header change.
  Verified when     a) The new function pointer is additive
                       to the C ABI.
                    b) `memopt/vmm/cuda_vmm.py` loads
                       unchanged.
                    c) The existing
                       `memopt_torch_step_boundary` semantic
                       is preserved when the callback is
                       NULL.
  Fallback          - HARD BLOCK FOR PHASE B (not Phase A).
                      If (a) fails — i.e., adding a callback
                      requires an ABI break — Phase B's
                      §2.4.1 plan must change. Either ship a
                      separate libmemopt_vmm v2 with the new
                      ABI, or have Layer 2 wrap the entire
                      `memopt_torch_step_boundary` rather
                      than hook into it.
                    - Phase A is not blocked because Phase A
                      does not change the C++ allocator
                      (§2.4.1 Phase A is "frozen").

#### Step Zero report

The verification checklist above produces
`docs/orchestrator_v1_step_zero_report.md`. Each S0.X entry is
recorded with one of: `PASS`, `DEGRADED-with-fallback-{N}`, or
`FAIL`. Phase A may proceed when:

  - All HARD-BLOCK items are `PASS`.
  - All other items are `PASS` or `DEGRADED-with-fallback`.
  - Each `DEGRADED` is documented with the §2 doc update it
    requires.

### 3.1 Unit test files per component

All new test files live under `memopt/orchestrator/tests/`
unless explicitly under `tests/`. Default bucket is REQUIRED-CI
(no marker). Approximate line counts let us estimate Phase A
scope; actual implementations may diverge by ±25 %.

#### 3.1.1 `test_access_tracker.py` — covers §2.3.1 (~250 lines)

  - `test_record_alloc_inserts_record` — alloc event
    creates an `AccessRecord` keyed by `(tenant, handle_id)`.
    [REQUIRED-CI]
  - `test_record_free_removes_record` — free event removes
    the record. [REQUIRED-CI]
  - `test_record_evict_updates_placement` — Layer-2 evict
    event updates `AccessRecord.placement`. [REQUIRED-CI]
  - `test_record_promote_updates_placement` — symmetric.
    [REQUIRED-CI]
  - `test_record_migrate_updates_placement` — Decision 4
    behaviour: `migrate` updates `placement` without
    invalidating the record. [REQUIRED-CI]
  - `test_lru_candidates_returns_oldest_first` — verifies
    the per-tenant LRU list ordering. [REQUIRED-CI]
  - `test_lru_candidates_respects_count` — bounded result
    set. [REQUIRED-CI]
  - `test_lru_candidates_isolated_per_tenant` — gap O7 / G1:
    tenant A's eviction candidates do not surface tenant B's
    handles. [REQUIRED-CI]
  - `test_forget_handle_removes_from_lru` — bookkeeping
    consistency. [REQUIRED-CI]
  - `test_forget_tenant_clears_all_state` — G3 cleanup.
    [REQUIRED-CI]
  - `test_concurrent_record_thread_safe` — two writer
    threads produce no torn updates under
    `threading.Lock`. [REQUIRED-CI]
  - `test_unknown_handle_id_silently_dropped` — best-effort
    delivery (C4). [REQUIRED-CI]
  - `test_snapshot_returns_consistent_view` — snapshot
    under lock. [REQUIRED-CI]

  Gaps covered: O2.

#### 3.1.2 `test_predictor.py` — covers §2.3.2 / DECISION 6 (~300 lines)

  - `test_observe_inserts_transition` — first-order Markov
    update. [REQUIRED-CI]
  - `test_observe_increments_existing_transition` — counter
    semantics. [REQUIRED-CI]
  - `test_predict_returns_top_k_sorted_by_confidence` —
    output ordering. [REQUIRED-CI]
  - `test_predict_includes_sequential_source` — `+1` / `+2`
    fallback (parity with `_oracle_py.py:134-141`).
    [REQUIRED-CI]
  - `test_predict_includes_recency_source` — recency
    fallback. [REQUIRED-CI]
  - `test_predict_fallback_when_cold_start` — cold-start
    behaviour. [REQUIRED-CI]
  - `test_min_confidence_filters_predictions` — the
    `predictor_min_confidence` config knob is honoured.
    [REQUIRED-CI]
  - `test_max_transitions_bounded` — LRU eviction at
    `predictor_max_transitions`. [REQUIRED-CI]
  - `test_predictor_keys_are_tenant_aware` — DECISION 6:
    keying on `(tenant, tag)` prevents cross-tenant
    transition pollution. Two tenants observing the same
    tag values produce disjoint transitions. [REQUIRED-CI]
  - `test_forget_tenant_removes_all_transitions` — G3
    cleanup; no orphaned counters. [REQUIRED-CI]
  - `test_predictor_thread_safe_under_rlock` —
    concurrent observe/predict produces consistent state.
    [REQUIRED-CI]
  - `test_predictor_stats_shape` — reportable counters.
    [REQUIRED-CI]
  - `test_predictor_does_not_share_state_with_memory_oracle`
    — verifies §2.4.5 (legacy oracle untouched).
    [REQUIRED-CI]

  Gaps covered: O3, O9. Decisions verified: D6.

#### 3.1.3 `test_policy_engine.py` — covers §2.3.3 (~250 lines)

  - `test_register_policy_appends_to_list` — basic
    registration. [REQUIRED-CI]
  - `test_evaluate_calls_each_policy` — fan-out semantics.
    [REQUIRED-CI]
  - `test_lru_watermark_policy_evicts_at_high_threshold` —
    when per-tenant `pressure >= high`, decisions are
    emitted. [REQUIRED-CI]
  - `test_lru_watermark_policy_stops_at_low_threshold` —
    decisions stop once below `low`. [REQUIRED-CI]
  - `test_lru_watermark_policy_uses_lru_candidates` —
    candidates come from `AccessTracker.lru_candidates`.
    [REQUIRED-CI]
  - `test_lru_watermark_policy_is_per_tenant` — tenant A's
    pressure does not produce decisions for tenant B
    (G2). [REQUIRED-CI]
  - `test_conflict_resolution_higher_priority_wins` — two
    policies emitting decisions for the same `handle_id`.
    [REQUIRED-CI]
  - `test_conflict_resolution_ties_break_on_registration_order`
    — deterministic. [REQUIRED-CI]
  - `test_buggy_policy_caught_and_logged` — failure mode:
    raise inside `evaluate` is caught. [REQUIRED-CI]
  - `test_buggy_policy_auto_unregistered_after_three_raises`
    — failure mode: persistent raises evict the policy.
    [REQUIRED-CI]
  - `test_user_policy_can_be_registered` — DECISION 4 / O4:
    user-supplied `Policy` works. [REQUIRED-CI]
  - `test_decision_dataclass_is_frozen` — invariant.
    [REQUIRED-CI]

  Gaps covered: O4, O7, O10.

#### 3.1.4 `test_coordinator.py` — covers §2.3.4 / DECISION 3 (~350 lines)

  - `test_start_creates_one_thread` — DECISION 3: single
    coordinator thread per `AllocationManager`.
    [REQUIRED-CI]
  - `test_start_subscribes_to_all_five_event_kinds` — gap
    O1: five `Dispatcher.subscribe` calls. [REQUIRED-CI]
  - `test_subscriber_callback_returns_immediately` — C4:
    callbacks do bounded work (enqueue), no policy work.
    [REQUIRED-CI]
  - `test_event_queue_drains_to_access_tracker` — wiring.
    [REQUIRED-CI]
  - `test_event_queue_full_drops_oldest` — failure mode;
    `events_dropped` increments. [REQUIRED-CI]
  - `test_cycle_period_observed` — coordinator wakes every
    `cycle_period_ms`. [REQUIRED-CI]
  - `test_tick_runs_one_full_cycle` — test-visible
    `tick()`. [REQUIRED-CI]
  - `test_decision_emits_orchestrator_event` — coordinator
    converts `Decision` to `Event` via
    `_emit_orchestrator_event`. [REQUIRED-CI]
  - `test_stop_joins_thread_in_2s` — clean shutdown.
    [REQUIRED-CI]
  - `test_stop_drops_subscriptions` — verified via the
    substrate's `events_dropped` not increasing post-stop.
    [REQUIRED-CI]
  - `test_idempotent_start_returns_same_handle` — §2.2.1
    idempotency. [REQUIRED-CI]
  - `test_coordinator_does_not_drive_eviction_without_flag`
    — DECISION 7: in v1.0, observation only — no
    `Decision` is applied unless explicitly requested via
    `tick()` in tests. [REQUIRED-CI]
  - `test_coordinator_emits_decisions_on_pressure` — under
    a synthetic high-pressure tick, decisions are
    emitted. [REQUIRED-CI]
  - `test_migrate_va_preserved` — DECISION 4: a `migrate`
    decision applied by the coordinator preserves the
    handle's `va`. [REQUIRED-LOCAL @gpu] (uses the CUDA
    backend; on Mac/CPU backend the assertion is "the
    handle's `_va` field equals the pre-migrate value" —
    can run REQUIRED-CI as well.)
  - `test_coordinator_thread_name` — `memopt-orchestrator-
    coordinator`. Eases debugging. [REQUIRED-CI]

  Gaps covered: O1, O11. Decisions verified: D3, D4, D7.

#### 3.1.5 `test_telemetry.py` — covers §2.3.5 / DECISION 8 (~150 lines)

  - `test_snapshot_empty_when_orchestrator_not_started` —
    `stats() == {}`. [REQUIRED-CI]
  - `test_snapshot_includes_running_flag` — basic shape.
    [REQUIRED-CI]
  - `test_snapshot_includes_events_ingested_per_kind` —
    counter shape. [REQUIRED-CI]
  - `test_snapshot_includes_decisions_per_kind` — counter
    shape. [REQUIRED-CI]
  - `test_snapshot_includes_predictor_block` — embedded
    predictor stats. [REQUIRED-CI]
  - `test_snapshot_includes_coordinator_block` — embedded
    coordinator stats. [REQUIRED-CI]
  - `test_snapshot_does_not_extend_substrate_stats` —
    DECISION 8: `memopt.stats()` does not grow.
    [REQUIRED-CI]
  - `test_snapshot_thread_safe` — concurrent reads.
    [REQUIRED-CI]
  - `test_per_tenant_decisions_require_admin_token` — G4:
    without `MEMOPT_ADMIN_TOKEN`, only aggregate counts
    are returned. [REQUIRED-CI]

  Gaps covered: O8. Decisions verified: D8.

#### 3.1.6 `test_config.py` — covers §2.3.6 (~120 lines)

  - `test_default_config_has_documented_values` — pins
    defaults to §2.3.6. [REQUIRED-CI]
  - `test_env_overrides_cycle_period` — `MEMOPT_ORCH_CYCLE_MS`.
    [REQUIRED-CI]
  - `test_env_overrides_queue_capacity` —
    `MEMOPT_ORCH_QUEUE_CAP`. [REQUIRED-CI]
  - `test_env_overrides_max_transitions` —
    `MEMOPT_ORCH_MAX_TRANSITIONS`. [REQUIRED-CI]
  - `test_env_overrides_evict_thresholds_shared_with_vmm`
    — `MEMOPT_EVICT_HIGH`/`_LOW` shared with `TierManager`
    (TV5). [REQUIRED-CI]
  - `test_env_invalid_falls_back_to_default` — defensive
    parsing. [REQUIRED-CI]
  - `test_config_is_frozen_dataclass` — invariant.
    [REQUIRED-CI]
  - `test_built_in_policy_can_be_disabled` — `built_in_policy=False`
    skips `LRUWatermarkPolicy` registration. [REQUIRED-CI]

#### 3.1.7 `test_public_api.py` — covers §2.2 (~250 lines)

  - `test_start_returns_orchestrator_handle` — §2.2.1.
    [REQUIRED-CI]
  - `test_start_is_idempotent` — §2.2.1. [REQUIRED-CI]
  - `test_start_lazy_subscribes_when_manager_constructed`
    — orchestrator does not force the substrate manager
    early. [REQUIRED-CI]
  - `test_stop_idempotent` — §2.2.2. [REQUIRED-CI]
  - `test_stop_drops_subscriptions` — §2.2.2.
    [REQUIRED-CI]
  - `test_stats_returns_dict` — §2.2.3. [REQUIRED-CI]
  - `test_stats_empty_before_start` — §2.2.3 default.
    [REQUIRED-CI]
  - `test_register_policy_appends` — §2.2.4. [REQUIRED-CI]
  - `test_register_policy_thread_safe` — §2.2.4 lock.
    [REQUIRED-CI]
  - `test_peek_handle_returns_handle_when_visible` — §2.2.5
    happy path. [REQUIRED-CI]
  - `test_peek_handle_returns_none_when_freed` — §2.2.5
    edge. [REQUIRED-CI]
  - `test_peek_handle_returns_none_when_unknown_id` —
    §2.2.5 edge. [REQUIRED-CI]
  - `test_peek_handle_raises_on_cross_tenant_with_context`
    — §2.5 G1 + §2.2.5 tenant scoping. [REQUIRED-CI]
  - `test_peek_handle_safe_in_cuda_callback` — §2.2.5
    safety. [REQUIRED-LOCAL @gpu]
  - `test_substrate_api_unchanged` — C1: `import memopt;
    memopt.alloc/free/context/stats/observe` import-shape
    snapshot test. [REQUIRED-CI]

  APIs covered: all of §2.2. Decisions verified: D5.

#### 3.1.8 `test_event_emission.py` — covers §2.2.6 (~180 lines)

  - `test_emit_orchestrator_event_evict` — basic emission.
    [REQUIRED-CI]
  - `test_emit_orchestrator_event_promote` — basic
    emission. [REQUIRED-CI]
  - `test_emit_orchestrator_event_migrate` — basic
    emission. [REQUIRED-CI]
  - `test_emit_rejects_alloc_kind` — only the three Layer-2
    kinds are accepted. [REQUIRED-CI]
  - `test_emit_rejects_free_kind` — symmetric.
    [REQUIRED-CI]
  - `test_evict_from_placement_hotter_than_to` — DECISION 4
    direction invariant. [REQUIRED-CI]
  - `test_promote_from_placement_colder_than_to` — DECISION 4.
    [REQUIRED-CI]
  - `test_migrate_lateral_or_explicit` — DECISION 4 lateral
    case. [REQUIRED-CI]
  - `test_migrate_preserves_handle_id` — DECISION 4
    distinguishing feature. [REQUIRED-CI]
  - `test_emitted_events_visible_to_external_subscriber`
    — round-trip via `memopt.observe`. [REQUIRED-CI]
  - `test_emit_does_not_raise_to_caller` — D4 (substrate)
    parity. [REQUIRED-CI]

  Gaps covered: O1 (producer side). Decisions verified: D4.

#### 3.1.9 `test_decision_d1_greenfield.py` — covers DECISION 1 (~80 lines)

  - `test_orchestrator_does_not_import_tier_manager` —
    static import scan via `ast`. [REQUIRED-CI]
  - `test_orchestrator_does_not_import_elastic_allocator`
    — static. [REQUIRED-CI]
  - `test_orchestrator_does_not_import_memory_governor` —
    static. [REQUIRED-CI]
  - `test_orchestrator_does_not_import_prefetch_engine`
    — static. [REQUIRED-CI]
  - `test_orchestrator_does_not_call_memopt_evict_to_target`
    — static (grep ctypes binding usage in
    `memopt/orchestrator/`). [REQUIRED-CI]

#### 3.1.10 `test_decision_d2_advisory.py` — covers DECISION 2 (~100 lines)

  - `test_alloc_succeeds_when_orchestrator_returns_none`
    — no opinion. [REQUIRED-CI]
  - `test_alloc_succeeds_when_orchestrator_returns_hint`
    — hint applied. [REQUIRED-CI]
  - `test_alloc_does_not_block_on_orchestrator_callback`
    — no wait verdict. [REQUIRED-CI]
  - `test_alloc_failure_still_raises_memory_error` — Layer
    1 owns hard-fail (C1). [REQUIRED-CI]
  - `test_orchestrator_callback_exception_does_not_break_alloc`
    — defensive. [REQUIRED-CI]

#### 3.1.11 `test_tenant_isolation.py` — covers §2.5 (~200 lines)

  - `test_g1_access_history_is_per_tenant` — G1.
    [REQUIRED-CI]
  - `test_g2_per_tenant_pressure_isolated` — G2.
    [REQUIRED-CI]
  - `test_g3_predictor_state_partitioned_by_tenant` — G3.
    [REQUIRED-CI]
  - `test_g4_aggregate_telemetry_only_without_admin_token`
    — G4. [REQUIRED-CI]
  - `test_g4_per_tenant_telemetry_with_admin_token` — G4
    happy path. [REQUIRED-CI]
  - `test_n4_queue_drops_oldest_under_starvation_workload`
    — N4 documented behaviour. [REQUIRED-CI]
  - `test_n3_register_policy_warns_on_unrestricted_use`
    — N3 documented behaviour. [REQUIRED-CI]
  - `test_lru_candidates_does_not_return_other_tenant`
    — G1 / O7 again. [REQUIRED-CI]
  - `test_forget_tenant_removes_all_state` — G3 +
    bookkeeping. [REQUIRED-CI]

#### 3.1.12 `test_perf_microbench.py` — covers §2.6 (~150 lines)

  Marker: `@perf` (OPTIONAL). The file prints measured
  numbers and asserts ordering only.

  - `test_event_ingest_p50_p99` — print-only. [@perf]
  - `test_predictor_query_p50_p99` — print-only. [@perf]
  - `test_policy_evaluation_p50_p99` — print-only.
    [@perf]
  - `test_decision_pump_cycle_steady_state` — print-only.
    [@perf]
  - `test_advisory_turnaround_p99` — print-only. [@perf]
  - `test_decision_to_emit_p50_p99` — print-only.
    [@perf]
  - `test_ordering_event_ingest_lt_predictor_query` —
    asserts ordering. [@perf]
  - `test_ordering_predictor_query_lt_policy_evaluation`
    — asserts ordering. [@perf]
  - `test_ordering_policy_evaluation_lt_cycle_period` —
    asserts ordering. [@perf]
  - `test_results_persisted_to_json` — writes
    `/tmp/memopt-orchestrator-bench/<sha>.json`.
    [@perf]

  Targets covered: all six of §2.6.

#### 3.1.13 Gap-to-test cross-check

```
Gap   Test files                                      Bucket
───── ──────────────────────────────────────────────── ───────
O1    test_coordinator.py, test_event_emission.py    REQUIRED-CI
O2    test_access_tracker.py                          REQUIRED-CI
O3    test_predictor.py                               REQUIRED-CI
O4    test_policy_engine.py                           REQUIRED-CI
O5    test_decision_d2_advisory.py,                  REQUIRED-CI
      test_orchestrator_legacy_parity.py (Phase B)
O6    test_decision_d2_advisory.py                    REQUIRED-CI
O7    test_tenant_isolation.py, test_policy_engine.py REQUIRED-CI
O8    test_telemetry.py                               REQUIRED-CI
O9    test_predictor.py                               REQUIRED-CI
      (test_does_not_share_state_with_memory_oracle)
O10   test_policy_engine.py, test_predictor.py        REQUIRED-CI
O11   test_coordinator.py                             REQUIRED-CI
```

Every gap O1–O11 is covered.

#### 3.1.14 Decision-to-test cross-check

```
Decision  Test                                                  Bucket
────────  ─────────────────────────────────────────────────────  ──────────
D1        test_decision_d1_greenfield.py                       REQUIRED-CI
D2        test_decision_d2_advisory.py                         REQUIRED-CI
D3        test_coordinator.py::test_start_creates_one_thread   REQUIRED-CI
D4        test_event_emission.py::test_migrate_preserves_*,    REQUIRED-CI
          test_coordinator.py::test_migrate_va_preserved       (+ @gpu)
D5        test_public_api.py::test_peek_handle_*               REQUIRED-CI
D6        test_predictor.py::test_predictor_keys_are_tenant_*  REQUIRED-CI
D7        test_coordinator.py::test_coordinator_does_not_*,    REQUIRED-CI
          test_orchestrator_legacy_parity.py (Phase A run)
D8        test_telemetry.py::test_snapshot_does_not_extend_*   REQUIRED-CI
```

Every decision D1–D8 has at least one test verifying the chosen
behaviour.

### 3.2 Bridge test specification

  Path: `tests/test_orchestrator_legacy_parity.py`

#### 3.2.1 Workload

  - 3 tenants: `alice`, `bob`, `carol`.
  - 200 allocations total, sizes drawn deterministically from
    a seeded PRNG (`random.Random(0xMEMOPT2)`):
    - 100 × 2 MiB (size class 2 MiB)
    - 60 × 8 MiB (size class 8 MiB)
    - 30 × 64 MiB
    - 10 × 256 MiB
    Distribution chosen to force eviction on a constrained
    pool (configured below).
  - Tenant assignment: round-robin in shuffled order.
  - Half (100) of the handles are stream-bound: `stream =
    MockStream()` per substrate `test_events.py`.
  - Pool: per-tenant arena ceiling set to 256 MiB via
    `OrchestratorConfig` overrides + `MEMOPT_EVICT_HIGH=0.85`,
    `MEMOPT_EVICT_LOW=0.65`. Total 768 MiB across three
    tenants — workload aggregate is ~2.5 GiB so eviction
    fires repeatedly.
  - Lifecycle phases (substrate bridge §3.3 model):
    1. **alloc** — issue all 200 allocations; record
       per-tenant byte counts.
    2. **access** — for each handle, call
       `memopt.peek_handle(id)` once (drives the predictor
       in Phase B; no-op observation in Phase A).
    3. **free60** — free 60 % of handles (round-robin per
       tenant).
    4. **realloc60** — re-allocate the same byte budget that
       was freed, with the same size mix.
    5. **final** — free everything; assert clean shutdown.

#### 3.2.2 Recorded metrics

  Per phase, record both legacy-only (orchestrator stopped)
  and orchestrator-on (Phase B) runs:

```
events_ingested_alloc        memopt.orchestrator.stats()
events_ingested_free
events_ingested_evict
events_ingested_promote
events_dropped               memopt.stats()['events_dropped']
double_free_count            memopt.stats()['double_free_count']
per_tenant_in_use_bytes      memopt.stats(tenant=t)['in_use_bytes']
per_tenant_high_water_bytes  memopt.stats(tenant=t)['high_water_bytes']
decisions_evict              memopt.orchestrator.stats()['decisions']['evict']
decisions_promote
decisions_migrate
predictor_transitions        memopt.orchestrator.stats()['predictor']['transitions']
coordinator_cycles           memopt.orchestrator.stats()['coordinator']['cycles']
queue_drops                  memopt.orchestrator.stats()['coordinator']['queue_drops']
```

#### 3.2.3 Phase A assertions (orchestrator started but does not drive eviction)

  Per DECISION 7 the orchestrator silently observes:

```
For each phase {alloc, access, free60, realloc60, final}:
    (1) per_tenant_in_use_bytes_orch == per_tenant_in_use_bytes_baseline   # exact
    (2) per_tenant_high_water_bytes_orch == ..._baseline                   # exact
    (3) events_dropped_orch == events_dropped_baseline                     # exact
    (4) double_free_count_orch == double_free_count_baseline               # exact
    (5) decisions_evict == 0                                               # observation only
    (6) decisions_promote == 0
    (7) decisions_migrate == 0
    (8) events_ingested_alloc == 200                                       # all events seen
    (9) events_ingested_free == 200 (sum across phases)
   (10) queue_drops == 0                                                   # queue not saturated
   (11) PermissionError on every cross-tenant peek                         # G1
```

  Phase A non-negotiable: assertions (1)–(4) MUST be exact —
  the orchestrator started but stopped MUST be byte-for-byte
  indistinguishable from the orchestrator never having existed.

#### 3.2.4 Phase B assertions (deferred to v1.1; spec only)

  With `MEMOPT_USE_ORCHESTRATOR=1`:

```
For each phase:
    (1) per_tenant_high_water_bytes_orch
            <= per_tenant_high_water_bytes_baseline * 1.05            # ±5%
    (2) abs(decisions_evict_orch - eviction_count_legacy)
            <= 0.10 * eviction_count_legacy                           # ±10%
    (3) abs(decisions_promote_orch - promotion_count_legacy)
            <= 0.10 * promotion_count_legacy                          # ±10%
    (4) events_dropped == 0                                           # in both runs
    (5) PermissionError on every cross-tenant peek                    # G1
    (6) coordinator_cycles >= total_runtime_s * (1000 / 50) * 0.9    # ≥90% of expected
    (7) queue_drops <= 0.001 * events_ingested_alloc                 # ≤0.1% under load
```

#### 3.2.5 Tolerance bounds rationale

  - **Phase A: exact.** The orchestrator must be invisible
    when stopped or in observation-only mode. Any drift here
    is a violation of C1 / DECISION 7.
  - **Phase B: ±5 % byte counts, ±10 % decision counts.**
    Mirrors substrate v1 §3.3 OPTION Y (±0.1 % byte
    counts) but relaxed because Layer 2 changes the eviction
    *policy*; the substrate v1 bridge changes only the
    *implementation*. Different policies legitimately produce
    different decision counts on the same workload.

#### 3.2.6 Order sensitivity (mirrors substrate v1 §3.3, OPTION Y)

  **OPTION Y is selected.** The bridge test does NOT assert
  the order of allocations, evictions, or decisions. It
  asserts exact metric counts (Phase A) or band-bounded
  metric counts (Phase B).

  Phase 1 evidence audit (re-checked for Phase 3):
    - The substrate's bridge audit (substrate v1 §3.3) found
      no test in the regression net asserting allocation
      order. Adding the orchestrator does not introduce new
      order-sensitive consumers.
    - `memopt/orchestrator/tests/` itself uses keyed lookups
      (`handle_id`), not iteration order, in every test
      planned in §3.1.

  Future-proofing comment in the bridge test source:

```python
# Bridge test assertions follow OPTION Y from
# docs/orchestrator_v1_design.md §3.2.6 (decided 2026-05-01).
# Order of allocations / evictions / decisions is NOT asserted
# because no existing test in the regression net depends on it
# (audit in §3.2.6 of the design doc). A future change that
# makes order observable to consumers should re-run that audit
# before relaxing this test's strictness.
```

### 3.3 Coexistence regression strategy

  Per §2.4, three legacy implementations stay frozen during
  Phase A. The strategy verifies each remains green under
  three orchestrator states.

#### 3.3.1 Scenario A — orchestrator NEVER started

  Command:

```
PYTHONPATH=. python3 -m pytest tests/ memopt/ -q -p no:cacheprovider \
  -k 'not gpu and not cuda' \
  --ignore=memopt/vmm/tests/test_cuda_vmm_smoke.py \
  --ignore=memopt/vmm/tests/test_torch_allocator_sanity.py
```

  Expected:
    - **752 passed, 22 skipped, 15 deselected** (the §1.6.1
      Mac baseline).
    - Plus N new passing tests from §3.1 where N is the
      sum of REQUIRED-CI test counts in §3.1 (≈140
      additional tests across 11 files).
    - Net: ~892 passed, 22 skipped, 15 deselected.

#### 3.3.2 Scenario B — orchestrator started but no flag set

  Command:

```
MEMOPT_AUTO_START_ORCHESTRATOR=1 \
PYTHONPATH=. python3 -m pytest tests/ memopt/ -q -p no:cacheprovider \
  -k 'not gpu and not cuda' \
  --ignore=memopt/vmm/tests/test_cuda_vmm_smoke.py \
  --ignore=memopt/vmm/tests/test_torch_allocator_sanity.py
```

  (`MEMOPT_AUTO_START_ORCHESTRATOR` is a Phase-A-only test
  knob, NOT a public env var. It is removed in Phase B in
  favour of `MEMOPT_USE_ORCHESTRATOR`.)

  Expected:
    - Same 752+N pass count as Scenario A.
    - The orchestrator subscribes silently per DECISION 7;
      no decision events flow.
    - `memopt.orchestrator.stats()['decisions']` returns
      all zeros.
    - `events_dropped == 0` end-to-end.

#### 3.3.3 Scenario C — `MEMOPT_USE_ORCHESTRATOR=1` (Phase B; spec only)

  Command (deferred to v1.1):

```
MEMOPT_USE_ORCHESTRATOR=1 \
PYTHONPATH=. python3 -m pytest tests/ memopt/ -q -p no:cacheprovider \
  -k 'not gpu and not cuda' \
  --ignore=memopt/vmm/tests/test_cuda_vmm_smoke.py \
  --ignore=memopt/vmm/tests/test_torch_allocator_sanity.py
```

  Expected (Phase B):
    - The §1.6.2 regression net (`test_vmm_smoke.py` 13,
      `test_vmm_hardening.py` 8, `test_oracle.py` 23,
      `test_access_log.py` 8, `test_layer3.py` 25,
      `test_events.py` 11, plus their cluster siblings) all
      remain green.
    - The bridge test (§3.2) Phase B assertions pass within
      tolerance.
    - The substrate's perf microbench `alloc(2 MiB) warm`
      P50 does not regress > 5 % vs. the substrate baseline
      (substrate §2.10).

  Phase A definition of done excludes Scenario C; it is
  Phase B's gating criterion.

#### 3.3.4 Non-negotiable

  The §1.6.1 baseline (752 PASSED on Mac) MUST hold in
  Scenarios A and B. New tests can ONLY add to the count.
  Any flip from PASSED to FAILED or SKIPPED in any of the
  752 fails the Phase A gate.

### 3.4 Performance regression strategy

  Mirrors substrate v1 §3.5 model. Microbenches print
  measured numbers; assertions are ordering-only.

#### 3.4.1 File structure

```
memopt/orchestrator/tests/test_perf_microbench.py
  ├── _bench_event_ingest()
  ├── _bench_predictor_query()
  ├── _bench_policy_evaluation()
  ├── _bench_cycle_period()
  ├── _bench_advisory_turnaround()
  ├── _bench_decision_to_emit()
  └── tests (each test calls one _bench_*)
```

  Marker: `@pytest.mark.perf`. Skipped by default in CI;
  run explicitly with `-m perf`.

#### 3.4.2 Measurement protocol

  - Each `_bench_*` does 10 000 trials, drops the first 1000
    as warmup, reports `(min, median, p99, max)` in ns.
  - Times are taken with `time.perf_counter_ns()`.
  - The bench function returns a dict; the test wraps it
    and asserts ordering invariants only (next subsection).
  - Measurements persist to
    `/tmp/memopt-orchestrator-bench/<git-sha>.json` so
    historical regressions can be plotted offline. The JSON
    is one record per bench function with `{name, median_ns,
    p99_ns, hardware, python_version, ts_ns}`.

#### 3.4.3 Ordering assertions

  ```
  assert event_ingest_p50          < predictor_query_p50
  assert predictor_query_p50       < policy_evaluation_p50
  assert policy_evaluation_p50     < cycle_period_target_ns
  assert decision_to_emit_p50      < advisory_turnaround_p99
  assert decision_pump_cycle_p99   < cycle_period_target_ns * 2
  ```

  Catches structural regressions (e.g. a predictor query
  becoming slower than a full policy evaluation) without
  pinning absolute thresholds that vary by hardware.

#### 3.4.4 Non-negotiable

  Tests in `test_perf_microbench.py` MUST NOT assert absolute
  thresholds. The only asserts are ordering invariants and
  "successfully wrote JSON." Hardware is too variable;
  absolute thresholds produce flakes that train teams to
  ignore the suite.

### 3.5 Implementation commit plan

Phase A is delivered as **14 commits** in **5 milestones**. Each
commit is reviewed by the maintainer before the next is launched.
No bulk commits; each is independently revertible.

#### Milestones

```
Milestone 1 — substrate-side prep (commits 0–1)
Milestone 2 — predictor + access tracker (commits 2–3)
Milestone 3 — policy engine + config (commits 4–5)
Milestone 4 — coordinator + telemetry + public API (commits 6–9)
Milestone 5 — bridge + docs + release (commits 10–13)
```

#### Commit table

```
| #  | Commit                                             | Touches                                                                  | New tests                                                          | Bucket                  |
| -- | -------------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------ | ----------------------- |
| 0  | Step Zero verification report                      | docs/orchestrator_v1_step_zero_report.md                                 | (none — verification only)                                         | —                       |
| 1  | substrate addition: peek_handle + emit hook        | memopt/substrate/manager.py, memopt/substrate/__init__.py, memopt/__init__.py | memopt/substrate/tests/test_peek_handle.py (10 tests)        | REQUIRED-CI             |
| 2  | AccessTracker + tenant LRU lists                   | memopt/orchestrator/access.py                                            | memopt/orchestrator/tests/test_access_tracker.py (~13 tests)       | REQUIRED-CI             |
| 3  | Predictor (Markov + sequential + recency)          | memopt/orchestrator/predict.py                                           | memopt/orchestrator/tests/test_predictor.py (~13 tests)            | REQUIRED-CI             |
| 4  | OrchestratorConfig + env-var plumbing              | memopt/orchestrator/config.py                                            | memopt/orchestrator/tests/test_config.py (~8 tests)                | REQUIRED-CI             |
| 5  | PolicyEngine + LRUWatermarkPolicy                  | memopt/orchestrator/policy.py                                            | memopt/orchestrator/tests/test_policy_engine.py (~12 tests)        | REQUIRED-CI             |
| 6  | TelemetryCollector                                 | memopt/orchestrator/telemetry.py                                         | memopt/orchestrator/tests/test_telemetry.py (~9 tests)             | REQUIRED-CI             |
| 7  | Coordinator + event-emission contract              | memopt/orchestrator/coordinator.py                                       | memopt/orchestrator/tests/test_coordinator.py (~15), test_event_emission.py (~11) | REQUIRED-CI (+ @gpu in 1) |
| 8  | Public API surface assembly                        | memopt/orchestrator/__init__.py                                          | memopt/orchestrator/tests/test_public_api.py (~15 tests)           | REQUIRED-CI (+ @gpu in 1) |
| 9  | Decision-verification + isolation tests            | memopt/orchestrator/tests/test_decision_d1_greenfield.py, test_decision_d2_advisory.py, test_tenant_isolation.py | (~5 + ~5 + ~9 tests)              | REQUIRED-CI             |
| 10 | Bridge test (legacy parity, Phase A assertions)    | tests/test_orchestrator_legacy_parity.py                                 | (the bridge test itself)                                           | REQUIRED-CI             |
| 11 | Performance microbench (ordering only)             | memopt/orchestrator/tests/test_perf_microbench.py                        | (~10 perf tests, all @perf)                                        | OPTIONAL                |
| 12 | Documentation: threat model, user guide, README    | docs/orchestrator_v1_threat_model.md, docs/orchestrator_v1_user_guide.md, README.md | (none — doc only)                                       | —                       |
| 13 | Final regression + version bump                    | pyproject.toml, memopt/__init__.py (version), CHANGELOG.md               | (none — release pass)                                              | —                       |
```

#### Per-commit acceptance criteria (AC1–AC6)

  AC1  The committed code parses with no new warnings under
       `python -W error -c "import memopt.orchestrator"` (or
       the relevant subpackage).
  AC2  Every test introduced in the commit passes locally on
       the Mac baseline.
  AC3  The full 752-pass Mac baseline regression suite passes
       (with the same 22 skipped, 15 deselected as the
       §1.6.1 baseline). The suite count grows by exactly the
       number of new tests added.
  AC4  No existing module under `memopt/vmm/`,
       `memopt/cluster/`, `memopt/serving/`, `memopt/api/`,
       `memopt/control_plane/`, `memopt/operator/`,
       `memopt/canary/`, `memopt/daemon/`, `memopt/profiler/`,
       `memopt/kernels/`, OR under `memopt/substrate/` (except
       commit 1's documented additions to `manager.py` and
       `__init__.py`) was edited.
  AC5  The commit message references the design-doc section
       it implements (e.g. "Implements §2.3.2 Predictor per
       docs/orchestrator_v1_design.md").
  AC6  The commit is independently revertible: reverting it
       does not break any earlier commit's tests.

#### Per-commit baseline contract status line

  Each commit ends its message with one line:

```
Baseline: NNN passed, 22 skipped, 15 deselected on Mac
(was: M passed, 22 skipped, 15 deselected at HEAD~1).
Net: +(NNN-M) tests, all in the orchestrator package.
```

  Commit 0: NNN == 752 (verification only; no code change).
  Commit 1: NNN == 752 + 10 (peek_handle tests).
  Commit 2: NNN == 762 + 13 = 775. ... etc.

#### Commit ordering rationale

  - **0 first**: nothing else may proceed if a TODO-VERIFY
    fails in a way that requires Phase 2 redesign. TV3, TV4,
    TV7 hard-block Phase A; the other four either pass or
    proceed with documented degradation.
  - **1 (substrate addition) before 2+**: every Layer 2
    component imports from `memopt.substrate` and uses
    `peek_handle`. Without commit 1, every later commit would
    require monkey-patching during tests.
  - **2 (AccessTracker) and 3 (Predictor) before 5
    (PolicyEngine)**: `LRUWatermarkPolicy.evaluate` reads
    `AccessTracker.lru_candidates` and `Predictor.predict` to
    rank candidates.
  - **4 (Config) before 5 (PolicyEngine) and 7 (Coordinator)**:
    both read `OrchestratorConfig`. Config has zero
    dependencies, so it could go earlier; placed at 4 so it
    immediately precedes its first consumer.
  - **5 (PolicyEngine) before 7 (Coordinator)**: coordinator's
    `tick()` builds `PolicySnapshot` and calls
    `PolicyEngine.evaluate`.
  - **6 (Telemetry) before 7 (Coordinator)**: coordinator
    increments telemetry counters every cycle.
  - **7 (Coordinator) before 8 (Public API)**: `start()`
    constructs and starts the coordinator.
  - **9 (decision tests + isolation) after 8 (Public API)**:
    the decision-verification tests use the public API to
    drive scenarios.
  - **10 (Bridge) after 9 (Public API + decision tests)**:
    the bridge test calls the public API and asserts
    behaviour pinned in commits 1–9.
  - **11 (Perf) after 10 (Bridge)**: perf benches are last
    of the test commits because they do not gate
    correctness.
  - **12 (Docs) after every code commit**: docs reference
    real implementation paths.
  - **13 last**: version bump only after everything else is
    green.

#### When a commit fails

  - AC1 / AC2 fail: the commit is fixed in the same
    invocation. No new commit.
  - AC3 fails (regression in the 752 baseline): the commit
    is reverted. Either the regression is real (Layer 2 has
    a bug — fix in same invocation) or the test was already
    flaky (record in flake-tracking issue; rerun).
  - AC4 fails (touched a forbidden module): the commit is
    rejected. Restart the invocation with corrected scope.
  - AC5 / AC6 fail: amend message / split commits.

### 3.6 Phase A acceptance gate

Phase A is complete when ALL of the following hold:

```
[ ] Step Zero report exists at
    docs/orchestrator_v1_step_zero_report.md and shows PASS
    or DEGRADED-with-fallback for every S0.1–S0.7 item.
[ ] All HARD-BLOCK items (S0.3 / TV3, S0.4 / TV4) are PASS.
[ ] memopt/orchestrator/ contains:
        __init__.py
        access.py
        predict.py
        policy.py
        coordinator.py
        telemetry.py
        config.py
        tests/__init__.py
        tests/test_access_tracker.py
        tests/test_predictor.py
        tests/test_policy_engine.py
        tests/test_coordinator.py
        tests/test_telemetry.py
        tests/test_config.py
        tests/test_public_api.py
        tests/test_event_emission.py
        tests/test_decision_d1_greenfield.py
        tests/test_decision_d2_advisory.py
        tests/test_tenant_isolation.py
        tests/test_perf_microbench.py
[ ] memopt/substrate/manager.py has the new peek_handle
    method and the internal _emit_orchestrator_event method
    (the only commit-1 substrate edits).
[ ] memopt/substrate/__init__.py re-exports peek_handle.
[ ] memopt/__init__.py re-exports peek_handle and exposes
    the orchestrator subpackage.
[ ] memopt/substrate/tests/test_peek_handle.py exists.
[ ] tests/test_orchestrator_legacy_parity.py exists and
    passes the §3.2 Phase A assertions.
[ ] docs/orchestrator_v1_design.md exists (Phases 1–3).
[ ] docs/orchestrator_v1_step_zero_report.md exists.
[ ] docs/orchestrator_v1_threat_model.md exists.
[ ] docs/orchestrator_v1_user_guide.md exists.
[ ] docs/orchestrator_v1_implementation_prompt.md exists
    (produced by Phase 4; placeholder during Phase A).
[ ] CHANGELOG.md has the orchestrator v1 entry.
[ ] pyproject.toml version is bumped (e.g. 1.1.0 → 1.2.0).
[ ] No existing module under memopt/vmm/, memopt/cluster/,
    memopt/serving/, memopt/api/, memopt/control_plane/,
    memopt/operator/, memopt/canary/, memopt/daemon/,
    memopt/profiler/, memopt/kernels/ was edited (verified
    by `git log --diff-filter=M --name-only` over the
    Phase A commits showing only the additive paths).
[ ] No existing file under memopt/substrate/ was edited
    EXCEPT memopt/substrate/manager.py and
    memopt/substrate/__init__.py in commit 1.
[ ] Mac baseline (§1.6.1): 752 originally passing tests STILL
    pass; orchestrator tests add to the PASSED count;
    SKIPPED stays at 22; DESELECTED stays at 15; FAILED == 0.
[ ] The substrate's existing GPU-rig baseline (753 PASSED on
    the substrate's @gpu CI) is preserved (commit 1 changes
    do not regress any substrate test).
[ ] Bridge test (§3.2.3) Phase A assertions all pass:
    exact equality on per-tenant byte counts and event-
    drop counters; decisions == 0.
[ ] Decision-to-test cross-check (§3.1.14): every D1–D8
    has at least one PASSING test verifying the chosen
    behaviour.
[ ] Gap-to-test cross-check (§3.1.13): every O1–O11 has at
    least one PASSING test.
[ ] Public-API cross-check (§2.2): every public symbol
    (start, stop, stats, register_policy, peek_handle,
    OrchestratorHandle, OrchestratorConfig) has at least
    one PASSING test in test_public_api.py or
    test_config.py.
[ ] Performance microbench (§3.4) writes a JSON record
    at /tmp/memopt-orchestrator-bench/<sha>.json on at
    least one rig; the ordering assertions pass.
[ ] `python -c "import memopt; memopt.orchestrator.start();
    memopt.orchestrator.stop()"` runs without error and
    leaves no zombie threads (`threading.enumerate()`
    pre-import == post-stop).
[ ] AC4 audit: `git diff --name-only main..HEAD` shows
    only paths under `memopt/orchestrator/`,
    `memopt/substrate/__init__.py`,
    `memopt/substrate/manager.py`,
    `memopt/substrate/tests/test_peek_handle.py`,
    `memopt/__init__.py`, `tests/test_orchestrator_legacy_parity.py`,
    `docs/orchestrator_v1_*`, `CHANGELOG.md`,
    `pyproject.toml`.
[ ] Phase A version tag created on `main` (e.g.
    `orchestrator-v1.0.0`).
[ ] CHANGELOG entry includes a "What Phase A does NOT
    deliver" section pointing at §3.7.
[ ] User guide (`docs/orchestrator_v1_user_guide.md`) has
    a worked example: `start()` → register a policy →
    observe a decision in `stats()` → `stop()`.
[ ] Threat model (`docs/orchestrator_v1_threat_model.md`)
    enumerates §2.5 G1–G4 and N1–N4 with mitigations.
[ ] No `TODO` or `FIXME` introduced in commits 1–13 that
    is not paired with an issue link.
```

(28 checkbox items.)

### 3.7 What Phase A does NOT deliver

Phase A delivers v1.0 of the orchestrator: parallel addition,
opt-in via explicit `start()`, observation-only by default.
The following are explicitly out of scope.

  - **Phase B** (`MEMOPT_USE_ORCHESTRATOR=1` flag flip).
    Separate engagement after v1.0 has soaked. Adds the
    advisory `placement="auto"` resolution per §1.5.4 and
    routes the C++ allocator's `step_boundary` through
    Layer 2 (§2.4.1). Gating criteria in §2.7 Phase B.
  - **Phase C** (default-ON).
    Separate engagement. Requires three months of Phase B
    production exposure with no regressions reported.
  - **Phase D** (remove legacy paths).
    Separate engagement. Deletes
    `TierManager._ensure_capacity`, `ElasticAllocator`,
    `MemoryGovernor`, the C++ direct-eviction path, and
    the VMM-internal Markov chain in `PrefetchEngine`.
    Gating criteria in §2.7 Phase D.
  - **Layer 3** (ML-driven prefetch oracle).
    Separate package `memopt-prefetch-oracle`; the §2.3.2
    Markov predictor stays classical in v1.0.
  - **Layer 4** (declarative policy DSL).
    Separate package `memopt-policy-dsl`; v1.0 ships only
    the `Policy` protocol for users to write policies in
    Python.
  - **Layer 5** (federated orchestration across nodes).
    The §2.3.2 predictor explicitly does not gossip across
    processes (DECISION 7 trade-off). Lives in a separate
    layer, analogous to `memopt/vmm/global_oracle.py` but
    for substrate handles.
  - **vLLM integration.** `memopt-vllm` (separate package)
    wires the orchestrator into vLLM's KV-cache loop. Out
    of scope for v1.0.
  - **Custom CUDA kernels for the predictor.** v1.0 is
    pure Python (DECISION 6 trade-off). Phase D may
    revisit consuming `memopt._memopt_core`.
  - **Persistence of learned patterns across restarts.**
    Predictor starts cold every process. The §2.8
    `MemoryOracle.warm_from_log` path can be reused in a
    future revision.
  - **Cost / FinOps integration.** Lives in `memopt-trust`.
    Substrate emits events; orchestrator emits events;
    finops consumes both out-of-process.
  - **Confidential computing / attestation.** Out of scope.
    Orchestrator cooperates with NVIDIA CC mode but does
    not provide attestation tokens.

### 3.8 Phase 3 stop

Phase 3 test plan complete. Awaiting review before Phase 4
(implementation prompt).

The Phase 4 prompt will be a separate, self-contained doc at
`docs/orchestrator_v1_implementation_prompt.md`. It will cite
this file as authoritative and walk through the 14 commits
above in prompt form, one prompt per commit. It will include
the guardrail "do not edit existing files in `memopt/vmm/`,
`memopt/cluster/`, `memopt/serving/`, or any other
non-orchestrator path outside the listed commits". It will
require the 752-pass Mac baseline to hold after every commit
and the orchestrator's own tests to pass per §3.1.

---

*End of Phase 3.*
