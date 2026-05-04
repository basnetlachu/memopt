# memopt Prefetch Oracle (Layer 3) — Design

| Field | Value |
| --- | --- |
| Status | Phase 1 — Discovery (read-only) |
| Builds on | Layer 1 substrate (`docs/substrate_v1_design.md`, v1.1) |
| Builds on | Layer 2 orchestrator (`docs/orchestrator_v1_design.md`, v1.2.0) |
| Repo | `/Users/lachumanbasnet/Personal/Sophisticates/MEMOPT/memopt` |
| Drafted | 2026-05-01 |

## Sections

  1. Discovery (Phase 1 — this section)
  2. Design (Phase 2 — pending)
  3. Test plan (Phase 3 — pending)
  4. Implementation prompt (Phase 4 — pending)

---

## Phase 1 — Discovery

This phase is read-only inventory. No design opinions; no
implementation. Every claim about existing code carries a `file:line`
citation. Where something does not exist, that is stated explicitly.

---

### 1.1 Layer 2 predictor protocol (what Layer 3 implements)

Layer 2 ships a tenant-keyed classical predictor at
`memopt/orchestrator/predict.py`. Layer 3's contract is to implement
the same public surface (or extend it backward-compatibly) so the
coordinator can swap implementations without changes upstream.

#### 1.1.1 The `Predictor` class

  Defined at `memopt/orchestrator/predict.py:31-50`. Constructor
  signature:

  ```python
  Predictor(max_transitions: int = 100_000,
            min_confidence: float = 0.3)
  ```

  Internal state (`predict.py:39-50`):
    - `_transitions: OrderedDict[(tenant, tag), Counter[(tenant, tag)]]`
    - `_recency: OrderedDict[(tenant, tag), int]`
    - `_steps: Dict[str, int]` — per-tenant step counter
    - `_last_tag: Dict[str, str]` — per-tenant last observed tag
    - `_lock: threading.RLock`

#### 1.1.2 Public methods

  - `observe(tenant: str, tag: str) -> None`
    `predict.py:52-72`. Updates the transition table from the previous
    tag for that tenant; bumps the per-tenant step counter; refreshes
    recency. Bounded: when `len(_transitions) > _max_transitions`,
    the least-recently-touched row is popped (`predict.py:65-66`).

  - `predict(tenant: str, tag: str, top_k: int = 10) -> List[Prediction]`
    `predict.py:74-147`. Returns up to `top_k` predictions sorted by
    confidence descending.

  - `forget_tenant(tenant: str) -> None`
    `predict.py:149-165`. Drops every transition row whose source key
    starts with that tenant; drops every recency entry; drops the
    tenant's step + last-tag scalars. Provides G3 cleanup.

  - `stats() -> dict`
    `predict.py:167-178`. Returns
    `{transition_rows, transition_cells, recency_size,
    tenants_tracked, max_transitions, min_confidence}`.

#### 1.1.3 The `Prediction` dataclass

  `predict.py:23-28`:

  ```python
  @dataclass(frozen=True)
  class Prediction:
      tenant: str
      tag: str
      confidence: float
      source: str  # "transition" | "sequential" | "recency" | "fallback"
  ```

  Frozen — Layer 3 must produce instances with the same field shape.

#### 1.1.4 The four prediction sources

  Implemented inside `predict()`:

  - **transition** (`predict.py:80-92`): top entries of
    `_transitions[(tenant, tag)]` Counter. Confidence = `count / total`.
  - **sequential** (`predict.py:94-111`): if `tag` matches
    `^(.*?)(\d+)$`, emit predictions for `+1` (conf 0.6) and `+2`
    (conf 0.4) with width-preserved zero-padding.
  - **recency** (`predict.py:113-131`): up to 5 other recently-observed
    tags for the same tenant (excluding `tag`). Confidence = 0.35.
  - **fallback** (`predict.py:133-141`): when no other source produced
    output, emit `(tenant, tag)` itself at confidence 0.3 — cold-start
    safety net.

  De-duplication is by next `tag`: the first source to claim a tag
  wins (`predict.py:78`, `results: Dict[str, Prediction]`).

#### 1.1.5 The `min_confidence` cutoff

  Source 1 (transition) is filtered at `predict.py:86`. Source 2
  (sequential) is filtered per-delta at `predict.py:101-102`. Source 3
  (recency) is filtered at `predict.py:124-125`. Source 4 (fallback)
  is filtered at `predict.py:135`. Setting `min_confidence > 0.6`
  silences sequential delta=2 (conf 0.4); setting `min_confidence > 0.6`
  also silences recency (conf 0.35); a value `> 0.3` silences fallback.

#### 1.1.6 The `(tenant, tag)` key invariant — DECISION 6

  Authoritative at `docs/orchestrator_v1_design.md` §2.1 DECISION 6
  and §2.5 G3 (lines 1577–1581). Every key in `_transitions` and
  `_recency` is the tuple `(tenant, tag)`; the value-side counter is
  also keyed by `(tenant, tag)`. Two tenants observing the same tag
  values produce disjoint transition state. The test enforcing this
  is at `memopt/orchestrator/tests/test_predictor.py:88-100`
  (`test_predictor_keys_are_tenant_aware`).

  Layer 3 inherits this invariant: any model state must be partitioned
  by tenant or proven not to cross-contaminate.

#### 1.1.7 How the coordinator calls `predict` and `observe`

  `OrchestratorCoordinator._apply_event` at
  `memopt/orchestrator/coordinator.py:160-170` calls
  `predictor.observe(event.tenant, event.tag)` for every
  `alloc` and `promote` event (no-op for `free`/`evict`/`migrate`).
  `observe` exceptions are caught and logged, never propagated.

  `OrchestratorCoordinator._build_snapshot` at `coordinator.py:187-219`
  embeds `predictor.stats()` (not `predict()` output) in the
  PolicySnapshot's `predictor` field. The Phase A coordinator
  **does not call `predict()` at all** — it only feeds `observe()`.
  Predictions are reserved for the Phase B `placement="auto"`
  advisory path (per `docs/orchestrator_v1_design.md` §2.4 and the
  policy engine's snapshot consumer).

  Singleton wiring lives at `memopt/orchestrator/__init__.py:46-71`
  (`_OrchestratorState`); the `Predictor` is constructed with
  `predictor_max_transitions` / `predictor_min_confidence` from
  `OrchestratorConfig`.

#### 1.1.8 Test surface that pins the protocol

  `memopt/orchestrator/tests/test_predictor.py` (13 tests) —
  enumerated in `docs/orchestrator_v1_design.md` §3.1.2. Notably:
    - `test_predict_returns_top_k_sorted_by_confidence` (lines 33-43)
    - `test_predict_includes_sequential_source` (lines 46-53)
    - `test_predict_fallback_when_cold_start` (lines 64-70)
    - `test_min_confidence_filters_predictions` (lines 73-80)
    - `test_max_transitions_bounded` (lines 83-90)
    - `test_predictor_keys_are_tenant_aware` (lines 88-100)
    - `test_predictor_does_not_share_state_with_memory_oracle`
      (lines 161-187)

  Layer 3's drop-in implementation must keep all 13 green.

---

### 1.2 Event stream and training data available

This is what Layer 3 will train and infer on.

#### 1.2.1 The substrate `Event` dataclass

  `memopt/substrate/events.py:33-46`:

  ```python
  _EVENT_KINDS = ("alloc", "free", "evict", "promote", "migrate")

  @dataclass(frozen=True)
  class Event:
      kind: str
      timestamp_ns: int
      handle_id: int
      tenant: str
      tag: str
      size_bytes: int
      from_placement: Optional[str] = None
      to_placement: Optional[str] = None
      reason: Optional[str] = None
  ```

  `timestamp_ns` is `time.monotonic_ns()` at emit time
  (`memopt/substrate/manager.py:247-257` for alloc;
  `manager.py:284-293` for free; `manager.py:240-249` from the Layer-2
  emission path in `_emit_orchestrator_event`).

#### 1.2.2 Event emission sites in Layer 1

  - **alloc**: `memopt/substrate/manager.py:247-257` — emitted
    immediately after a fresh handle is bound; `from_placement=None`,
    `to_placement=handle.placement`, `reason="cold"|"freelist"`.
  - **free**: `memopt/substrate/manager.py:284-293` — emitted in
    `AllocationManager.free`; `from_placement=handle.placement`,
    `to_placement=None`, `reason="immediate"|"stream_locked"`.
  - **evict / promote / migrate**: emitted only by Layer 2 via
    `AllocationManager._emit_orchestrator_event`
    (`memopt/substrate/manager.py:398-435`), which validates the
    placement-rank invariant before forwarding to `_dispatcher.emit`.
    In Phase A no Layer-2 caller routinely fires these
    (`docs/orchestrator_v1_design.md` DECISION 7); the bridge test
    confirms `decisions == 0` in observation-only mode.

  Subscription happens via `Dispatcher.subscribe`
  (`memopt/substrate/events.py:150-163`) — per-kind filtering, one
  background dispatcher thread per subscription.

#### 1.2.3 The `AccessRecord` dataclass

  `memopt/orchestrator/access.py:20-55`. Slot-based (mutable; cannot
  be frozen because `lru_prev/lru_next` move under the LRU list):

  ```python
  AccessRecord:
      tenant, handle_id, tag, placement, size_bytes,
      last_seen_ns, hit_count,
      lru_prev, lru_next      # intrusive LRU links
  ```

  `hit_count` is incremented on every `evict/promote/migrate` for the
  record (`access.py:183`). `last_seen_ns` is refreshed on the same
  events (`access.py:182`). `alloc` initializes the record;
  `free` removes it (`access.py:147-173`).

#### 1.2.4 Temporal information available to Layer 3

  - **Per-event monotonic timestamp**: `Event.timestamp_ns`
    (`events.py:39`). Same source on alloc + free, so inter-arrival
    times within a sequence are recoverable.
  - **Per-handle last-seen**: `AccessRecord.last_seen_ns`
    (`access.py:29`). Updated on placement-changing events only.
  - **Per-handle hit count**: `AccessRecord.hit_count`
    (`access.py:30`). Updated on placement-changing events only.
  - **Per-tenant step counter** (`Predictor._steps`): monotonic
    integer per tenant (`predict.py:46`, `predict.py:68`). Provides a
    relative ordinal for the recency table.
  - **Per-handle creation time**: `MemoryHandle.created_at`
    (`memopt/substrate/manager.py:229`) — `time.monotonic()`. Live on
    the handle, accessible via `peek_handle`.

#### 1.2.5 What is NOT in events (gaps Layer 3 may need surfaced)

  - **No `read` event.** The substrate emits no event on data read /
    write to a `MemoryHandle`. `docs/orchestrator_v1_design.md`
    §2.3.1 (lines 1294-1297) calls this out explicitly — instrumenting
    reads is "out of scope (would require a substrate change)."
    Layer 3 can only observe the alloc/free/placement-change events.
  - **No token position / generation step.** The legacy
    `BlockAccessEvent` carries `token_position` and `attention_layer`
    (`memopt/vmm/access_log.py:25-35`); the Layer 1 `Event` does not.
    Layer 3 must derive sequence ordering from `tag` patterns
    (e.g. numeric suffixes) or from the per-tenant step counter, OR
    propose a substrate addition to surface this.
  - **No request / sequence_id boundary.** The legacy oracle keys
    transitions on `sequence_id` (`_oracle_py.py:77`). Layer 1's
    `Event.tag` is the analogue but the substrate does not enforce
    that a `sequence_id` and a `tag` mean the same thing — that
    convention is left to the user.
  - **No outcome feedback.** `MemoryOracle.record_outcome`
    (`_oracle_py.py:182-188`) is the legacy oracle's "did this
    prediction come true?" signal. Layer 2's `Predictor` does not
    accept this signal today; the coordinator does not produce it.
  - **No working-set window**: events are point-in-time. To compute
    e.g. "tenant alice's resident-set turnover rate over the last
    100 ms" Layer 3 must aggregate inside its own state.

---

### 1.3 Existing ML / oracle infrastructure

These predate Layer 2 and live under `memopt/vmm/` and
`csrc/core/oracle.{h,cpp}`. They serve the legacy VMM, not the
orchestrator. Per `docs/orchestrator_v1_design.md` §2.4.5 they are
frozen-in-place during Phase A and must remain green forever.

#### 1.3.1 `memopt/vmm/_oracle_py.py` — legacy classical oracle

  Pure Python; no torch, no numpy (file header,
  `_oracle_py.py:1-7`). Contains:

  - `BlockPrediction` dataclass (`_oracle_py.py:20-26`): fields
    `(sequence_id, block_index, confidence, predicted_at_step, source)`.
    Source ∈ `{"transition","sequential","recency","fallback"}` —
    same four sources as Layer 2's `Prediction`.
  - `OracleStats` dataclass (`_oracle_py.py:29-37`).
  - `MemoryOracle` class (`_oracle_py.py:40-251`):
    - `__init__(horizon=50, max_transitions=100_000, min_confidence=0.3, hardware_profile=None)` (`_oracle_py.py:42-73`).
    - `observe(sequence_id, block_index, step=None)` (`_oracle_py.py:77-113`).
    - `predict(sequence_id, current_block, top_k=10)` (`_oracle_py.py:115-180`).
    - `record_outcome(sequence_id, block_index)` (`_oracle_py.py:182-188`).
    - `stats()` (`_oracle_py.py:190-211`).
    - `reset(sequence_id=None)` (`_oracle_py.py:213-227`).
    - `warm_from_log(log_path, max_events=50_000)` (`_oracle_py.py:229-251`).
  - **Disposition for Layer 3**: COEXIST. Legacy VMM still depends on
    it; not Layer 3's path. Layer 3 must NOT import it
    (`memopt/orchestrator/tests/test_predictor.py:172-181` already
    enforces this for Layer 2's predictor — Layer 3 inherits the
    constraint).

#### 1.3.2 `memopt/vmm/access_log.py` — JSONL persisted traces

  - `BlockAccessEvent` dataclass (`access_log.py:25-35`): fields
    `(sequence_id, block_index, token_position, attention_layer,
    timestamp, tier_at_access, promotion_latency_ms, tenant_id)`.
  - `AccessLog` class (`access_log.py:37-141`):
    - Background daemon thread `access-log-writer`
      (`access_log.py:62-65`).
    - Default log dir: `~/.memopt/access_logs/`
      (`access_log.py:53`); rotated daily (`access_log.py:55`);
      file pattern `access_log_<YYYYMMDD>.jsonl`.
    - Bounded queue capacity 10 000 (`access_log.py:40`); full →
      drop + counter increment (`access_log.py:71-73`).
    - Used today only by `memopt/vmm/prefetch_engine.py:18,143,158,
      204-215` (constructor-injected, not auto-wired).
  - **Disposition for Layer 3**: REUSE the JSONL schema as a
    candidate persistence format; the writer infrastructure itself is
    not wired to substrate events. Layer 3 either (a) writes its own
    log from substrate events or (b) proposes that Layer 2 emit
    BlockAccessEvent-shaped records. Either is a Phase 2 decision.

#### 1.3.3 `memopt/vmm/oracle_trainer.py` — warm-from-log path

  - `OracleTrainer` class (`oracle_trainer.py:25-185`): background
    daemon `oracle-trainer` (`oracle_trainer.py:48-49`); polls
    `log_dir` every `poll_interval_s` (default 5.0,
    `oracle_trainer.py:31`); calls `oracle.observe()` for each clean
    event (`oracle_trainer.py:91`, `oracle_trainer.py:152`).
  - Cleaning is delegated to `OracleDataCleaner.clean_file()`
    (`oracle_trainer.py:86-87`).
  - **Disposition for Layer 3**: COEXIST as an algorithmic reference
    for incremental on-disk training. Layer 3's training pipeline
    will likely need an analogue but not depend on this class —
    this class targets `MemoryOracle.observe`, not Layer 2's
    `Predictor.observe`.

#### 1.3.4 `memopt/vmm/oracle_data_cleaner.py` — 7-step cleaner

  - `CleaningStats` dataclass (`oracle_data_cleaner.py:19-28`).
  - `OracleDataCleaner` class (`oracle_data_cleaner.py:31-263`):
    - `__init__(min_sequence_length=3, max_sequence_length=10_000,
       min_block_index=0, max_block_index=1_000_000,
       max_promotion_latency_ms=10_000.0, valid_tiers=...,
       deduplicate_window=3)` (`oracle_data_cleaner.py:33-49`).
    - `clean_file(jsonl_path) -> (events, stats)`
      (`oracle_data_cleaner.py:53-end`). Steps 1-7: parse, validate,
      group by `sequence_id`, sort, dedupe, sequence-length filter,
      outlier filter.
  - **Disposition for Layer 3**: REUSE as a reference for data
    hygiene; the schema (`sequence_id` + `block_index` +
    `token_position`) does not match the substrate `Event` schema
    one-to-one, so reuse will require a translation layer.

#### 1.3.5 `memopt/vmm/global_oracle.py` — control-plane aggregator

  - `GlobalOracleConfig` (`global_oracle.py:53-73`): env-driven
    (`MEMOPT_GLOBAL_ORACLE_PULL_S`, `MEMOPT_GLOBAL_ORACLE_TOP_K`,
    etc.).
  - `GlobalOracle` class (`global_oracle.py:76-end`): runs in the
    control plane process; pulls from pod oracles every 60 s
    (default); pushes top-K transitions back. Aggregates across pods,
    not per-request.
  - The three-tier hierarchy (Global / Pod / Node) is documented in
    the file header (`global_oracle.py:1-19`).
  - **Disposition for Layer 3**: COEXIST. Per
    `docs/orchestrator_v1_design.md` §2.4.5 (lines 1547-1556)
    `global_oracle.py` and `pod_controller.py` are the cluster-wide
    legacy oracle infrastructure and "must remain green forever";
    cross-pollination with Layer 3 is a Layer 5 concern.

#### 1.3.6 `memopt/vmm/pod_controller.py` — pod-level oracle

  570 lines (`wc -l` 2026-05-01). Same comment block at the top of
  `global_oracle.py:1-19` documents this as the middle tier of the
  three-tier hierarchy, pulling from node oracles every 1 s. Same
  disposition as 1.3.5: COEXIST, untouched by Layer 3.

#### 1.3.7 `csrc/core/oracle.{h,cpp}` — C++ accelerated oracle

  - `oracle.h:33-40`: `BlockPrediction` struct mirrors the Python
    dataclass exactly (per the comment).
  - `oracle.h:44-52`: `OracleStats` struct mirrors Python.
  - `oracle.h:72-125`: `MemoryOracle` C++ class — 256-bucket striped
    locking (`oracle.h:138-143`), `std::partial_sort` for
    `predict()` (per file header comment, `oracle.h:10`),
    `warm_from_log` released-GIL implementation
    (`oracle.cpp:416`). pybind11-exposed.
  - **Disposition for Layer 3**: COEXIST. C++ ABI parity is the
    legacy VMM's contract; Layer 3 is greenfield Python. Reuse would
    require a new pybind11 surface and is out of Phase A scope.

#### 1.3.8 Training-data infrastructure useful to Layer 3

  Summary of what already exists that Layer 3 *could* reuse without
  modifying Layer 1 or Layer 2:

  | Asset | File | Useful for Layer 3? |
  | --- | --- | --- |
  | JSONL schema (BlockAccessEvent) | `access_log.py:25-35` | Schema reference; not auto-wired to substrate |
  | Background JSONL writer | `access_log.py:37-141` | Reusable as a private helper if rooted under a Layer-3 module |
  | 7-step cleaner | `oracle_data_cleaner.py` | Reference; field-shape mismatch needs translation |
  | warm-from-log poll loop | `oracle_trainer.py:69-185` | Reference; targets `MemoryOracle.observe`, not `Predictor.observe` |
  | C++ accelerated transition table | `csrc/core/oracle.{h,cpp}` | Out of scope for Phase A; requires pybind11 surface and ABI work |

---

### 1.4 Layer 3 gaps (what does NOT exist today)

Following `docs/orchestrator_v1_design.md` §1.4 shape: each gap is a
thing the codebase does NOT have that Layer 3 needs. Each entry has
either a `file:line` "could live here but doesn't" pointer or
"greenfield."

  - **O1 — No persistent storage of Layer 2's access traces.**
    `memopt/orchestrator/predict.py:31-50` shows the `Predictor` is
    in-memory only. There is no flush / load / serialize hook on
    Layer 2's predictor; `forget_tenant` discards state without
    persisting (`predict.py:149-165`). The legacy `AccessLog`
    (`access_log.py`) is not wired to substrate events and does not
    persist `Event` records. Greenfield for Layer 3 from the substrate
    event side.

  - **O2 — No training pipeline for ML models.** No file under
    `memopt/orchestrator/` imports `torch`, `numpy`, or `sklearn`
    (`grep -rn "torch\|numpy\|sklearn" memopt/orchestrator/` returns
    no hits in source files). `oracle_trainer.py:25-185` exists but
    targets the legacy `MemoryOracle.observe` API and counter-based
    transitions, not gradient training. Greenfield.

  - **O3 — No model serialization / loading format.**
    `Predictor.stats()` (`predict.py:167-178`) returns counters but
    no weights / parameters / vocabulary mapping. No `save()` /
    `load()` / `from_file()` method exists on `Predictor`. The C++
    oracle exposes `warm_from_log` (`csrc/core/oracle.h:108`) but no
    binary state save. Greenfield.

  - **O4 — No inference path within the coordinator latency budget
    (cycle is 50 ms by default; per-event budget is sub-1 ms).**
    `OrchestratorCoordinator._cycle_period_ms` defaults to 50
    (`coordinator.py:64`); the per-event handler at
    `coordinator.py:160-170` is currently `tracker.record` +
    `predictor.observe` + telemetry — pure Python dict ops.
    Inference of an RNN / transformer at this rate from the
    coordinator thread has no current implementation. The §2.6 perf
    targets in `docs/orchestrator_v1_design.md` (lines 1622-1664) are
    written for the classical predictor (predictor query P50 < 20 µs).
    Greenfield for ML-grade inference.

  - **O5 — No A/B testing harness (classical vs ML).** No file under
    `memopt/orchestrator/` or `tests/` contains a switch between two
    `Predictor` implementations on the same workload. The bridge
    test (`tests/test_orchestrator_legacy_parity.py:1-219`) compares
    orchestrator-on vs orchestrator-off, not classical vs ML.
    Greenfield.

  - **O6 — No fallback mechanism when model confidence is low.**
    `Predictor.predict` already has a four-source fallback chain
    (`predict.py:74-147`), but the *fallback to a different
    Predictor instance* does not exist. `OrchestratorCoordinator`
    holds a single `_predictor` reference (`coordinator.py:53`); no
    multiplexer / cascade exists. Greenfield.

  - **O7 — No model versioning.** `Predictor.__init__`
    (`predict.py:34-50`) takes only `max_transitions` and
    `min_confidence` — no version tag. No `get_version()` method.
    `OrchestratorConfig` (`memopt/orchestrator/config.py:23-37`)
    has no `predictor_model_version` field. Greenfield.

  - **O8 — No GPU acceleration for inference.** `Predictor` is pure
    Python under a `threading.RLock` (`predict.py:50, 53, 77`). No
    CUDA / Triton / accelerator code path exists in the orchestrator
    package. The C++ `csrc/core/oracle.cpp` is CPU-only too (no CUDA
    kernels). Greenfield.

  - **O9 — No eval harness with held-out workloads.** The bridge
    test (`tests/test_orchestrator_legacy_parity.py`) drives a single
    workload with seed `0xBE60E20` (line 23). The substrate has no
    workload library; `memopt/orchestrator/tests/` contains only
    unit-style coverage. Greenfield.

  - **O10 — No privacy / multi-tenancy guard for cross-tenant
    training.** `Predictor` enforces `(tenant, tag)` keying
    (`predict.py:80-92`) but provides no mechanism for *aggregating
    safely across tenants for model training*. Today's predictor is
    per-process and per-tenant. The substrate's G1 / G3 contracts
    (`docs/substrate_v1_threat_model.md` G1 + orchestrator §2.5 G1/G3)
    reject cross-tenant reads at the API surface but do not constrain
    what an in-process Layer-3 trainer would do with the events.
    Greenfield + design risk: explicitly called out as N1 in
    `docs/orchestrator_v1_threat_model.md` §N1 (cross-tenant pattern
    leakage via shared aggregation is OUT OF SCOPE today).

  - **O11 — No `record_outcome` feedback signal in Layer 2.** Legacy
    has `MemoryOracle.record_outcome` (`_oracle_py.py:182-188`);
    Layer 2's `Predictor` does not expose this method. The
    coordinator never calls back to the predictor with
    "your prediction was right / wrong." Without a label signal, an
    ML model can be trained on the access stream itself
    (next-token-prediction style) but cannot be supervised on
    prefetch correctness. Greenfield.

  - **O12 — No `read` event in the Layer 1 `Event` schema.**
    `events.py:33-46` enumerates exactly five kinds, none of which
    fire on data access. `docs/orchestrator_v1_design.md` §2.3.1
    (lines 1294-1297) calls this out as an intentional gap.
    Substrate change required if Layer 3 needs read-time signals.

---

### 1.5 Layer 2 / Layer 3 boundary

This subsection makes the boundary explicit so the design phase has
no ambiguity about who owns what.

#### 1.5.1 What Layer 2 owns (forever)

  - **Event subscription**: `OrchestratorCoordinator.start`
    (`coordinator.py:72-85`) fans out 5 `Dispatcher.subscribe` calls;
    `_enqueue` (`coordinator.py:122-135`) bounds the queue. Layer 3
    does NOT subscribe to substrate events directly.
  - **Decision pump**: `OrchestratorCoordinator._run`
    (`coordinator.py:139-158`) drives the cycle loop.
  - **Policy engine**: `memopt/orchestrator/policy.py` —
    PolicyEngine, conflict resolution, buggy-policy auto-unregister.
    Layer 3 may register a Policy but does not own the engine.
  - **Telemetry namespace**: `memopt/orchestrator/telemetry.py` —
    aggregate counters; G4-gated per-tenant counts.
  - **The `Predictor` protocol contract**: the public surface
    (observe / predict / forget_tenant / stats), the `Prediction`
    dataclass shape, the `(tenant, tag)` key invariant, the four
    source labels, the `min_confidence` cutoff. (Ratified in §1.1.)
  - **The classical Markov fallback at `predict.py`**: NEVER
    deleted. Always available as the safety net when Layer 3's
    model is unavailable, low-confidence, or fails to load.

#### 1.5.2 What Layer 3 owns

  - Model architecture (RNN / transformer / sequence model).
  - Training pipeline (data ingest, cleaning, batching, optimization).
  - Inference path (model invocation, batching, latency budget).
  - Model versioning + serialization format.
  - Eval harness (held-out workloads, accuracy metric, latency
    metric).
  - Per-tenant / cross-tenant training-data policy (must satisfy
    §2.5 G3 / N1 from the orchestrator threat model).
  - The Layer-3 implementation of the `Predictor` protocol.
  - Layer-3-specific config (extends `OrchestratorConfig` or a
    sibling).

#### 1.5.3 The interface

  Layer 3 ships a class — call it `OracleV1` (working name) — that
  exposes the same four methods as `memopt/orchestrator/predict.py`
  Predictor:

  ```python
  class OracleV1:
      def observe(self, tenant: str, tag: str) -> None: ...
      def predict(self, tenant: str, tag: str,
                  top_k: int = 10) -> List[Prediction]: ...
      def forget_tenant(self, tenant: str) -> None: ...
      def stats(self) -> dict: ...
  ```

  `Prediction` is the same frozen dataclass at `predict.py:23-28`.
  Layer 3 may add fields to its `stats()` return value (additive); it
  may NOT change the `Prediction` shape (frozen, contract).

  Per `docs/orchestrator_v1_design.md` §2.5 G3 the predictor's
  per-tenant key invariant is part of the contract. Layer 3's
  `forget_tenant(tenant)` MUST drop all per-tenant model state for
  that tenant (training cache, recent context, etc.).

  The wiring point is `_OrchestratorState.__init__`
  (`memopt/orchestrator/__init__.py:46-71`), which today instantiates
  `Predictor(...)` at line 53. Phase 2 will define how Layer 3's
  implementation is selected (env var? config field?). Layer 2 source
  changes are out of scope for Layer 3's Phase A — this is a future
  extension.

#### 1.5.4 What stays in Layer 2 forever (the safety net)

  The classical predictor at `memopt/orchestrator/predict.py` is the
  always-available fallback. It is:

  - Pure Python; no torch / numpy / sklearn dependency.
  - Tested by 13 tests in
    `memopt/orchestrator/tests/test_predictor.py` (REQUIRED-CI).
  - Bounded memory (`max_transitions`, `predict.py:48`).
  - Sub-millisecond per call (per `docs/orchestrator_v1_design.md`
    §2.6 target predictor query P50 < 20 µs).

  Layer 3 must never delete or rename this file. The Phase A
  contract here mirrors substrate v1's "legacy paths frozen during
  Phase A" model.

---

### 1.6 Test surface and regression net (876 baseline)

The Mac baseline that Layer 3's own Phase A must preserve.

#### 1.6.1 Recorded baseline

  Command (per `docs/orchestrator_v1_implementation_prompt.md` §A.2):

  ```
  PYTHONPATH=. python3 -m pytest tests/ memopt/ -q -p no:cacheprovider \
    -k 'not gpu and not cuda' \
    --ignore=memopt/vmm/tests/test_cuda_vmm_smoke.py \
    --ignore=memopt/vmm/tests/test_torch_allocator_sanity.py
  ```

  Result on Darwin 25.4.0 / Python 3.12 / no CUDA, run 2026-05-01
  at HEAD `2e2fd9a` (orchestrator-v1.0.0):

  ```
  876 passed, 34 skipped, 15 deselected, 8 warnings
  ```

  This is the contract: Layer 3 may grow PASSED only; SKIPPED /
  DESELECTED / FAILED must not regress.

#### 1.6.2 Critical regression net for Layer 3

  - **Layer 1 substrate tests** under `memopt/substrate/tests/`
    (counted into the 876). Untouched by Layer 3.
  - **Layer 2 orchestrator tests** under
    `memopt/orchestrator/tests/` — 13 of these
    (`test_predictor.py`) directly pin the protocol Layer 3
    implements. Untouched by Layer 3 source; Layer 3 may add
    sibling test files for its own implementation.
  - **Bridge test** `tests/test_orchestrator_legacy_parity.py:1-219`.
    Phase A assertions (lines 156-200) must still pass: per-tenant
    byte counts and event-drop counters byte-for-byte equal between
    orchestrator-stopped and orchestrator-on (observation-only)
    runs; `decisions == 0`. Layer 3 in observation-only mode must
    preserve the same parity.
  - **Legacy VMM oracle tests** `memopt/vmm/tests/test_oracle.py`,
    `test_access_log.py` — green forever per
    `docs/orchestrator_v1_design.md` §2.4.5 (lines 1555-1556).
    Layer 3 does not touch these.
  - **Decision-isolation tests**
    `memopt/orchestrator/tests/test_decision_d1_greenfield.py:1-67`
    asserts (via `ast` import scan) that the orchestrator does not
    import legacy VMM modules. Layer 3 inherits this discipline —
    its source files must pass the same scan.

#### 1.6.3 Categorization

  Per `docs/orchestrator_v1_implementation_prompt.md` §A.3 the
  regression buckets are:

  - **REQUIRED-CI** — runs on Mac/Linux without CUDA. The 876
    PASSED above are all in this bucket.
  - **REQUIRED-LOCAL @gpu** — needs a CUDA rig; skipped on Mac.
    2 such tests today
    (`test_migrate_va_preserved`,
    `test_peek_handle_safe_in_driver_callback`); skipped under
    `pytest.importorskip("torch") + torch.cuda.is_available()` gates.
  - **OPTIONAL @perf** — 10 perf microbenches under
    `memopt/orchestrator/tests/test_perf_microbench.py`; auto-skipped
    by `memopt/orchestrator/tests/conftest.py:6-15` unless `-m perf`
    is passed.

---

### 1.7 Open questions for Phase 2

These are the questions Phase 2 must decide. None are answered here
(per the non-negotiables — Phase 1 is read-only inventory, no design
opinions).

  - **Q1 — Model architecture.** RNN (LSTM/GRU)? Transformer?
    Simpler n-gram beyond the first-order Markov already in Layer 2?
    Each has different latency / memory / training-data envelopes.
    Trade-off: classical Layer-2 predictor query P50 < 20 µs
    (`docs/orchestrator_v1_design.md` §2.6) — what fraction of that
    budget can Layer 3 spend at inference?

  - **Q2 — Training location.** On-device (in the same Python
    process as the coordinator)? Separate Python worker process?
    Off-device cloud (uploading anonymized traces)? Cross-tenant
    risk varies by location; G3 / N1 from the orchestrator threat
    model bound the answer.

  - **Q3 — Inference location.** On the coordinator thread (must
    fit the 50 ms cycle budget per `coordinator.py:64`)? On a
    dedicated inference thread? On a per-tenant pool? On GPU?

  - **Q4 — Latency budget for ML inference.** Today's classical
    `predict()` is in the µs range. An RNN / transformer is not.
    What is the upper bound (µs / ms) before Layer 3 must cascade
    to the Layer-2 fallback?

  - **Q5 — Confidence-driven fallback.** When does Layer 3 yield
    to Layer 2's classical predictor? Mean confidence below
    threshold? No predictions above `min_confidence`? Cascade
    every call vs gate by some signal?

  - **Q6 — Pre-trained vs train-locally.** Does v1.0 ship
    pre-trained model checkpoints (and if so, which workloads)? Or
    does it ship only the training pipeline and require each user
    to bootstrap from their own access traces?

  - **Q7 — Cross-tenant training policy.** How does Layer 3 honor
    G3 (per-tenant key isolation, `predict.py:88-100`) while
    learning shared patterns useful across tenants? Per-tenant
    fine-tuning over a shared backbone? Disjoint per-tenant models?
    Differential-privacy-style noise on cross-tenant aggregations?

  - **Q8 — Minimum useful trace size.** What is the smallest
    workload (in events / unique tags / unique tenants) before
    Layer 3 begins to outperform Layer 2's classical predictor?
    Below that threshold, Layer 3 should silently defer to Layer 2
    rather than degrade.

  - **Q9 — Persistence across restarts.** `docs/orchestrator_v1_design.md`
    §3.7 explicitly excludes "persistence of learned patterns
    across restarts" from Phase A — Predictor starts cold every
    process. Does Layer 3 lift that restriction (load from disk on
    start)? If yes, where on disk, with what schema, and gated by
    which env var?

  - **Q10 — Substrate change for `read` events.** §1.2.5 / O12 —
    Layer 3 may need a `read` event the substrate does not emit.
    Phase 2 must decide whether to add it (substrate change,
    extends the long-term contract in
    `docs/orchestrator_v1_implementation_prompt.md` §H.4 "the five
    event kinds") or work around its absence.

  - **Q11 — `record_outcome` feedback (O11).** Phase 2 must
    decide whether Layer 3 grows the protocol with a
    `record_outcome` method (additive — backward-compatible;
    Layer 2's classical predictor would no-op it) or trains
    unsupervised on the raw access stream only.

---

*End of Phase 1. Phases 2 (Design), 3 (Test plan), and 4
(Implementation prompt) are deferred — to be drafted after user
review of this discovery section.*
