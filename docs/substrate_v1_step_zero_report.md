# Substrate v1 — Step Zero Verification Report

| Field        | Value                                                       |
| ------------ | ----------------------------------------------------------- |
| Run by       | Lachu Man Basnet                                            |
| Run date     | 2026-04-29                                                  |
| Test rigs    | Lachus-MacBook-Pro.local (Darwin 25.4.0, arm64, Apple Silicon) — single dev host. No NVIDIA GPU, no AMD GPU, no CXL hardware, no Linux `/sys` interface available. |
| CUDA version | N/A (no nvcc on this host; CUDA-side checks deferred to a GPU rig pre-Commit 5/10) |
| ROCm version | N/A (no `/opt/rocm`; HIP-side checks deferred to an AMD rig pre-Commit 6) |
| Python       | Python 3.12.6                                               |
| PyTorch      | torch 2.9.1, `torch.cuda.is_available()` = False             |

> Honesty note: this is a macOS Apple-Silicon dev box without GPU, AMD, CXL, or
> Linux. Five of the eight items (S0.1, S0.2, S0.4, S0.5, S0.8) require
> hardware or kernel surfaces this host does not have, and one (S0.3) is a
> deep PyTorch upstream source check that we approximated via a single
> WebFetch summarizer pass rather than a pinned-commit grep. The honest
> outcome is therefore DEGRADED on six items, with each fallback being the
> documented one in design §3.0. None of the failures are S0.3 *mismatches*
> (we found no contradiction with §2.5; we just couldn't fully verify
> everything from this host), so the redesign protocol B.4 is not triggered.
> The verifications marked DEGRADED-deferred MUST be re-run on the
> appropriate rig before the corresponding commit lands (cited in each
> NEXT line).

---

## S0.1 — CUDA fabric handle (G5)

STATUS:   DEGRADED

EVIDENCE:
- `which nvcc` → not found.
- `$CUDA_HOME` is unset. No `cuda.h` available on host.
- Could not run the `grep -nE "CU_MEM_HANDLE_TYPE_FABRIC|cuMemExportToShareableHandle"` step from §3.0 S0.1 Method (1).
- Could not build/run the cuMemCreate + cuMemExportToShareableHandle probe from Method (2) on either an IMEX or non-IMEX host.
- NVIDIA documents `CU_MEM_HANDLE_TYPE_FABRIC` and `cuMemExportToShareableHandle` in the CUDA Driver API VA group (cited at https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html); no contradicting evidence found.

IMPACT:   §2.3 (BackendStrategy.export_fabric_handle) and §2.4 (CUDABackend) — both are written to tolerate a None return, so behaviour is unchanged on a non-fabric host. No design-doc edit needed.

NEXT STEP: Apply §3.0 S0.1 fallback bullet 1 ("CUDA <12.4 only on test rigs: keep `export_fabric_handle()` but it returns None unconditionally; no design change"). Re-run the probe on a CUDA-12.4+ rig (and an IMEX-enabled host if available) BEFORE Commit 5 lands; record results in the Commit-5 PR description. If the probe fails on CUDA ≥12.4 with an unexpected CUresult, escalate per §3.0 S0.1 fallback bullets 2 or 3.

---

## S0.2 — HIP VMM capability and granularity (G8)

STATUS:   DEGRADED

EVIDENCE:
- `ls /opt/rocm` → "No such file or directory".
- No `hip_runtime_api.h` on host.
- Could not run `grep -nE "hipMemGetAllocationGranularity|hipDeviceAttributeVirtualMemory" /opt/rocm/include/hip/*.h` (Method 1) or build the HIP probe (Method 2).
- No AMD CI rig is named in this environment; Method 3 is not executable here.

IMPACT:   §2.4 HIPBackend. Per the §3.0 S0.2 fallback, if the capability bit cannot be confirmed VERIFIED on at least one MI300X/MI200 rig, Commit 6 must ship the same-shape stub as LevelZeroBackend (raises NotImplementedError on every method except `is_available()`).

NEXT STEP: Apply §3.0 S0.2 fallback bullet 2 by default ("HIPBackend ships as the same-shape stub … future ROCm releases may flip the bit"). The Commit 6 prompt explicitly branches on this report's S0.2 status — current entry directs the prompt down the stub path. If an AMD rig becomes available, re-run the probe before Commit 6 and update this section to VERIFIED to flip the prompt back to the real-impl path.

---

## S0.3 — PyTorch CCA semantics (G3, DECISION 2)

STATUS:   DEGRADED-deferred

EVIDENCE:
- Local `torch.__version__` = 2.9.1; this is the version that will run on the test rigs.
- Best-effort doc-only check via WebFetch on
  https://raw.githubusercontent.com/pytorch/pytorch/main/c10/cuda/CUDACachingAllocator.cpp
  (i.e. the `main` branch HEAD as of 2026-04-29; the exact upstream commit
  hash was not pinned this session and MUST be pinned before Commit 10):
    - `record_stream`         → present (~line 3847)
    - `stream_uses`           → present (~line 322)
    - `event_pool`            → present (~line 1460)
    - `BlockPool`             → present (~line 446); `large_blocks` and
                                 `small_blocks` instances (~lines 2667–2668)
    - Size-class scheme       → confirmed: small (≤ 1 MiB, packed into 2 MiB
                                 buffers) vs large (> 1 MiB) — matches the
                                 conservative-rounding tolerance §2.5 cites.
    - `pending_events`        → not surfaced by the summarizer pass; the
                                 concept is implied by `event_pool` + record-
                                 stream code paths in §3847+ but a direct
                                 grep on a clone is required to confirm.
    - `get_allocation_size`   → not surfaced by the summarizer pass; may be
                                 inlined or renamed in 2026-vintage CCA.
                                 Direct grep required.
    - `cudaStreamIsCapturing` → not surfaced; likely lives on the runtime
                                 API path (`cuda_runtime_api.h` consumer
                                 sites in CCA) rather than CCA core. Direct
                                 grep required.
- No mismatch between observed CCA behaviour and §2.5 was found; the
  stream-locked free / record_stream / event-drain semantic in §2.5
  matches the structure observed in the summarized CCA.
- The E1–E9 audit in §2.5 was NOT exhaustively re-walked against the
  pinned-commit source this session.

IMPACT:   §2.5 StreamRegistry (foundational). No design contradiction was
observed, so B.4 redesign protocol is NOT triggered (the protocol fires on
*mismatch*, not on incomplete verification). However, partial verification
is not sufficient to claim VERIFIED for a hard-block item.

NEXT STEP: Re-run S0.3 on a Linux + CUDA + GPU rig BEFORE Commit 10 lands,
with these concrete steps:
  (a) Pin the exact PyTorch upstream commit hash that the rig's installed
      torch was built from (`torch.version.git_version` if exposed, else
      record `torch.__version__` and the PyTorch tag for that release).
  (b) Clone PyTorch at that commit and run the literal greps from §3.0 S0.3
      Method (2) for `pending_events`, `get_allocation_size`, and
      `cudaStreamIsCapturing` (treat renames as VERIFIED-with-rename).
  (c) Walk E1–E9 in §2.5 against the actual code paths and add E10+ for any
      new edge case observed.
  (d) If a *semantic* mismatch is found, halt Commit 10 and follow B.4. If
      only renames are found, update §2.5 citations and proceed.

This DEGRADED-deferred status is not a HALT — it is an explicit deferral to
the rig where StreamRegistry will actually be exercised. Commit 10 is gated
on this re-run completing successfully.

---

## S0.4 — CUDA Graphs capture APIs (E1 in §2.5)

STATUS:   DEGRADED

EVIDENCE:
- No `cuda_runtime_api.h` on host (no CUDA toolkit installed).
- Could not run `grep -nE "cudaStreamIsCapturing|cudaStreamCaptureStatus" $CUDA_HOME/include/cuda_runtime_api.h` (Method 1) or check the three enum members.
- NVIDIA's CUDA Runtime API stream-management group documents these symbols (https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html); no evidence of removal/rename in the docs cutoff knowledge available in this session.

IMPACT:   §2.5 E1 (CUDA Graphs capture handling). No code change implied; the symbol set is expected to be present.

NEXT STEP: Apply §3.0 S0.4 fallback bullet 1 by default ("APIs renamed: update §2.5 E1 to cite current symbols"). Run the literal grep on a CUDA-equipped rig before Commit 10 (same gate as S0.3), since both feed the StreamRegistry. If the APIs are absent, escalate per fallback bullet 2 (redesign E1 to disable capture-aware behaviour and document the regression in §2.11).

---

## S0.5 — CXL detection heuristic (G6)

STATUS:   DEGRADED

EVIDENCE:
- Host is macOS (Darwin 25.4.0); `/sys/devices/system/node` does not exist.
- No CXL deployment available in this environment.
- Could not run the cpulist-empty + non-zero-MemTotal heuristic (Method 1) or check `/sys/bus/cxl/` alignment (Method 2) or the no-false-positive control on a non-CXL Linux host (Method 3).

IMPACT:   §2.4 CXLBackend. Auto-detection must be disabled by default in v1.

NEXT STEP: Apply §3.0 S0.5 fallback bullet 3 ("No CXL deployment available for empirical check: ship CXLBackend with auto-detection disabled and gate it behind `MEMOPT_CXL_NODES`. Mark G6 as 'partial v1; auto-detect deferred to v1.1'."). The Commit 8 prompt already branches on this: with the current report status, Commit 8 ships the env-var-gated form. If a CXL host becomes available before Commit 8 lands, re-run the heuristic and update this section.

---

## S0.6 — Event ring atomicity primitive (§2.5 EventRing)

STATUS:   DEGRADED

EVIDENCE:
- `python3 -c "import threading; print(hasattr(threading, 'atomic'))"` → `False` on Python 3.12.6 (the actual interpreter on the dev rig).
- Confirms the §3.0 S0.6 HONEST NOTE: `threading.atomic` is NOT in stdlib in 3.12, and not available as a usable abstraction in the project's support matrix (3.10–3.13).
- Of the three alternatives in Method (2):
    - (a) `ctypes.c_uint64` + platform atomic intrinsics: requires per-platform code; not portable to Apple Silicon ARM64 + Linux x86_64 + Linux aarch64 uniformly without a third-party dep — rejected.
    - (b) `threading.Lock` around head/tail integers: works on every supported Python; well-defined semantics; ~50 ns per emission cited in the design — selected.
    - (c) Single-producer-single-consumer lock-free ring relying on bytecode atomicity: PEP 703 makes this fragile under free-threaded CPython; the HONEST NOTE explicitly warns against it — rejected.
- Microbenchmark of the chosen primitive at < 200 ns on a real test rig was NOT run (the µbench rig is the GPU host; this is a dev box).

IMPACT:   §2.5 EventRing primitive choice = `threading.Lock`. §2.10 event-emission target is at risk of relaxing to "< 1 µs" if the µbench misses 200 ns; the ordering invariant (`event_emit < alloc_warm`) still holds either way.

NEXT STEP: Apply §3.0 S0.6 fallback line 1 conditionally ("All three alternatives miss the < 200 ns target: relax the §2.10 target for event emission to '< 1 µs' and document the reason"). Concretely, Commit 11 implements EventRing with `threading.Lock`. The µbench (REQUIRED-LOCAL on Commit 11) will measure latency on the GPU rig; if the Lock-based ring misses 200 ns there, edit §2.10 to relax the event-emission target to < 1 µs and record the bench numbers in the Commit 11 PR description.

---

## S0.7 — CVE identifiers (§2.6 N3)

STATUS:   VERIFIED

EVIDENCE: Each CVE looked up at https://nvd.nist.gov/vuln/detail/<id> via WebFetch (2026-04-29):

- CVE-2025-23266 — present on NVD. NVIDIA Container Toolkit, allows an attacker to "execute arbitrary code with elevated permissions". Container-level vulnerability — driver-/container-adjacent and relevant to §2.6 N3's claim that container-escape / driver-level vulnerabilities are out of scope of memopt's tenant model.
- CVE-2025-33220 — present on NVD. NVIDIA vGPU software, heap use-after-free in the Virtual GPU Manager (CWE-416). Memory-safety issue affecting GPU memory subsystem — directly relevant to N3.
- CVE-2025-23352 — present on NVD. NVIDIA vGPU software, uninitialized pointer access in Virtual GPU Manager — GPU/memory-relevant.

All three CVEs resolve, are NVIDIA-published, and align with the §2.6 N3 framing ("driver-level / vGPU-level / container-level vulnerabilities are out of scope"). No replacement or removal needed.

IMPACT:   None — §2.6 N3 stands as written. Commit 14 (threat model doc) cites these three identifiers verbatim.

NEXT STEP: Nothing. The Commit 14 threat-model doc cites the verified set as-is.

---

## S0.8 — MI300X granularity (§2.7)

STATUS:   DEGRADED

EVIDENCE:
- No AMD GPU available on this host. Cannot run the HIP probe from S0.2 to capture `hipMemGetAllocationGranularity(MINIMUM)` and `(RECOMMENDED)`.
- §2.7's "2 MiB typical" assumption is documented in the design from public ROCm sources; we have no contradicting empirical data, but also no on-rig confirmation.

IMPACT:   §2.7 (granularity policy doc) and §2.9 (bridge-test workload sizes). HIPBackend already uses the runtime probe value rather than a hardcoded constant, so there is no code-behaviour risk; this is purely a doc-claim verification.

NEXT STEP: Apply §3.0 S0.8 fallback bullet 2 ("No MI300X available for the probe: run probe on whatever AMD card we have … and tag the result with the hardware. Mark §2.7 with a TODO-NEXT-RIG note for when MI300X access is obtained"). Concretely, when an AMD rig becomes available (whether MI300X / MI200 / RX 7900 XT), capture both granularity values, write them into §2.7 with a hardware tag, and adjust the §2.9 bridge-test workload sizes if the actual granularity differs from 2 MiB by more than a power-of-two factor.

---

## Summary

DEGRADED with 6 items (S0.1, S0.2, S0.3, S0.4, S0.5, S0.6, S0.8 — that's
7; S0.7 is the only fully-VERIFIED item).

Wait — recount: VERIFIED = {S0.7}. DEGRADED = {S0.1, S0.2, S0.3, S0.4,
S0.5, S0.6, S0.8} = 7 items. FAILED = {} = 0 items.

**SUMMARY: DEGRADED with 7 items.**

No FAILED items, no S0.3 *mismatch* — the B.4 redesign protocol is NOT
triggered. Each DEGRADED item has the documented §3.0 fallback recorded in
its NEXT STEP, and several (S0.1, S0.3, S0.4, S0.6, S0.8) require
re-verification on the appropriate rig before specific commits land
(citations inline). Phase A may proceed to Commit 1 with these caveats
recorded.
