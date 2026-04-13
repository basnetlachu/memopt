# Hardware Support Matrix

Honest status. No marketing. No performance claims without a
measurement source. When something is labeled `Untested`, it means
the code compiles / imports but no one has run it against the
target hardware.

## Status definitions

| Status       | Meaning                                                |
|--------------|--------------------------------------------------------|
| Implemented  | Code written and tested on hardware                    |
| Validated    | Code tested, performance measured                      |
| Stub         | Interface defined, no real implementation              |
| Planned      | On roadmap, not started                                |
| Untested     | Code exists but never exercised against the target HW  |

---

## NVIDIA CUDA

| Component          | Status       | Notes                                  |
|--------------------|--------------|----------------------------------------|
| HAL detection      | Implemented  | pynvml primary, torch.cuda fallback    |
| VMM tiers          | Implemented  | HBM → DRAM → NVMe                       |
| C++ data plane     | Implemented  | sm_86 / sm_90 / sm_100                 |
| Paged attention    | Implemented  | gather kernel                          |
| GDS NVMe           | Implemented  | CUDA >= 11.8 required                  |
| RDMA transport     | Implemented  | Untested on bare metal                 |
| GKD                | Implemented  | Used in every test run                 |
| Silicon cert       | Validated    | A100, Blackwell lab harness            |
| Performance        | Untested     | No external benchmark yet              |

Datasheet-only `GPU_SPECS` entries (`measured=False`): H100, H100-SXM5,
H100-PCIe, H200, A100, A100-SXM4-80GB, A100-PCIe, A30, and all RTX.
Replacing `measured=False` with real numbers is a per-GPU task run
on actual hardware.

---

## AMD ROCm

| Component          | Status       | Notes                                    |
|--------------------|--------------|------------------------------------------|
| HAL detection      | Implemented  | rocm-smi primary, torch HIP fallback     |
| VMM tiers          | Implemented  | Python fallback works without ROCm SDK   |
| C++ data plane     | Implemented  | gfx90a / gfx940 / gfx941 / gfx942        |
| Unified memory     | Implemented  | gfx94x only (MI300X unified HBM)         |
| RDMA transport     | Stub         | No HIP RDMA code path yet                |
| GKD                | Implemented  | Shared transport / hashing with CUDA     |
| Silicon cert       | Stub         | Kernel battery is CUDA-specific today    |
| Performance        | Untested     | No AMD hardware access yet               |

**Honest note:** the AMD backend has never been run on real hardware.
The Python shim is exercised in CI on every PR (fallback path), and
`csrc/rocm/rocm_backend.cpp` compiles to a stub on CUDA-only hosts.
Real validation requires an AMD MI300X (or equivalent) system with
ROCm 5.x+ installed.

GPU_SPECS entries (all `measured=False`, sourced from datasheets):
- `AMD Instinct MI300X` — 5300 GB/s HBM3, 192 GB
- `AMD Instinct MI250X` — 3276.8 GB/s HBM2e, 128 GB
- `AMD Instinct MI210` — 1638.4 GB/s HBM2e, 64 GB

---

## Intel Gaudi

| Component          | Status   | Notes                                      |
|--------------------|----------|--------------------------------------------|
| HAL detection      | Stub     | `habana_frameworks` / `/dev/accel/` / `hl-smi` |
| VMM tiers          | Stub     |                                            |
| C++ data plane     | Planned  | No SynapseAI wrappers yet                  |
| GKD                | Planned  |                                            |
| Silicon cert       | Planned  | Needs Gaudi-native correctness battery      |

Stub class: `memopt.vmm.backends._gaudi_backend_py.GaudiBackend`.
All memory operations return `None` / `False` with a warning log.
When Gaudi hardware and SynapseAI become available, replace the stub
methods with `habana_frameworks.torch` equivalents (see the stub's
docstring for the concrete API map).

---

## Google TPU

| Component          | Status   | Notes                                       |
|--------------------|----------|---------------------------------------------|
| HAL detection      | Stub     | JAX / torch_xla TPU device probe             |
| VMM tiers          | Stub     | XLA memory model is not imperative           |
| GKD                | Planned  | Needs JAX-array-aware hash/lookup path       |
| Silicon cert       | Planned  |                                             |

Stub class: `memopt.vmm.backends._tpu_backend_py.TPUBackend`. TPU VMs
have no NVMe tier — replace the "NVMe" rung with GCS or Persistent
Disk when serious TPU support is implemented.

---

## CPU-only (fallback)

Every control-plane and serving feature that doesn't strictly require
GPU memory runs in CPU-only mode:

- GKD store (`Redis` / `ScyllaDB` / local backend)
- Oracle transition tracking
- Control plane (FastAPI)
- Kubernetes operator (Phase 5b)
- Canary rollout controller (Phase 7a)
- Federation gossip
- Heartbeat batcher (Phase 8a)
- Boot / PXE / image-version endpoints

Performance without GPU:
- KV cache lives in DRAM only — no HBM tier
- NVMe tier still available
- End-to-end inference is CPU-bound and not useful for production.
  The value of CPU-only mode is that it lets the control-plane,
  operator, and test suite run on any developer laptop.

---

## Adding a new hardware backend

Checklist — keep in the same order as the existing backends:

1. **HAL detection** (`memopt/vmm/hal.py`)
   - Add a `HardwareBackend` enum value
   - Add a `_try_detect_X()` method
   - Call it from `_detect()` in priority order

2. **Python backend** (`memopt/vmm/backends/_X_backend_py.py`)
   - Implement `allocate_hbm`, `free_hbm`, `copy_to_device`,
     `copy_from_device`, `synchronize`, `stats`
   - Implement `detect_tiers()` returning
     `[MemoryTier("hbm"), MemoryTier("dram"), MemoryTier("nvme")]`

3. **C++ backend** (`csrc/X/X_backend.cpp`)
   - Match the pybind11 surface of `_memopt_rocm` (alloc/free/copy/sync)
   - Guard the real code behind `#ifdef MEMOPT_X_AVAILABLE`; compile
     a stub module otherwise so `import memopt._memopt_X` always
     succeeds

4. **CMake option** (`csrc/CMakeLists.txt`)
   - `option(MEMOPT_ENABLE_X "..." OFF)`
   - `find_package()` the vendor SDK; fall back to stub if missing

5. **Hardware specs** (`memopt/profiler/hardware_counters.py`)
   - Add `GPU_SPECS["X"]` entry with `vendor`, `architecture`,
     `arch_tag`, `hbm_size_gb`, `measured=False`, and `source`
     pointing at the datasheet URL / PDF.

6. **Silicon cert profile** (`memopt/kernels/certification.py`)
   - Add a correctness battery (rope / layer-norm / softmax) and a
     throughput battery (memcpy / matmul) for the new arch

7. **Update this matrix** with honest status columns.
