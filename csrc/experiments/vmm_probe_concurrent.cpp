// vmm_probe_concurrent.cpp — Phase 2 of the VMM go/no-go test.
//
// Question 1: does remapping a VA while an unrelated kernel is running on
//             another stream fault, or stall, or corrupt anything?
// Question 2: what's the latency budget for
//             (a) remap-only (no content move), and
//             (b) full eviction round trip (save DtoH + new handle + restore HtoD)?
//
// Design:
//   Stream A (busy): runs a burst of PTX "spin" kernels on a SEPARATE
//                    allocation. We queue ~5 seconds worth of work and
//                    let it drain while Stream B hammers the VA.
//   Stream B (remap): runs REMAP_CYCLES of cuMemUnmap + cuMemMap(same hA)
//                     + cuMemSetAccess. Measures host-side latency per cycle.
//                     Verifies the 2 MiB page's first u32 is still PATTERN
//                     after each cycle.
//
//   Then Phase 2: EVICT_CYCLES of the full eviction pipeline:
//     save (DtoH) → unmap → release old handle → create new handle →
//       map new → setAccess → restore (HtoD). Measures total + each step.
//
// Reordering note vs. the spec: the spec listed
//     cuMemUnmap → cuMemcpy VA→host → cuMemCreate → cuMemMap → setAccess
//   which is impossible (you can't memcpy from a VA with no backing).
//   Actual order used: save → unmap → release → create → map → setAccess →
//   restore. This is the only order that preserves contents.
//
// Uses the CUDA Driver API only (no cudart, no nvcc required at build time).

#include <cuda.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

// ── Test sizing ──────────────────────────────────────────────────────────
constexpr size_t PAGE_BYTES       = 2ull << 20;    // 2 MiB — VMM min granularity
constexpr size_t VA_RESERVE_BYTES = 1ull << 30;    // 1 GiB reservation
constexpr size_t BUSY_BUF_BYTES   = 512ull << 10;  // 512 KiB output for busy kernel
constexpr int    REMAP_CYCLES     = 1000;
constexpr int    EVICT_CYCLES     = 100;
constexpr unsigned int SPIN_ITERS = 10'000'000u;   // per-thread loop count
constexpr int    BUSY_LAUNCHES    = 500;           // ≈5 s of concurrent work
constexpr uint32_t PATTERN        = 0xDEADBEEFu;

// ── PTX source for a pure-compute busy kernel ────────────────────────────
//
// Each thread runs a tight mul/add loop for `spin_iters` iterations, then
// stores the final accumulator at out[idx]. Targets sm_80 PTX so it JITs
// on Ada (sm_89), Hopper (sm_90), and Blackwell (sm_100).
static const char* kBusyPtx = R"PTX(
.version 7.8
.target sm_80
.address_size 64

.visible .entry spin(
    .param .u32 spin_iters,
    .param .u64 out_addr
)
{
    .reg .pred  %p<2>;
    .reg .u32   %r<12>;
    .reg .u64   %rd<5>;

    ld.param.u32    %r1, [spin_iters];
    ld.param.u64    %rd1, [out_addr];

    // idx = ctaid.x * ntid.x + tid.x
    mov.u32         %r2, %ntid.x;
    mov.u32         %r3, %ctaid.x;
    mov.u32         %r4, %tid.x;
    mad.lo.u32      %r5, %r2, %r3, %r4;

    mov.u32         %r6, 0;                 // counter
    mov.u32         %r7, 1;                 // acc

$L_loop:
    mul.lo.u32      %r7, %r7, 2654435761;
    add.u32         %r7, %r7, %r6;
    add.u32         %r6, %r6, 1;
    setp.lt.u32     %p1, %r6, %r1;
    @%p1 bra        $L_loop;

    // out[idx] = acc
    mul.wide.u32    %rd2, %r5, 4;
    add.u64         %rd3, %rd1, %rd2;
    st.global.u32   [%rd3], %r7;
    ret;
}
)PTX";

// ── Error checking ───────────────────────────────────────────────────────
#define CU_CHECK(expr, step)                                                  \
    do {                                                                      \
        CUresult _r = (expr);                                                 \
        if (_r != CUDA_SUCCESS) {                                             \
            const char *_n="?", *_s="?";                                      \
            cuGetErrorName(_r, &_n); cuGetErrorString(_r, &_s);               \
            std::fprintf(stderr,                                              \
                "  [FAIL] step=%s cuda_err=%d (%s): %s\n",                    \
                step, static_cast<int>(_r), _n, _s);                          \
            return 1;                                                         \
        }                                                                     \
    } while (0)

// ── Timing + stats ───────────────────────────────────────────────────────
using clk = std::chrono::steady_clock;

double us_between(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

struct Stats { size_t n=0; double min=0, max=0, mean=0, p50=0, p99=0; };

Stats summarize(std::vector<double> v) {
    Stats s; s.n = v.size();
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    double sum = 0; for (double x : v) sum += x;
    s.min  = v.front();
    s.max  = v.back();
    s.mean = sum / v.size();
    s.p50  = v[v.size() / 2];
    size_t i99 = v.size() * 99 / 100; if (i99 >= v.size()) i99 = v.size() - 1;
    s.p99  = v[i99];
    return s;
}

void print_stats(const char* label, const Stats& s) {
    if (s.n == 0) { std::printf("  %-30s no samples\n", label); return; }
    std::printf("  %-30s n=%4zu  min=%8.2f  mean=%8.2f  p50=%8.2f  "
                "p99=%8.2f  max=%8.2f  (us)\n",
                label, s.n, s.min, s.mean, s.p50, s.p99, s.max);
}

// ── The actual probe ─────────────────────────────────────────────────────
int run() {
    CU_CHECK(cuInit(0), "cuInit");
    CUdevice dev; CU_CHECK(cuDeviceGet(&dev, 0), "cuDeviceGet");

    char name[128] = {0};
    cuDeviceGetName(name, sizeof(name), dev);
    int ccM = 0, ccm = 0;
    cuDeviceGetAttribute(&ccM, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&ccm, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    int n_sm = 0;
    cuDeviceGetAttribute(&n_sm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    std::printf("device: %s  sm_%d%d  multiprocessors=%d\n", name, ccM, ccm, n_sm);

    CUcontext ctx;
    CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, dev), "ctxRetain");
    CU_CHECK(cuCtxSetCurrent(ctx), "ctxSetCurrent");

    // JIT-compile PTX
    CUmodule mod;
    char jit_log[4096] = {0};
    CUjit_option jit_opts[] = { CU_JIT_ERROR_LOG_BUFFER,
                                CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES };
    void* jit_vals[] = { jit_log, reinterpret_cast<void*>(sizeof(jit_log)) };
    CUresult jr = cuModuleLoadDataEx(&mod, kBusyPtx, 2, jit_opts, jit_vals);
    if (jr != CUDA_SUCCESS) {
        const char *_n="?", *_s="?";
        cuGetErrorName(jr, &_n); cuGetErrorString(jr, &_s);
        std::fprintf(stderr, "  [FAIL] cuModuleLoadDataEx err=%d (%s): %s\n"
                     "  JIT log:\n%s\n", (int)jr, _n, _s, jit_log);
        return 1;
    }
    CUfunction fn_spin;
    CU_CHECK(cuModuleGetFunction(&fn_spin, mod, "spin"), "getFn");

    // VMM properties
    CUmemAllocationProp prop = {};
    prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id   = 0;
    CUmemAccessDesc acc = {};
    acc.location = prop.location;
    acc.flags    = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    // Reserve VA, create hA, map first 2 MiB, fill with PATTERN
    CUdeviceptr va = 0;
    CU_CHECK(cuMemAddressReserve(&va, VA_RESERVE_BYTES, 0, 0, 0), "reserveVA");
    CUmemGenericAllocationHandle hA = 0;
    CU_CHECK(cuMemCreate(&hA, PAGE_BYTES, &prop, 0), "create hA");
    CU_CHECK(cuMemMap(va, PAGE_BYTES, 0, hA, 0),     "map hA");
    CU_CHECK(cuMemSetAccess(va, PAGE_BYTES, &acc, 1),"access hA");

    CUstream sA, sB;
    CU_CHECK(cuStreamCreate(&sA, CU_STREAM_NON_BLOCKING), "streamA");
    CU_CHECK(cuStreamCreate(&sB, CU_STREAM_NON_BLOCKING), "streamB");

    CU_CHECK(cuMemsetD32Async(va, PATTERN, PAGE_BYTES / 4, sB), "prefill");
    CU_CHECK(cuStreamSynchronize(sB), "sync prefill");

    uint32_t verify = 0;
    CU_CHECK(cuMemcpyDtoH(&verify, va, 4), "verify pre");
    std::printf("prefill: VA[0]=0x%08x (expected 0x%08x) -> %s\n",
                verify, PATTERN, verify == PATTERN ? "OK" : "MISMATCH");
    if (verify != PATTERN) return 1;

    // Busy buffer (separate allocation — Stream A touches this, never the VMM VA)
    CUdeviceptr busy_buf = 0;
    CU_CHECK(cuMemAlloc(&busy_buf, BUSY_BUF_BYTES), "busyBufAlloc");
    CU_CHECK(cuMemsetD32Async(busy_buf, 0, BUSY_BUF_BYTES / 4, sA), "busyZero");

    // ═══════════════════════════════════════════════════════════════════
    // Phase 1: concurrent remap-only latency
    // ═══════════════════════════════════════════════════════════════════
    std::printf("\n=== Phase 1: %d remap cycles on Stream B; %d spin kernels "
                "on Stream A ===\n", REMAP_CYCLES, BUSY_LAUNCHES);

    int grid = 256, block = 128; // 32768 threads × 4B = 128 KiB output
    unsigned int spin_iters = SPIN_ITERS;
    void* spin_args[] = { &spin_iters, &busy_buf };

    auto busy_start = clk::now();
    for (int i = 0; i < BUSY_LAUNCHES; ++i) {
        CU_CHECK(cuLaunchKernel(fn_spin, grid, 1, 1, block, 1, 1,
                                0, sA, spin_args, nullptr),
                 "launch spin");
    }
    std::printf("queued %d spin kernels on Stream A  (grid=%d block=%d iters=%u)\n",
                BUSY_LAUNCHES, grid, block, spin_iters);

    CUresult qA0 = cuStreamQuery(sA);
    std::printf("cuStreamQuery(sA) right after enqueue = %d  %s\n", (int)qA0,
                qA0 == CUDA_ERROR_NOT_READY ? "(NOT_READY — expected, stream is busy)"
                : qA0 == CUDA_SUCCESS       ? "(already drained — kernels too short)"
                : "(UNEXPECTED)");

    std::vector<double> remap_us; remap_us.reserve(REMAP_CYCLES);
    int faults = 0, mismatches = 0;
    bool aborted = false;

    auto phase1_t0 = clk::now();
    for (int i = 0; i < REMAP_CYCLES; ++i) {
        auto t0 = clk::now();
        CUresult r1 = cuMemUnmap(va, PAGE_BYTES);
        CUresult r2 = cuMemMap(va, PAGE_BYTES, 0, hA, 0);
        CUresult r3 = cuMemSetAccess(va, PAGE_BYTES, &acc, 1);
        auto t1 = clk::now();

        if (r1 != CUDA_SUCCESS || r2 != CUDA_SUCCESS || r3 != CUDA_SUCCESS) {
            faults++;
            const char *n1="?", *n2="?", *n3="?";
            cuGetErrorName(r1,&n1); cuGetErrorName(r2,&n2); cuGetErrorName(r3,&n3);
            std::fprintf(stderr, "  cycle %d FAULT: unmap=%d(%s) map=%d(%s) access=%d(%s)\n",
                         i, (int)r1, n1, (int)r2, n2, (int)r3, n3);
            aborted = true;
            break;
        }
        remap_us.push_back(us_between(t0, t1));

        uint32_t v = 0;
        CUresult r4 = cuMemcpyDtoH(&v, va, 4);
        if (r4 != CUDA_SUCCESS || v != PATTERN) {
            mismatches++;
            if (mismatches <= 5) {
                std::fprintf(stderr, "  cycle %d: readback=0x%08x (want 0x%08x) r4=%d\n",
                             i, v, PATTERN, (int)r4);
            }
        }
    }
    double phase1_s = us_between(phase1_t0, clk::now()) / 1e6;

    CUresult qA1 = cuStreamQuery(sA);
    std::printf("\nafter %d cycles in %.3f s:  cuStreamQuery(sA) = %d  %s\n",
                REMAP_CYCLES, phase1_s, (int)qA1,
                qA1 == CUDA_ERROR_NOT_READY ? "(still running — expected)"
                : qA1 == CUDA_SUCCESS       ? "(drained during remap loop)"
                :                             "(UNEXPECTED — possible fault)");

    CUresult syncA = cuStreamSynchronize(sA);
    double busy_total_s = us_between(busy_start, clk::now()) / 1e6;
    const char *syncA_n="?", *syncA_s="?";
    cuGetErrorName(syncA, &syncA_n); cuGetErrorString(syncA, &syncA_s);
    std::printf("cuStreamSynchronize(sA) = %d (%s) after %.3f s total\n",
                (int)syncA, syncA_n, busy_total_s);

    // Sanity check: the busy kernels actually wrote output
    std::vector<uint32_t> busy_sample(16, 0);
    cuMemcpyDtoH(busy_sample.data(), busy_buf, sizeof(uint32_t) * busy_sample.size());
    int nonzero = 0;
    for (uint32_t x : busy_sample) if (x != 0) nonzero++;
    std::printf("busy buffer sample (first 16 words): %d/16 non-zero -> %s\n",
                nonzero, nonzero > 0 ? "kernels ran" : "SUSPICIOUS (none ran?)");

    std::printf("\n--- remap-only latency (cuMemUnmap + cuMemMap + cuMemSetAccess) ---\n");
    print_stats("remap_pair", summarize(remap_us));
    std::printf("  faults=%d  readback_mismatches=%d\n", faults, mismatches);

    if (aborted) {
        std::fprintf(stderr, "\n[ABORT] Phase 1 faulted — not proceeding to Phase 2.\n");
        return 1;
    }

    // ═══════════════════════════════════════════════════════════════════
    // Phase 2: full eviction round trip
    // ═══════════════════════════════════════════════════════════════════
    std::printf("\n=== Phase 2: %d full eviction round trips "
                "(save DtoH + unmap + release + create + map + access + restore HtoD) ===\n",
                EVICT_CYCLES);

    void* host_pinned = nullptr;
    CU_CHECK(cuMemHostAlloc(&host_pinned, PAGE_BYTES, CU_MEMHOSTALLOC_PORTABLE),
             "hostAlloc");

    std::vector<double> ev_total, ev_save, ev_unmap, ev_release,
                         ev_create, ev_map, ev_access, ev_restore;
    ev_total.reserve(EVICT_CYCLES);
    ev_save.reserve(EVICT_CYCLES);
    ev_unmap.reserve(EVICT_CYCLES);
    ev_release.reserve(EVICT_CYCLES);
    ev_create.reserve(EVICT_CYCLES);
    ev_map.reserve(EVICT_CYCLES);
    ev_access.reserve(EVICT_CYCLES);
    ev_restore.reserve(EVICT_CYCLES);

    CUmemGenericAllocationHandle h_cur = hA;
    int evict_mismatches = 0;

    for (int i = 0; i < EVICT_CYCLES; ++i) {
        auto t0 = clk::now();

        // 1. save contents D→H
        CU_CHECK(cuMemcpyDtoHAsync(host_pinned, va, PAGE_BYTES, sB), "save");
        CU_CHECK(cuStreamSynchronize(sB), "sync save");
        auto t_save = clk::now();

        // 2. unmap (detach physical from VA)
        CU_CHECK(cuMemUnmap(va, PAGE_BYTES), "evict unmap");
        auto t_unmap = clk::now();

        // 3. release old physical (return HBM to driver's pool)
        CU_CHECK(cuMemRelease(h_cur), "evict release");
        auto t_rel = clk::now();

        // 4. create fresh physical handle
        CUmemGenericAllocationHandle h_new = 0;
        CU_CHECK(cuMemCreate(&h_new, PAGE_BYTES, &prop, 0), "evict create");
        auto t_cre = clk::now();

        // 5. map new handle at same VA
        CU_CHECK(cuMemMap(va, PAGE_BYTES, 0, h_new, 0), "evict map");
        auto t_map = clk::now();

        // 6. set access
        CU_CHECK(cuMemSetAccess(va, PAGE_BYTES, &acc, 1), "evict access");
        auto t_acc = clk::now();

        // 7. restore H→D
        CU_CHECK(cuMemcpyHtoDAsync(va, host_pinned, PAGE_BYTES, sB), "restore");
        CU_CHECK(cuStreamSynchronize(sB), "sync restore");
        auto t_rest = clk::now();

        ev_total.push_back(  us_between(t0,      t_rest));
        ev_save.push_back(   us_between(t0,      t_save));
        ev_unmap.push_back(  us_between(t_save,  t_unmap));
        ev_release.push_back(us_between(t_unmap, t_rel));
        ev_create.push_back( us_between(t_rel,   t_cre));
        ev_map.push_back(    us_between(t_cre,   t_map));
        ev_access.push_back( us_between(t_map,   t_acc));
        ev_restore.push_back(us_between(t_acc,   t_rest));

        h_cur = h_new;

        // Verify first word still matches (proves save+restore preserved content)
        uint32_t v = 0;
        cuMemcpyDtoH(&v, va, 4);
        if (v != PATTERN) {
            evict_mismatches++;
            if (evict_mismatches <= 5) {
                std::fprintf(stderr, "  evict cycle %d: readback=0x%08x (want 0x%08x)\n",
                             i, v, PATTERN);
            }
        }
    }

    Stats s_total   = summarize(ev_total);
    Stats s_save    = summarize(ev_save);
    Stats s_unmap   = summarize(ev_unmap);
    Stats s_rel     = summarize(ev_release);
    Stats s_cre     = summarize(ev_create);
    Stats s_map     = summarize(ev_map);
    Stats s_acc     = summarize(ev_access);
    Stats s_rest    = summarize(ev_restore);

    std::printf("\n--- full eviction round trip ---\n");
    print_stats("evict_total", s_total);
    std::printf("\n--- breakdown per step ---\n");
    print_stats("  save    (DtoH 2 MiB)", s_save);
    print_stats("  unmap", s_unmap);
    print_stats("  release (old hA)", s_rel);
    print_stats("  create  (new handle)", s_cre);
    print_stats("  map     (new handle)", s_map);
    print_stats("  set_access", s_acc);
    print_stats("  restore (HtoD 2 MiB)", s_rest);
    std::printf("  evict_restore_mismatches=%d\n", evict_mismatches);

    double mean_bytes_moved = 2.0 * PAGE_BYTES; // DtoH + HtoD
    double throughput_gib_s = (mean_bytes_moved / (s_total.mean * 1e-6))
                              / double(1ull << 30);
    std::printf("\nthroughput: %.3f GiB/s  (moved %.1f MiB per cycle, mean %.2f us)\n",
                throughput_gib_s, mean_bytes_moved / (1024.0 * 1024.0), s_total.mean);

    // ── Verdict ─────────────────────────────────────────────────────────
    std::printf("\n=== verdict ===\n");
    std::printf("remap-only mean:       %8.2f us  (pure VMM overhead, no DMA)\n",
                summarize(remap_us).mean);
    std::printf("full eviction mean:    %8.2f us  (with 2 MiB DtoH + 2 MiB HtoD)\n",
                s_total.mean);
    if (s_total.mean < 100.0)
        std::printf("  <  100 us    → transparent eviction viable at token-gen frequency\n");
    else if (s_total.mean < 1000.0)
        std::printf("  100us–1ms    → viable BETWEEN decode steps, not inside them\n");
    else
        std::printf("  >  1 ms      → too slow for transparent use; need app cooperation\n");
    std::printf("concurrent faults:    %d   readback mismatches: %d   evict mismatches: %d\n",
                faults, mismatches, evict_mismatches);

    // ── Cleanup ─────────────────────────────────────────────────────────
    cuMemUnmap(va, PAGE_BYTES);
    cuMemRelease(h_cur);
    cuMemAddressFree(va, VA_RESERVE_BYTES);
    cuMemFree(busy_buf);
    cuMemFreeHost(host_pinned);
    cuStreamDestroy(sA);
    cuStreamDestroy(sB);
    cuModuleUnload(mod);
    cuDevicePrimaryCtxRelease(dev);
    return 0;
}

} // namespace

int main() {
    std::printf("=== MEMOPT CUDA VMM concurrent probe ===\n");
    return run();
}
