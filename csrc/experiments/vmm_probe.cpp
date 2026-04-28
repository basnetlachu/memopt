// vmm_probe.cpp — Prove the CUDA VMM primitive works on this GPU/driver.
//
// This is the go/no-go test for MEMOPT's transparent HBM remapping design.
// If every step passes: VA↔PA decoupling works, we can build the allocator.
// If any step fails: the error code + step tells us exactly what's blocked.
//
// Test plan:
//   1. cuMemAddressReserve 1 GiB of virtual address space.
//   2. cuMemCreate two 512 MiB physical HBM handles: A and B.
//   3. Map A at VA[0..512MiB], fill with pattern 0xAAAAAAAA.
//   4. Sync, read back — must see 0xAAAAAAAA.
//   5. Unmap A, map B at the same VA, fill with 0xBBBBBBBB.
//   6. Sync, read back — must see 0xBBBBBBBB.
//   7. Unmap B, remap A at the same VA.
//   8. Read back — must see 0xAAAAAAAA again (physical A's contents
//      must survive while it was unmapped).
//
// Build:  see CMakeLists.txt in this directory.
// Run:    ./vmm_probe [device_id]     (default device 0)
//
// Uses only the CUDA Driver API (libcuda). No nvcc required.
// Kernel launches happen inside cuMemsetD32Async — same code path as
// any other kernel touching these VAs, so this is a faithful test.

#include <cuda.h>

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr size_t VA_RESERVE_BYTES  = 1ull << 30;   // 1 GiB
constexpr size_t CHUNK_BYTES       = 512ull << 20; // 512 MiB each
constexpr uint32_t PATTERN_A       = 0xAAAAAAAAu;
constexpr uint32_t PATTERN_B       = 0xBBBBBBBBu;

#define CU_CHECK(expr, step_name)                                             \
    do {                                                                      \
        CUresult _r = (expr);                                                 \
        if (_r != CUDA_SUCCESS) {                                             \
            const char* _n = "?"; const char* _s = "?";                       \
            cuGetErrorName(_r, &_n); cuGetErrorString(_r, &_s);               \
            std::fprintf(stderr,                                              \
                "  [FAIL] step=%s cuda_err=%d (%s): %s\n",                    \
                step_name, static_cast<int>(_r), _n, _s);                     \
            return _r;                                                        \
        }                                                                     \
    } while (0)

CUresult readback_and_check(CUdeviceptr va, size_t bytes, uint32_t expected,
                            const char* label) {
    // Sample 3 points: first u32, middle u32, last u32. If all three match,
    // we trust the mapping — no need to DMA 512 MiB every time.
    std::vector<uint32_t> host(3, 0);
    size_t mid_off = (bytes / 2) & ~size_t(3);
    size_t end_off = bytes - sizeof(uint32_t);

    CU_CHECK(cuMemcpyDtoH(&host[0], va + 0,       sizeof(uint32_t)), label);
    CU_CHECK(cuMemcpyDtoH(&host[1], va + mid_off, sizeof(uint32_t)), label);
    CU_CHECK(cuMemcpyDtoH(&host[2], va + end_off, sizeof(uint32_t)), label);

    bool ok = (host[0] == expected) && (host[1] == expected) && (host[2] == expected);
    std::printf("      readback[%s]: first=0x%08x mid=0x%08x last=0x%08x "
                "expected=0x%08x -> %s\n",
                label, host[0], host[1], host[2], expected,
                ok ? "OK" : "MISMATCH");
    return ok ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}

CUresult run(int device_id) {
    std::printf("[0] cuInit + pick device %d\n", device_id);
    CU_CHECK(cuInit(0), "cuInit");

    int dev_count = 0;
    CU_CHECK(cuDeviceGetCount(&dev_count), "cuDeviceGetCount");
    if (device_id < 0 || device_id >= dev_count) {
        std::fprintf(stderr, "  [FAIL] device_id %d out of range (count=%d)\n",
                     device_id, dev_count);
        return CUDA_ERROR_INVALID_DEVICE;
    }

    CUdevice dev;
    CU_CHECK(cuDeviceGet(&dev, device_id), "cuDeviceGet");

    char name[128] = {0};
    CU_CHECK(cuDeviceGetName(name, sizeof(name), dev), "cuDeviceGetName");

    int cc_major = 0, cc_minor = 0;
    CU_CHECK(cuDeviceGetAttribute(&cc_major,
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev), "cc_major");
    CU_CHECK(cuDeviceGetAttribute(&cc_minor,
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev), "cc_minor");

    int vmm_supported = 0;
    CU_CHECK(cuDeviceGetAttribute(&vmm_supported,
                CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, dev),
             "vmm_attr");

    std::printf("      device: \"%s\" sm_%d%d  VMM_SUPPORTED=%d\n",
                name, cc_major, cc_minor, vmm_supported);
    if (!vmm_supported) {
        std::fprintf(stderr,
            "  [FAIL] device reports CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_"
            "MANAGEMENT_SUPPORTED=0 — VMM API unusable here.\n");
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    CUcontext ctx;
    CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, dev), "ctxRetain");
    CU_CHECK(cuCtxSetCurrent(ctx), "ctxSetCurrent");

    // ── Granularity query ────────────────────────────────────────────
    CUmemAllocationProp prop = {};
    prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id   = device_id;

    size_t gran_min = 0, gran_rec = 0;
    CU_CHECK(cuMemGetAllocationGranularity(&gran_min, &prop,
                CU_MEM_ALLOC_GRANULARITY_MINIMUM), "gran_min");
    CU_CHECK(cuMemGetAllocationGranularity(&gran_rec, &prop,
                CU_MEM_ALLOC_GRANULARITY_RECOMMENDED), "gran_rec");
    std::printf("      granularity: min=%zu bytes (%.1f MiB)  "
                "recommended=%zu bytes (%.1f MiB)\n",
                gran_min, gran_min / (1024.0 * 1024.0),
                gran_rec, gran_rec / (1024.0 * 1024.0));

    if (CHUNK_BYTES % gran_min != 0) {
        std::fprintf(stderr,
            "  [FAIL] CHUNK_BYTES (%zu) not a multiple of min granularity "
            "(%zu) — refusing to proceed.\n", CHUNK_BYTES, gran_min);
        return CUDA_ERROR_INVALID_VALUE;
    }

    // ── Step 1: reserve 1 GiB VA ─────────────────────────────────────
    std::printf("[1] cuMemAddressReserve %zu bytes (%.2f GiB)\n",
                VA_RESERVE_BYTES, VA_RESERVE_BYTES / double(1ull << 30));
    CUdeviceptr va = 0;
    CU_CHECK(cuMemAddressReserve(&va, VA_RESERVE_BYTES, 0, 0, 0),
             "cuMemAddressReserve");
    std::printf("      reserved VA = 0x%016" PRIx64 "\n",
                static_cast<uint64_t>(va));

    // ── Step 2: create two physical handles A, B ─────────────────────
    std::printf("[2] cuMemCreate A (%.0f MiB) and B (%.0f MiB)\n",
                CHUNK_BYTES / (1024.0 * 1024.0),
                CHUNK_BYTES / (1024.0 * 1024.0));
    CUmemGenericAllocationHandle hA = 0, hB = 0;
    CU_CHECK(cuMemCreate(&hA, CHUNK_BYTES, &prop, 0), "cuMemCreate(A)");
    CU_CHECK(cuMemCreate(&hB, CHUNK_BYTES, &prop, 0), "cuMemCreate(B)");

    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id   = device_id;
    access.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    CUstream stream;
    CU_CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate");

    // ── Step 3: map A, fill with PATTERN_A ───────────────────────────
    std::printf("[3] cuMemMap VA -> A, set access, memset PATTERN_A\n");
    CU_CHECK(cuMemMap(va, CHUNK_BYTES, 0, hA, 0), "cuMemMap(A)");
    CU_CHECK(cuMemSetAccess(va, CHUNK_BYTES, &access, 1), "cuMemSetAccess(A)");
    CU_CHECK(cuMemsetD32Async(va, PATTERN_A,
                CHUNK_BYTES / sizeof(uint32_t), stream), "memset(A)");

    // ── Step 4: sync, verify A's pattern ─────────────────────────────
    std::printf("[4] cuStreamSynchronize, readback\n");
    CU_CHECK(cuStreamSynchronize(stream), "sync(A)");
    CU_CHECK(readback_and_check(va, CHUNK_BYTES, PATTERN_A, "A@VA"),
             "verify_A_first");

    // ── Step 5: unmap A, map B, fill with PATTERN_B ──────────────────
    std::printf("[5] cuMemUnmap A, cuMemMap B, memset PATTERN_B\n");
    CU_CHECK(cuMemUnmap(va, CHUNK_BYTES), "cuMemUnmap(A)");
    CU_CHECK(cuMemMap(va, CHUNK_BYTES, 0, hB, 0), "cuMemMap(B)");
    CU_CHECK(cuMemSetAccess(va, CHUNK_BYTES, &access, 1), "cuMemSetAccess(B)");
    CU_CHECK(cuMemsetD32Async(va, PATTERN_B,
                CHUNK_BYTES / sizeof(uint32_t), stream), "memset(B)");

    // ── Step 6: sync, verify B's pattern ─────────────────────────────
    std::printf("[6] cuStreamSynchronize, readback (expect PATTERN_B)\n");
    CU_CHECK(cuStreamSynchronize(stream), "sync(B)");
    CU_CHECK(readback_and_check(va, CHUNK_BYTES, PATTERN_B, "B@VA"),
             "verify_B");

    // ── Step 7: unmap B, remap A ─────────────────────────────────────
    std::printf("[7] cuMemUnmap B, cuMemMap A (remap the original physical)\n");
    CU_CHECK(cuMemUnmap(va, CHUNK_BYTES), "cuMemUnmap(B)");
    CU_CHECK(cuMemMap(va, CHUNK_BYTES, 0, hA, 0), "cuMemMap(A-again)");
    CU_CHECK(cuMemSetAccess(va, CHUNK_BYTES, &access, 1),
             "cuMemSetAccess(A-again)");

    // ── Step 8: readback, verify A's pattern PRESERVED ───────────────
    std::printf("[8] readback (expect PATTERN_A still intact on hA)\n");
    CU_CHECK(readback_and_check(va, CHUNK_BYTES, PATTERN_A, "A@VA-again"),
             "verify_A_preserved");

    // ── Cleanup ──────────────────────────────────────────────────────
    std::printf("[9] cleanup\n");
    CU_CHECK(cuMemUnmap(va, CHUNK_BYTES),           "final unmap");
    CU_CHECK(cuMemRelease(hA),                      "release A");
    CU_CHECK(cuMemRelease(hB),                      "release B");
    CU_CHECK(cuMemAddressFree(va, VA_RESERVE_BYTES), "address free");
    CU_CHECK(cuStreamDestroy(stream),               "stream destroy");
    CU_CHECK(cuDevicePrimaryCtxRelease(dev),        "ctx release");

    std::printf("\n[PASS] CUDA VMM primitive works on this GPU/driver.\n");
    std::printf("       Remap preserves backing contents. Safe to proceed.\n");
    return CUDA_SUCCESS;
}

} // namespace

int main(int argc, char** argv) {
    int device_id = 0;
    if (argc >= 2) device_id = std::atoi(argv[1]);

    std::printf("=== MEMOPT CUDA VMM probe ===\n");
    CUresult r = run(device_id);
    if (r != CUDA_SUCCESS) {
        std::printf("\n[FAIL] probe aborted. See [FAIL] line above for the "
                    "failing step + CUDA error.\n");
        return 1;
    }
    return 0;
}
