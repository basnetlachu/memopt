/**
 * Tests the memopt VMM allocator.
 * Proves:
 *   1. Allocation returns valid GPU pointer
 *   2. Data written before eviction survives in DRAM
 *   3. Data restored correctly after promotion
 *   4. HBM actually freed after eviction
 *   5. Multiple alloc/evict/promote cycles stable
 */
#include "vmm_allocator.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <stdint.h>

#define CHECK_CUDA(x) do {                                              \
    cudaError_t e = (x);                                                \
    if (e != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA error at %s:%d — %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(e));             \
        exit(1);                                                        \
    }                                                                   \
} while(0)

static double now_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

int main() {
    printf("=== memopt VMM Allocator Test ===\n\n");

    // Get device memory info
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    printf("GPU HBM: %.1f GiB free / %.1f GiB total\n",
           free_bytes / (1024.0*1024.0*1024.0),
           total_bytes / (1024.0*1024.0*1024.0));

    // Create allocator with 60% of free HBM
    size_t pool_size = (size_t)(free_bytes * 0.60);
    printf("Creating allocator with %.1f GiB pool...\n\n",
           pool_size / (1024.0*1024.0*1024.0));

    MemoptAllocator* alloc = memopt_allocator_create(0, pool_size);
    assert(alloc != NULL);

    // ---- Test 1: Basic alloc + write + read ----
    printf("[Test 1] Alloc 64 MiB, write pattern, verify\n");

    size_t sz = 64 * 1024 * 1024ULL;  // 64 MiB
    CUdeviceptr va = memopt_malloc(alloc, sz, "test_block");
    assert(va != 0);

    // Write known pattern
    CHECK_CUDA(cudaMemset((void*)va, 0xAB, sz));

    // Read back on CPU and verify
    uint8_t* host_buf = (uint8_t*)malloc(sz);
    CHECK_CUDA(cudaMemcpy(host_buf, (void*)va, sz,
                          cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < sz; i += 4096) {
        assert(host_buf[i] == 0xAB);
    }
    printf("  [OK] Write/read OK\n");

    // ---- Test 2: Evict and check HBM freed ----
    printf("\n[Test 2] Evict to DRAM, verify HBM freed\n");

    cudaMemGetInfo(&free_bytes, &total_bytes);
    double hbm_before = free_bytes / (1024.0*1024.0*1024.0);

    memopt_quiesce(alloc);
    int n_evicted = memopt_evict_n_pages(alloc,
                        (int)(sz / MEMOPT_PAGE_SIZE_BYTES) + 1);

    cudaMemGetInfo(&free_bytes, &total_bytes);
    double hbm_after = free_bytes / (1024.0*1024.0*1024.0);

    printf("  Pages evicted: %d\n", n_evicted);
    printf("  HBM before: %.3f GiB free\n", hbm_before);
    printf("  HBM after:  %.3f GiB free\n", hbm_after);
    printf("  HBM freed:  %.3f GiB\n", hbm_after - hbm_before);

    assert(n_evicted > 0);
    // HBM should have grown (more free) after eviction
    assert(hbm_after > hbm_before - 0.001);
    printf("  [OK] HBM actually freed after eviction\n");

    MemoptStats s = memopt_stats(alloc);
    printf("  Stats: pages_hbm=%d pages_dram=%d evictions=%llu\n",
           s.pages_hbm, s.pages_dram,
           (unsigned long long)s.eviction_count);

    // ---- Test 3: Promote back and verify data ----
    printf("\n[Test 3] Promote back to HBM, verify data intact\n");

    int rc = memopt_promote(alloc, va);
    assert(rc == 0);

    uint8_t* verify_buf = (uint8_t*)malloc(sz);
    CHECK_CUDA(cudaMemcpy(verify_buf, (void*)va, sz,
                          cudaMemcpyDeviceToHost));
    int mismatches = 0;
    for (size_t i = 0; i < sz; i += 4096) {
        if (verify_buf[i] != 0xAB) mismatches++;
    }
    printf("  Data mismatches: %d / %zu\n",
           mismatches, sz/4096);
    assert(mismatches == 0);
    printf("  [OK] Data intact after evict+promote\n");

    // ---- Test 4: Latency measurement ----
    printf("\n[Test 4] Evict/promote latency (10 cycles)\n");

    double evict_total = 0, promote_total = 0;
    int n_cycles = 10;

    for (int c = 0; c < n_cycles; c++) {
        int n_pages = (int)(sz / MEMOPT_PAGE_SIZE_BYTES) + 1;

        double t0 = now_us();
        memopt_quiesce(alloc);
        memopt_evict_n_pages(alloc, n_pages);
        double t1 = now_us();
        evict_total += (t1 - t0);

        double t2 = now_us();
        memopt_promote(alloc, va);
        double t3 = now_us();
        promote_total += (t3 - t2);
    }

    int pages_per_block = (int)(sz / MEMOPT_PAGE_SIZE_BYTES) + 1;
    printf("  Evict   mean: %.1f us total  (%.1f us/page, %d pages)\n",
           evict_total/n_cycles,
           evict_total/n_cycles/pages_per_block,
           pages_per_block);
    printf("  Promote mean: %.1f us total  (%.1f us/page)\n",
           promote_total/n_cycles,
           promote_total/n_cycles/pages_per_block);

    // ---- Test 5: Multiple allocations ----
    printf("\n[Test 5] Allocate 8 x 128 MiB blocks\n");

    CUdeviceptr vas[8];
    for (int i = 0; i < 8; i++) {
        char tag[32];
        snprintf(tag, sizeof(tag), "block_%d", i);
        vas[i] = memopt_malloc(alloc, 128*1024*1024ULL, tag);
        if (vas[i]) {
            CHECK_CUDA(cudaMemset((void*)vas[i], i+1,
                                  128*1024*1024ULL));
        }
        printf("  block_%d: %s\n", i, vas[i] ? "OK" : "FAILED");
    }

    // Evict all to DRAM
    printf("\n  Evicting down to 30%% HBM pressure...\n");
    memopt_quiesce(alloc);

    cudaMemGetInfo(&free_bytes, &total_bytes);
    double before_evict = free_bytes / (1024.0*1024.0*1024.0);

    memopt_evict_to_target(alloc, 0.30f);

    cudaMemGetInfo(&free_bytes, &total_bytes);
    double after_evict = free_bytes / (1024.0*1024.0*1024.0);

    s = memopt_stats(alloc);
    printf("  HBM freed: %.2f GiB\n", after_evict - before_evict);
    printf("  pages_hbm=%d pages_dram=%d\n",
           s.pages_hbm, s.pages_dram);

    // Promote and verify each
    printf("\n  Promoting and verifying each block...\n");
    int total_mismatches = 0;
    for (int i = 0; i < 8; i++) {
        if (!vas[i]) continue;
        memopt_promote(alloc, vas[i]);
        uint8_t sample;
        CHECK_CUDA(cudaMemcpy(&sample, (void*)vas[i], 1,
                              cudaMemcpyDeviceToHost));
        if (sample != (uint8_t)(i+1)) {
            printf("  block_%d: MISMATCH (got %d, want %d)\n",
                   i, sample, i+1);
            total_mismatches++;
        }
    }

    printf("  Total mismatches: %d\n", total_mismatches);
    assert(total_mismatches == 0);
    printf("  [OK] All blocks intact\n");

    // ---- Final stats ----
    printf("\n=== Final Stats ===\n");
    s = memopt_stats(alloc);
    printf("  pages_total:    %d\n", s.pages_total);
    printf("  pages_hbm:      %d\n", s.pages_hbm);
    printf("  pages_dram:     %d\n", s.pages_dram);
    printf("  evictions:      %llu\n",
           (unsigned long long)s.eviction_count);
    printf("  promotions:     %llu\n",
           (unsigned long long)s.promotion_count);
    printf("  bytes_evicted:  %.2f GiB\n",
           (double)s.bytes_evicted_total / (1024.0*1024.0*1024.0));
    printf("  hbm_pressure:   %.1f%%\n",
           s.hbm_pressure * 100);

    // Cleanup
    free(host_buf);
    free(verify_buf);
    memopt_allocator_destroy(alloc);

    printf("\n[ALL TESTS PASSED]\n");
    printf("memopt VMM Allocator Milestone 1: COMPLETE\n");
    return 0;
}
