/**
 * memopt VMM Allocator — Milestone 1 implementation
 *
 * Strategy: VA pool pre-reserved at startup.
 * All allocations return VAs from the pool.
 * Physical HBM handles allocated per page.
 * Eviction: copy HBM → pinned DRAM, release handle,
 *           VA remains valid (backed by nothing — access
 *           would fault, but we only evict at step boundary
 *           when no kernel runs).
 * Promotion: alloc new HBM handle, copy DRAM → HBM,
 *            remap VA to new handle.
 */

#include "vmm_allocator.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <assert.h>

// ============================================================
// Internal structures
// ============================================================

// Cached-free entry (M2.2).
//
// On free, we DO NOT unmap or release physical HBM — we stash the whole
// (va, handle, size) so that sub-allocators (cuBLAS workspace cache etc.)
// don't find their cached pointers suddenly unmapped. Same semantics as
// PyTorch's native caching allocator.
//
// On alloc, we look for an exact-size match and recycle the mapped region
// as-is (no cuMemCreate / cuMemMap / cuMemSetAccess — already valid).
struct VaFree {
    CUdeviceptr                   va;
    size_t                        size;
    CUmemGenericAllocationHandle  handle;
};

struct MemoptAllocator {
    // Device
    int          device_idx;
    CUdevice     cu_device;
    CUcontext    cu_context;

    // VA pool
    CUdeviceptr  va_pool_base;   // Start of reserved VA range
    size_t       va_pool_size;   // Total reserved
    size_t       va_pool_used;   // Next allocation offset

    // Page table
    MemoptPage   pages[MEMOPT_MAX_PAGES];
    int          n_pages;
    pthread_mutex_t lock;

    // LRU counter
    uint64_t     tick;
    uint64_t     alloc_seq;

    // Stats
    uint64_t     eviction_count;
    uint64_t     promotion_count;
    size_t       bytes_evicted_total;

    // Allocation properties (for cuMemCreate)
    CUmemAllocationProp alloc_prop;
    size_t       granularity;  // 2MiB

    // Pinned host DRAM slab (M2): single cudaHostAlloc at startup, then
    // bump-allocated per eviction. Skips the ~1 ms per-eviction pin cost.
    // LIFO reclaim in promote/free; non-top-of-stack returns leak until the
    // allocator is destroyed. Milestone 3 replaces the bump with a freelist.
    void*        host_pool;
    size_t       host_pool_size;
    size_t       host_pool_used;

    // VA reclaim list (M2.1): exact-size match when memopt_free releases a
    // VA region, so torch-style alloc/free churn doesn't grow va_pool_used
    // monotonically. Without this, bnb model loading exhausts the pool even
    // though nothing is live.
    VaFree*      va_free;
    int          n_va_free;
    int          va_free_cap;
    size_t       cache_bytes;       // currently held in the free cache
    size_t       cache_bytes_limit; // eager-release when exceeded
};

// ============================================================
// Helper: round up to page boundary
// ============================================================

static size_t align_up(size_t n, size_t align) {
    return (n + align - 1) & ~(align - 1);
}

// ============================================================
// Helper: HBM pressure
// ============================================================

static float hbm_pressure_internal() {
    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    if (total_bytes == 0) return 0.0f;
    return 1.0f - ((float)free_bytes / (float)total_bytes);
}

// ============================================================
// M2: pinned host slab helpers — caller must hold a->lock
// ============================================================

static void* host_mirror_acquire(MemoptAllocator* a, size_t size) {
    if (a->host_pool) {
        if (a->host_pool_used + size > a->host_pool_size) {
            return NULL;  // slab full — caller handles the miss
        }
        void* p = (char*)a->host_pool + a->host_pool_used;
        a->host_pool_used += size;
        return p;
    }
    // Fallback (slab allocation failed at create time): per-call pin.
    void* p = NULL;
    if (cudaHostAlloc(&p, size, cudaHostAllocPortable) != cudaSuccess) {
        return NULL;
    }
    return p;
}

static void host_mirror_release(MemoptAllocator* a, void* mirror, size_t size) {
    if (!mirror) return;
    if (a->host_pool) {
        char* base = (char*)a->host_pool;
        char* end  = base + a->host_pool_size;
        if ((char*)mirror >= base && (char*)mirror < end) {
            // LIFO: only reclaim if this mirror is at the top of the bump.
            char* top = base + a->host_pool_used;
            if ((char*)mirror + size == top) {
                a->host_pool_used -= size;
            }
            // else: leaked until destroy. Acceptable for M2; fixed in M3.
            return;
        }
    }
    cudaFreeHost(mirror);
}

// Fraction of LIVE allocations currently in HBM (0.0 to 1.0).
// "live-data ratio" — evict_to_target(0.30) means keep at most 30 % of the
// currently-allocated data hot. This matches caller intent better than a
// fraction of the (arbitrarily sized) pool.
// Caller must hold a->lock.
static float pool_pressure_locked(MemoptAllocator* a) {
    size_t hbm = 0, dram = 0;
    for (int i = 0; i < a->n_pages; i++) {
        if (!a->pages[i].in_use) continue;
        if (a->pages[i].tier == TIER_HBM)
            hbm += a->pages[i].size_bytes;
        else if (a->pages[i].tier == TIER_DRAM)
            dram += a->pages[i].size_bytes;
    }
    size_t total = hbm + dram;
    return total == 0 ? 0.0f : (float)hbm / (float)total;
}

// ============================================================
// Create allocator
// ============================================================

MemoptAllocator* memopt_allocator_create(
        int device_idx, size_t pool_size_bytes) {

    // Initialize CUDA driver API
    CUresult cr = cuInit(0);
    if (cr != CUDA_SUCCESS) {
        fprintf(stderr, "[memopt] cuInit failed: %d\n", cr);
        return NULL;
    }

    MemoptAllocator* a = (MemoptAllocator*)calloc(
        1, sizeof(MemoptAllocator));
    if (!a) return NULL;

    pthread_mutex_init(&a->lock, NULL);
    a->device_idx = device_idx;
    a->tick = 0;
    a->alloc_seq = 0;
    a->eviction_count = 0;
    a->promotion_count = 0;
    a->bytes_evicted_total = 0;

    // Get device and context
    cr = cuDeviceGet(&a->cu_device, device_idx);
    if (cr != CUDA_SUCCESS) {
        fprintf(stderr, "[memopt] cuDeviceGet failed: %d\n", cr);
        free(a); return NULL;
    }
    cr = cuCtxGetCurrent(&a->cu_context);
    if (cr != CUDA_SUCCESS || a->cu_context == NULL) {
        // Retain the primary context (works across CUDA 11/12/13 and matches
        // what the runtime API uses, so cudaMemcpy et al. share this ctx).
        cr = cuDevicePrimaryCtxRetain(&a->cu_context, a->cu_device);
        if (cr != CUDA_SUCCESS) {
            fprintf(stderr,
                "[memopt] cuDevicePrimaryCtxRetain failed: %d\n", cr);
            free(a); return NULL;
        }
        cuCtxSetCurrent(a->cu_context);
    }

    // Set up allocation properties
    memset(&a->alloc_prop, 0, sizeof(a->alloc_prop));
    a->alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    a->alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    a->alloc_prop.location.id = device_idx;

    // Get granularity
    cr = cuMemGetAllocationGranularity(
        &a->granularity, &a->alloc_prop,
        CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
    if (cr != CUDA_SUCCESS) {
        a->granularity = MEMOPT_PAGE_SIZE_BYTES;
    }

    // Align pool size to granularity
    pool_size_bytes = align_up(pool_size_bytes, a->granularity);

    // Reserve virtual address range
    cr = cuMemAddressReserve(&a->va_pool_base, pool_size_bytes,
                              a->granularity, 0, 0);
    if (cr != CUDA_SUCCESS) {
        fprintf(stderr,
            "[memopt] cuMemAddressReserve(%zu bytes) failed: %d\n",
            pool_size_bytes, cr);
        free(a); return NULL;
    }

    a->va_pool_size = pool_size_bytes;
    a->va_pool_used = 0;
    a->n_pages = 0;

    // M2.1: VA freelist. Cap at 8× max pages — torch churns far more
    // alloc/free cycles than live pages (bnb staging, scratch workspaces).
    a->va_free_cap = MEMOPT_MAX_PAGES * 8;
    a->va_free = (VaFree*)calloc(a->va_free_cap, sizeof(VaFree));
    a->n_va_free = 0;
    if (!a->va_free) {
        fprintf(stderr, "[memopt] WARN: va_free alloc failed; VA will leak\n");
        a->va_free_cap = 0;
    }

    // Cache size cap (M2.3). Too large and we starve HBM of live allocations;
    // too small and sub-allocators (cuBLAS workspace) get their pointers
    // invalidated every call. 4 GiB default — tunable via env var.
    a->cache_bytes = 0;
    a->cache_bytes_limit = 4ULL * 1024 * 1024 * 1024;
    const char* env_cap = getenv("MEMOPT_CACHE_GB");
    if (env_cap && *env_cap) {
        double g = atof(env_cap);
        if (g > 0.0) a->cache_bytes_limit = (size_t)(g * (1ULL << 30));
    }
    fprintf(stderr, "[memopt] free cache limit: %.1f GiB\n",
            a->cache_bytes_limit / (1024.0*1024.0*1024.0));

    fprintf(stderr,
        "[memopt] allocator ready. VA pool: 0x%llx, "
        "size=%.1f GiB, granularity=%zu MiB\n",
        (unsigned long long)a->va_pool_base,
        (double)pool_size_bytes / (1024.0*1024.0*1024.0),
        a->granularity / (1024*1024));

    // ── M2 FIX 1: pre-pinned host DRAM slab for bump-allocated mirrors ──
    // Target: slab = min(pool/2, 32 GiB). If the pin fails (memlock limits
    // or low system RAM), halve and retry until we get ≥64 MiB, else fall
    // back to per-eviction cudaHostAlloc.
    a->host_pool      = NULL;
    a->host_pool_size = 0;
    a->host_pool_used = 0;

    size_t slab_cap  = 32ULL * 1024 * 1024 * 1024;    // 32 GiB hard cap
    size_t slab_size = pool_size_bytes / 2;
    if (slab_size > slab_cap) slab_size = slab_cap;

    while (slab_size >= (64ULL * 1024 * 1024)) {  // floor: 64 MiB
        cudaError_t e = cudaHostAlloc(&a->host_pool, slab_size,
                                      cudaHostAllocPortable);
        if (e == cudaSuccess && a->host_pool) {
            a->host_pool_size = slab_size;
            break;
        }
        a->host_pool = NULL;
        slab_size /= 2;
    }
    if (a->host_pool) {
        fprintf(stderr, "[memopt] host slab pinned: %.1f GiB\n",
                a->host_pool_size / (1024.0*1024.0*1024.0));
    } else {
        fprintf(stderr, "[memopt] WARN: host slab pin failed — "
                        "falling back to per-eviction cudaHostAlloc\n");
    }

    return a;
}

// ============================================================
// Allocate
// ============================================================

CUdeviceptr memopt_malloc(MemoptAllocator* a,
                           size_t size_bytes,
                           const char* tag) {
    if (!a || size_bytes == 0) return 0;

    // Round up to page boundary
    size_t alloc_size = align_up(size_bytes, a->granularity);

    pthread_mutex_lock(&a->lock);

    if (a->n_pages >= MEMOPT_MAX_PAGES) {
        fprintf(stderr, "[memopt] page table full\n");
        pthread_mutex_unlock(&a->lock);
        return 0;
    }

    // ── M2.2: cached-free reuse ────────────────────────────────────────
    // Exact-size match: grab the entry and skip cuMemCreate/Map/SetAccess
    // entirely — the VMM region is still live and still mapped.
    CUdeviceptr va = 0;
    CUmemGenericAllocationHandle handle = 0;
    bool recycled = false;
    for (int i = 0; i < a->n_va_free; i++) {
        if (a->va_free[i].size == alloc_size) {
            va       = a->va_free[i].va;
            handle   = a->va_free[i].handle;
            a->cache_bytes -= a->va_free[i].size;
            a->va_free[i] = a->va_free[a->n_va_free - 1]; // swap-remove
            a->n_va_free--;
            recycled = true;
            break;
        }
    }

    bool need_map = false;
    if (!recycled) {
        // Look for a "VA-only" freelist entry (handle==0) — VA is already
        // reserved from our pool but needs a fresh physical handle + mapping.
        for (int i = 0; i < a->n_va_free; i++) {
            if (a->va_free[i].handle == 0 && a->va_free[i].size == alloc_size) {
                va = a->va_free[i].va;
                a->va_free[i] = a->va_free[a->n_va_free - 1];
                a->n_va_free--;
                need_map = true;
                break;
            }
        }
        if (va == 0) {
            // Bump the VA pool.
            if (a->va_pool_used + alloc_size > a->va_pool_size) {
                fprintf(stderr,
                    "[memopt] VA pool exhausted (used=%.1f GiB / total=%.1f GiB, "
                    "freelist=%d entries)\n",
                    a->va_pool_used  / (1024.0*1024.0*1024.0),
                    a->va_pool_size  / (1024.0*1024.0*1024.0),
                    a->n_va_free);
                pthread_mutex_unlock(&a->lock);
                return 0;
            }
            va = a->va_pool_base + a->va_pool_used;
            a->va_pool_used += alloc_size;
            need_map = true;
        }
    }

    if (need_map) {
        CUresult cr = cuMemCreate(&handle, alloc_size, &a->alloc_prop, 0);
        if (cr != CUDA_SUCCESS) {
            fprintf(stderr, "[memopt] cuMemCreate(%zu) failed: %d\n",
                    alloc_size, cr);
            // VA remains reserved in the pool; return it to freelist as
            // handle=0 so a future alloc can retry.
            if (a->n_va_free < a->va_free_cap) {
                a->va_free[a->n_va_free].va     = va;
                a->va_free[a->n_va_free].size   = alloc_size;
                a->va_free[a->n_va_free].handle = 0;
                a->n_va_free++;
            }
            pthread_mutex_unlock(&a->lock);
            return 0;
        }
        cr = cuMemMap(va, alloc_size, 0, handle, 0);
        if (cr != CUDA_SUCCESS) {
            cuMemRelease(handle);
            pthread_mutex_unlock(&a->lock);
            return 0;
        }
        CUmemAccessDesc desc;
        desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        desc.location.id   = a->device_idx;
        desc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        cr = cuMemSetAccess(va, alloc_size, &desc, 1);
        if (cr != CUDA_SUCCESS) {
            cuMemUnmap(va, alloc_size);
            cuMemRelease(handle);
            pthread_mutex_unlock(&a->lock);
            return 0;
        }
    }

    // Register in page table
    MemoptPage* page = &a->pages[a->n_pages++];
    page->va          = va;
    page->hbm_handle  = handle;
    page->dram_mirror = NULL;
    page->size_bytes  = alloc_size;
    page->tier        = TIER_HBM;
    page->last_access = ++a->tick;
    page->alloc_seq   = ++a->alloc_seq;
    page->in_use      = true;
    snprintf(page->tag, sizeof(page->tag),
             "%s", tag ? tag : "unknown");

    pthread_mutex_unlock(&a->lock);
    return va;
}

// ============================================================
// Free
// ============================================================

void memopt_free(MemoptAllocator* a, CUdeviceptr va) {
    if (!a || !va) return;

    pthread_mutex_lock(&a->lock);

    for (int i = 0; i < a->n_pages; i++) {
        if (a->pages[i].va == va && a->pages[i].in_use) {
            MemoptPage* p = &a->pages[i];

            // Free DRAM mirror if evicted (LIFO-return to slab, M2)
            if (p->dram_mirror) {
                host_mirror_release(a, p->dram_mirror, p->size_bytes);
                p->dram_mirror = NULL;
            }

            // M2.2/M2.3: cache HBM-tier frees for fast reuse, BUT cap the
            // total cache size so it doesn't starve live allocations.
            bool cacheable = (p->tier == TIER_HBM)
                          && (a->n_va_free < a->va_free_cap)
                          && (a->cache_bytes + p->size_bytes
                                <= a->cache_bytes_limit);
            if (cacheable) {
                a->va_free[a->n_va_free].va     = p->va;
                a->va_free[a->n_va_free].size   = p->size_bytes;
                a->va_free[a->n_va_free].handle = p->hbm_handle;
                a->n_va_free++;
                a->cache_bytes += p->size_bytes;
            } else if (p->tier == TIER_HBM) {
                // Hard release — cache full or size-capped.
                cuMemUnmap(p->va, p->size_bytes);
                cuMemRelease(p->hbm_handle);
                // M2.3: return the VA to the freelist with handle=0 so a
                // future alloc of matching size can reuse the VA with a
                // fresh mapping — otherwise the bump pointer grows without
                // bound under torch's alloc/free churn.
                if (a->n_va_free < a->va_free_cap) {
                    a->va_free[a->n_va_free].va     = p->va;
                    a->va_free[a->n_va_free].size   = p->size_bytes;
                    a->va_free[a->n_va_free].handle = 0;
                    a->n_va_free++;
                }
            }
            // DRAM-tier frees are unchanged: mirror already released by
            // host_mirror_release below, handle was zeroed on evict.

            p->in_use = false;

            // Compact page table (swap with last)
            if (i < a->n_pages - 1) {
                a->pages[i] = a->pages[a->n_pages - 1];
            }
            a->n_pages--;
            break;
        }
    }

    pthread_mutex_unlock(&a->lock);
}

// ============================================================
// Quiesce — MUST call before eviction
// ============================================================

void memopt_quiesce(MemoptAllocator* a) {
    (void)a;
    // Sync all streams on all devices
    cudaDeviceSynchronize();
}

// ============================================================
// Evict one page (internal, lock held)
// ============================================================

static int evict_page_locked(MemoptAllocator* a, int page_idx) {
    MemoptPage* p = &a->pages[page_idx];

    if (!p->in_use || p->tier != TIER_HBM) return 0;

    // M2: acquire from pre-pinned slab (bump) — skips per-eviction pin cost.
    void* host_buf = host_mirror_acquire(a, p->size_bytes);
    if (!host_buf) {
        return 0;
    }

    // Copy HBM → DRAM (synchronous, streams are quiesced)
    cudaError_t err = cudaMemcpy(host_buf, (void*)p->va,
                                 p->size_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        host_mirror_release(a, host_buf, p->size_bytes);
        return 0;
    }

    // Unmap VA from HBM (VA remains reserved, just unmapped)
    CUresult cr = cuMemUnmap(p->va, p->size_bytes);
    if (cr != CUDA_SUCCESS) {
        host_mirror_release(a, host_buf, p->size_bytes);
        return 0;
    }

    // Release HBM physical handle (frees the HBM)
    cuMemRelease(p->hbm_handle);
    memset(&p->hbm_handle, 0, sizeof(p->hbm_handle));

    p->dram_mirror = host_buf;
    p->tier = TIER_DRAM;

    a->eviction_count++;
    a->bytes_evicted_total += p->size_bytes;

    return 1;
}

// ============================================================
// Evict n pages (LRU)
// ============================================================

int memopt_evict_n_pages(MemoptAllocator* a, int n) {
    if (!a || n <= 0) return 0;

    pthread_mutex_lock(&a->lock);

    int evicted = 0;
    for (int round = 0; round < n; round++) {
        // Find LRU HBM page
        int lru_idx = -1;
        uint64_t lru_tick = UINT64_MAX;

        for (int i = 0; i < a->n_pages; i++) {
            if (a->pages[i].in_use &&
                a->pages[i].tier == TIER_HBM &&
                a->pages[i].last_access < lru_tick) {
                lru_idx = i;
                lru_tick = a->pages[i].last_access;
            }
        }

        if (lru_idx < 0) break;
        if (evict_page_locked(a, lru_idx)) evicted++;
    }

    pthread_mutex_unlock(&a->lock);
    return evicted;
}

// ============================================================
// Evict until target pressure
// ============================================================

size_t memopt_evict_to_target(MemoptAllocator* a,
                               float target_fraction) {
    if (!a) return 0;

    size_t total_freed = 0;
    int safety = MEMOPT_MAX_PAGES + 1;  // bound — one page per iter at most

    // M2 FIX 2: pressure is measured against OUR VA pool, not the whole GPU.
    // "target 0.30" means "keep at most 30 % of the pool in HBM".
    while (safety-- > 0) {
        pthread_mutex_lock(&a->lock);

        if (pool_pressure_locked(a) <= target_fraction) {
            pthread_mutex_unlock(&a->lock);
            break;
        }

        int lru_idx = -1;
        uint64_t lru_tick = UINT64_MAX;
        for (int i = 0; i < a->n_pages; i++) {
            if (a->pages[i].in_use &&
                a->pages[i].tier == TIER_HBM &&
                a->pages[i].last_access < lru_tick) {
                lru_idx = i;
                lru_tick = a->pages[i].last_access;
            }
        }

        if (lru_idx < 0) {
            pthread_mutex_unlock(&a->lock);
            break;
        }

        size_t page_size = a->pages[lru_idx].size_bytes;
        if (evict_page_locked(a, lru_idx)) {
            total_freed += page_size;
        } else {
            // Eviction failed (e.g., host slab full) — stop, avoid spinning.
            pthread_mutex_unlock(&a->lock);
            break;
        }

        pthread_mutex_unlock(&a->lock);
    }

    return total_freed;
}

// ============================================================
// Promote: DRAM → HBM
// ============================================================

int memopt_promote(MemoptAllocator* a, CUdeviceptr va) {
    if (!a || !va) return -1;

    pthread_mutex_lock(&a->lock);

    MemoptPage* p = NULL;
    for (int i = 0; i < a->n_pages; i++) {
        if (a->pages[i].va == va && a->pages[i].in_use) {
            p = &a->pages[i];
            break;
        }
    }

    if (!p || p->tier != TIER_DRAM || !p->dram_mirror) {
        pthread_mutex_unlock(&a->lock);
        return -1;
    }

    // Allocate new HBM physical handle
    CUmemGenericAllocationHandle new_handle;
    CUresult cr = cuMemCreate(&new_handle, p->size_bytes,
                               &a->alloc_prop, 0);
    if (cr != CUDA_SUCCESS) {
        pthread_mutex_unlock(&a->lock);
        return -1;
    }

    // Map VA → new HBM handle
    cr = cuMemMap(p->va, p->size_bytes, 0, new_handle, 0);
    if (cr != CUDA_SUCCESS) {
        cuMemRelease(new_handle);
        pthread_mutex_unlock(&a->lock);
        return -1;
    }

    // Set access
    CUmemAccessDesc desc;
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    desc.location.id   = a->device_idx;
    desc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    cuMemSetAccess(p->va, p->size_bytes, &desc, 1);

    // Restore contents: DRAM → HBM
    cudaMemcpy((void*)p->va, p->dram_mirror,
               p->size_bytes, cudaMemcpyHostToDevice);

    // M2: return the mirror to the slab (LIFO) instead of cudaFreeHost.
    host_mirror_release(a, p->dram_mirror, p->size_bytes);
    p->dram_mirror = NULL;
    p->hbm_handle = new_handle;
    p->tier = TIER_HBM;
    p->last_access = ++a->tick;

    a->promotion_count++;

    pthread_mutex_unlock(&a->lock);
    return 0;
}

// ============================================================
// Stats
// ============================================================

MemoptStats memopt_stats(MemoptAllocator* a) {
    MemoptStats s = {};
    if (!a) return s;

    pthread_mutex_lock(&a->lock);

    for (int i = 0; i < a->n_pages; i++) {
        if (!a->pages[i].in_use) continue;
        s.pages_total++;
        if (a->pages[i].tier == TIER_HBM) {
            s.pages_hbm++;
            s.bytes_hbm += a->pages[i].size_bytes;
        } else {
            s.pages_dram++;
            s.bytes_dram += a->pages[i].size_bytes;
        }
    }

    s.eviction_count  = a->eviction_count;
    s.promotion_count = a->promotion_count;
    s.bytes_evicted_total = a->bytes_evicted_total;
    s.hbm_pressure    = hbm_pressure_internal();

    pthread_mutex_unlock(&a->lock);
    return s;
}

void memopt_dump_pages(MemoptAllocator* a) {
    if (!a) return;
    pthread_mutex_lock(&a->lock);
    fprintf(stderr, "=== memopt page dump (%d pages) ===\n",
            a->n_pages);
    for (int i = 0; i < a->n_pages; i++) {
        MemoptPage* p = &a->pages[i];
        if (!p->in_use) continue;
        fprintf(stderr,
            "  [%d] va=0x%llx size=%zuMiB tier=%s "
            "tag=%s lru=%llu\n",
            i, (unsigned long long)p->va,
            p->size_bytes/(1024*1024),
            p->tier == TIER_HBM ? "HBM" : "DRAM",
            p->tag, (unsigned long long)p->last_access);
    }
    pthread_mutex_unlock(&a->lock);
}

// ============================================================
// Destroy
// ============================================================

void memopt_allocator_destroy(MemoptAllocator* a) {
    if (!a) return;

    pthread_mutex_lock(&a->lock);
    for (int i = 0; i < a->n_pages; i++) {
        if (!a->pages[i].in_use) continue;
        MemoptPage* p = &a->pages[i];
        if (p->dram_mirror) {
            host_mirror_release(a, p->dram_mirror, p->size_bytes);
            p->dram_mirror = NULL;
        }
        if (p->tier == TIER_HBM) {
            cuMemUnmap(p->va, p->size_bytes);
            cuMemRelease(p->hbm_handle);
        }
        p->in_use = false;
    }
    a->n_pages = 0;

    // M2.2/M2.3: drain the freelist. Entries with handle != 0 are still
    // mapped; entries with handle == 0 are VA-only placeholders.
    for (int i = 0; i < a->n_va_free; i++) {
        if (a->va_free[i].handle != 0) {
            cuMemUnmap(a->va_free[i].va, a->va_free[i].size);
            cuMemRelease(a->va_free[i].handle);
        }
    }
    a->n_va_free = 0;
    a->cache_bytes = 0;

    // Release host slab (single cudaFreeHost for the whole thing, M2)
    if (a->host_pool) {
        cudaFreeHost(a->host_pool);
        a->host_pool = NULL;
        a->host_pool_size = 0;
        a->host_pool_used = 0;
    }
    pthread_mutex_unlock(&a->lock);

    // Release VA pool
    if (a->va_pool_base) {
        cuMemAddressFree(a->va_pool_base, a->va_pool_size);
    }

    if (a->va_free) {
        free(a->va_free);
        a->va_free = NULL;
    }

    pthread_mutex_destroy(&a->lock);
    free(a);
}

// ============================================================
// PyTorch CUDAPluggableAllocator entry points (M2 / M3 bridge)
//
// PyTorch 2.1+ ABI:
//   MallocFuncType = void*(size_t size, int device, cudaStream_t stream)
//   FreeFuncType   = void (void* ptr, size_t size, int device,
//                          cudaStream_t stream)
// Note: free DOES take a size parameter. We ignore it because the page
// table already knows each VA's size — but the parameter MUST be in the
// signature or the ABI is broken.
// ============================================================

static MemoptAllocator* g_torch_allocator = NULL;
static pthread_once_t   g_torch_once      = PTHREAD_ONCE_INIT;
static int              g_torch_device    = 0;

static void init_torch_allocator(void) {
    // Pool sizing: MEMOPT_TORCH_POOL_GB env var (set by the Python wrapper)
    // overrides; otherwise fall back to 85 % of current free HBM.
    size_t pool_size = 0;
    const char* env = getenv("MEMOPT_TORCH_POOL_GB");
    if (env && *env) {
        double g = atof(env);
        if (g > 0.0) {
            pool_size = (size_t)(g * 1024.0 * 1024.0 * 1024.0);
        }
    }
    if (pool_size == 0) {
        size_t free_bytes = 0, total_bytes = 0;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        pool_size = (size_t)(free_bytes * 0.85);
    }

    g_torch_allocator = memopt_allocator_create(g_torch_device, pool_size);
    if (!g_torch_allocator) {
        fprintf(stderr,
            "[memopt] WARN: torch allocator init failed — "
            "memopt_torch_malloc will fall back to cudaMalloc\n");
    } else {
        fprintf(stderr,
            "[memopt] torch allocator active (device=%d, pool=%.1f GiB)\n",
            g_torch_device,
            pool_size / (1024.0*1024.0*1024.0));
    }
}

extern "C" {

// PyTorch malloc entry
void* memopt_torch_malloc(size_t size, int device, cudaStream_t stream) {
    (void)stream;
    g_torch_device = device;
    pthread_once(&g_torch_once, init_torch_allocator);

    if (!g_torch_allocator) {
        void* ptr = NULL;
        cudaMalloc(&ptr, size);
        return ptr;
    }

    // IMPORTANT: zero-byte allocations must NOT trigger the eviction retry.
    // memopt_malloc(size=0) returns 0 as a sentinel ("nothing to allocate"),
    // not as a capacity failure — if we treated it as a failure we'd kick
    // off a spurious evict that trashes live pages.
    if (size == 0) return NULL;

    if (getenv("MEMOPT_TORCH_TRACE")) {
        fprintf(stderr, "[memopt.trace] malloc size=%zu device=%d\n",
                size, device);
    }

    CUdeviceptr va = memopt_malloc(g_torch_allocator, size, "torch");
    if (va == 0) {
        // Allocator full — try evicting the coldest pages and retry once.
        if (getenv("MEMOPT_TORCH_TRACE")) {
            fprintf(stderr,
                "[memopt.trace] malloc size=%zu FULL — quiesce + evict\n",
                size);
        }
        memopt_quiesce(g_torch_allocator);
        memopt_evict_to_target(g_torch_allocator, 0.70f);
        va = memopt_malloc(g_torch_allocator, size, "torch_retry");
    }
    return (void*)va;
}

// PyTorch free entry — MUST take (ptr, size, device, stream) per ABI
void memopt_torch_free(void* ptr, size_t size, int device,
                       cudaStream_t stream) {
    (void)device;
    if (!ptr) return;

    if (getenv("MEMOPT_TORCH_TRACE")) {
        fprintf(stderr, "[memopt.trace] free ptr=0x%llx size=%zu\n",
                (unsigned long long)ptr, size);
    }

    // Stream-ordering: torch is asking us to free memory that was last
    // used on `stream`. If we cuMemUnmap immediately, OR if we recycle
    // the VA via the freelist, any kernel still queued that references
    // this VA will fault. Torch's native allocator handles this via
    // per-stream events + lazy free.
    // Conservative fix (M2.2): synchronize the stream before tearing
    // down or caching. Slower than event-tracked lazy free, but correct.
    // Note: stream==NULL means the legacy/default stream — must still
    // synchronize. cudaStreamSynchronize(0) syncs the default stream,
    // which is what we want. The earlier `if (stream)` guard was
    // unsound — it skipped the sync on default-stream frees and let
    // a recycled VA get handed to a new alloc while a kernel on the
    // default stream was still using the old contents → CUDA "illegal
    // memory access" in the next kernel.
    // TODO(M3): replace with event-record + deferred release list.
    cudaStreamSynchronize(stream);

    if (!g_torch_allocator) {
        cudaFree(ptr);
        return;
    }
    memopt_free(g_torch_allocator, (CUdeviceptr)ptr);
}

// Stats probe for Python
void memopt_torch_get_stats(int* pages_hbm, int* pages_dram,
                             float* hbm_pressure) {
    if (!g_torch_allocator) {
        if (pages_hbm)     *pages_hbm     = 0;
        if (pages_dram)    *pages_dram    = 0;
        if (hbm_pressure)  *hbm_pressure  = 0.0f;
        return;
    }
    MemoptStats s = memopt_stats(g_torch_allocator);
    if (pages_hbm)    *pages_hbm    = s.pages_hbm;
    if (pages_dram)   *pages_dram   = s.pages_dram;
    if (hbm_pressure) *hbm_pressure = s.hbm_pressure;
}

// Called between decode steps: quiesce, then evict if live-data ratio >
// threshold. Target is threshold - 0.10 (i.e., evict with some hysteresis).
void memopt_torch_step_boundary(float threshold) {
    if (!g_torch_allocator) return;
    memopt_quiesce(g_torch_allocator);
    // Gate on GPU-wide pressure (same semantics as the Python reference
    // step_boundary). If the device isn't under pressure we skip eviction
    // entirely — don't do work when we don't have to.
    MemoptStats s = memopt_stats(g_torch_allocator);
    if (s.hbm_pressure > threshold) {
        float target = threshold - 0.10f;
        if (target < 0.0f) target = 0.0f;
        memopt_evict_to_target(g_torch_allocator, target);
    }
}

// Expose global allocator to Python bindings (for stats() on the torch-
// owned instance, without a separate handle).
MemoptAllocator* memopt_torch_get_allocator(void) {
    return g_torch_allocator;
}

} // extern "C"
