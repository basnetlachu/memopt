/*
 * gkd_map.cpp
 *
 * C++ hash map for GKD content-hash lookups.
 * Exposes a plain C API consumed by Python via ctypes.
 *
 * Uses std::unordered_map with a per-map std::mutex.
 * If abseil is available at build time, uses absl::flat_hash_map
 * instead (faster for string keys, better memory layout).
 *
 * The handle-based API avoids exposing raw C++ pointers to Python.
 * Each map is identified by a uint64_t handle assigned at creation.
 *
 * API:
 *   uint64_t gkd_map_create()
 *   int      gkd_map_insert(h, key, klen, val, vlen)  → 0 or -1
 *   int      gkd_map_lookup(h, key, klen, buf, bufsz) → bytes or -1
 *   int      gkd_map_erase(h, key, klen)              → 1 if found
 *   int64_t  gkd_map_size(h)                          → count or -1
 *   void     gkd_map_destroy(h)
 *
 * Build:
 *   g++ -O3 -std=c++17 -shared -fPIC \
 *       -o libgkd_map.so gkd_map.cpp
 *
 * With abseil (faster):
 *   g++ -O3 -std=c++17 -shared -fPIC \
 *       -labsl_container -labsl_hash \
 *       -o libgkd_map.so gkd_map.cpp
 */

#include <cstring>
#include <cstdint>
#include <string>
#include <mutex>
#include <atomic>
#include <new>
#include <unordered_map>

#ifdef __has_include
#  if __has_include(<absl/container/flat_hash_map.h>)
#    include <absl/container/flat_hash_map.h>
#    define GKD_HASHMAP absl::flat_hash_map
#  else
#    define GKD_HASHMAP std::unordered_map
#  endif
#else
#  define GKD_HASHMAP std::unordered_map
#endif

struct GKDMap {
    GKD_HASHMAP<std::string, std::string> data;
    std::mutex                            lock;
};

/*
 * Handle registry — maps uint64_t handles to GKDMap*.
 * Separated from GKDMap so we never expose raw pointers to Python.
 * Protected by its own mutex — distinct from per-map locks.
 */
static std::unordered_map<uint64_t, GKDMap*> _registry;
static std::mutex                             _registry_lock;
static std::atomic<uint64_t>                  _next_handle{1};

static GKDMap* get(uint64_t h) {
    std::lock_guard<std::mutex> g(_registry_lock);
    auto it = _registry.find(h);
    return (it != _registry.end()) ? it->second : nullptr;
}

extern "C" {

uint64_t gkd_map_create(void) {
    GKDMap* m = new (std::nothrow) GKDMap;
    if (!m) return 0;
    uint64_t h = _next_handle.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> g(_registry_lock);
    _registry[h] = m;
    return h;
}

int gkd_map_insert(uint64_t h,
                   const char* key, size_t klen,
                   const char* val, size_t vlen) {
    GKDMap* m = get(h);
    if (!m) return -1;
    std::lock_guard<std::mutex> g(m->lock);
    m->data[std::string(key, klen)] = std::string(val, vlen);
    return 0;
}

/*
 * Returns bytes written to buf on hit, -1 on miss.
 * Truncates to bufsz if value is larger.
 */
int gkd_map_lookup(uint64_t h,
                   const char* key, size_t klen,
                   char*       buf, size_t bufsz) {
    GKDMap* m = get(h);
    if (!m) return -1;
    std::lock_guard<std::mutex> g(m->lock);
    auto it = m->data.find(std::string(key, klen));
    if (it == m->data.end()) return -1;
    size_t n = it->second.size() < bufsz ? it->second.size() : bufsz;
    std::memcpy(buf, it->second.data(), n);
    return (int)n;
}

/* Returns 1 if key was present and erased, 0 if not found, -1 on bad handle */
int gkd_map_erase(uint64_t h, const char* key, size_t klen) {
    GKDMap* m = get(h);
    if (!m) return -1;
    std::lock_guard<std::mutex> g(m->lock);
    return (int)m->data.erase(std::string(key, klen));
}

int64_t gkd_map_size(uint64_t h) {
    GKDMap* m = get(h);
    if (!m) return -1;
    std::lock_guard<std::mutex> g(m->lock);
    return (int64_t)m->data.size();
}

void gkd_map_destroy(uint64_t h) {
    GKDMap* m = nullptr;
    {
        std::lock_guard<std::mutex> g(_registry_lock);
        auto it = _registry.find(h);
        if (it == _registry.end()) return;
        m = it->second;
        _registry.erase(it);
    }
    delete m;
}

} /* extern "C" */
