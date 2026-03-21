/*
 * kernel_shm.cpp
 *
 * Shared memory segment for the compiled kernel handle registry.
 * Multiple serving processes share one cache — zero re-compilation
 * when a worker restarts.
 *
 * The shared segment stores a fixed-size table of entries:
 *   struct ShmEntry { char key[256]; uint8_t valid; }
 *
 * The compiled module objects themselves cannot be stored in shared
 * memory (they contain Python C extension state). What we store is
 * the set of keys for which a compiled kernel exists on disk, so
 * any worker can find and load it without re-invoking the compiler.
 *
 * API:
 *   int  ksm_open(const char* name, int max_entries)
 *        Opens or creates the shared segment. Returns fd or -errno.
 *   int  ksm_insert(int fd, const char* key)  → 0 or -errno
 *   int  ksm_contains(int fd, const char* key) → 1/0 or -errno
 *   void ksm_close(int fd)
 *   void ksm_unlink(const char* name)   (cleanup, call once on shutdown)
 *
 * Build:
 *   g++ -O3 -std=c++17 -shared -fPIC \
 *       -o libkernel_shm.so kernel_shm.cpp
 */

#include <cstring>
#include <cerrno>
#include <cstdint>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

static const int KEY_LEN = 255;

struct ShmEntry {
    char    key[KEY_LEN + 1];
    uint8_t valid;
};

struct ShmHeader {
    int      max_entries;
    int      count;
    /* ShmEntry entries[] follow immediately */
};

static ShmEntry* entries_ptr(ShmHeader* h) {
    return reinterpret_cast<ShmEntry*>(h + 1);
}

static size_t shm_size(int max_entries) {
    return sizeof(ShmHeader) + (size_t)max_entries * sizeof(ShmEntry);
}

extern "C" {

/*
 * ksm_open — open or create a shared memory segment.
 * Returns a file descriptor that must be passed to other functions.
 * Returns -errno on failure.
 *
 * The segment is initialised to zero on first creation.
 * Subsequent opens map to the existing segment.
 */
int ksm_open(const char* name, int max_entries) {
    if (!name || max_entries <= 0) return -EINVAL;

    int fd = shm_open(name, O_CREAT | O_RDWR, 0600);
    if (fd < 0) return -errno;

    size_t sz = shm_size(max_entries);
    if (ftruncate(fd, (off_t)sz) < 0) {
        int e = errno;
        close(fd);
        return -e;
    }

    void* ptr = mmap(nullptr, sz,
                     PROT_READ | PROT_WRITE,
                     MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        int e = errno;
        close(fd);
        return -e;
    }

    ShmHeader* h = static_cast<ShmHeader*>(ptr);
    /* Initialise on first creation (count == 0 and max == 0) */
    if (h->max_entries == 0) {
        h->max_entries = max_entries;
        h->count       = 0;
    }

    munmap(ptr, sz);
    return fd;
}

/*
 * ksm_insert — record that a kernel with this key exists on disk.
 * Idempotent — inserting an existing key is a no-op.
 */
int ksm_insert(int fd, const char* key) {
    if (fd < 0 || !key) return -EINVAL;

    /* Re-map on every call — safe across processes */
    struct stat st;
    if (fstat(fd, &st) < 0) return -errno;
    size_t sz = (size_t)st.st_size;

    void* ptr = mmap(nullptr, sz,
                     PROT_READ | PROT_WRITE,
                     MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) return -errno;

    ShmHeader* h   = static_cast<ShmHeader*>(ptr);
    ShmEntry*  ent = entries_ptr(h);
    int ret = -ENOSPC;

    for (int i = 0; i < h->max_entries; ++i) {
        if (ent[i].valid && strncmp(ent[i].key, key, KEY_LEN) == 0) {
            ret = 0; /* already present */
            break;
        }
        if (!ent[i].valid) {
            strncpy(ent[i].key, key, KEY_LEN);
            ent[i].key[KEY_LEN] = '\0';
            ent[i].valid = 1;
            h->count++;
            ret = 0;
            break;
        }
    }

    munmap(ptr, sz);
    return ret;
}

/* Returns 1 if key is present, 0 if not, -errno on error */
int ksm_contains(int fd, const char* key) {
    if (fd < 0 || !key) return -EINVAL;

    struct stat st;
    if (fstat(fd, &st) < 0) return -errno;
    size_t sz = (size_t)st.st_size;

    void* ptr = mmap(nullptr, sz, PROT_READ, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) return -errno;

    ShmHeader* h   = static_cast<ShmHeader*>(ptr);
    ShmEntry*  ent = entries_ptr(h);
    int found = 0;

    for (int i = 0; i < h->max_entries && !found; ++i) {
        if (ent[i].valid && strncmp(ent[i].key, key, KEY_LEN) == 0)
            found = 1;
    }

    munmap(ptr, sz);
    return found;
}

void ksm_close(int fd) {
    if (fd >= 0) close(fd);
}

void ksm_unlink(const char* name) {
    if (name) shm_unlink(name);
}

} /* extern "C" */
