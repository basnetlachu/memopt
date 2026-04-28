"""Quick smoke test for the ctypes VMM bindings — matches the M2 spec."""
import sys
sys.path.insert(0, "/opt/memopt")

from memopt.vmm.cuda_vmm import CUDAVMMAllocator

alloc = CUDAVMMAllocator(device=0, pool_gb=20.0, evict_threshold=0.80)

va = alloc.malloc(512 * 1024 * 1024, tag="test")
print(f"Allocated VA: 0x{va:x}")

stats = alloc.stats()
print(f'pages_hbm: {stats["pages_hbm"]}')
print(f'hbm_pressure: {stats["hbm_pressure"]:.1%}')
print(f'pool_pressure (hbm/pool): '
      f'{stats["bytes_hbm"] / (20 * 1024**3):.1%}')

alloc.quiesce()
freed = alloc.evict_to_target(0.30)
print(f"Freed: {freed / 1e6:.0f} MB  (with target 0.30 of 20 GiB pool)")

stats_after = alloc.stats()
print(f'After evict - pages_dram: {stats_after["pages_dram"]}')

# Second attempt with a target actually below current pool pressure
freed2 = alloc.evict_to_target(0.001)
stats_final = alloc.stats()
print(f"Freed2: {freed2 / 1e6:.0f} MB  (with target 0.001 — forces eviction)")
print(f'After evict2 - pages_dram: {stats_final["pages_dram"]}')

# Promote to verify full round trip
alloc.promote(va)
stats_promoted = alloc.stats()
print(f'After promote - pages_hbm: {stats_promoted["pages_hbm"]}, '
      f'pages_dram: {stats_promoted["pages_dram"]}')

alloc.free(va)
print("Python bindings: WORKING")
