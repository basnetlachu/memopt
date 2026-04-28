"""Bridge sanity: install MUST happen before any CUDA interaction."""
import sys
sys.path.insert(0, "/opt/memopt")

import torch  # import is fine; CUDA not initialized yet.

from memopt.vmm.torch_allocator import install_memopt_allocator

# Install FIRST, before anything touches CUDA.
alloc = install_memopt_allocator(pool_gb=4.0, evict_threshold=0.80,
                                 verbose=True)
print(f"installed: {alloc.is_installed()}")
assert alloc.is_installed(), "bridge install failed — bailing"

# Now it's safe to use CUDA.
print(f"torch: {torch.__version__}  cuda: {torch.version.cuda}  "
      f"device: {torch.cuda.get_device_name(0)}")

x = torch.empty(1024 * 1024, dtype=torch.float32, device="cuda")
print(f"x.data_ptr() = 0x{x.data_ptr():x}")
y = torch.arange(1024 * 1024, dtype=torch.float32, device="cuda")
x.copy_(y)
z = x.sum().item()
expected = (1024 * 1024 - 1) * 1024 * 1024 / 2.0
print(f"x.sum() = {z:.3e}   expected={expected:.3e}   match={abs(z-expected)<1e6}")

w = torch.empty(16 * 1024 * 1024, dtype=torch.float32, device="cuda")
print(f"w.data_ptr() = 0x{w.data_ptr():x}")

s = alloc.stats()
print("stats after allocs:")
for k in ("pages_total", "pages_hbm", "pages_dram", "bytes_hbm",
          "eviction_count", "hbm_pressure"):
    if k in s:
        print(f"  {k} = {s[k]}")

big = torch.empty(256 * 1024 * 1024, dtype=torch.float32, device="cuda")
print(f"big.data_ptr() = 0x{big.data_ptr():x}")

s = alloc.step_boundary()
print("after step_boundary:")
for k in ("pages_hbm", "pages_dram", "eviction_count",
          "bytes_evicted_total", "hbm_pressure"):
    if k in s:
        print(f"  {k} = {s[k]}")

del big, w, x, y
torch.cuda.empty_cache()
print("bridge sanity: OK (memopt owned every alloc above)")
