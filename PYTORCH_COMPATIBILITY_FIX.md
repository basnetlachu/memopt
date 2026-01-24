# PyTorch Compatibility Fix

## Issue
```
AttributeError: 'FunctionEventAvg' object has no attribute 'cuda_time_total'. Did you mean: 'cpu_time_total'?
```

This error occurs because PyTorch changed the profiler API between versions. The attribute `cuda_time_total` was renamed to `device_time_total` in newer PyTorch versions.

## Solution

Apply this fix to `memopt/bandwidth_profiler.py` around line 226-234:

### BEFORE (Broken):
```python
for event in events:
    # Sum CUDA time
    if event.device_type == torch.profiler.DeviceType.CUDA:
        total_cuda_time_us += event.cuda_time_total

        # Estimate memory traffic from tensor sizes
        # This is approximate - PyTorch doesn't expose exact HBM traffic
        if hasattr(event, 'cuda_memory_usage') and event.cuda_memory_usage > 0:
            total_memory_bytes += event.cuda_memory_usage
```

### AFTER (Fixed):
```python
for event in events:
    # Sum CUDA time
    if event.device_type == torch.profiler.DeviceType.CUDA:
        # Handle both old and new PyTorch API
        if hasattr(event, 'cuda_time_total'):
            total_cuda_time_us += event.cuda_time_total
        elif hasattr(event, 'device_time_total'):
            total_cuda_time_us += event.device_time_total
        else:
            # Fallback to self_cuda_time_total if available
            total_cuda_time_us += getattr(event, 'self_cuda_time_total', 0)

        # Estimate memory traffic from tensor sizes
        # This is approximate - PyTorch doesn't expose exact HBM traffic
        if hasattr(event, 'cuda_memory_usage') and event.cuda_memory_usage > 0:
            total_memory_bytes += event.cuda_memory_usage
```

## How to Apply on GPU Server

### Option 1: Manual Edit
```bash
cd /path/to/memopt
nano memopt/bandwidth_profiler.py
# Go to line 226
# Replace the code as shown above
# Save and exit
```

### Option 2: Use sed (Quick)
```bash
cd /path/to/memopt

# Backup original file
cp memopt/bandwidth_profiler.py memopt/bandwidth_profiler.py.backup

# Apply fix (this is a multi-line replacement, so do it manually or use the patch below)
```

### Option 3: Download Fixed File
If you can transfer files, I've already fixed this in the Mac version. Just re-transfer the file:

```bash
# From Mac
scp memopt/bandwidth_profiler.py user@gpu-server:/path/to/memopt/memopt/

# Or re-sync entire directory
rsync -avz memopt/ user@gpu-server:/path/to/memopt/
```

## Verification

After applying the fix:

```bash
# Test basic functionality
python3 test_installation.py

# Should now pass without AttributeError
python3 -m memopt.cli --demo lazy-kv
```

## Why This Happened

PyTorch profiler API changed between versions:
- **PyTorch < 2.0**: Used `cuda_time_total`
- **PyTorch >= 2.0**: Uses `device_time_total` (more general, supports ROCm, etc.)
- **PyTorch >= 2.1**: May use different attributes

The fix uses `hasattr()` to check which attribute is available and gracefully falls back, making it compatible with all PyTorch versions.

## Testing After Fix

Run these commands to verify:

```bash
# 1. Basic test
python3 test_installation.py

# 2. Demo without validation
python3 -m memopt.cli --demo lazy-kv

# 3. Check PyTorch version
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

Expected PyTorch version on your GPU server: 2.0.0 or newer

## Alternative: Use Compatible PyTorch Version

If you want to avoid the fix, downgrade PyTorch (not recommended):

```bash
pip install torch==1.13.1 --index-url https://download.pytorch.org/whl/cu118
```

But the fix above is better - it works with all PyTorch versions.
