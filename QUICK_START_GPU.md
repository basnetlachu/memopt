# Quick Start Guide for GPU Testing

**IMPORTANT: Read this first before running tests on GPU server**

---

## 🚨 Known Issue & Fix

Your GPU server has PyTorch 2.x which uses a different API. You'll encounter this error:

```
AttributeError: 'FunctionEventAvg' object has no attribute 'cuda_time_total'
```

**FIX IS ALREADY INCLUDED!** Just run:

```bash
cd /path/to/memopt
python3 apply_pytorch_fix.py
```

This takes 2 seconds and fixes the compatibility issue.

---

## 📋 3-Step Quick Start

### Step 1: Transfer & Install (2 minutes)

```bash
# On GPU server
cd ~
# (assume code is already transferred)
cd memopt

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install
pip install -e .

# Apply PyTorch compatibility fix
python3 apply_pytorch_fix.py
```

---

### Step 2: Run Tests (5 minutes)

```bash
# Test 1: Installation (must pass 3/3)
python3 test_installation.py

# Test 2: Correctness (must pass 11/11)
python3 tests/test_memory_coalescing.py

# Test 3: GPU detection
python3 -m memopt.cli --gpu-info
```

**Expected: All tests PASS ✅**

---

### Step 3: Run Demo (2 minutes)

```bash
# WITHOUT hardware validation (fast)
python3 -m memopt.cli --demo lazy-kv

# WITH hardware validation (slow, requires Nsight Compute)
python3 -m memopt.cli --demo lazy-kv --validate-hardware
```

---

## ✅ Success Criteria

After completing the 3 steps above:

- [ ] `test_installation.py` shows 3/3 PASS
- [ ] `test_memory_coalescing.py` shows 11/11 PASS
- [ ] GPU info displays your GPU model correctly
- [ ] Demo runs without errors
- [ ] HTML report generated (`bandwidth_report.html`)

**If all checkboxes checked: Platform is working!** 🎉

---

## 📊 What You'll See

### Installation Test Output:
```
======================================================================
MEMOPT INSTALLATION TEST
======================================================================

Testing imports...
✅ Phase 1: Core profiling
✅ Phase 2: Visualization & Reporting
✅ Phase 3: Optimization Engine
✅ Utilities: KV Cache

3/3 tests passed

🎉 All tests passed! MemOpt is ready to use.
```

### Correctness Test Output:
```
Testing correctness...
✅ Exact match test passed
✅ Subset reuse test passed
✅ Autoregressive pattern test passed
✅ Full KV cache test passed
✅ Multiple layers test passed

...

11/11 PASS
```

### GPU Info Output:
```
GPU 0: NVIDIA A100-SXM4-80GB
  Memory: 80.0 GB
  Compute Capability: 8.0
  Multi-processors: 108
```

---

## 🐛 If Something Fails

### If `apply_pytorch_fix.py` fails:
```bash
# Check if already fixed
grep "device_time_total" memopt/bandwidth_profiler.py

# If not found, see PYTORCH_COMPATIBILITY_FIX.md for manual fix
```

### If tests fail:
```bash
# Check PyTorch version
python3 -c "import torch; print(torch.__version__)"

# Should be 2.0.0 or higher with CUDA support
python3 -c "import torch; print(torch.cuda.is_available())"
```

### If demo fails with OOM:
Your GPU might not have enough memory. The demo uses:
- 32 layers
- Batch size 4
- 1024 sequence length

This is normal for testing. The platform will work with your actual models.

---

## 📁 Files You Need

**Transfer these from Mac to GPU server:**

```
memopt/
├── memopt/                          # Main package
├── tests/                           # Test files
├── examples/                        # Example scripts
├── setup.py                         # Installation
├── requirements.txt                 # Dependencies
├── test_installation.py             # Installation test
├── apply_pytorch_fix.py             # ⭐ Compatibility fix
├── PYTORCH_COMPATIBILITY_FIX.md     # Fix documentation
└── GPU_TESTING_GUIDE.md             # Full guide
```

**Transfer command from Mac:**
```bash
cd /Users/lachumanbasnet/Personal/Sophisticates/memory-optimization
rsync -avz --exclude='*.pyc' --exclude='__pycache__' --exclude='.git' \
      memopt/ user@gpu-server:/root/memopt/
```

---

## 🎯 Enterprise Demo Ready?

**Minimum requirements:**
- ✅ All tests pass (3/3 + 11/11)
- ✅ Demo runs without errors
- ✅ HTML report generated
- ⚠️  Yellow disclaimer shown (without Nsight Compute)

**Ideal for customer demos:**
- ✅ All tests pass
- ✅ Nsight Compute installed
- ✅ Demo with `--validate-hardware` works
- ✅ Green ✅ validation banner in reports

---

## 📞 Quick Commands Reference

```bash
# Fix PyTorch compatibility
python3 apply_pytorch_fix.py

# Test installation
python3 test_installation.py

# Test correctness
python3 tests/test_memory_coalescing.py

# Check GPU
python3 -m memopt.cli --gpu-info

# Run demo (fast)
python3 -m memopt.cli --demo lazy-kv

# Run demo (with validation - slow)
python3 -m memopt.cli --demo lazy-kv --validate-hardware

# Get validation setup instructions
python3 -m memopt.cli --validation-setup
```

---

## 🚀 Next Steps

Once tests pass:

1. **For development:** Run examples in `examples/` directory
2. **For customer demos:** Install Nsight Compute and test `--validate-hardware`
3. **For production:** Integrate into your inference pipeline

---

## 📖 Full Documentation

- **GPU_TESTING_GUIDE.md** - Complete testing procedures
- **PYTORCH_COMPATIBILITY_FIX.md** - Detailed fix explanation
- **DEPLOYMENT_GUIDE.txt** - Full deployment documentation

---

**Good luck! The fix is simple and tests should pass in under 10 minutes.** 🎉
