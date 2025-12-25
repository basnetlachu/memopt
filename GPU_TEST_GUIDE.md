# GPU Test Guide - torch.compile Validation

## Current Status

Your Windows benchmark results show:
- **Stage 0 (Conservative):** 228.2 tok/s - baseline with optimizations (FP16, paged KV, flash attention)
- **Stage 1 (Balanced):** 230.9 tok/s - 1.01× speedup (~same performance)
- **Stage 2 (High):** 228.3 tok/s - 1.00× speedup (~same performance)

**Why no improvement?**
- torch.compile doesn't work on Windows (expected 15-30% gain missing)
- Adaptive allocation: ~1% gain (seen in Stage 1)
- Continuous batching: no gain with sequential requests (expected for single-threaded test)

---

## What to Test on GPU (Linux)

### 1. Confirm torch.compile Works

Run the same benchmark on Linux/Mac with GPU:

```bash
python benchmark_all_stages.py --model gpt2-xl --num-prompts 8
```

**Expected output:**
```
Loading model with balanced optimization...
  Applying torch.compile optimization...
  ✓ torch.compile enabled  # ← Should NOT show warning on Linux
```

### 2. Expected Performance with torch.compile

**Stage 1 (Balanced) - Expected:**
- Throughput: **262-296 tok/s** (1.15-1.30× vs Stage 0)
- torch.compile working ✓
- Adaptive allocation working ✓

**Stage 2 (High) - Expected:**
- Throughput: **274-320 tok/s** (1.20-1.40× vs Stage 0)
- torch.compile working ✓
- Continuous batching working ✓

**If torch.compile works:**
- Stage 1: Should show 15-30% gain over Stage 0
- Stage 2: Should show additional 5-10% gain over Stage 1

---

## Decision Tree After GPU Test

### ✅ If torch.compile Shows 15-30% Gain:

**Action:** Proceed with Stage 3 implementation

**Stage 3 target:**
- Additional 1.2-1.5× speedup from prefix sharing
- Total cumulative: Stage 0 → Stage 1 (1.2×) → Stage 2 (1.3×) → Stage 3 (1.4×) = **2.2× total**

**I will implement:**
- KV cache prefix sharing
- Hash-based prefix detection
- Block sharing for common prefixes
- New "maximum" preset
- Full testing and benchmarking

### ❌ If torch.compile Shows <10% Gain:

**Action:** Investigate why torch.compile isn't helping

**Possible causes:**
- Model too small (GPT-2 XL = 1.5B params, torch.compile overhead may dominate)
- GPU memory bandwidth not the bottleneck
- Need different torch.compile settings

**Options:**
1. Try larger model (Llama-2-7B or Llama-2-13B)
2. Tune torch.compile settings (mode="max-autotune")
3. Skip torch.compile, focus on other optimizations

### ⚠️ If torch.compile Still Fails on Linux:

**Action:** Remove torch.compile from Stage 1/Stage 2

**Fallback plan:**
- Keep adaptive allocation (minimal gain but safe)
- Keep continuous batching (for API server use case)
- Focus Stage 3 on prefix sharing (doesn't depend on torch.compile)

---

## Test Commands

### Quick Test (GPT-2 XL, 5 minutes)
```bash
python benchmark_all_stages.py --model gpt2-xl --num-prompts 8 --max-tokens 256
```

### Full Test (Llama-2-7B, 15 minutes)
```bash
python benchmark_all_stages.py --model meta-llama/Llama-2-7b-hf --num-prompts 8 --max-tokens 512
```

### Comprehensive Test (Llama-2-13B, 30 minutes)
```bash
python benchmark_all_stages.py --model meta-llama/Llama-2-13b-hf --num-prompts 8 --max-tokens 512
```

---

## What to Look For

### 1. torch.compile Status

**Success:**
```
Loading model with balanced optimization...
  Applying torch.compile optimization...
  ✓ Model compiled successfully
```

**Failure:**
```
Loading model with balanced optimization...
  Applying torch.compile optimization...
  ⚠️  torch.compile failed (...), continuing without it
```

### 2. Performance Numbers

**Stage 1 vs Stage 0:**
- Look for 1.15-1.30× speedup
- If less than 1.10×, torch.compile not helping much

**Stage 2 vs Stage 1:**
- Look for 1.05-1.15× speedup
- Modest gain expected (sequential workload)

### 3. Correctness

**Critical:**
```
✅ ALL STAGES PRODUCE IDENTICAL OUTPUTS
```

**If correctness fails:**
- STOP - do not proceed to Stage 3
- Report which stage failed
- Investigate issue before continuing

---

## Benchmark Results Format

Save the output and send me:

```
Stage 0 (Conservative): XXX tok/s
Stage 1 (Balanced): XXX tok/s (X.XX× speedup)
Stage 2 (High): XXX tok/s (X.XX× speedup)

torch.compile status: [working/failed]
Correctness: [PASSED/FAILED]
```

**Example good result:**
```
Stage 0: 228 tok/s
Stage 1: 296 tok/s (1.30× speedup) ✓
Stage 2: 320 tok/s (1.40× speedup) ✓

torch.compile: working
Correctness: PASSED
→ PROCEED TO STAGE 3
```

**Example bad result:**
```
Stage 0: 228 tok/s
Stage 1: 232 tok/s (1.02× speedup)
Stage 2: 230 tok/s (1.01× speedup)

torch.compile: working but no gain
Correctness: PASSED
→ INVESTIGATE why torch.compile not helping
```

---

## After You Confirm Results

### If torch.compile shows 15%+ gain:

**Reply:** "torch.compile working, proceed stage 3"

**I will:**
1. Implement KV cache prefix sharing
2. Add "maximum" preset
3. Create Stage 3 tests
4. Create Stage 3 benchmark
5. Deliver in ~30 minutes

### If torch.compile shows <10% gain:

**Reply:** "torch.compile gain only X%, what next?"

**I will:**
- Analyze why gain is low
- Suggest alternative optimizations
- Possibly skip torch.compile, focus on prefix sharing

### If any issues:

**Reply:** "issue: <description>"

**I will:**
- Debug the specific issue
- Fix without breaking existing code
- Ensure correctness maintained

---

## Stage 3 Preview (If Proceeding)

**What Stage 3 adds:**
- KV cache prefix sharing
- Detects common prompt prefixes (e.g., system prompts)
- Shares KV blocks across requests with same prefix
- Eliminates redundant computation

**Expected gain:**
- 1.2-1.5× additional speedup
- Best for chatbots with system prompts
- Best for multi-turn conversations

**Implementation:**
- ~200-300 lines of code
- All in kv_cache.py + model.py
- Feature flag controlled
- Full testing included

**Risk level:** 🟢 Low (backward compatible, feature flag)

---

## Summary

**Current state:**
- Stage 1 and Stage 2 implemented ✓
- Tests passing ✓
- Correctness validated ✓
- Performance limited by Windows (no torch.compile)

**Next step:**
- Test on Linux GPU
- Confirm torch.compile provides 15-30% gain
- If yes → proceed to Stage 3
- If no → investigate or pivot

**Stage 3 ready to implement when you confirm torch.compile gains.**

---

## Questions to Answer from GPU Test

1. Does torch.compile work on Linux? (yes/no)
2. What is Stage 1 speedup vs Stage 0? (X.XX×)
3. What is Stage 2 speedup vs Stage 1? (X.XX×)
4. Did correctness tests pass? (yes/no)
5. Proceed to Stage 3? (yes/no)

**I'll wait for your GPU test results before implementing Stage 3.**

No exaggeration. Just real numbers and honest assessment.
