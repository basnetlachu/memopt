# Stage 3: KV Cache Prefix Sharing - Implementation Plan

## Overview

**Goal:** Add prefix sharing to KV cache to eliminate redundant computation for common prompt prefixes.

**Target Speedup:** 1.2-1.5× additional improvement (on top of Stage 1 + Stage 2)

**Risk Level:** 🟢 Low (all changes behind feature flags, backward compatible)

---

## What is Prefix Sharing?

When multiple requests share the same prompt prefix (e.g., system prompts):

```
Request 1: "[SYSTEM] You are a helpful assistant. [USER] What is 2+2?"
Request 2: "[SYSTEM] You are a helpful assistant. [USER] What is 3+3?"
```

Both share the prefix: `"[SYSTEM] You are a helpful assistant. [USER]"`

**Without prefix sharing:**
- Compute KV cache for full prompt in Request 1
- Compute KV cache for full prompt in Request 2 (redundant!)

**With prefix sharing:**
- Compute KV cache for shared prefix once
- Share the same KV cache blocks between requests
- Only compute KV for unique suffix

---

## Implementation Strategy

### 1. Add Prefix Detection to KV Cache

**File:** `memopt/kv_cache.py`

**Changes:**
- Add `prefix_cache` dict to track prefix hashes → block IDs
- Add `find_prefix_match()` method to detect shared prefixes
- Modify `allocate_blocks()` to reuse prefix blocks when match found
- Increment ref counts for shared blocks

**New fields:**
```python
# In PagedKVCache.__init__()
self.enable_prefix_sharing = enable_prefix_sharing  # Feature flag
self.prefix_cache: Dict[str, List[int]] = {}  # prefix_hash -> block_ids
self.prefix_min_length = 32  # Minimum tokens for prefix sharing
```

**New methods:**
```python
def compute_prefix_hash(self, token_ids: List[int]) -> str:
    """Compute hash of token sequence for prefix matching"""
    # Use first N tokens as prefix identifier
    # Hash to string for fast dict lookup

def find_prefix_match(self, token_ids: List[int]) -> Optional[Tuple[str, List[int]]]:
    """Find longest matching prefix in cache"""
    # Check progressively shorter prefixes
    # Return (prefix_hash, shared_block_ids) if match found

def register_prefix(self, prefix_hash: str, block_ids: List[int]):
    """Register a new prefix for future sharing"""
    # Store prefix_hash -> block_ids mapping
    # Increment ref counts on shared blocks
```

### 2. Integrate with Model Generation

**File:** `memopt/model.py`

**Changes:**
- Pass `enable_prefix_sharing` flag to KV cache
- Detect prefix matches before generation
- Skip computation for shared prefix tokens

**Modified logic in `generate()`:**
```python
# Before forward pass
if self.kv_cache and self.kv_cache.enable_prefix_sharing:
    prefix_match = self.kv_cache.find_prefix_match(input_ids[0].tolist())

    if prefix_match:
        prefix_hash, shared_blocks = prefix_match
        # Skip prefix tokens, only generate from suffix
        # KV cache blocks already populated for prefix
```

### 3. Add Feature Flag to Presets

**File:** `memopt/model.py`

**Changes to OPTIMIZATION_PRESETS:**

```python
"conservative": {
    ...
    "enable_prefix_sharing": False,  # Stage 3 disabled
},
"balanced": {
    ...
    "enable_prefix_sharing": False,  # Stage 3 disabled
},
"high": {
    ...
    "enable_prefix_sharing": False,  # Stage 3 disabled (Stage 2 only)
},
"aggressive": {
    ...
    "enable_prefix_sharing": True,  # Stage 3 enabled
},
```

**Add new preset:**
```python
"maximum": {  # New preset for Stage 3
    "quantize_kv": False,
    "use_paged_cache": True,
    "use_flash_attention": True,
    "kv_block_size": 16,
    "enable_adaptive_allocation": True,
    "enable_workspace_reuse": True,
    "use_torch_compile": True,
    "use_continuous_batching": True,
    "enable_prefix_sharing": True,  # Stage 3
},
```

---

## When Prefix Sharing Helps

### ✅ Use Cases with High Benefit

1. **Chatbots with System Prompts**
   ```
   Every request: "[SYSTEM] You are a helpful assistant. [USER] <question>"
   Prefix: "[SYSTEM] You are a helpful assistant."
   Savings: 50-100+ tokens per request
   ```

2. **API with Templates**
   ```
   Every request: "Translate to French: <text>"
   Prefix: "Translate to French: "
   Savings: 4-8 tokens per request
   ```

3. **Multi-turn Conversations**
   ```
   Turn 1: "Hello"
   Turn 2: "Hello\nHow are you?"  ← reuses "Hello" prefix
   Turn 3: "Hello\nHow are you?\nWhat's the weather?"  ← reuses longer prefix
   ```

### ❌ Use Cases with No Benefit

1. **Unique prompts** - no shared prefixes
2. **Short prompts** - prefix sharing overhead not worth it
3. **Single-shot requests** - no repeat prefixes

---

## Performance Analysis

### Expected Speedup Calculation

**Assumptions:**
- Average prompt length: 100 tokens
- Shared prefix length: 50 tokens (system prompt)
- Prefix sharing hit rate: 80% (8 out of 10 requests share prefix)

**Without prefix sharing:**
- Compute 100 tokens × 10 requests = 1000 token computations

**With prefix sharing:**
- Compute 100 tokens for first request (no match yet)
- Compute 50 tokens × 9 subsequent requests (reuse prefix)
- Total: 100 + 450 = 550 token computations
- **Speedup: 1000 / 550 = 1.82×**

**Conservative estimate (50% hit rate, 30% prefix ratio):**
- Speedup: ~1.2-1.3×

**Realistic estimate (80% hit rate, 50% prefix ratio):**
- Speedup: ~1.4-1.5×

---

## Implementation Checklist

### Phase 1: Core Implementation
- [ ] Add prefix tracking fields to `PagedKVCache`
- [ ] Implement `compute_prefix_hash()`
- [ ] Implement `find_prefix_match()`
- [ ] Implement `register_prefix()`
- [ ] Modify `allocate_blocks()` to handle shared blocks
- [ ] Update ref counting for shared blocks

### Phase 2: Integration
- [ ] Add `enable_prefix_sharing` parameter to model
- [ ] Pass flag from preset to KV cache
- [ ] Integrate prefix matching into generation loop
- [ ] Add "maximum" preset for Stage 3

### Phase 3: Testing
- [ ] Unit tests for prefix detection
- [ ] Unit tests for block sharing
- [ ] Correctness tests (outputs must be identical)
- [ ] Performance benchmark (Stage 2 vs Stage 3)

### Phase 4: Documentation
- [ ] STAGE3_README.md - implementation details
- [ ] STAGE3_SUMMARY.md - quick reference
- [ ] Update benchmark scripts
- [ ] Usage examples

---

## Safety Guarantees

✅ **Backward Compatible:**
- Feature flag defaults to `False` (disabled)
- Stage 0, 1, 2 behavior unchanged
- Can disable via `enable_prefix_sharing=False`

✅ **Correctness Preserved:**
- Block sharing uses reference counting (already implemented)
- No data corruption - blocks are immutable once written
- Prefix matching is exact (hash-based)

✅ **No Breaking Changes:**
- API unchanged
- Existing presets unchanged (conservative, balanced, high)
- New preset only ("maximum")

✅ **Rollback Available:**
- Set `enable_prefix_sharing=False`
- Use existing presets (conservative, balanced, high)
- No data migration needed

---

## Testing Strategy

### Unit Tests

**File:** `tests/test_stage3.py`

```python
def test_prefix_hash_computation():
    """Test prefix hash is consistent"""

def test_prefix_match_detection():
    """Test finding matching prefixes"""

def test_block_sharing():
    """Test blocks are shared correctly"""

def test_ref_counting():
    """Test ref counts updated correctly"""

def test_correctness_with_sharing():
    """Test outputs identical with/without sharing"""
```

### Performance Benchmark

**File:** `benchmark_stage3.py`

Compare Stage 2 (high) vs Stage 3 (maximum):
- Same prompts with shared prefixes
- Measure speedup
- Validate correctness
- Check memory usage
- Verify cache hit rates

---

## Expected Benchmark Results

### Stage 2 (high) - Baseline
```
Throughput: 264 tok/s (with torch.compile on Linux)
Latency: 3.79 ms/tok
Memory: 3.81 GB
```

### Stage 3 (maximum) - With Prefix Sharing
```
Throughput: 317-396 tok/s (1.2-1.5× faster)
Latency: 2.53-3.16 ms/tok
Memory: 3.60-3.70 GB (shared blocks save memory)
Cache hit rate: 70-85%
```

**Windows without torch.compile:**
- Stage 2: ~228 tok/s
- Stage 3: ~274-342 tok/s (1.2-1.5× from prefix sharing alone)

---

## Limitations

### Current Limitations

1. **Exact Prefix Matching Only**
   - Requires exact token-for-token match
   - Small variations break sharing (e.g., "Hello" vs "Hi")
   - Future: Semantic similarity matching

2. **Fixed Prefix Length**
   - Currently checks fixed-length prefixes
   - Future: Variable-length prefix detection

3. **No Eviction Policy**
   - Prefix cache grows indefinitely
   - Future: LRU eviction for prefix cache

### When NOT to Use Stage 3

- Unique prompts every request (no shared prefixes)
- Very short prompts (<32 tokens)
- Memory-constrained systems (prefix cache overhead)
- Single-request workloads

---

## Next Steps After Stage 3

**Stage 4:** Speculative Decoding (~1.3-1.5× additional)
**Stage 5:** Attention Kernel Specialization (~1.2× additional)

**Cumulative speedup path:**
- Stage 0: 1.0× (conservative baseline)
- Stage 1: 1.2-1.3× (with torch.compile)
- Stage 2: 1.3-1.4× (continuous batching)
- Stage 3: 1.2-1.5× (prefix sharing)
- **Total: 1.87-2.73× over conservative baseline**

Combined with existing optimizations (FP16, paged KV, flash attention):
- **Total target: 12-15× over unoptimized baseline**

---

## Files to Modify

### Core Implementation
1. **`memopt/kv_cache.py`** - Add prefix sharing logic
2. **`memopt/model.py`** - Integrate prefix sharing, add "maximum" preset

### Testing
3. **`tests/test_stage3.py`** - Unit tests (new file)
4. **`benchmark_stage3.py`** - Performance benchmark (new file)

### Documentation
5. **`STAGE3_README.md`** - Implementation guide (new file)
6. **`STAGE3_SUMMARY.md`** - Quick summary (new file)

### No Changes Required
- ✅ `memopt/scheduler.py` - Unchanged
- ✅ `memopt/attention.py` - Unchanged
- ✅ `memopt/memory_manager.py` - Unchanged
- ✅ All existing tests and benchmarks

---

## Summary

**Stage 3 adds KV cache prefix sharing:**
- ✅ Safe: Feature flag controlled, backward compatible
- ✅ Targeted: Only helps when prompts share prefixes
- ✅ Measurable: 1.2-1.5× speedup for typical chatbot workloads
- ✅ Non-breaking: No API changes, full rollback available

**Implementation approach:**
- Minimal code changes (mostly in kv_cache.py)
- Reuses existing block sharing infrastructure (ref counting)
- Hash-based prefix detection (fast)
- Comprehensive testing (correctness + performance)

**Ready to implement when torch.compile gains are confirmed.**

---

**Status:** 📋 Planning Complete, Ready for Implementation
**Risk Level:** 🟢 Low
**Estimated LOC:** ~200-300 lines (mostly in kv_cache.py)
**Testing Time:** ~30 minutes
