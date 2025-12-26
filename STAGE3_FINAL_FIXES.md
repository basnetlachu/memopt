# Stage 3 Final Fixes - Reference Counting Correction

## Issue Identified

The Stage 3 implementation had an incorrect reference counting strategy that would cause test failures and potential memory management issues.

### Problem

**Incorrect behavior**:
- `register_prefix()` was incrementing reference counts when storing prefix metadata
- This caused ref counts to be too high (double-counting)
- Test "Free with shared blocks" would fail because ref counts were 2 instead of expected 1

**Example scenario**:
```python
# Allocate blocks for seq_id_1
blocks = [0, 1, 2, 3]  # ref_counts = {0:1, 1:1, 2:1, 3:1}

# Register prefix
register_prefix(tokens, [0, 1])  # WRONG: incremented to {0:2, 1:2, 2:1, 3:1}

# Allocate with sharing for seq_id_2
allocate_with_prefix_sharing(seq_id_2, tokens, 4)  # WRONG: incremented to {0:3, 1:3, ...}

# Free seq_id_1
free_sequence(seq_id_1)  # ref_counts = {0:2, 1:2, ...}
# Test expected {0:1, 1:1} but got {0:2, 1:2} ❌
```

## Solution

**Correct behavior**: Reference counts should **only** be incremented when blocks are **actually allocated to a sequence**, not when just registering metadata.

### Changes Made

#### 1. Fixed `register_prefix()` in [memopt/kv_cache.py](memopt/kv_cache.py#L449-L463)

**Before**:
```python
def register_prefix(self, token_ids: List[int], block_ids: List[int]):
    if not self.enable_prefix_sharing or len(token_ids) < self.prefix_min_length:
        return

    prefix_hash = self.compute_prefix_hash(token_ids)
    if prefix_hash and prefix_hash not in self.prefix_cache:
        # Store prefix mapping
        self.prefix_cache[prefix_hash] = block_ids.copy()

        # Increment ref counts for shared blocks  ❌ WRONG
        for block_id in block_ids:
            if block_id in self.block_ref_counts:
                self.block_ref_counts[block_id] += 1
```

**After**:
```python
def register_prefix(self, token_ids: List[int], block_ids: List[int]):
    if not self.enable_prefix_sharing or len(token_ids) < self.prefix_min_length:
        return

    prefix_hash = self.compute_prefix_hash(token_ids)
    if prefix_hash and prefix_hash not in self.prefix_cache:
        # Store prefix mapping (just metadata, ref counts unchanged) ✅ CORRECT
        self.prefix_cache[prefix_hash] = block_ids.copy()
```

#### 2. Fixed test `test_prefix_reference_counting()` in [tests/test_stage3.py](tests/test_stage3.py#L167-L196)

**Updated test to verify correct behavior**:
1. Register prefix → ref counts should **stay at 1** (not increment)
2. Find prefix match → ref counts should **stay at 1** (just a lookup)
3. Allocate with prefix sharing → ref counts should **increment to 2** (actual usage)

**After**:
```python
def test_prefix_reference_counting(self, kv_cache):
    """Test that shared blocks are reference counted."""
    tokens = list(range(64))
    blocks = [0, 1, 2, 3]

    # Allocate blocks first
    for block_id in blocks:
        kv_cache.free_blocks.discard(block_id)
        kv_cache.block_ref_counts[block_id] = 1

    kv_cache.register_prefix(tokens, blocks)

    # Find match (should NOT increment ref count - just returns match)
    match = kv_cache.find_prefix_match(tokens)
    assert match is not None

    # Ref counts should still be 1 (find doesn't increment)
    for block_id in blocks:
        assert kv_cache.block_ref_counts[block_id] == 1

    # Now actually use the prefix via allocate_with_prefix_sharing
    seq_id = 1
    allocated_blocks, shared_count = kv_cache.allocate_with_prefix_sharing(
        seq_id, tokens, 4
    )

    # NOW ref counts should be incremented for shared blocks
    assert shared_count == 4
    for block_id in blocks:
        assert kv_cache.block_ref_counts[block_id] == 2
```

## Correct Reference Counting Strategy

### When Reference Counts Are Incremented

1. **`allocate_blocks(seq_id, num_blocks)`**: Sets ref_count = 1 for newly allocated blocks
2. **`allocate_with_prefix_sharing(seq_id, token_ids, num_blocks)`**: Increments ref_count for shared blocks (line 523-525)

### When Reference Counts Are Decremented

1. **`free_sequence(seq_id)`**: Decrements ref_count for all blocks in the sequence

### When Reference Counts Are NOT Changed

1. **`register_prefix(token_ids, block_ids)`**: Just stores metadata mapping
2. **`find_prefix_match(token_ids)`**: Just returns matching prefix (lookup only)

## Test Trace Example

**Correct trace with fix**:

```python
# Step 1: Allocate seq_id_1
blocks_1 = allocate_blocks(0, 4)  # [0, 1, 2, 3]
# ref_counts = {0:1, 1:1, 2:1, 3:1}

# Step 2: Register prefix (metadata only)
register_prefix(tokens, [0, 1])
# ref_counts = {0:1, 1:1, 2:1, 3:1}  ← Unchanged ✅

# Step 3: Allocate seq_id_2 with sharing
blocks_2, shared = allocate_with_prefix_sharing(1, tokens, 4)
# shared_blocks = [0, 1], new_blocks = [4, 5]
# blocks_2 = [0, 1, 4, 5]
# ref_counts = {0:2, 1:2, 2:1, 3:1, 4:1, 5:1}  ← Only shared blocks incremented ✅

# Step 4: Free seq_id_1
free_sequence(0)
# Decrements: 0(2→1), 1(2→1), 2(1→0), 3(1→0)
# ref_counts = {0:1, 1:1, 4:1, 5:1}  ← Matches test expectation ✅
# free_blocks gains: {2, 3}

# Step 5: Free seq_id_2
free_sequence(1)
# Decrements: 0(1→0), 1(1→0), 4(1→0), 5(1→0)
# ref_counts = {}  ← All freed ✅
# free_blocks gains: {0, 1, 4, 5}
```

## Additional Fix: Test Assertion Update

### Issue in `test_free_with_shared_blocks`

The test expected `block_ref_counts[block_id] == 0` for freed blocks, but the implementation **deletes** the ref count entry when it reaches 0 (line 185 in kv_cache.py):

```python
if self.block_ref_counts[block_id] == 0:
    self.free_blocks.add(block_id)
    self.stats.used_pages -= 1
    del self.block_ref_counts[block_id]  # Entry deleted
```

This is **correct behavior** - no need to track ref counts for freed blocks.

**Updated test** (lines 416-420):
```python
# NOW shared blocks should be freed
for block_id in prefix_blocks:
    assert block_id in kv_cache.free_blocks
    # Ref count should be deleted (not tracked for freed blocks)
    assert block_id not in kv_cache.block_ref_counts
```

## Additional Fix: Test Fixture Isolation

### Issue in Standalone Test Execution

The standalone test runner (lines 434-453) was reusing the same `kv_cache_int` instance across multiple integration tests. This caused **state pollution** where blocks allocated in "Write and read with prefix sharing" affected "Free with shared blocks".

**Updated test runner** (lines 451-452):
```python
# Each integration test gets a fresh cache instance
("Write and read with prefix sharing", lambda: test_integration.test_write_and_read_with_prefix_sharing(test_integration.kv_cache())),
("Free with shared blocks", lambda: test_integration.test_free_with_shared_blocks(test_integration.kv_cache())),
```

## Impact

### Fixed Tests
- ✅ `test_prefix_reference_counting`: Now tests correct behavior
- ✅ `test_free_with_shared_blocks`: Fixed assertion (check deleted, not == 0) + fresh cache
- ✅ All 11 Stage 3 tests should pass

### Memory Safety
- ✅ Ref counts accurately track block usage
- ✅ Blocks freed when no longer referenced
- ✅ No memory leaks from incorrect counting
- ✅ No premature frees from under-counting

### Correctness
- ✅ Shared blocks properly reference counted
- ✅ Free operations handle shared blocks correctly
- ✅ Multiple sequences can safely share prefix blocks

## Summary

**The fix ensures**:
1. Reference counts only change when blocks are **allocated** or **freed**
2. Metadata operations (`register_prefix`, `find_prefix_match`) don't affect ref counts
3. Shared blocks are properly tracked when actually used
4. Memory management is safe and correct

**All Stage 3 tests should now pass** with correct reference counting semantics.
