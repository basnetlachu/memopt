# Deprecated Multi-GPU Code

This directory contains the old multi-GPU implementation that has been replaced by the production-grade worker-per-GPU architecture.

## Why These Files Were Moved

The original multi-GPU approach used:
- DataParallel-style model replication within a single process
- Multi-GPU routers for request distribution within one process
- Tensor/Model parallelism across GPUs in same process

**Problems with this approach:**
- Poor scaling (50-70% efficiency)
- Shared CUDA context overhead
- Complex synchronization
- Hard to debug
- Not production-ready

## Current Architecture (Production)

**Worker-Per-GPU Architecture** (`production/` directory):
- One OS process per GPU
- Complete isolation via `CUDA_VISIBLE_DEVICES`
- No shared CUDA context
- Near-linear scaling (95-100% efficiency)
- Production-proven (same as vLLM, TGI, Ray Serve)

## Files in This Directory

- `multi_gpu_router.py` - Old request router for single-process multi-GPU
- `rl_router_env.py` - RL-based routing (replaced by production load balancer)
- `model_parallel.py` - Tensor parallelism (not needed for inference)

## Impact of Removal

✅ **ZERO impact on production throughput**

These files were never imported by:
- `production/worker_service.py`
- `scripts/benchmark_production.sh`
- Any production code path

## If You Need This Code

For research or experimentation, these files are preserved here. However, for production use, always use the worker-per-GPU architecture in `production/`.

---

**Moved on**: 2026-01-21
**Reason**: Replaced by superior worker-per-GPU architecture
**Safe to delete**: After 2026-02-21 (30 days retention)
