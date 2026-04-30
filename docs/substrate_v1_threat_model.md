# memopt Substrate v1 — Threat Model

| Field | Value |
| --- | --- |
| Status | v1.0 |
| Authoritative | `docs/substrate_v1_design.md` §2.6 |
| Step Zero report | `docs/substrate_v1_step_zero_report.md` |
| CVE references verified | 2026-04-29 |

This document lists what the substrate guarantees against multi-tenant
misuse, and what it does not. It is derived directly from design §2.6
and the Step Zero verification report (S0.7 VERIFIED).

## Guarantees

### G1 — Cross-tenant read/write denied

`MemoryHandle.read`, `write`, `as_tensor`, and `as_numpy` validate
`handle.tenant` against the calling thread's `memopt.context(tenant=...)`.
A mismatch raises `PermissionError` before any backend driver call runs.

> Example: alice allocates `h`. bob calls `h.read()` inside
> `memopt.context(tenant="bob")`. Result: `PermissionError`.

The check is one dict lookup + one string compare. Target overhead:
< 100 ns on the read/write path (§2.10).

### G2 — Stats namespacing

`memopt.stats(tenant="alice")` returns alice's data only.
`memopt.stats(tenant=None)` requires both:

  1. `MEMOPT_ADMIN_TOKEN` env var configured.
  2. The calling thread holds a context that resolves to that token.

The token is compared in constant time via `hmac.compare_digest`.

> Example: a non-admin worker thread calling `memopt.stats(tenant=None)`
> raises `PermissionError`. Adding the env var alone is not sufficient
> if the worker's context does not resolve to it.

### G3 — NVMe paths are tenant-namespaced

Spilled pages live under `<MEMOPT_NVME_DIR>/<tenant_hash>/...` where
`tenant_hash` is `hashlib.sha256(tenant.encode()).hexdigest()[:32]`.
The directory is created with mode `0700`. The legacy `_sanitize_id`
regex (`^[A-Za-z0-9_-]{1,128}$`) is applied BEFORE hashing as defence in
depth. Final paths are checked with `os.path.realpath` to catch symlink
escapes.

> Example: tenant id `"../../etc"` is rejected by the regex before any
> path is built (`ValueError`).
> Example: a symlink inside `<MEMOPT_NVME_DIR>/<tenant_hash>` that
> points outside the root raises `ValueError("escape")`.

### G4 — Tags are NOT a security boundary

Two tenants may use the same tag string (e.g. `"kv_cache"`). Tags are
routing/observability keys; they do not isolate. This is documented at
the top of the `memopt.alloc` docstring.

> Example: alice and bob both `memopt.alloc(N, tag="kv")` — both
> succeed; no collision error. Their handles still cannot read each
> other (G1 still applies via tenant check).

## Non-guarantees

### N1 — Hardware side channels

Cache timing, power analysis, Rowhammer-class attacks, and GPU memory
reuse residue are out of scope. For adversarial multi-tenant
deployments, use NVIDIA Confidential Computing (CC mode on H100/H200/
B200) or Multi-Instance GPU (MIG). The substrate cooperates with MIG
(one substrate per MIG slice) but does not provide MIG-equivalent
isolation on its own.

### N2 — Direct driver bypass

A tenant with `cudaMalloc` / `cuMemCreate` access can allocate outside
memopt entirely. memopt is opt-in. To force opt-in, install memopt as
PyTorch's allocator (existing `torch_allocator` path) and run inside a
process namespace where the allocator cannot be replaced.

> Example: a tenant calls `torch.empty(N, device='cuda')` directly. The
> resulting buffer is invisible to `memopt.stats()`.

### N3 — Driver-level vulnerabilities

The substrate is not a substitute for keeping drivers patched. Verified
references (S0.7 VERIFIED, 2026-04-29):

  - **CVE-2025-23266** — NVIDIA Container Toolkit: arbitrary code
    execution with elevated permissions.
  - **CVE-2025-33220** — NVIDIA vGPU software: heap use-after-free in
    the Virtual GPU Manager (CWE-416).
  - **CVE-2025-23352** — NVIDIA vGPU software: uninitialized pointer
    access in the Virtual GPU Manager.

All three are container-level / vGPU-level / driver-level issues.
memopt's tenant model is not a defence against them.

### N4 — Bus-level peer access

The substrate controls `cuMemSetAccess` for handles it manages. A
process with direct CUDA driver access can call `cuMemSetAccess` on its
own allocations independently. Same caveat as N2.

## Summary

The substrate provides **logical** tenant isolation suitable for
cooperative multi-tenant workloads (e.g. multiple training jobs sharing
a host). It is **not** a hardware security boundary.

For adversarial multi-tenancy, layer the substrate inside MIG slices or
Confidential Computing partitions; the substrate's guarantees compose
with hardware isolation, they do not replace it.
