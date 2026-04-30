r"""Tenant identity, validation, and per-tenant NVMe path generation.

Per design §2.6 G3, NVMe paths are namespaced via
`<MEMOPT_NVME_DIR>/<sha256(tenant)[:32]>/...` with mode 0700 and a
realpath escape-check. The legacy `_sanitize_id` regex
`^[A-Za-z0-9_\-]{1,128}$` is applied BEFORE hashing as defence in depth.
"""
from __future__ import annotations

import contextvars
import hashlib
import os
import re
from contextlib import contextmanager
from typing import Iterator, Optional


_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,128}$")


def _sanitize_id(kind: str, value: str) -> str:
    if not isinstance(value, str) or not _ID_RE.match(value):
        raise ValueError(f"invalid {kind}: must match [A-Za-z0-9_-]{{1,128}}")
    return value


_current_tenant: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "memopt_current_tenant", default=None
)
_current_tag: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "memopt_current_tag", default=None
)
_current_placement: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "memopt_current_placement", default=None
)


@contextmanager
def tenant_context(
    tenant: Optional[str] = None,
    tag: Optional[str] = None,
    placement: Optional[str] = None,
) -> Iterator[None]:
    if tenant is not None:
        _sanitize_id("tenant", tenant)
    tokens = []
    if tenant is not None:
        tokens.append(_current_tenant.set(tenant))
    if tag is not None:
        tokens.append(_current_tag.set(tag))
    if placement is not None:
        tokens.append(_current_placement.set(placement))
    try:
        yield
    finally:
        for tok in reversed(tokens):
            tok.var.reset(tok)


def current_tenant() -> Optional[str]:
    return _current_tenant.get()


def current_tag() -> Optional[str]:
    return _current_tag.get()


def current_placement() -> Optional[str]:
    return _current_placement.get()


def assert_tenant_match(handle_tenant: str) -> None:
    """G1 enforcement: the calling thread's tenant context, if set, must
    match the handle's tenant. PermissionError on mismatch."""
    cur = _current_tenant.get()
    if cur is None:
        return
    if cur != handle_tenant:
        raise PermissionError(
            f"tenant context '{cur}' does not match handle tenant "
            f"'{handle_tenant}' (G1 isolation)"
        )


def tenant_hash(tenant: str) -> str:
    """sha256 prefix of the tenant id, used as the on-disk namespace."""
    _sanitize_id("tenant", tenant)
    return hashlib.sha256(tenant.encode("utf-8")).hexdigest()[:32]


def nvme_block_path(
    nvme_dir: str,
    tenant: str,
    sequence_id: str,
    block_index: int,
) -> str:
    """G3-compliant tenant-namespaced NVMe block path.

    Layout: <nvme_dir>/<tenant_hash>/<sequence_id>_<block_index>.vmm_block
    The directory is created with mode 0700. The realpath of the result
    must be inside realpath(nvme_dir); a symlink that escapes raises
    ValueError.
    """
    safe_seq = _sanitize_id("sequence_id", sequence_id)
    if not isinstance(block_index, int) or block_index < 0:
        raise ValueError("block_index must be non-negative int")
    th = tenant_hash(tenant)
    tenant_dir = os.path.join(nvme_dir, th)
    os.makedirs(tenant_dir, mode=0o700, exist_ok=True)
    try:
        os.chmod(tenant_dir, 0o700)
    except OSError:
        pass
    candidate = os.path.join(tenant_dir, f"{safe_seq}_{block_index}.vmm_block")
    real_root = os.path.realpath(nvme_dir)
    real_path = os.path.realpath(candidate)
    if not real_path.startswith(real_root + os.sep) and real_path != real_root:
        raise ValueError(
            "tenant nvme path escapes the configured root via symlink "
            "(G3 path-traversal guard)"
        )
    return candidate
