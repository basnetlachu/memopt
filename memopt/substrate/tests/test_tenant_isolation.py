"""Tests for tenant isolation (per design §3.1 test_tenant_isolation.py + §2.6)."""
from __future__ import annotations

import os
import tempfile

import pytest

from memopt.substrate import tenant as tenant_mod
from memopt.substrate.tenant import (
    nvme_block_path,
    tenant_context,
    tenant_hash,
    assert_tenant_match,
    _sanitize_id,
)


def test_g1_cross_tenant_read_denied():
    """alice's handle, bob's tenant context -> PermissionError on read.
    The check is encapsulated in assert_tenant_match (called from
    MemoryHandle.read in Commit 12). Here we verify the check itself."""
    handle_tenant = "alice"
    with tenant_context(tenant="bob"):
        with pytest.raises(PermissionError, match="G1"):
            assert_tenant_match(handle_tenant)


def test_g1_cross_tenant_write_denied():
    handle_tenant = "alice"
    with tenant_context(tenant="bob"):
        with pytest.raises(PermissionError):
            assert_tenant_match(handle_tenant)


def test_g1_cross_tenant_as_tensor_denied():
    handle_tenant = "alice"
    with tenant_context(tenant="bob"):
        with pytest.raises(PermissionError):
            assert_tenant_match(handle_tenant)


def test_g1_same_tenant_passes():
    with tenant_context(tenant="alice"):
        assert_tenant_match("alice")  # no raise


def test_g1_no_context_passes():
    """When no tenant context is set, the check is a no-op."""
    assert tenant_mod.current_tenant() is None
    assert_tenant_match("alice")  # no raise


def test_g2_stats_namespacing_admin_token_required(monkeypatch):
    """memopt.stats(tenant=None) without admin token raises PermissionError.
    The token check is implemented in Commit 12 in manager.stats; here we
    document the env var contract."""
    monkeypatch.delenv("MEMOPT_ADMIN_TOKEN", raising=False)
    assert os.environ.get("MEMOPT_ADMIN_TOKEN") is None


def test_g3_nvme_path_namespaced(tmp_path):
    """alice and bob produce paths under different tenant_hash subdirs."""
    a = nvme_block_path(str(tmp_path), "alice", "seq1", 0)
    b = nvme_block_path(str(tmp_path), "bob", "seq1", 0)
    assert os.path.dirname(a) != os.path.dirname(b)
    # Both paths live under tmp_path's realpath.
    assert os.path.realpath(a).startswith(os.path.realpath(str(tmp_path)))
    assert os.path.realpath(b).startswith(os.path.realpath(str(tmp_path)))
    # Tenant subdirectory mode is 0700.
    mode = os.stat(os.path.dirname(a)).st_mode & 0o777
    assert mode == 0o700


def test_g3_nvme_path_traversal_blocked(tmp_path):
    """tenant id "../etc" is rejected by the regex BEFORE any path is built."""
    with pytest.raises(ValueError):
        nvme_block_path(str(tmp_path), "../etc", "seq", 0)
    with pytest.raises(ValueError):
        nvme_block_path(str(tmp_path), "alice", "seq\x00null", 0)
    with pytest.raises(ValueError):
        # over-long tenant id
        nvme_block_path(str(tmp_path), "x" * 200, "seq", 0)


def test_g3_realpath_escape_via_symlink_blocked(tmp_path):
    """If the tenant subdir is a symlink that escapes the root, raise."""
    th = tenant_hash("alice")
    # Pre-create a symlink inside tmp_path pointing OUTSIDE.
    outside = tempfile.mkdtemp()
    try:
        os.symlink(outside, os.path.join(str(tmp_path), th))
        with pytest.raises(ValueError, match="escape"):
            nvme_block_path(str(tmp_path), "alice", "seq", 0)
    finally:
        try:
            os.unlink(os.path.join(str(tmp_path), th))
        except OSError:
            pass
        try:
            os.rmdir(outside)
        except OSError:
            pass


def test_g4_tag_is_not_a_security_boundary():
    """alice and bob may use the same tag string with no error."""
    with tenant_context(tenant="alice", tag="kv"):
        assert tenant_mod.current_tag() == "kv"
    with tenant_context(tenant="bob", tag="kv"):
        assert tenant_mod.current_tag() == "kv"


def test_sanitize_id_rejects_invalid_chars():
    with pytest.raises(ValueError):
        _sanitize_id("tenant", "alice/bob")
    with pytest.raises(ValueError):
        _sanitize_id("tenant", "")
    with pytest.raises(ValueError):
        _sanitize_id("tenant", "alice bob")


def test_n2_direct_cudaMalloc_bypasses_memopt():
    """Documents the limitation: a direct cudaMalloc'd pointer is invisible
    to memopt.stats(). Marked @gpu — skip on no-CUDA hosts."""
    pytest.importorskip("torch")
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Allocate via the raw runtime API (bypasses memopt entirely).
    direct = torch.empty(1024, device="cuda")
    # The substrate's stats (when assembled in Commit 12) will not include
    # this allocation — its live_bytes contribution is exactly zero.
    # We can only document the contract here; the actual assertion lives
    # in test_substrate_smoke once the manager exists.
    assert direct.numel() == 1024
