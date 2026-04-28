"""
Tier manager — promotes and evicts KV blocks between memory tiers.

Promotion:  NVMe → DRAM → HBM  (page-fault path, triggered by fetch())
Eviction:   HBM → DRAM → NVMe  (pressure path, triggered by allocate())

Eviction policy: LRU among unpinned blocks.
Watermarks: configurable via MEMOPT_EVICT_HIGH (default 0.90)
and MEMOPT_EVICT_LOW (default 0.75).

Hardware-agnostic: all memory ops go through hal.backend.

GUM integration (optional):
  When remote_client and block_directory are provided, fetch() attempts
  cross-node block retrieval on local miss before raising KeyError.
  Eviction to NVMe registers the block in block_directory for peer access.
"""
from __future__ import annotations
import logging
import os
import time
from typing import Optional

# Import hal as a module so backend/tiers/tier_names resolve lazily
# via hal.__getattr__ on first attribute access. Doing
# `from .hal import backend` here would call hal._detect_backend()
# at import time, which calls torch.cuda.is_available() and
# initializes PyTorch's CUDA primary context — breaking the
# pluggable-allocator install path.
from . import hal as _hal
from .page_table import PageTable, PageTableEntry


def __getattr__(name):
    """Module-level lazy alias for backend/tiers/tier_names —
    keeps existing imports `from memopt.vmm.tier_manager import backend`
    working without forcing CUDA init at import time."""
    if name in ("backend", "tiers", "tier_names"):
        return getattr(_hal, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

logger = logging.getLogger(__name__)

# Default eviction watermarks (overridable via env vars).
# MEMOPT_EVICT_HIGH: fraction of tier capacity that triggers eviction (default 0.90)
# MEMOPT_EVICT_LOW:  fraction to evict down to (default 0.75)
# Tune per deployment:
#   HBM:  0.85/0.70 (aggressive — low latency cost for eviction)
#   NVMe: 0.95/0.80 (conservative — high latency cost for eviction)
_DEFAULT_EVICT_HIGH = 0.90
_DEFAULT_EVICT_LOW  = 0.75


class TierManager:

    def __init__(
        self,
        page_table: PageTable,
        remote_client=None,
        block_directory=None,
        node_id: str = "",
    ) -> None:
        self.page_table = page_table
        self._tier_capacity: dict[str, int] = {t.name: t.capacity_bytes for t in _hal.tiers}

        # GUM cross-node block sharing (optional)
        self._remote_client = remote_client
        self._block_directory = block_directory
        self._node_id = node_id or os.environ.get("MEMOPT_NODE_ID", "")

        # Eviction watermarks — configurable via environment
        evict_high = float(os.environ.get(
            'MEMOPT_EVICT_HIGH', str(_DEFAULT_EVICT_HIGH)))
        evict_low = float(os.environ.get(
            'MEMOPT_EVICT_LOW', str(_DEFAULT_EVICT_LOW)))

        if not (0.5 <= evict_low < evict_high <= 1.0):
            logger.warning(
                "Invalid eviction thresholds: low=%.2f high=%.2f. "
                "Requirement: 0.5 <= low < high <= 1.0. "
                "Using defaults %.2f/%.2f.",
                evict_low, evict_high,
                _DEFAULT_EVICT_LOW, _DEFAULT_EVICT_HIGH)
            evict_low = _DEFAULT_EVICT_LOW
            evict_high = _DEFAULT_EVICT_HIGH

        self._evict_high = evict_high
        self._evict_low = evict_low

    # ── Public API ────────────────────────────────────────────────────────

    def allocate(
        self,
        sequence_id: str,
        block_index: int,
        size_bytes: int,
        tenant_id: str = "_default",
    ) -> PageTableEntry:
        """
        Allocate a new block in the hottest available tier.
        Evicts LRU blocks first if tier is above the high watermark.
        """
        hot = _hal.tier_names[0]
        self._ensure_capacity(hot, size_bytes)
        handle = _hal.backend.allocate(size_bytes, hot, tenant_id=tenant_id)
        return self.page_table.insert(sequence_id, block_index, hot, handle, size_bytes)

    def fetch(self, sequence_id: str, block_index: int) -> PageTableEntry:
        """
        Ensure the block is in the hottest tier, promoting if needed.
        This is the page-fault handler.

        On local miss: attempts GUM remote fetch from peer nodes
        (if remote_client and block_directory are configured).
        Raises KeyError only after all local and remote sources exhausted.
        """
        entry = self.page_table.lookup(sequence_id, block_index)

        if entry is not None:
            # Block found locally — promote to hot tier if needed
            hot = _hal.tier_names[0]
            if entry.tier != hot:
                self._promote(entry, hot)
                entry = self.page_table.lookup(sequence_id, block_index)
            return entry

        # Block not in local page table — try GUM remote fetch
        if self._remote_client is not None \
                and self._block_directory is not None:
            remote_data = self._try_gum_fetch(sequence_id, block_index)
            if remote_data is not None:
                # Remote fetch succeeded — allocate locally and return
                hot = _hal.tier_names[0]
                self._ensure_capacity(hot, len(remote_data))
                handle = _hal.backend.allocate(len(remote_data), hot)
                # Copy remote data into local allocation
                try:
                    import torch
                    if isinstance(handle, torch.Tensor):
                        src = torch.frombuffer(
                            bytearray(remote_data), dtype=torch.uint8)
                        handle.copy_(src)
                    elif isinstance(handle, bytearray):
                        handle[:] = remote_data
                except Exception:
                    pass  # best effort copy
                new_entry = self.page_table.insert(
                    sequence_id, block_index, hot,
                    handle, len(remote_data))
                logger.debug(
                    "GUM: restored (%r, %d) from remote peer",
                    sequence_id, block_index)
                return new_entry

        raise KeyError(
            f"Block ({sequence_id!r}, {block_index}) "
            f"not in page table or remote peers")

    def evict_sequence(self, sequence_id: str) -> None:
        """Free all blocks for a completed sequence."""
        entries = self.page_table.remove_sequence(sequence_id)
        for e in entries:
            _hal.backend.free(e.handle, e.tier)
        if entries:
            logger.debug("Evicted %d blocks for sequence %r", len(entries), sequence_id)

    def stats(self) -> dict:
        """Return page table stats augmented with backend and tier info."""
        s = self.page_table.stats()
        s["backend"] = type(_hal.backend).__name__
        s["tiers_available"] = _hal.tier_names
        s["gum_enabled"] = (self._remote_client is not None
                            and self._block_directory is not None)
        try:
            from memopt.vmm.backends._cuda_backend_py import get_async_nvme_manager
            s["nvme_async"] = get_async_nvme_manager().stats()
        except Exception:
            s["nvme_async"] = {"async_available": False}
        return s

    # ── Internal ──────────────────────────────────────────────────────────

    def _promote(self, entry: PageTableEntry, target_tier: str) -> None:
        self._ensure_capacity(target_tier, entry.size_bytes)
        new_handle = _hal.backend.allocate(entry.size_bytes, target_tier)
        self._sync_copy(entry.handle, new_handle)
        old_tier, old_handle = entry.tier, entry.handle
        self.page_table.update_tier(entry.sequence_id, entry.block_index, target_tier, new_handle)
        _hal.backend.free(old_handle, old_tier)
        logger.debug(
            "Promoted (%r, %d): %s → %s",
            entry.sequence_id, entry.block_index, old_tier, target_tier,
        )

    def _evict_one(self, from_tier: str) -> bool:
        idx = _hal.tier_names.index(from_tier)
        if idx + 1 >= len(_hal.tier_names):
            logger.warning("Cannot evict from %r: no colder tier available", from_tier)
            return False

        cold_tier = _hal.tier_names[idx + 1]
        candidates = self.page_table.lru_candidates(from_tier, count=1)
        if not candidates:
            return False

        entry = candidates[0]
        new_handle = _hal.backend.allocate(entry.size_bytes, cold_tier)
        self._sync_copy(entry.handle, new_handle)
        old_handle = entry.handle
        self.page_table.update_tier(entry.sequence_id, entry.block_index, cold_tier, new_handle)
        _hal.backend.free(old_handle, from_tier)
        logger.debug(
            "Evicted (%r, %d): %s → %s",
            entry.sequence_id, entry.block_index, from_tier, cold_tier,
        )

        # GUM: register evicted block in directory for peer access
        if cold_tier == "nvme" and self._block_directory is not None:
            self._register_in_directory(entry, new_handle)

        return True

    def _register_in_directory(self, entry: PageTableEntry,
                                nvme_handle) -> None:
        """Register an NVMe-evicted block in the block directory."""
        try:
            from memopt.cluster.block_directory import BlockEntry
            content_hash = self._compute_block_hash(
                entry.sequence_id, entry.block_index)
            path = str(nvme_handle) if isinstance(nvme_handle, str) else ""
            self._block_directory.register(BlockEntry(
                content_hash=content_hash,
                node_id=self._node_id,
                tier="nvme",
                path=path,
                size_bytes=entry.size_bytes,
                registered_at=time.time(),
                lease_count=0,
            ))
            logger.debug(
                "GUM: registered evicted block %s in directory",
                content_hash[:16])
        except Exception as e:
            logger.debug("GUM register failed: %s", e)

    def _try_gum_fetch(self, sequence_id: str,
                        block_index: int) -> Optional[bytes]:
        """Try to fetch a block from a remote peer via GUM with geo routing."""
        try:
            content_hash = self._compute_block_hash(
                sequence_id, block_index)
            dir_entry = self._block_directory.lookup(content_hash)

            if dir_entry is None:
                return None
            if dir_entry.node_id == self._node_id:
                return None  # local entry — already checked

            remote_host = self._get_node_host(dir_entry.node_id)
            if not remote_host:
                return None

            logger.debug(
                "GUM: fetching block %s from peer %s",
                content_hash[:16], dir_entry.node_id)

            # Use geo-routed failover fetch
            try:
                from memopt.cluster.remote_block import GeoRouter
                from memopt.cluster.gum_metrics import get_gum_metrics

                geo_router = GeoRouter(
                    local_node_id=self._node_id,
                    local_rack=os.environ.get("MEMOPT_RACK", "unknown"),
                    local_pod=os.environ.get("MEMOPT_POD", "unknown"),
                    local_region=os.environ.get("MEMOPT_REGION", "unknown"))

                candidates = [{
                    "node_id": dir_entry.node_id,
                    "host": remote_host,
                    "port": int(os.environ.get("MEMOPT_RBP_PORT", "18516")),
                    "rack": getattr(dir_entry, "rack", "unknown"),
                    "pod": getattr(dir_entry, "pod", "unknown"),
                    "region": getattr(dir_entry, "region", "unknown"),
                }]

                data = self._remote_client.fetch_block_with_failover(
                    content_hash=content_hash,
                    candidates=candidates,
                    geo_router=geo_router,
                    metrics=get_gum_metrics(),
                    max_attempts=int(os.environ.get(
                        "MEMOPT_GUM_MAX_ATTEMPTS", "3")))

                if data is not None:
                    logger.info(
                        "GUM: fetched block %s from %s (%d bytes)",
                        content_hash[:16], dir_entry.node_id, len(data))
                return data

            except ImportError:
                # GeoRouter/GUMMetrics not available — direct fetch fallback
                data = self._remote_client.fetch_block(
                    content_hash, remote_host)
                if data is not None:
                    logger.info(
                        "GUM: fetched block %s from %s (%d bytes)",
                        content_hash[:16], dir_entry.node_id, len(data))
                return data

        except Exception as e:
            logger.debug("GUM fetch failed: %s", e)
        return None

    def _compute_block_hash(self, sequence_id: str,
                             block_index: int) -> str:
        """Compute a content hash for a (sequence, block) pair."""
        import hashlib
        raw = f"{sequence_id}:{block_index}".encode()
        return hashlib.sha256(raw).hexdigest()

    def _ensure_capacity(self, tier: str, needed_bytes: int) -> None:
        capacity = self._tier_capacity.get(tier, float("inf"))
        used = self.page_table.stats()["bytes_per_tier"].get(tier, 0)

        if used + needed_bytes < capacity * self._evict_high:
            return

        evicted = 0
        while True:
            used = self.page_table.stats()["bytes_per_tier"].get(tier, 0)
            if used + needed_bytes < capacity * self._evict_low:
                break
            if not self._evict_one(tier):
                break
            evicted += 1

        if evicted:
            logger.info("Evicted %d blocks from %r to make room", evicted, tier)

    def _fetch_from_lower_tier(
        self,
        sequence_id:    str,
        block_index:    int,
        content_hash:   str = "",
        remote_client=  None,
        block_directory=None,
    ) -> Optional[bytes]:
        """
        Legacy method — kept for backward compatibility.
        New code uses _try_gum_fetch() which reads from instance attrs.
        """
        if content_hash and block_directory is not None \
                and remote_client is not None:
            entry = block_directory.lookup(content_hash)
            if entry is not None and entry.node_id != getattr(
                self, "_node_id", ""
            ):
                remote_host = self._get_node_host(entry.node_id)
                if remote_host:
                    leased = remote_client.acquire_lease(
                        content_hash, remote_host
                    )
                    try:
                        data = remote_client.fetch_block(
                            content_hash, remote_host
                        )
                        if data is not None:
                            logger.info(
                                f"GUM: fetched block {content_hash[:16]}... "
                                f"from {entry.node_id} "
                                f"({len(data)} bytes)"
                            )
                            return data
                    finally:
                        if leased:
                            remote_client.release_lease(
                                content_hash, remote_host
                            )

        return None

    def _get_node_host(self, node_id: str) -> Optional[str]:
        """
        Resolve node_id to host string.

        Resolution order:
          1. NodeDiscovery peers (Redis-backed, dynamic)
          2. MEMOPT_NODE_HOSTS env var (static fallback)
        Returns None if node_id not resolvable.
        """
        # Try discovery first (dynamic peer resolution)
        try:
            from memopt.vmm.discovery import NodeDiscovery, NodeCapabilities
            # Check if any running discovery instance has this peer
            # Use static peers from env as discovery fallback
            caps = NodeCapabilities()
            caps.detect()
            discovery = NodeDiscovery(caps)
            for peer in discovery.peers():
                peer_id = peer.node_id if hasattr(peer, 'node_id') else peer.get("node_id", "")
                peer_host = peer.host if hasattr(peer, 'host') else peer.get("host", "")
                if peer_id == node_id and peer_host:
                    return peer_host
        except Exception:
            pass

        # Fall back to static env var
        hosts_env = os.environ.get("MEMOPT_NODE_HOSTS", "")
        if not hosts_env:
            return None
        for pair in hosts_env.split(","):
            parts = pair.strip().split(":")
            if len(parts) == 2 and parts[0].strip() == node_id:
                return parts[1].strip()
        return None

    def _hot_tier(self) -> str:
        return _hal.tier_names[0]

    @staticmethod
    def _sync_copy(src, dst) -> None:
        # Use AsyncNVMeManager for NVMe file paths when available
        if isinstance(src, str) or isinstance(dst, str):
            try:
                from memopt.vmm.backends._cuda_backend_py import get_async_nvme_manager
                mgr = get_async_nvme_manager()

                if isinstance(src, str):
                    # NVMe → tensor: read via async manager
                    data = mgr.read_block(src, os.path.getsize(src))
                    if data is not None:
                        import torch
                        if hasattr(dst, 'copy_'):
                            src_tensor = torch.frombuffer(
                                bytearray(data), dtype=torch.uint8)
                            dst.copy_(src_tensor)
                            return
                elif isinstance(dst, str):
                    # tensor → NVMe: write via async manager
                    if hasattr(src, 'cpu'):
                        cpu = src.cpu() if src.is_cuda else src
                        block_data = cpu.numpy().tobytes()
                    else:
                        block_data = bytes(src)
                    if mgr.write_block(dst, block_data):
                        return
            except Exception:
                pass  # fall through to original path

        result = _hal.backend.async_copy(src, dst)
        if hasattr(_hal.backend, "record_event"):
            _hal.backend.record_event(result).synchronize()
        elif hasattr(result, "wait"):         # threading.Event (unified)
            result.wait()
