"""
Elastic Allocator — decides WHERE to promote predicted blocks.

Given a prediction confidence (urgency) and current memory state,
returns an AllocationDecision specifying the target tier and node.

Decision logic (priority order):
  urgency >= 0.8 + local HBM free  -> local HBM
  urgency >= 0.7 + remote HBM free -> remote HBM
  urgency >= 0.5 + local DRAM free -> local DRAM
  else                              -> DRAM fallback or NVMe

No external dependencies.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

from .universal_profile import UniversalMemoryProfile

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NodeMemoryState:
    node_id: str
    hbm_used_gb: float
    hbm_total_gb: float
    dram_used_gb: float
    dram_total_gb: float
    is_local: bool


@dataclass(frozen=True)
class AllocationDecision:
    target_tier: str      # "hbm" | "dram" | "remote_hbm" | "nvme"
    target_node: str
    urgency: float
    reason: str


@dataclass
class AllocatorStats:
    decisions_made: int = 0
    local_hbm: int = 0
    local_dram: int = 0
    remote_hbm: int = 0
    fallback_nvme: int = 0


class ElasticAllocator:

    def __init__(
        self,
        hw_profile: UniversalMemoryProfile,
        node_id: str = "",
        remote_nodes: Optional[List[NodeMemoryState]] = None,
    ) -> None:
        self._hw_profile = hw_profile
        self._node_id = node_id or "local"
        self._remote_nodes: Dict[str, NodeMemoryState] = {}
        self._lock = threading.Lock()
        self._stats = AllocatorStats()

        hbm_total = 0.0
        dram_total = 0.0
        for tier in hw_profile.tiers:
            if tier.name == "hbm":
                hbm_total = tier.capacity_gb
            elif tier.name == "dram":
                dram_total = tier.capacity_gb

        self._local_state = NodeMemoryState(
            node_id=self._node_id,
            hbm_used_gb=0.0,
            hbm_total_gb=hbm_total,
            dram_used_gb=0.0,
            dram_total_gb=dram_total,
            is_local=True,
        )

        if remote_nodes:
            for ns in remote_nodes:
                self._remote_nodes[ns.node_id] = ns

    def decide(
        self,
        confidence: float,
        block_size_bytes: int = 131_072,
    ) -> AllocationDecision:
        urgency = confidence
        block_gb = block_size_bytes / (1024 ** 3)

        with self._lock:
            local = self._local_state
            hbm_free = local.hbm_total_gb - local.hbm_used_gb
            dram_free = local.dram_total_gb - local.dram_used_gb

            # Priority 1: local HBM for high urgency
            if urgency >= 0.8 and hbm_free > block_gb and local.hbm_total_gb > 0:
                self._stats.decisions_made += 1
                self._stats.local_hbm += 1
                return AllocationDecision(
                    target_tier="hbm",
                    target_node=self._node_id,
                    urgency=urgency,
                    reason="high_urgency_local_hbm",
                )

            # Priority 2: remote HBM when local HBM unavailable
            if urgency >= 0.7:
                for nid, ns in self._remote_nodes.items():
                    remote_free = ns.hbm_total_gb - ns.hbm_used_gb
                    if remote_free > block_gb and ns.hbm_total_gb > 0:
                        self._stats.decisions_made += 1
                        self._stats.remote_hbm += 1
                        return AllocationDecision(
                            target_tier="remote_hbm",
                            target_node=nid,
                            urgency=urgency,
                            reason="high_urgency_remote_hbm",
                        )

            # Priority 3: local DRAM for medium urgency
            if urgency >= 0.5 and dram_free > block_gb:
                self._stats.decisions_made += 1
                self._stats.local_dram += 1
                return AllocationDecision(
                    target_tier="dram",
                    target_node=self._node_id,
                    urgency=urgency,
                    reason="medium_urgency_local_dram",
                )

            # Fallback: DRAM if space, else NVMe
            if dram_free > block_gb:
                self._stats.decisions_made += 1
                self._stats.local_dram += 1
                return AllocationDecision(
                    target_tier="dram",
                    target_node=self._node_id,
                    urgency=urgency,
                    reason="low_urgency_dram_fallback",
                )

            self._stats.decisions_made += 1
            self._stats.fallback_nvme += 1
            return AllocationDecision(
                target_tier="nvme",
                target_node=self._node_id,
                urgency=urgency,
                reason="no_space_nvme_fallback",
            )

    def update_node_state(self, state: NodeMemoryState) -> None:
        with self._lock:
            if state.is_local:
                self._local_state = state
            else:
                self._remote_nodes[state.node_id] = state

    def stats(self) -> AllocatorStats:
        with self._lock:
            return AllocatorStats(
                decisions_made=self._stats.decisions_made,
                local_hbm=self._stats.local_hbm,
                local_dram=self._stats.local_dram,
                remote_hbm=self._stats.remote_hbm,
                fallback_nvme=self._stats.fallback_nvme,
            )
