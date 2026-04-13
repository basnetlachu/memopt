"""
Python dataclasses that mirror the MemoptCluster and MemoptNode
CRD schemas.

Used by the operator controller to work with Kubernetes objects
without raw dict access.

All `from_dict` methods tolerate None / missing keys and fall back
to defaults that match the CRD schema defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ─────────────────────────────────────────────
# ClusterSpec subsections
# ─────────────────────────────────────────────


@dataclass
class ImageSpec:
    repository: str = "memopt/serving"
    tag: str = "latest"
    pull_policy: str = "IfNotPresent"

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "ImageSpec":
        if not d:
            return cls()
        return cls(
            repository=d.get("repository", "memopt/serving"),
            tag=d.get("tag", "latest"),
            pull_policy=d.get("pullPolicy", "IfNotPresent"),
        )


@dataclass
class ResourceRequirements:
    cpu_request: str = "1000m"
    memory_request: str = "4Gi"
    cpu_limit: str = "4000m"
    memory_limit: str = "16Gi"

    @classmethod
    def from_dict(
        cls, d: Optional[dict]
    ) -> "ResourceRequirements":
        if not d:
            return cls()
        req = d.get("requests", {}) or {}
        lim = d.get("limits", {}) or {}
        return cls(
            cpu_request=req.get("cpu", "1000m"),
            memory_request=req.get("memory", "4Gi"),
            cpu_limit=lim.get("cpu", "4000m"),
            memory_limit=lim.get("memory", "16Gi"),
        )


@dataclass
class ServingSpec:
    port: int = 8080
    resources: ResourceRequirements = field(
        default_factory=ResourceRequirements)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "ServingSpec":
        if not d:
            return cls()
        return cls(
            port=d.get("port", 8080),
            resources=ResourceRequirements.from_dict(
                d.get("resources", {})),
        )


@dataclass
class CertificationSpec:
    enabled: bool = True
    schedule: str = "0 0 * * *"

    @classmethod
    def from_dict(
        cls, d: Optional[dict]
    ) -> "CertificationSpec":
        if not d:
            return cls()
        return cls(
            enabled=d.get("enabled", True),
            schedule=d.get("schedule", "0 0 * * *"),
        )


@dataclass
class TransportSpec:
    mode: str = "auto"

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "TransportSpec":
        if not d:
            return cls()
        return cls(mode=d.get("mode", "auto"))


@dataclass
class GlobalOracleSpec:
    enabled: bool = False
    pull_interval_seconds: int = 60

    @classmethod
    def from_dict(
        cls, d: Optional[dict]
    ) -> "GlobalOracleSpec":
        if not d:
            return cls()
        return cls(
            enabled=d.get("enabled", False),
            pull_interval_seconds=d.get(
                "pullIntervalSeconds", 60),
        )


# ─────────────────────────────────────────────
# MemoptCluster top-level
# ─────────────────────────────────────────────


@dataclass
class MemoptClusterSpec:
    """Spec section of MemoptCluster CRD."""

    node_selector: dict = field(default_factory=dict)
    image: ImageSpec = field(default_factory=ImageSpec)
    redis_url: str = ""
    scylladb_hosts: str = ""
    serving: ServingSpec = field(default_factory=ServingSpec)
    certification: CertificationSpec = field(
        default_factory=CertificationSpec)
    transport: TransportSpec = field(
        default_factory=TransportSpec)
    global_oracle: GlobalOracleSpec = field(
        default_factory=GlobalOracleSpec)

    @classmethod
    def from_dict(
        cls, d: Optional[dict]
    ) -> "MemoptClusterSpec":
        if not d:
            return cls()
        redis_url = ""
        if "redis" in d and d["redis"]:
            redis_url = d["redis"].get("url", "")
        scylladb_hosts = ""
        if "scylladb" in d and d["scylladb"]:
            scylladb_hosts = d["scylladb"].get("hosts", "")
        return cls(
            node_selector=d.get("nodeSelector", {}) or {},
            image=ImageSpec.from_dict(d.get("image", {})),
            redis_url=redis_url,
            scylladb_hosts=scylladb_hosts,
            serving=ServingSpec.from_dict(
                d.get("serving", {})),
            certification=CertificationSpec.from_dict(
                d.get("certification", {})),
            transport=TransportSpec.from_dict(
                d.get("transport", {})),
            global_oracle=GlobalOracleSpec.from_dict(
                d.get("globalOracle", {})),
        )


@dataclass
class StatusCondition:
    type: str
    status: str
    last_transition_time: str
    reason: str = ""
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "type":               self.type,
            "status":             self.status,
            "lastTransitionTime": self.last_transition_time,
            "reason":             self.reason,
            "message":            self.message,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StatusCondition":
        return cls(
            type=d.get("type", ""),
            status=d.get("status", "Unknown"),
            last_transition_time=d.get(
                "lastTransitionTime", ""),
            reason=d.get("reason", ""),
            message=d.get("message", ""),
        )


@dataclass
class MemoptClusterStatus:
    phase: str = "Pending"
    ready_nodes: int = 0
    total_nodes: int = 0
    conditions: List[StatusCondition] = field(
        default_factory=list)
    observed_generation: int = 0

    def to_dict(self) -> dict:
        return {
            "phase":              self.phase,
            "readyNodes":         self.ready_nodes,
            "totalNodes":         self.total_nodes,
            "conditions": [
                c.to_dict() if hasattr(c, "to_dict") else c
                for c in self.conditions],
            "observedGeneration": self.observed_generation,
        }


# ─────────────────────────────────────────────
# MemoptNode top-level
# ─────────────────────────────────────────────


@dataclass
class MemoptNodeSpec:
    node_name: str = ""
    cluster_ref: str = ""
    gpu_count: int = 0
    hbm_total_gb: float = 0.0

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "MemoptNodeSpec":
        if not d:
            return cls()
        return cls(
            node_name=d.get("nodeName", ""),
            cluster_ref=d.get("clusterRef", ""),
            gpu_count=d.get("gpuCount", 0),
            hbm_total_gb=d.get("hbmTotalGb", 0.0),
        )


@dataclass
class MemoptNodeStatus:
    phase: str = "Pending"
    certification_hash: str = ""
    certification_time: str = ""
    drift_pct: float = 0.0
    hbm_free_gb: float = 0.0
    last_heartbeat: str = ""
    conditions: List[StatusCondition] = field(
        default_factory=list)

    def to_dict(self) -> dict:
        return {
            "phase":             self.phase,
            "certificationHash": self.certification_hash,
            "certificationTime": self.certification_time,
            "driftPct":          self.drift_pct,
            "hbmFreeGb":         self.hbm_free_gb,
            "lastHeartbeat":     self.last_heartbeat,
            "conditions": [
                c.to_dict() if hasattr(c, "to_dict") else c
                for c in self.conditions],
        }
