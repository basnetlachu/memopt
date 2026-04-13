"""
Tests for operator CRD Python models.
"""
from memopt.operator.models import (
    CertificationSpec,
    GlobalOracleSpec,
    ImageSpec,
    MemoptClusterSpec,
    MemoptClusterStatus,
    MemoptNodeSpec,
    MemoptNodeStatus,
    ResourceRequirements,
    ServingSpec,
    StatusCondition,
    TransportSpec,
)


# ══════════════════════════════════════════════════════════════════════════
#  MemoptClusterSpec
# ══════════════════════════════════════════════════════════════════════════


def test_cluster_spec_from_dict_full():
    spec = MemoptClusterSpec.from_dict({
        "nodeSelector": {"gpu": "true"},
        "image": {
            "repository": "myrepo/memopt",
            "tag": "v2.0",
        },
        "redis": {"url": "redis://r:6379"},
        "scylladb": {"hosts": "s1,s2,s3"},
        "serving": {"port": 9090},
        "certification": {"enabled": False},
        "transport": {"mode": "rdma"},
        "globalOracle": {
            "enabled": True,
            "pullIntervalSeconds": 30,
        },
    })
    assert spec.node_selector == {"gpu": "true"}
    assert spec.image.repository == "myrepo/memopt"
    assert spec.image.tag == "v2.0"
    assert spec.redis_url == "redis://r:6379"
    assert spec.scylladb_hosts == "s1,s2,s3"
    assert spec.serving.port == 9090
    assert spec.certification.enabled is False
    assert spec.transport.mode == "rdma"
    assert spec.global_oracle.enabled is True
    assert spec.global_oracle.pull_interval_seconds == 30


def test_cluster_spec_from_dict_empty():
    spec = MemoptClusterSpec.from_dict({})
    assert spec.serving.port == 8080
    assert spec.certification.enabled is True
    assert spec.transport.mode == "auto"
    assert spec.global_oracle.enabled is False
    assert spec.redis_url == ""


def test_cluster_spec_from_dict_none():
    spec = MemoptClusterSpec.from_dict(None)
    assert spec.serving.port == 8080
    assert spec.node_selector == {}


# ══════════════════════════════════════════════════════════════════════════
#  MemoptClusterStatus
# ══════════════════════════════════════════════════════════════════════════


def test_cluster_status_to_dict():
    status = MemoptClusterStatus(
        phase="Running",
        ready_nodes=8,
        total_nodes=10)
    d = status.to_dict()
    assert d["phase"] == "Running"
    assert d["readyNodes"] == 8
    assert d["totalNodes"] == 10
    assert isinstance(d["conditions"], list)
    assert d["observedGeneration"] == 0


def test_cluster_status_conditions_to_dict():
    cond = StatusCondition(
        type="Ready",
        status="True",
        last_transition_time="2026-04-12T00:00:00Z")
    status = MemoptClusterStatus(
        phase="Running",
        conditions=[cond])
    d = status.to_dict()
    assert len(d["conditions"]) == 1
    assert d["conditions"][0]["type"] == "Ready"


# ══════════════════════════════════════════════════════════════════════════
#  MemoptNodeSpec / Status
# ══════════════════════════════════════════════════════════════════════════


def test_node_spec_from_dict():
    spec = MemoptNodeSpec.from_dict({
        "nodeName": "gpu-node-01",
        "clusterRef": "prod-cluster",
        "gpuCount": 8,
        "hbmTotalGb": 640.0,
    })
    assert spec.node_name == "gpu-node-01"
    assert spec.cluster_ref == "prod-cluster"
    assert spec.gpu_count == 8
    assert spec.hbm_total_gb == 640.0


def test_node_spec_from_dict_empty():
    spec = MemoptNodeSpec.from_dict({})
    assert spec.node_name == ""
    assert spec.gpu_count == 0


def test_node_status_to_dict():
    status = MemoptNodeStatus(
        phase="Ready",
        certification_hash="abc123def456",
        drift_pct=0.5,
        hbm_free_gb=72.3)
    d = status.to_dict()
    assert d["phase"] == "Ready"
    assert d["certificationHash"] == "abc123def456"
    assert d["driftPct"] == 0.5
    assert d["hbmFreeGb"] == 72.3


# ══════════════════════════════════════════════════════════════════════════
#  StatusCondition round-trip
# ══════════════════════════════════════════════════════════════════════════


def test_status_condition_roundtrip():
    cond = StatusCondition(
        type="Certified",
        status="True",
        last_transition_time="2026-04-12T00:00:00Z",
        reason="CertificationPassed",
        message="All tests passed")
    d = cond.to_dict()
    restored = StatusCondition.from_dict(d)
    assert restored.type == cond.type
    assert restored.status == cond.status
    assert restored.reason == cond.reason
    assert restored.message == cond.message
    assert restored.last_transition_time == \
        cond.last_transition_time


# ══════════════════════════════════════════════════════════════════════════
#  Sub-specs
# ══════════════════════════════════════════════════════════════════════════


def test_resource_requirements_defaults():
    req = ResourceRequirements.from_dict({})
    assert req.cpu_request == "1000m"
    assert req.memory_request == "4Gi"
    assert req.cpu_limit == "4000m"
    assert req.memory_limit == "16Gi"


def test_resource_requirements_custom():
    req = ResourceRequirements.from_dict({
        "requests": {
            "cpu": "2000m",
            "memory": "8Gi",
        },
        "limits": {
            "cpu": "8000m",
            "memory": "32Gi",
        },
    })
    assert req.cpu_request == "2000m"
    assert req.memory_request == "8Gi"
    assert req.cpu_limit == "8000m"
    assert req.memory_limit == "32Gi"


def test_image_spec_defaults():
    img = ImageSpec.from_dict(None)
    assert img.repository == "memopt/serving"
    assert img.tag == "latest"
    assert img.pull_policy == "IfNotPresent"


def test_image_spec_custom():
    img = ImageSpec.from_dict({
        "repository": "ghcr.io/foo/bar",
        "tag": "2026.04",
        "pullPolicy": "Always",
    })
    assert img.repository == "ghcr.io/foo/bar"
    assert img.tag == "2026.04"
    assert img.pull_policy == "Always"


def test_serving_spec_nested_resources():
    serving = ServingSpec.from_dict({
        "port": 9000,
        "resources": {
            "requests": {"cpu": "500m"},
        },
    })
    assert serving.port == 9000
    assert serving.resources.cpu_request == "500m"
    # Other defaults preserved
    assert serving.resources.memory_request == "4Gi"


def test_certification_spec_defaults():
    c = CertificationSpec.from_dict(None)
    assert c.enabled is True
    assert c.schedule == "0 0 * * *"


def test_transport_spec_defaults():
    t = TransportSpec.from_dict(None)
    assert t.mode == "auto"


def test_global_oracle_spec_defaults():
    g = GlobalOracleSpec.from_dict(None)
    assert g.enabled is False
    assert g.pull_interval_seconds == 60


# ══════════════════════════════════════════════════════════════════════════
#  CRD YAML sanity (loaded alongside models)
# ══════════════════════════════════════════════════════════════════════════


def test_cluster_crd_yaml_loads():
    import os
    import yaml

    repo_root = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))
    path = os.path.join(
        repo_root, "deploy/crds/memoptcluster.yaml")
    with open(path) as f:
        doc = yaml.safe_load(f)

    assert doc["kind"] == "CustomResourceDefinition"
    assert doc["metadata"]["name"] == "memoptclusters.memopt.io"
    assert doc["spec"]["names"]["kind"] == "MemoptCluster"
    assert "mc" in doc["spec"]["names"]["shortNames"]


def test_node_crd_yaml_loads():
    import os
    import yaml

    repo_root = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))
    path = os.path.join(
        repo_root, "deploy/crds/memoptnode.yaml")
    with open(path) as f:
        doc = yaml.safe_load(f)

    assert doc["kind"] == "CustomResourceDefinition"
    assert doc["metadata"]["name"] == "memoptnodes.memopt.io"
    assert doc["spec"]["names"]["kind"] == "MemoptNode"
    assert "mn" in doc["spec"]["names"]["shortNames"]
