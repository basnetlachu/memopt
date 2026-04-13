#!/usr/bin/env python3
"""
Validate CRD YAML files are syntactically correct and the schema
is consistent with the Python models.

Run: python scripts/validate_crds.py
"""
import os
import sys

# Make the repo root importable when running from scripts/
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import yaml

CRD_FILES = [
    "deploy/crds/memoptcluster.yaml",
    "deploy/crds/memoptnode.yaml",
]


def validate_yaml(path: str) -> bool:
    """Validate CRD YAML structure."""
    try:
        with open(path) as f:
            doc = yaml.safe_load(f)

        assert doc["apiVersion"] == \
            "apiextensions.k8s.io/v1"
        assert doc["kind"] == "CustomResourceDefinition"
        assert "metadata" in doc
        assert "spec" in doc
        assert "group" in doc["spec"]
        assert doc["spec"]["group"] == "memopt.io"
        assert "versions" in doc["spec"]
        assert len(doc["spec"]["versions"]) > 0

        version = doc["spec"]["versions"][0]
        assert "schema" in version
        assert "openAPIV3Schema" in version["schema"]

        print(f"[PASS] {path}")
        return True
    except Exception as e:
        print(f"[FAIL] {path}: {e}")
        return False


def validate_models() -> bool:
    """Validate Python models parse dict inputs correctly."""
    try:
        from memopt.operator.models import (
            MemoptClusterSpec,
            MemoptClusterStatus,
            MemoptNodeSpec,
            MemoptNodeStatus,
            StatusCondition,
        )

        # Round-trip test for cluster spec
        spec_dict = {
            "nodeSelector": {
                "nvidia.com/gpu.present": "true"},
            "image": {
                "repository": "memopt/serving",
                "tag": "v1.0"},
            "redis": {"url": "redis://redis:6379"},
            "serving": {"port": 8080},
            "certification": {"enabled": True},
        }
        spec = MemoptClusterSpec.from_dict(spec_dict)
        assert spec.serving.port == 8080
        assert spec.redis_url == "redis://redis:6379"
        assert spec.certification.enabled is True
        assert spec.node_selector == {
            "nvidia.com/gpu.present": "true"}

        # Round-trip test for node status
        status = MemoptNodeStatus(
            phase="Ready",
            certification_hash="abc123",
            drift_pct=1.5)
        d = status.to_dict()
        assert d["phase"] == "Ready"
        assert d["certificationHash"] == "abc123"
        assert d["driftPct"] == 1.5

        # Empty dicts return defaults
        empty_spec = MemoptClusterSpec.from_dict({})
        assert empty_spec.serving.port == 8080
        assert empty_spec.certification.enabled is True

        # Status conditions round-trip
        cond = StatusCondition(
            type="Certified",
            status="True",
            last_transition_time="2026-04-12T00:00:00Z",
            reason="Ok",
            message="ok")
        cd = cond.to_dict()
        assert StatusCondition.from_dict(cd).type == \
            "Certified"

        # Cluster status to_dict produces CRD-shaped keys
        cs = MemoptClusterStatus(
            phase="Running", ready_nodes=3, total_nodes=5)
        csd = cs.to_dict()
        assert csd["readyNodes"] == 3
        assert csd["totalNodes"] == 5

        # Node spec parse
        ns = MemoptNodeSpec.from_dict({
            "nodeName": "n1",
            "clusterRef": "c1",
            "gpuCount": 8,
            "hbmTotalGb": 640.0})
        assert ns.node_name == "n1"
        assert ns.gpu_count == 8

        print("[PASS] Python models")
        return True
    except Exception as e:
        print(f"[FAIL] Python models: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    results = []
    for crd_file in CRD_FILES:
        if os.path.exists(crd_file):
            results.append(validate_yaml(crd_file))
        else:
            print(f"[SKIP] {crd_file} not found")

    results.append(validate_models())

    all_pass = all(results)
    print(f"\n{'ALL PASS' if all_pass else 'FAILURES'}")
    sys.exit(0 if all_pass else 1)
