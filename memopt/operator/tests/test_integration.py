"""
Integration tests for the Helm chart + example manifests + CRD
validation script.

All tests are skip-safe: if helm/kubectl/python aren't available,
tests skip cleanly.
"""
import glob
import os
import shutil
import subprocess

import pytest
import yaml


_REPO_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))


def _has_helm() -> bool:
    return shutil.which("helm") is not None


# ══════════════════════════════════════════════════════════════════════
#  Helm chart rendering
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not _has_helm(),
                    reason="helm not installed")
def test_helm_chart_with_operator_enabled():
    """Helm template renders with operator enabled."""
    result = subprocess.run(
        ["helm", "template", "memopt",
         "deploy/helm/memopt/",
         "--set", "operator.enabled=true",
         "--set", "redis.url=redis://r:6379"],
        capture_output=True, text=True,
        cwd=_REPO_ROOT)

    if result.returncode != 0:
        pytest.fail(result.stderr)

    assert "memopt-operator" in result.stdout
    assert "ClusterRole" in result.stdout
    assert "ServiceAccount" in result.stdout


@pytest.mark.skipif(not _has_helm(),
                    reason="helm not installed")
def test_helm_chart_without_operator():
    """Helm template renders without operator resources."""
    result = subprocess.run(
        ["helm", "template", "memopt",
         "deploy/helm/memopt/",
         "--set", "operator.enabled=false",
         "--set", "redis.url=redis://r:6379"],
        capture_output=True, text=True,
        cwd=_REPO_ROOT)

    if result.returncode != 0:
        pytest.fail(result.stderr)

    # No operator Deployment or ClusterRole when disabled
    assert "memopt-operator" not in result.stdout


@pytest.mark.skipif(not _has_helm(),
                    reason="helm not installed")
def test_helm_chart_lint_clean():
    """`helm lint` on the chart must not fail."""
    result = subprocess.run(
        ["helm", "lint", "deploy/helm/memopt/"],
        capture_output=True, text=True,
        cwd=_REPO_ROOT)
    assert "0 chart(s) failed" in result.stdout, \
        f"lint failed: {result.stdout}\n{result.stderr}"


# ══════════════════════════════════════════════════════════════════════
#  Example manifests
# ══════════════════════════════════════════════════════════════════════


def test_example_manifests_valid_yaml():
    """All example manifests are valid YAML with apiVersion."""
    examples = glob.glob(
        os.path.join(_REPO_ROOT, "deploy/examples/*.yaml"))
    assert len(examples) >= 3, \
        "Expected at least 3 example files"

    for path in examples:
        with open(path) as f:
            docs = list(yaml.safe_load_all(f))
        assert len(docs) >= 1, f"Empty YAML in {path}"
        for doc in docs:
            if doc:
                assert "apiVersion" in doc, \
                    f"Missing apiVersion in {path}"


def test_example_cluster_spec_parseable():
    """Example cluster specs parse to MemoptClusterSpec models."""
    from memopt.operator.models import MemoptClusterSpec

    path = os.path.join(
        _REPO_ROOT,
        "deploy/examples/single-node-cluster.yaml")
    with open(path) as f:
        doc = yaml.safe_load(f)

    spec = MemoptClusterSpec.from_dict(doc.get("spec", {}))

    assert spec.serving.port == 8080
    assert spec.certification.enabled is True
    assert spec.transport.mode == "tcp"


def test_design_partner_example_parses():
    from memopt.operator.models import MemoptClusterSpec

    path = os.path.join(
        _REPO_ROOT,
        "deploy/examples/design-partner-cluster.yaml")
    with open(path) as f:
        doc = yaml.safe_load(f)

    spec = MemoptClusterSpec.from_dict(doc.get("spec", {}))

    assert spec.transport.mode == "auto"
    assert spec.redis_url == "redis://redis-service:6379"
    assert spec.certification.enabled is True


def test_zettascale_example_parses():
    from memopt.operator.models import MemoptClusterSpec

    path = os.path.join(
        _REPO_ROOT,
        "deploy/examples/zettascale-cluster.yaml")
    with open(path) as f:
        doc = yaml.safe_load(f)

    spec = MemoptClusterSpec.from_dict(doc.get("spec", {}))

    assert spec.transport.mode == "rdma"
    assert "scylla" in spec.scylladb_hosts
    assert spec.global_oracle.enabled is True


def test_example_apiversion_matches_crd():
    """Examples use the API version the CRD serves."""
    with open(os.path.join(
            _REPO_ROOT,
            "deploy/crds/memoptcluster.yaml")) as f:
        crd = yaml.safe_load(f)
    crd_version = crd["spec"]["versions"][0]["name"]
    crd_group = crd["spec"]["group"]
    expected = f"{crd_group}/{crd_version}"

    for path in glob.glob(os.path.join(
            _REPO_ROOT, "deploy/examples/*.yaml")):
        with open(path) as f:
            doc = yaml.safe_load(f)
        assert doc["apiVersion"] == expected, \
            f"{path} uses {doc['apiVersion']} but CRD serves {expected}"


# ══════════════════════════════════════════════════════════════════════
#  CRD validation script
# ══════════════════════════════════════════════════════════════════════


def test_validate_crds_script():
    """CRD validation script passes."""
    import sys
    # Use sys.executable so the subprocess inherits the same Python
    # environment as the test runner (which has the test deps such as
    # `yaml` installed). A bare `python3` would resolve via PATH and
    # may pick a different interpreter without the test deps.
    result = subprocess.run(
        [sys.executable, "scripts/validate_crds.py"],
        capture_output=True, text=True,
        cwd=_REPO_ROOT)
    assert result.returncode == 0, \
        f"CRD validation failed: {result.stdout}\n{result.stderr}"
    assert "ALL PASS" in result.stdout


# ══════════════════════════════════════════════════════════════════════
#  values.yaml includes operator section
# ══════════════════════════════════════════════════════════════════════


def test_values_yaml_has_operator_section():
    with open(os.path.join(
            _REPO_ROOT,
            "deploy/helm/memopt/values.yaml")) as f:
        values = yaml.safe_load(f)
    assert "operator" in values
    assert values["operator"]["enabled"] is False
    assert "reconcileIntervalSeconds" in values["operator"]
    assert "resources" in values["operator"]
    assert "crds" in values
