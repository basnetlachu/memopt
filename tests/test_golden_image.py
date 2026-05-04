"""
Tests for the golden image build pipeline.

What these tests cover:
  - memopt/image_version.py — version/commit/source detection
  - NodeCapabilities — includes image info after detect()
  - scripts/build_golden_image.sh — syntax + --help
  - scripts/first_boot.sh — syntax only (cannot run without target env)
  - .github/workflows/build_golden_image.yml — valid YAML
  - Dockerfile.golden — on-disk presence and key instructions

What these tests DO NOT cover (REQUIRES hardware):
  - Real CUDA kernel compilation
  - Real image boot time
  - Real PXE server delivery
  - Real image size
  - Real first-boot execution
"""
import os
import subprocess
import sys
from unittest import mock

import pytest


_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))


# ══════════════════════════════════════════════════════════════════════
#  image_version module
# ══════════════════════════════════════════════════════════════════════


def test_image_version_from_env():
    old = os.environ.get("MEMOPT_IMAGE_VERSION")
    os.environ["MEMOPT_IMAGE_VERSION"] = "v1.2.3"
    try:
        # Must avoid the /etc/memopt/version path short-circuiting.
        with mock.patch(
                "memopt.image_version.os.path.exists",
                return_value=False):
            from memopt.image_version import get_image_version
            assert get_image_version() == "v1.2.3"
    finally:
        if old is not None:
            os.environ["MEMOPT_IMAGE_VERSION"] = old
        else:
            os.environ.pop("MEMOPT_IMAGE_VERSION", None)


def test_image_version_fallback_to_package():
    env = {k: v for k, v in os.environ.items()
           if k != "MEMOPT_IMAGE_VERSION"}
    with mock.patch.dict(os.environ, env, clear=True):
        with mock.patch(
                "memopt.image_version.os.path.exists",
                return_value=False):
            from memopt.image_version import get_image_version
            version = get_image_version()
            # Falls back to package __version__ or "unknown"
            assert isinstance(version, str)
            assert len(version) > 0


def test_image_version_never_raises():
    from memopt.image_version import get_image_version
    with mock.patch(
            "memopt.image_version.os.path.exists",
            side_effect=PermissionError):
        version = get_image_version()
        assert isinstance(version, str)


def test_get_git_commit_from_env():
    old = os.environ.get("MEMOPT_GIT_COMMIT")
    os.environ["MEMOPT_GIT_COMMIT"] = "abc1234"
    try:
        with mock.patch(
                "memopt.image_version.os.path.exists",
                return_value=False):
            from memopt.image_version import get_git_commit
            assert get_git_commit() == "abc1234"
    finally:
        if old is not None:
            os.environ["MEMOPT_GIT_COMMIT"] = old
        else:
            os.environ.pop("MEMOPT_GIT_COMMIT", None)


def test_get_git_commit_never_raises():
    from memopt.image_version import get_git_commit
    with mock.patch(
            "memopt.image_version.os.path.exists",
            side_effect=OSError):
        assert isinstance(get_git_commit(), str)


def test_detect_image_source_development():
    from memopt.image_version import _detect_image_source
    with mock.patch(
            "memopt.image_version.os.path.exists",
            return_value=False):
        assert _detect_image_source() == "development"


def test_detect_image_source_golden():
    from memopt.image_version import _detect_image_source

    def mock_exists(path):
        return path == "/etc/memopt/version"

    with mock.patch(
            "memopt.image_version.os.path.exists",
            side_effect=mock_exists):
        assert _detect_image_source() == "golden"


def test_detect_image_source_docker():
    from memopt.image_version import _detect_image_source

    def mock_exists(path):
        return path == "/.dockerenv"

    with mock.patch(
            "memopt.image_version.os.path.exists",
            side_effect=mock_exists):
        assert _detect_image_source() == "docker"


def test_get_node_image_info_keys():
    from memopt.image_version import get_node_image_info
    info = get_node_image_info()
    assert "image_version" in info
    assert "git_commit" in info
    assert "image_source" in info
    assert info["image_source"] in (
        "golden", "docker", "development")


# ══════════════════════════════════════════════════════════════════════
#  NodeCapabilities integration
# ══════════════════════════════════════════════════════════════════════


def test_node_capabilities_has_image_info():
    from memopt.vmm.discovery import NodeCapabilities
    caps = NodeCapabilities()
    caps.detect()
    assert hasattr(caps, "image_version")
    assert hasattr(caps, "git_commit")
    assert hasattr(caps, "image_source")
    assert isinstance(caps.image_version, str)
    assert isinstance(caps.image_source, str)


def test_node_capabilities_to_dict_includes_image():
    from memopt.vmm.discovery import NodeCapabilities
    caps = NodeCapabilities()
    caps.detect()
    d = caps.to_dict()
    for key in ("image_version", "git_commit", "image_source"):
        assert key in d


def test_node_capabilities_roundtrip_image_info():
    from memopt.vmm.discovery import NodeCapabilities
    caps = NodeCapabilities()
    caps.detect()
    d = caps.to_dict()
    restored = NodeCapabilities.from_dict(d)
    assert restored.image_version == caps.image_version
    assert restored.git_commit == caps.git_commit
    assert restored.image_source == caps.image_source


# ══════════════════════════════════════════════════════════════════════
#  Shell scripts
# ══════════════════════════════════════════════════════════════════════


def test_build_script_help():
    result = subprocess.run(
        ["bash",
         os.path.join(_REPO_ROOT,
                      "scripts/build_golden_image.sh"),
         "--help"],
        capture_output=True, text=True)
    assert result.returncode == 0
    out = result.stdout + result.stderr
    assert "Usage" in out or "usage" in out


def test_build_script_requires_tag():
    """--tag is mandatory; script must exit 1 without it."""
    result = subprocess.run(
        ["bash",
         os.path.join(_REPO_ROOT,
                      "scripts/build_golden_image.sh")],
        capture_output=True, text=True)
    assert result.returncode == 1
    assert "tag" in (result.stdout + result.stderr).lower()


def test_build_script_syntax():
    result = subprocess.run(
        ["bash", "-n",
         os.path.join(_REPO_ROOT,
                      "scripts/build_golden_image.sh")],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_first_boot_script_syntax():
    result = subprocess.run(
        ["bash", "-n",
         os.path.join(_REPO_ROOT, "scripts/first_boot.sh")],
        capture_output=True, text=True)
    assert result.returncode == 0, \
        f"Syntax error: {result.stderr}"


# ══════════════════════════════════════════════════════════════════════
#  CI workflow YAML
# ══════════════════════════════════════════════════════════════════════


def test_golden_image_workflow_valid_yaml():
    import yaml
    import pytest
    # The build_golden_image workflow was disabled in the v1.3.0 OSS
    # release prep (renamed to .disabled). When/if it's re-enabled,
    # this test re-asserts that the YAML still parses. Skip cleanly
    # while it's disabled.
    base = os.path.join(_REPO_ROOT, ".github/workflows")
    candidates = ["build_golden_image.yml", "build_golden_image.yml.disabled"]
    path = next((os.path.join(base, c) for c in candidates
                 if os.path.exists(os.path.join(base, c))), None)
    if path is None:
        pytest.skip("build_golden_image workflow not present (deferred)")
    with open(path) as f:
        doc = yaml.safe_load(f)
    # YAML parses "on:" as key True when on is unquoted — accept either
    assert "on" in doc or True in doc
    assert "jobs" in doc
    jobs = doc["jobs"]
    assert "build-golden" in jobs


# ══════════════════════════════════════════════════════════════════════
#  Dockerfile.golden
# ══════════════════════════════════════════════════════════════════════


def test_dockerfile_golden_exists():
    path = os.path.join(_REPO_ROOT, "Dockerfile.golden")
    assert os.path.exists(path), "Dockerfile.golden not created"
    with open(path) as f:
        content = f.read()

    assert "FROM ubuntu:22.04" in content
    assert "memopt-transport" in content
    assert "LABEL memopt.version" in content
    # Systemd is the init process for PXE boot
    assert "systemd" in content
    # First-boot wiring
    assert "first_boot.sh" in content or \
        "first-boot" in content
    # Version stamps baked into the image
    assert "/etc/memopt/version" in content
