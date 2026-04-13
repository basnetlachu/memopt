"""
Image version reporting.

Used by certification, discovery, and the control plane to track
which image version a node is running. Works in three contexts:

  - Golden image: reads /etc/memopt/version (written at bake time)
  - Docker dev:   falls back to MEMOPT_IMAGE_VERSION / pkg version
  - Local dev:    reports package version or "unknown"

Never raises.
"""
from __future__ import annotations

import os
from typing import Dict


_VERSION_FILE = "/etc/memopt/version"
_GIT_COMMIT_FILE = "/etc/memopt/git-commit"
_DOCKER_MARKER = "/.dockerenv"


def get_image_version() -> str:
    """
    Returns the image version this node is running.

    Priority:
      1. /etc/memopt/version (golden image)
      2. MEMOPT_IMAGE_VERSION env var
      3. memopt.__version__ (development)
      4. "unknown"
    """
    try:
        if os.path.exists(_VERSION_FILE):
            with open(_VERSION_FILE) as f:
                version = f.read().strip()
            if version:
                return version
    except Exception:
        pass

    env_version = os.getenv("MEMOPT_IMAGE_VERSION", "")
    if env_version:
        return env_version

    try:
        import memopt
        return getattr(memopt, "__version__", "unknown")
    except Exception:
        pass

    return "unknown"


def get_git_commit() -> str:
    """
    Returns the git commit of the running image. Never raises.
    """
    try:
        if os.path.exists(_GIT_COMMIT_FILE):
            with open(_GIT_COMMIT_FILE) as f:
                commit = f.read().strip()
            if commit:
                return commit
    except Exception:
        pass

    return os.getenv("MEMOPT_GIT_COMMIT", "unknown")


def _detect_image_source() -> str:
    """
    Detect whether we are running from a golden image, a Docker
    container, or a development environment.

    Returns: "golden" | "docker" | "development"
    """
    try:
        if os.path.exists(_VERSION_FILE):
            return "golden"
    except Exception:
        pass

    try:
        if os.path.exists(_DOCKER_MARKER):
            return "docker"
    except Exception:
        pass

    return "development"


def get_node_image_info() -> Dict[str, str]:
    """
    Returns complete image information for this node.
    Used by discovery registration and control plane reporting.
    """
    return {
        "image_version": get_image_version(),
        "git_commit":    get_git_commit(),
        "image_source":  _detect_image_source(),
    }
