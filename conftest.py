"""
Root pytest configuration.

Shared fixtures available to all test suites under memopt/.
"""
import pytest


@pytest.fixture(scope="session")
def cuda_available():
    """True when a CUDA device is reachable; skip if caller needs one."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.fixture(scope="session")
def cuda_device(cuda_available):
    """Return 'cuda' if available, else skip the test."""
    if not cuda_available:
        pytest.skip("CUDA not available")
    return "cuda"
