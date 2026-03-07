"""
Global pytest configuration.

Mocks heavy GPU/torch dependencies before any test collects,
so that control-plane and pure-Python tests can run without a GPU or
torch installation.  Tests that explicitly import torch (phase2, phase3,
agent, etc.) must mark themselves with @pytest.mark.requires_gpu or
install torch themselves.
"""

import sys
from unittest.mock import MagicMock

# ── Guard: only mock if torch is not already installed ────────────────────
try:
    import torch as _torch_check  # noqa: F401 — already present, nothing to do
    del _torch_check
except ModuleNotFoundError:
    def _make_mock(name: str) -> MagicMock:
        m = MagicMock(name=name)
        m.__name__    = name
        m.__package__ = name.split(".")[0]
        m.__spec__    = None
        # Make nn.Module a real base class so type annotations work
        if name in ("torch.nn", "torch"):
            class _Module:
                pass
            m.Module = _Module
        return m

    _TORCH_MODS = [
        "torch", "torch.nn", "torch.nn.functional", "torch.cuda",
        "torch.amp", "torch.utils", "torch.utils.checkpoint",
        "torch.optim", "torch.optim.lr_scheduler",
        "torchvision", "torchvision.transforms",
        "triton",
        "pynvml", "nvitop",
        "transformers",
        "torchao", "torchao.quantization",
    ]
    for _mod in _TORCH_MODS:
        if _mod not in sys.modules:
            sys.modules[_mod] = _make_mock(_mod)
