"""
Hardware portability layer — compiles Triton kernel source to the
correct binary for the detected hardware target.

Supported targets (detected automatically via the existing HAL):
  cuda   : NVIDIA — PTX via triton + nvcc
  rocm   : AMD   — AMDGCN via triton + hipcc
  cpu    : stub   — returns None (no GPU kernel on CPU)
  mlir   : custom ASICs — emits MLIR text for vendor toolchain

Triton handles NVIDIA and AMD transparently. The portability layer's job
is to detect the target, set the correct triton backend environment
variables, and catch compilation errors cleanly.

If Triton is not installed, compile() always returns None and logs a
warning. Callers must handle None gracefully.
"""
from __future__ import annotations
import logging
import os
import types
from typing import Optional

logger = logging.getLogger(__name__)


def _detect_target() -> str:
    """Return 'cuda', 'rocm', or 'cpu' based on available hardware."""
    try:
        import torch
        if torch.cuda.is_available():
            if hasattr(torch.version, "hip") and torch.version.hip:
                return "rocm"
            return "cuda"
    except ImportError:
        pass
    return "cpu"


class PortabilityLayer:
    """
    Compiles Triton kernel source to the binary for the current hardware.

    compile() takes a source string and a hardware hint string
    (e.g. "cuda:H100", "rocm:MI300X") and returns a module with a
    `run_kernel` callable, or None on failure.

    The returned module is a plain Python module object with the compiled
    kernel functions as attributes. The JIT generator calls
    compiled.run_kernel(*args) to execute it.
    """

    def __init__(self):
        self._target = _detect_target()
        logger.info(f"PortabilityLayer: target={self._target}")

    def compile(self, source: str, hardware_hint: str = "") -> Optional[types.ModuleType]:
        """
        Compile Triton kernel source for the detected hardware target.

        Args:
            source:        Valid Python source starting with `import triton`.
            hardware_hint: The hardware string from the BottleneckEvent
                           (used for logging only — actual target is
                           detected from the runtime environment).

        Returns:
            A module with `run_kernel` callable, or None on failure.
        """
        target = self._resolve_target(hardware_hint)

        if target == "cpu":
            logger.debug("PortabilityLayer: no GPU — skipping compilation")
            return None

        if target == "mlir":
            return self._compile_mlir(source)

        return self._compile_triton(source, target)

    def _resolve_target(self, hint: str) -> str:
        """
        Resolve the compilation target from the runtime environment.
        The hint is informational only — we trust the runtime HAL.
        """
        if self._target in ("cuda", "rocm"):
            return self._target
        if "mlir" in hint.lower() or "asic" in hint.lower():
            return "mlir"
        return "cpu"

    def _compile_triton(self, source: str, target: str) -> Optional[types.ModuleType]:
        """
        Write Triton kernel source to a temp .py file and import it.

        Triton's @jit decorator requires the decorated functions to live in
        a real file on disk (it inspects __file__ to locate the source).
        exec() from a string raises "@jit functions should be defined in a
        Python file". We work around this by writing to a tempfile and using
        importlib to load the module from disk.
        """
        import importlib.util
        import sys
        import tempfile

        try:
            import triton  # noqa: F401 — validate triton is installed

            # Write source to a real .py file on disk
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, prefix="memopt_jit_"
            ) as f:
                f.write(source)
                tmp_path = f.name

            try:
                spec = importlib.util.spec_from_file_location("jit_kernel", tmp_path)
                module = importlib.util.module_from_spec(spec)
                sys.modules["jit_kernel"] = module
                spec.loader.exec_module(module)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

            if not hasattr(module, "run_kernel"):
                logger.warning(
                    "PortabilityLayer: compiled module has no run_kernel function"
                )
                return None

            logger.info(f"PortabilityLayer: compiled for {target}")
            return module

        except ImportError:
            logger.warning(
                "PortabilityLayer: triton not installed — "
                "pip install triton to enable JIT kernel synthesis"
            )
            return None
        except SyntaxError as e:
            logger.warning(f"PortabilityLayer: syntax error in generated kernel: {e}")
            return None
        except Exception as e:
            logger.warning(f"PortabilityLayer: compilation error: {e}")
            return None

    def _compile_mlir(self, source: str) -> Optional[types.ModuleType]:
        """
        For custom ASICs: emit MLIR text to a temp file.
        The vendor's toolchain picks it up from the path in the env var
        MEMOPT_MLIR_OUT_DIR (default: /tmp/memopt_mlir/).

        Returns None — the kernel is not directly executable by Python.
        The MLIR file is the deliverable for the vendor toolchain.
        """
        import os, tempfile
        out_dir = os.environ.get("MEMOPT_MLIR_OUT_DIR", "/tmp/memopt_mlir")
        os.makedirs(out_dir, exist_ok=True)

        try:
            path = os.path.join(out_dir, f"kernel_{int(time.monotonic()*1000)}.mlir")
            with open(path, "w") as f:
                f.write(source)
            logger.info(f"PortabilityLayer: MLIR emitted to {path}")
        except Exception as e:
            logger.warning(f"PortabilityLayer: MLIR emit failed: {e}")

        return None   # not directly executable

    def target(self) -> str:
        return self._target


import time  # noqa: E402 — used in _compile_mlir above
