"""
Hardware portability layer — BackendStrategy pattern.

Dispatches Triton kernel compilation to the correct backend based on the
detected hardware at runtime. Each strategy encapsulates the compile path
for one hardware family.

Supported strategies (auto-detected):
  TritonCUDAStrategy   : NVIDIA — PTX via triton + nvcc
  TritonROCmStrategy   : AMD    — AMDGCN via triton + hipcc (smoke-tested)
  TorchCompileStrategy : CPU / unsupported GPU — torch.compile fallback
  MLIRStrategy         : Custom ASICs — emits MLIR text for vendor toolchain

Usage:
    layer = PortabilityLayer()
    hw    = layer.profile()          # HardwareProfile for prompt builders
    mod   = layer.compile(source, hardware_hint)  # module or None
"""
from __future__ import annotations

import logging
import os
import time
import types
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ── Hardware capability tables ────────────────────────────────────────────────

# (major, minor) → (arch_name, max_block_size, safe_tile_m, safe_tile_n, safe_tile_k)
# max_block_size: largest power-of-2 block that avoids register spill on this arch
# safe_tile_*   : tested tile dimensions for GEMM-style kernels
_ARCH_PROFILES: dict = {
    (12, 0): ("blackwell",       256, 128, 256, 64),  # RTX PRO 6000, B100, B200
    (9,  0): ("hopper",          256, 128, 256, 64),  # H100, H200
    (8,  9): ("ada_lovelace",    256, 128, 128, 64),  # RTX 4090
    (8,  0): ("ampere",          128, 128, 128, 32),  # A100, RTX 3090
    (7,  5): ("turing",          128,  64, 128, 32),  # T4, RTX 2080
    (7,  0): ("volta",           128,  64, 128, 32),  # V100
    (6,  0): ("pascal",           64,  64,  64, 32),  # P100
}
_DEFAULT_PROFILE = ("unknown",  64,  64,  64, 16)


# ── HardwareProfile ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HardwareProfile:
    """Immutable snapshot of the detected hardware capabilities."""
    backend:       str    # "triton_cuda" | "triton_rocm" | "torch_compile" | "mlir" | "cpu"
    arch_name:     str    # "blackwell" | "hopper" | "ampere" | "rocm_cdna" | ...
    compute_cap:   str    # "12.0" | "9.0" | "8.0" | "" (non-CUDA)
    device_name:   str    # human-readable device name
    max_block_size: int   # largest safe tl.arange / BLOCK_SIZE for this arch
    safe_tile_m:   int    # safe GEMM tile M dimension
    safe_tile_n:   int    # safe GEMM tile N dimension
    safe_tile_k:   int    # safe GEMM tile K dimension
    use_triton:    bool   # True if Triton is available AND target supports it


def detect_hardware() -> HardwareProfile:
    """
    Probe the runtime environment and return a HardwareProfile.

    Detection order:
      1. NVIDIA CUDA  → TritonCUDA
      2. AMD ROCm     → TritonROCm
      3. Apple M-series (MPS) → TorchCompile
      4. CPU / unknown → TorchCompile
    """
    try:
        import torch

        # ── NVIDIA CUDA ──────────────────────────────────────────────
        if torch.cuda.is_available() and not (
            hasattr(torch.version, "hip") and torch.version.hip
        ):
            major, minor = torch.cuda.get_device_capability(0)
            name         = torch.cuda.get_device_name(0)
            arch_name, max_blk, tm, tn, tk = _ARCH_PROFILES.get(
                (major, minor),
                _ARCH_PROFILES.get((major, 0), _DEFAULT_PROFILE),
            )
            has_triton = _triton_available()
            return HardwareProfile(
                backend        = "triton_cuda",
                arch_name      = arch_name,
                compute_cap    = f"{major}.{minor}",
                device_name    = name,
                max_block_size = max_blk,
                safe_tile_m    = tm,
                safe_tile_n    = tn,
                safe_tile_k    = tk,
                use_triton     = has_triton,
            )

        # ── AMD ROCm ─────────────────────────────────────────────────
        if torch.cuda.is_available() and hasattr(torch.version, "hip") and torch.version.hip:
            name = torch.cuda.get_device_name(0)
            has_triton = _triton_available()
            return HardwareProfile(
                backend        = "triton_rocm",
                arch_name      = "rocm_cdna",
                compute_cap    = "",
                device_name    = name,
                max_block_size = 128,
                safe_tile_m    = 64,
                safe_tile_n    = 64,
                safe_tile_k    = 32,
                use_triton     = has_triton,
            )

        # ── Apple MPS ────────────────────────────────────────────────
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return HardwareProfile(
                backend        = "torch_compile",
                arch_name      = "apple_m",
                compute_cap    = "",
                device_name    = "Apple M-series GPU",
                max_block_size = 64,
                safe_tile_m    = 32,
                safe_tile_n    = 32,
                safe_tile_k    = 16,
                use_triton     = False,
            )

    except ImportError:
        pass

    # ── CPU / unknown ─────────────────────────────────────────────────
    return HardwareProfile(
        backend        = "cpu",
        arch_name      = "cpu",
        compute_cap    = "",
        device_name    = "CPU",
        max_block_size = 32,
        safe_tile_m    = 16,
        safe_tile_n    = 16,
        safe_tile_k    = 16,
        use_triton     = False,
    )


def _triton_available() -> bool:
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False


# ── Backend strategies ────────────────────────────────────────────────────────

class TritonCUDAStrategy:
    """
    Compile Triton kernel source for NVIDIA CUDA.

    Triton's @jit requires the decorated functions to live in a real .py file
    on disk (it inspects __file__). exec() from a string raises:
      "@jit functions should be defined in a Python file".
    We write to a tempfile and import via importlib.
    """

    def compile(self, source: str) -> Optional[types.ModuleType]:
        import importlib.util
        import sys
        import tempfile

        try:
            import triton  # noqa: F401 — validate triton is installed

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, prefix="memopt_jit_"
            ) as f:
                f.write(source)
                tmp_path = f.name

            try:
                spec   = importlib.util.spec_from_file_location("jit_kernel", tmp_path)
                module = importlib.util.module_from_spec(spec)
                sys.modules["jit_kernel"] = module
                spec.loader.exec_module(module)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

            if not hasattr(module, "run_kernel"):
                logger.warning("TritonCUDA: compiled module has no run_kernel")
                return None

            logger.info("TritonCUDA: compilation succeeded")
            return module

        except ImportError:
            logger.warning("TritonCUDA: triton not installed — pip install triton")
            return None
        except SyntaxError as e:
            logger.warning(f"TritonCUDA: syntax error in generated kernel: {e}")
            return None
        except Exception as e:
            logger.warning(f"TritonCUDA: compilation error: {e}")
            return None


class TritonROCmStrategy:
    """
    Compile Triton kernel source for AMD ROCm.

    After compilation, runs a smoke test (4×4 NaN/Inf check) before accepting
    the kernel. AMD Triton occasionally produces silent NaN outputs on certain
    kernel patterns that NVIDIA Triton handles correctly — the smoke test
    catches these before they enter the cache.
    """

    def compile(self, source: str) -> Optional[types.ModuleType]:
        import importlib.util
        import sys
        import tempfile

        try:
            import triton  # noqa: F401

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, prefix="memopt_rocm_"
            ) as f:
                f.write(source)
                tmp_path = f.name

            try:
                spec   = importlib.util.spec_from_file_location("jit_kernel_rocm", tmp_path)
                module = importlib.util.module_from_spec(spec)
                sys.modules["jit_kernel_rocm"] = module
                spec.loader.exec_module(module)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

            if not hasattr(module, "run_kernel"):
                logger.warning("TritonROCm: compiled module has no run_kernel")
                return None

            if not self._rocm_smoke_test(module):
                logger.warning(
                    "TritonROCm: smoke test failed (NaN/Inf detected) — kernel rejected"
                )
                return None

            logger.info("TritonROCm: compilation and smoke test passed")
            return module

        except ImportError:
            logger.warning("TritonROCm: triton not installed")
            return None
        except SyntaxError as e:
            logger.warning(f"TritonROCm: syntax error: {e}")
            return None
        except Exception as e:
            logger.warning(f"TritonROCm: compilation error: {e}")
            return None

    def _rocm_smoke_test(self, module: types.ModuleType) -> bool:
        """
        Run run_kernel on a tiny 4×4 float16 tensor and reject if NaN/Inf.
        Returns True if the output is finite, False otherwise.
        Uses CUDA if available, CPU otherwise (for local testing).
        """
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype  = torch.float16 if device == "cuda" else torch.float32
            t = torch.ones(4, 4, dtype=dtype, device=device)
            out = module.run_kernel(t)
            if isinstance(out, (tuple, list)):
                return all(torch.isfinite(o).all().item() for o in out
                           if isinstance(o, torch.Tensor))
            if isinstance(out, torch.Tensor):
                return bool(torch.isfinite(out).all().item())
            return True   # non-tensor output — pass
        except Exception as e:
            logger.debug(f"TritonROCm smoke test exception: {e}")
            return False   # exception during smoke → reject


class TorchCompileStrategy:
    """
    Fallback: wrap the source as a torch.compile'd callable.

    Used when: Triton is unavailable, or hardware is CPU/Apple M.
    The source must define a `run_kernel` function that uses plain PyTorch.
    torch.compile then applies its own fusion passes.
    """

    def compile(self, source: str) -> Optional[types.ModuleType]:
        try:
            import torch

            ns: dict = {}
            exec(compile(source, "<memopt_torch_compile>", "exec"), ns)  # noqa: S102

            if "run_kernel" not in ns:
                logger.warning("TorchCompile: source has no run_kernel")
                return None

            compiled_fn = torch.compile(ns["run_kernel"])
            module = types.ModuleType("torch_compile_kernel")
            module.run_kernel = compiled_fn
            logger.info("TorchCompile: kernel compiled with torch.compile")
            return module

        except SyntaxError as e:
            logger.warning(f"TorchCompile: syntax error: {e}")
            return None
        except Exception as e:
            logger.warning(f"TorchCompile: compile error: {e}")
            return None


class MLIRStrategy:
    """
    For custom ASICs: emit MLIR text to disk for the vendor toolchain.

    The MLIR file is the deliverable — this strategy always returns None
    because the kernel is not directly callable from Python.
    Set MEMOPT_MLIR_OUT_DIR (default: /tmp/memopt_mlir) to control output dir.
    """

    def compile(self, source: str) -> Optional[types.ModuleType]:
        out_dir = os.environ.get("MEMOPT_MLIR_OUT_DIR", "/tmp/memopt_mlir")
        os.makedirs(out_dir, exist_ok=True)

        try:
            import uuid
            path = os.path.join(
                out_dir,
                f"kernel_{int(time.monotonic() * 1000)}_{uuid.uuid4().hex[:8]}.mlir"
            )
            with open(path, "w") as f:
                f.write(source)
            logger.info(f"MLIR: emitted to {path}")
        except Exception as e:
            logger.warning(f"MLIR: emit failed: {e}")

        return None   # not directly executable


# ── PortabilityLayer ──────────────────────────────────────────────────────────

class PortabilityLayer:
    """
    Compiles Triton kernel source to the binary for the current hardware.

    compile() takes a source string and a hardware hint string and returns
    a module with a `run_kernel` callable, or None on failure.

    profile() returns a HardwareProfile that prompt builders can use to
    generate architecture-aware kernels with correct block sizes and tile dims.
    """

    def __init__(self):
        self._hw      = detect_hardware()
        self._strategy = self._pick_strategy(self._hw)
        logger.info(
            f"PortabilityLayer: backend={self._hw.backend} "
            f"arch={self._hw.arch_name} "
            f"max_block={self._hw.max_block_size}"
        )

    def _pick_strategy(self, hw: HardwareProfile):
        if hw.backend == "triton_cuda":
            return TritonCUDAStrategy()
        if hw.backend == "triton_rocm":
            return TritonROCmStrategy()
        if hw.backend == "mlir":
            return MLIRStrategy()
        if hw.backend == "torch_compile":
            return TorchCompileStrategy()
        # cpu / unknown
        return TorchCompileStrategy()

    def compile(self, source: str, hardware_hint: str = "") -> Optional[types.ModuleType]:
        """
        Compile Triton (or plain PyTorch) kernel source for the detected target.

        Args:
            source:        Valid Python source. For GPU targets: import triton.
                           For CPU/Apple M targets: plain PyTorch ops.
            hardware_hint: Informational only — actual target is detected
                           from the runtime environment.

        Returns:
            A module with `run_kernel` callable, or None on failure.
        """
        if self._hw.backend == "cpu" and not isinstance(self._strategy, TorchCompileStrategy):
            logger.debug("PortabilityLayer: CPU target — no GPU kernel")
            return None

        return self._strategy.compile(source)

    def profile(self) -> HardwareProfile:
        """Return the detected hardware profile (for prompt builders)."""
        return self._hw

    def target(self) -> str:
        """Legacy shim: return 'cuda', 'rocm', or 'cpu' for existing tests."""
        if self._hw.backend in ("triton_cuda",):
            return "cuda"
        if self._hw.backend == "triton_rocm":
            return "rocm"
        return "cpu"
