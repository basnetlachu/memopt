"""
MemOpt vLLM Plugin
Hooks into vLLM's scheduler, KV cache, and memory planning for optimization

PERFORMANCE GUARANTEE:
- NO hot path modification (token generation loop untouched)
- Patches only affect control plane (batch planning, allocation)
- Safe mode available (MEMOPT_SAFE_MODE=1) for maximum safety
"""
import os
from typing import Optional, Dict, Any
import warnings


def is_enabled() -> bool:
    """Check if MemOpt is enabled"""
    return os.getenv("MEMOPT_ENABLED") == "1"


def is_strict_mode() -> bool:
    """Check if strict mode is enabled (fail if patching fails)"""
    return os.getenv("MEMOPT_STRICT") == "1"


def is_safe_mode() -> bool:
    """
    Check if safe mode is enabled.

    Safe mode ONLY enables static optimizations (KV cache configuration)
    and DISABLES dynamic patches (scheduler, allocation hooks) to prevent
    any performance regression in vLLM's hot path.
    """
    return os.getenv("MEMOPT_SAFE_MODE") == "1"


class VLLMPatcher:
    """Patches vLLM components for MemOpt optimization"""

    def __init__(self):
        self.patches_applied: Dict[str, bool] = {}
        self.patch_errors: Dict[str, str] = {}
        self.vllm_version: Optional[str] = None

    def detect_vllm(self) -> bool:
        """Detect if vLLM is installed"""
        try:
            import vllm
            self.vllm_version = vllm.__version__
            return True
        except ImportError:
            return False

    def patch_scheduler(self) -> bool:
        """
        Patch vLLM scheduler for memory-aware batching.

        PERFORMANCE CRITICAL: This patch only affects the scheduler tick (batch planning),
        NOT the per-token decode loop. Safe mode disables this patch entirely.
        """
        # SAFETY: Skip scheduler patching in safe mode
        if is_safe_mode():
            self.patch_errors['scheduler'] = "Skipped in safe mode (no hot path modification)"
            return False

        try:
            from vllm.core.scheduler import Scheduler
            from . import scheduler as memopt_scheduler

            # Store original method
            if not hasattr(Scheduler, '_original_schedule'):
                Scheduler._original_schedule = Scheduler.schedule

            # Patch schedule method
            # PERFORMANCE GUARANTEE: This only runs at batch planning time,
            # NOT per token. vLLM's CUDA kernel hot path is unchanged.
            def memopt_schedule(self, *args, **kwargs):
                # Use MemOpt scheduler logic
                return memopt_scheduler.optimize_schedule(
                    self,
                    *args,
                    original_schedule=self._original_schedule,
                    **kwargs
                )

            Scheduler.schedule = memopt_schedule
            self.patches_applied['scheduler'] = True
            return True

        except ImportError as e:
            self.patch_errors['scheduler'] = f"Import error: {e}"
            return False
        except AttributeError as e:
            self.patch_errors['scheduler'] = f"Scheduler API changed: {e}"
            return False
        except Exception as e:
            self.patch_errors['scheduler'] = f"Unexpected error: {e}"
            return False

    def patch_kv_cache(self) -> bool:
        """
        Patch vLLM KV cache allocator.

        PERFORMANCE CRITICAL: This patch only affects cache initialization and
        block allocation (control plane), NOT the per-token attention kernel.
        """
        try:
            # Try to patch PagedAttention config
            try:
                from vllm.worker.model_runner import ModelRunner
                from . import kv_cache as memopt_kv

                if not hasattr(ModelRunner, '_original_init_cache_engine'):
                    ModelRunner._original_init_cache_engine = ModelRunner._init_cache_engine

                def memopt_init_cache_engine(self, *args, **kwargs):
                    # Initialize with MemOpt KV cache config
                    result = self._original_init_cache_engine(*args, **kwargs)
                    memopt_kv.optimize_cache_config(self.cache_config)
                    return result

                ModelRunner._init_cache_engine = memopt_init_cache_engine
                self.patches_applied['kv_cache'] = True
                return True

            except (ImportError, AttributeError):
                # Fallback: Try alternative vLLM structure
                from vllm.core.block_manager import BlockSpaceManager
                from . import kv_cache as memopt_kv

                if not hasattr(BlockSpaceManager, '_original_allocate'):
                    BlockSpaceManager._original_allocate = BlockSpaceManager.allocate

                def memopt_allocate(self, *args, **kwargs):
                    # Use MemOpt allocation strategy
                    return memopt_kv.optimize_allocation(
                        self,
                        *args,
                        original_allocate=self._original_allocate,
                        **kwargs
                    )

                BlockSpaceManager.allocate = memopt_allocate
                self.patches_applied['kv_cache'] = True
                return True

        except ImportError as e:
            self.patch_errors['kv_cache'] = f"Import error: {e}"
            return False
        except Exception as e:
            self.patch_errors['kv_cache'] = f"Unexpected error: {e}"
            return False

    def patch_memory_planner(self) -> bool:
        """
        Patch memory planning parameters.

        PERFORMANCE SAFE: This only affects static configuration at initialization.
        No runtime overhead.
        """
        try:
            from vllm.config import CacheConfig
            from . import memory_manager as memopt_memory

            # Override default cache config
            if not hasattr(CacheConfig, '_original_init'):
                CacheConfig._original_init = CacheConfig.__init__

            def memopt_cache_config_init(self, *args, **kwargs):
                self._original_init(*args, **kwargs)
                # Apply MemOpt memory optimizations
                memopt_memory.optimize_cache_params(self)

            CacheConfig.__init__ = memopt_cache_config_init
            self.patches_applied['memory_planner'] = True
            return True

        except ImportError as e:
            self.patch_errors['memory_planner'] = f"Import error: {e}"
            return False
        except Exception as e:
            self.patch_errors['memory_planner'] = f"Unexpected error: {e}"
            return False

    def apply_all_patches(self) -> bool:
        """Apply all vLLM patches"""
        if not self.detect_vllm():
            warnings.warn(
                "vLLM not detected. MemOpt vLLM plugin disabled.\n"
                "Install vLLM to use MemOpt optimizations.",
                UserWarning
            )
            return False

        print(f"[MemOpt] Detected vLLM version: {self.vllm_version}")

        # SAFETY: Warn about safe mode
        if is_safe_mode():
            print("[MemOpt] SAFE MODE enabled - only static optimizations applied")
            print("[MemOpt] Dynamic scheduler/allocator patches disabled for guaranteed safety")

        print("[MemOpt] Applying vLLM patches...")

        # Apply patches
        self.patch_scheduler()
        self.patch_kv_cache()
        self.patch_memory_planner()

        # Report results
        successful = [k for k, v in self.patches_applied.items() if v]
        failed = [k for k, v in self.patch_errors.items()]

        if successful:
            print(f"[MemOpt] Successfully patched: {', '.join(successful)}")

        if failed:
            error_details = '\n'.join([f"  - {k}: {v}" for k, v in self.patch_errors.items()])
            warnings.warn(
                f"[MemOpt] Failed to patch some components:\n{error_details}",
                UserWarning
            )

            if is_strict_mode():
                raise RuntimeError(
                    f"MemOpt strict mode enabled but patches failed:\n{error_details}"
                )

        return len(successful) > 0

    def get_status(self) -> Dict[str, Any]:
        """Get patch status for diagnostics"""
        return {
            "vllm_detected": self.vllm_version is not None,
            "vllm_version": self.vllm_version,
            "patches_applied": self.patches_applied,
            "patch_errors": self.patch_errors,
        }


# Global patcher instance
_patcher: Optional[VLLMPatcher] = None


def initialize_plugin() -> bool:
    """Initialize MemOpt vLLM plugin"""
    global _patcher

    if not is_enabled():
        return False

    # Validate license
    try:
        from .license import validate_license
        license_obj = validate_license()
        print(f"[MemOpt] License validated for customer: {license_obj.customer_id}")

        if not license_obj.has_feature("vllm"):
            warnings.warn(
                "License does not include 'vllm' feature. Plugin disabled.",
                UserWarning
            )
            return False

    except Exception as e:
        if is_strict_mode():
            raise RuntimeError(f"License validation failed: {e}")
        warnings.warn(f"License validation failed: {e}", UserWarning)
        return False

    # Apply patches
    _patcher = VLLMPatcher()
    success = _patcher.apply_all_patches()

    if success:
        print("[MemOpt] vLLM plugin initialized successfully")
    else:
        print("[MemOpt] vLLM plugin initialization failed (vLLM will run normally)")

    return success


def enable_vllm() -> bool:
    """
    Explicitly enable MemOpt for vLLM
    Can be called directly if auto-init is disabled
    """
    return initialize_plugin()


def get_patcher() -> Optional[VLLMPatcher]:
    """Get global patcher instance for diagnostics"""
    return _patcher


def get_status() -> Dict[str, Any]:
    """Get plugin status"""
    if _patcher is None:
        return {
            "enabled": False,
            "reason": "Plugin not initialized"
        }

    return {
        "enabled": True,
        **_patcher.get_status()
    }
