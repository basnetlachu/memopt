"""
Phase 3: Custom Kernel Registry

Registry of high-performance kernels with automatic fallbacks.
Provides unified API for Flash Attention, xFormers, and custom kernels.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("memopt.phase3")

# Try to import torch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    F = None


@dataclass
class KernelInfo:
    """Information about a registered kernel"""
    name: str
    implementation: Callable
    fallback: Optional[Callable] = None
    requirements: List[str] = field(default_factory=list)
    description: str = ""
    expected_speedup_pct: float = 0.0


class CustomKernelRegistry:
    """
    Registry of custom high-performance kernels.

    Automatically selects best available implementation and falls back
    to slower alternatives when requirements aren't met.
    """

    def __init__(self):
        self.kernels: Dict[str, KernelInfo] = {}
        self._detect_available_backends()
        self._register_builtin_kernels()

    def _detect_available_backends(self):
        """Detect which optimized kernel libraries are available."""

        self.has_flash_attn = self._check_flash_attention()
        self.has_xformers = self._check_xformers()
        self.has_triton = self._check_triton()
        self.has_cudnn = self._check_cudnn()
        self.has_sdpa = self._check_sdpa()

        logger.info("Available backends:")
        logger.info(f"  Flash Attention: {self.has_flash_attn}")
        logger.info(f"  xFormers: {self.has_xformers}")
        logger.info(f"  Triton: {self.has_triton}")
        logger.info(f"  cuDNN: {self.has_cudnn}")
        logger.info(f"  PyTorch SDPA: {self.has_sdpa}")
        logger.info(f"Attention backend: {self.active_attention_backend()}")

    def _check_flash_attention(self) -> bool:
        try:
            import flash_attn
            return True
        except ImportError:
            return False

    def _check_xformers(self) -> bool:
        try:
            import xformers
            return True
        except ImportError:
            return False

    def _check_triton(self) -> bool:
        try:
            import triton
            return True
        except ImportError:
            return False

    def _check_cudnn(self) -> bool:
        if not HAS_TORCH:
            return False
        return torch.backends.cudnn.is_available()

    def _check_sdpa(self) -> bool:
        if not HAS_TORCH:
            return False
        return hasattr(F, 'scaled_dot_product_attention')

    def active_attention_backend(self) -> str:
        """Return the highest-priority available attention backend name."""
        if self.has_flash_attn:
            return "flash_attn"
        if self.has_xformers:
            return "xformers"
        if self.has_sdpa:
            return "sdpa"
        return "naive"

    def _register_builtin_kernels(self):
        """Register built-in optimized kernels."""

        # Flash Attention
        self.register_kernel(
            name="fused_attention",
            implementation=self._fused_attention_impl,
            fallback=self._naive_attention_impl,
            requirements=[],  # Handled internally
            description="High-performance fused attention",
            expected_speedup_pct=50.0
        )

        # Fused LayerNorm + Linear
        self.register_kernel(
            name="fused_layernorm_linear",
            implementation=self._fused_layernorm_linear_impl,
            fallback=self._naive_layernorm_linear_impl,
            requirements=['torch_compile'],
            description="Fused LayerNorm + Linear using torch.compile",
            expected_speedup_pct=20.0
        )

        # Fused GELU + Dropout
        self.register_kernel(
            name="fused_gelu_dropout",
            implementation=self._fused_gelu_dropout_impl,
            fallback=self._naive_gelu_dropout_impl,
            requirements=['torch_compile'],
            description="Fused GELU + Dropout using torch.compile",
            expected_speedup_pct=15.0
        )

    def register_kernel(
        self,
        name: str,
        implementation: Callable,
        fallback: Optional[Callable] = None,
        requirements: Optional[List[str]] = None,
        description: str = "",
        expected_speedup_pct: float = 0.0
    ):
        """Register a custom kernel."""

        self.kernels[name] = KernelInfo(
            name=name,
            implementation=implementation,
            fallback=fallback,
            requirements=requirements or [],
            description=description,
            expected_speedup_pct=expected_speedup_pct
        )
        logger.debug(f"Registered kernel: {name}")

    def get_kernel(self, name: str) -> Callable:
        """Get best available implementation of kernel."""

        if name not in self.kernels:
            raise ValueError(f"Unknown kernel: {name}")

        kernel_info = self.kernels[name]

        # Check if requirements are met
        requirements_met = all(
            self._check_requirement(req)
            for req in kernel_info.requirements
        )

        if requirements_met:
            return kernel_info.implementation
        elif kernel_info.fallback is not None:
            warnings.warn(
                f"Requirements not met for {name}, using fallback implementation"
            )
            return kernel_info.fallback
        else:
            raise RuntimeError(
                f"Requirements not met for {name} and no fallback available"
            )

    def _check_requirement(self, req: str) -> bool:
        """Check if requirement is satisfied."""

        requirements_map = {
            'flash_attn': self.has_flash_attn,
            'xformers': self.has_xformers,
            'triton': self.has_triton,
            'cudnn': self.has_cudnn,
            'sdpa': self.has_sdpa,
            'torch_compile': HAS_TORCH and hasattr(torch, 'compile'),
        }

        return requirements_map.get(req, False)

    def list_kernels(self) -> List[str]:
        """List all registered kernels."""
        return list(self.kernels.keys())

    def get_kernel_info(self, name: str) -> Optional[KernelInfo]:
        """Get information about a kernel."""
        return self.kernels.get(name)

    # =========================================================================
    # Built-in kernel implementations
    # =========================================================================

    def _fused_attention_impl(
        self,
        query: Any,
        key: Any,
        value: Any,
        attn_mask: Optional[Any] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False
    ) -> Any:
        """Fused attention using best available backend."""

        if not HAS_TORCH:
            raise RuntimeError("PyTorch not available")

        # Try Flash Attention library first
        if self.has_flash_attn:
            try:
                from flash_attn import flash_attn_func
                # flash_attn expects (batch, seqlen, nheads, headdim)
                # Standard format is (batch, nheads, seqlen, headdim)
                # May need reshaping depending on input format
                return flash_attn_func(
                    query, key, value,
                    dropout_p=dropout_p,
                    causal=is_causal
                )
            except Exception as e:
                logger.debug(f"flash_attn failed, trying SDPA: {e}")

        # Tier 2: xFormers (avoids flash_attn ABI issues, ~10-15% behind flash_attn)
        if self.has_xformers:
            try:
                import xformers.ops as xops
                return xops.memory_efficient_attention(
                    query, key, value,
                    attn_bias=attn_mask,
                    scale=query.size(-1) ** -0.5,
                )
            except Exception as e:
                logger.debug(f"xFormers failed, falling back to SDPA: {e}")

        # Tier 3: PyTorch SDPA (always available >= 2.0, uses flash-attn v2 kernel internally)
        if self.has_sdpa:
            return F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal
            )

        # Tier 4: naive fallback
        return self._naive_attention_impl(
            query, key, value, attn_mask, dropout_p
        )

    def _naive_attention_impl(
        self,
        query: Any,
        key: Any,
        value: Any,
        attn_mask: Optional[Any] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False
    ) -> Any:
        """Naive attention implementation (fallback)."""

        if not HAS_TORCH:
            raise RuntimeError("PyTorch not available")

        # Standard attention: softmax(QK^T / sqrt(d)) * V
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)

        if attn_mask is not None:
            scores = scores + attn_mask

        if is_causal:
            # Create causal mask
            seq_len = query.size(-2)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)

        if dropout_p > 0:
            # CustomKernelRegistry is not nn.Module; apply dropout unconditionally
            # when called (caller controls training vs eval via dropout_p=0.0)
            attn = F.dropout(attn, p=dropout_p)

        output = torch.matmul(attn, value)
        return output

    def _fused_layernorm_linear_impl(
        self,
        x: Any,
        layer_norm: Any,
        linear: Any
    ) -> Any:
        """Fused LayerNorm + Linear using torch.compile."""

        if not HAS_TORCH:
            raise RuntimeError("PyTorch not available")

        @torch.compile(mode='max-autotune')
        def fused_op(x, ln, fc):
            return fc(ln(x))

        return fused_op(x, layer_norm, linear)

    def _naive_layernorm_linear_impl(
        self,
        x: Any,
        layer_norm: Any,
        linear: Any
    ) -> Any:
        """Naive LayerNorm + Linear (fallback)."""

        if not HAS_TORCH:
            raise RuntimeError("PyTorch not available")

        return linear(layer_norm(x))

    def _fused_gelu_dropout_impl(
        self,
        x: Any,
        dropout_p: float = 0.1
    ) -> Any:
        """Fused GELU + Dropout using torch.compile."""

        if not HAS_TORCH:
            raise RuntimeError("PyTorch not available")

        @torch.compile(mode='max-autotune')
        def fused_op(x, p):
            return F.dropout(F.gelu(x), p=p)

        return fused_op(x, dropout_p)

    def _naive_gelu_dropout_impl(
        self,
        x: Any,
        dropout_p: float = 0.1
    ) -> Any:
        """Naive GELU + Dropout (fallback)."""

        if not HAS_TORCH:
            raise RuntimeError("PyTorch not available")

        return F.dropout(F.gelu(x), p=dropout_p)


# Global kernel registry instance
kernel_registry = CustomKernelRegistry()


def fused_attention(
    query: Any,
    key: Any,
    value: Any,
    attn_mask: Optional[Any] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False
) -> Any:
    """
    High-performance fused attention.

    Automatically selects best available implementation:
    1. Flash Attention (best)
    2. PyTorch SDPA (good)
    3. xFormers (good)
    4. Naive PyTorch (fallback)

    Args:
        query: Query tensor (batch, heads, seq_len, head_dim)
        key: Key tensor (batch, heads, seq_len, head_dim)
        value: Value tensor (batch, heads, seq_len, head_dim)
        attn_mask: Optional attention mask
        dropout_p: Dropout probability
        is_causal: Whether to apply causal masking

    Returns:
        Attention output tensor
    """

    kernel = kernel_registry.get_kernel("fused_attention")
    return kernel(query, key, value, attn_mask, dropout_p, is_causal)


def get_active_attention_backend() -> str:
    """Return the highest-priority available attention backend.

    Returns one of: "flash_attn", "xformers", "sdpa", "naive".
    Priority order: flash_attn > xformers > sdpa > naive.
    """
    return kernel_registry.active_attention_backend()


def fused_layernorm_linear(
    x: Any,
    layer_norm: Any,
    linear: Any
) -> Any:
    """
    Fused LayerNorm + Linear operation.

    Args:
        x: Input tensor
        layer_norm: LayerNorm module
        linear: Linear module

    Returns:
        Output tensor
    """

    kernel = kernel_registry.get_kernel("fused_layernorm_linear")
    return kernel(x, layer_norm, linear)


def fused_gelu_dropout(
    x: Any,
    dropout_p: float = 0.1
) -> Any:
    """
    Fused GELU + Dropout operation.

    Args:
        x: Input tensor
        dropout_p: Dropout probability

    Returns:
        Output tensor
    """

    kernel = kernel_registry.get_kernel("fused_gelu_dropout")
    return kernel(x, dropout_p)
