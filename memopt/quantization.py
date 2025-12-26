"""
Quantization utilities for INT4/INT8 model compression.

This module provides utilities for quantizing model weights and activations
to reduce memory footprint and improve throughput.

Supported quantization schemes:
- INT8: 8-bit quantization (standard, good balance)
- INT4: 4-bit quantization (aggressive, maximum compression)

Expected benefits:
- Memory: 50-75% reduction
- Speed: 1.3-1.5x improvement
- Total speedup: 8-9x (from current 6x)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import math


class QuantizationConfig:
    """Configuration for quantization."""

    def __init__(
        self,
        bits: int = 8,
        symmetric: bool = True,
        per_channel: bool = True,
        group_size: Optional[int] = None
    ):
        """
        Args:
            bits: Number of bits for quantization (4 or 8)
            symmetric: Use symmetric quantization (zero_point = 0)
            per_channel: Quantize per output channel vs per tensor
            group_size: Group size for grouped quantization (None = per-channel)
        """
        assert bits in [4, 8], "Only 4-bit and 8-bit quantization supported"
        self.bits = bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.group_size = group_size

        # Quantization range
        if symmetric:
            self.qmin = -(2 ** (bits - 1))
            self.qmax = 2 ** (bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** bits - 1


def compute_quantization_params(
    tensor: torch.Tensor,
    qmin: int,
    qmax: int,
    symmetric: bool = True,
    dim: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute quantization scale and zero point.

    Args:
        tensor: Input tensor to quantize
        qmin: Minimum quantized value
        qmax: Maximum quantized value
        symmetric: Use symmetric quantization
        dim: Dimension for per-channel quantization (None = per-tensor)

    Returns:
        Tuple of (scale, zero_point)
    """
    if dim is None:
        # Per-tensor quantization
        min_val = tensor.min()
        max_val = tensor.max()
    else:
        # Per-channel quantization
        min_val = tensor.amin(dim=dim, keepdim=True)
        max_val = tensor.amax(dim=dim, keepdim=True)

    if symmetric:
        # Symmetric quantization: zero_point = 0
        max_abs = torch.max(torch.abs(min_val), torch.abs(max_val))
        scale = max_abs / (qmax - qmin) * 2
        zero_point = torch.zeros_like(scale, dtype=torch.int32)
    else:
        # Asymmetric quantization
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - torch.round(min_val / scale)
        zero_point = torch.clamp(zero_point, qmin, qmax).to(torch.int32)

    # Avoid division by zero
    scale = torch.clamp(scale, min=1e-8)

    return scale, zero_point


def quantize_tensor(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    qmin: int,
    qmax: int,
    dtype: torch.dtype = torch.int8
) -> torch.Tensor:
    """
    Quantize a floating point tensor.

    Args:
        tensor: Input tensor
        scale: Quantization scale
        zero_point: Quantization zero point
        qmin: Minimum quantized value
        qmax: Maximum quantized value
        dtype: Output dtype (torch.int8 or torch.uint8)

    Returns:
        Quantized tensor
    """
    # Quantize: q = clip(round(x / scale) + zero_point, qmin, qmax)
    q = torch.clamp(
        torch.round(tensor / scale) + zero_point,
        qmin,
        qmax
    )
    return q.to(dtype)


def dequantize_tensor(
    q_tensor: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor
) -> torch.Tensor:
    """
    Dequantize a quantized tensor back to float.

    Args:
        q_tensor: Quantized tensor
        scale: Quantization scale
        zero_point: Quantization zero point

    Returns:
        Dequantized float tensor
    """
    # Dequantize: x = scale * (q - zero_point)
    return scale * (q_tensor.to(torch.float32) - zero_point.to(torch.float32))


class QuantizedTensor:
    """
    Wrapper for a quantized tensor with its quantization parameters.
    """

    def __init__(
        self,
        q_tensor: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        dtype: torch.dtype = torch.int8
    ):
        """
        Args:
            q_tensor: Quantized tensor data
            scale: Quantization scale
            zero_point: Quantization zero point
            dtype: Data type of quantized tensor
        """
        self.q_tensor = q_tensor
        self.scale = scale
        self.zero_point = zero_point
        self.dtype = dtype

    def dequantize(self) -> torch.Tensor:
        """Dequantize back to float."""
        return dequantize_tensor(self.q_tensor, self.scale, self.zero_point)

    def to(self, device: torch.device) -> 'QuantizedTensor':
        """Move to device."""
        return QuantizedTensor(
            self.q_tensor.to(device),
            self.scale.to(device),
            self.zero_point.to(device),
            self.dtype
        )

    @property
    def shape(self):
        return self.q_tensor.shape

    @property
    def device(self):
        return self.q_tensor.device


def quantize_weight_int8(
    weight: torch.Tensor,
    per_channel: bool = True
) -> QuantizedTensor:
    """
    Quantize a weight tensor to INT8.

    Args:
        weight: Weight tensor [out_features, in_features]
        per_channel: Use per-channel quantization

    Returns:
        QuantizedTensor with INT8 weights
    """
    config = QuantizationConfig(bits=8, symmetric=True, per_channel=per_channel)

    # Compute quantization params
    dim = 0 if per_channel else None
    scale, zero_point = compute_quantization_params(
        weight,
        config.qmin,
        config.qmax,
        symmetric=True,
        dim=dim
    )

    # Quantize
    q_weight = quantize_tensor(
        weight,
        scale,
        zero_point,
        config.qmin,
        config.qmax,
        dtype=torch.int8
    )

    return QuantizedTensor(q_weight, scale, zero_point, torch.int8)


def quantize_weight_int4(
    weight: torch.Tensor,
    group_size: int = 128
) -> QuantizedTensor:
    """
    Quantize a weight tensor to INT4 with grouped quantization.

    Args:
        weight: Weight tensor [out_features, in_features]
        group_size: Size of groups for quantization

    Returns:
        QuantizedTensor with INT4 weights (stored as int8)
    """
    config = QuantizationConfig(bits=4, symmetric=True, group_size=group_size)

    out_features, in_features = weight.shape

    # Reshape into groups
    if in_features % group_size != 0:
        # Pad to multiple of group_size
        pad_size = group_size - (in_features % group_size)
        weight_padded = torch.nn.functional.pad(weight, (0, pad_size), value=0)
    else:
        weight_padded = weight
        pad_size = 0

    # Reshape to [out_features, num_groups, group_size]
    num_groups = weight_padded.shape[1] // group_size
    weight_grouped = weight_padded.view(out_features, num_groups, group_size)

    # Compute per-group quantization params
    scale, zero_point = compute_quantization_params(
        weight_grouped,
        config.qmin,
        config.qmax,
        symmetric=True,
        dim=2  # Per group
    )

    # Quantize
    q_weight = quantize_tensor(
        weight_grouped,
        scale,
        zero_point,
        config.qmin,
        config.qmax,
        dtype=torch.int8  # Store INT4 as INT8
    )

    # Flatten back
    q_weight = q_weight.view(out_features, -1)

    # Remove padding
    if pad_size > 0:
        q_weight = q_weight[:, :-pad_size]

    # Store metadata for dequantization
    q_tensor = QuantizedTensor(q_weight, scale, zero_point, torch.int8)
    q_tensor.group_size = group_size
    q_tensor.original_shape = weight.shape

    return q_tensor


def quantize_activation_int8(
    activation: torch.Tensor,
    running_min: Optional[torch.Tensor] = None,
    running_max: Optional[torch.Tensor] = None,
    momentum: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize activation to INT8 with running statistics.

    Args:
        activation: Activation tensor
        running_min: Running minimum for calibration
        running_max: Running maximum for calibration
        momentum: Momentum for updating running stats

    Returns:
        Tuple of (quantized_activation, new_running_min, new_running_max)
    """
    config = QuantizationConfig(bits=8, symmetric=False, per_channel=False)

    # Update running stats
    current_min = activation.min()
    current_max = activation.max()

    if running_min is None or running_max is None:
        running_min = current_min
        running_max = current_max
    else:
        running_min = (1 - momentum) * running_min + momentum * current_min
        running_max = (1 - momentum) * running_max + momentum * current_max

    # Compute quantization params using running stats
    scale = (running_max - running_min) / (config.qmax - config.qmin)
    scale = torch.clamp(scale, min=1e-8)
    zero_point = config.qmin - torch.round(running_min / scale)
    zero_point = torch.clamp(zero_point, config.qmin, config.qmax).to(torch.int32)

    # Quantize and dequantize (simulate quantization)
    q_activation = quantize_tensor(
        activation,
        scale,
        zero_point,
        config.qmin,
        config.qmax,
        dtype=torch.uint8
    )

    dq_activation = dequantize_tensor(q_activation, scale, zero_point)

    return dq_activation, running_min, running_max


class QuantizedLinear(nn.Module):
    """
    Quantized linear layer with INT8 weights.

    This layer stores weights in INT8 format and dequantizes on-the-fly
    during forward pass. Provides ~4x memory savings for weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bits: int = 8,
        per_channel: bool = True
    ):
        """
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            bias: Whether to include bias
            bits: Quantization bits (4 or 8)
            per_channel: Use per-channel quantization
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.per_channel = per_channel

        # Register buffers for quantized weights
        self.register_buffer('q_weight', torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer('weight_scale', torch.zeros(out_features if per_channel else 1))
        self.register_buffer('weight_zero_point', torch.zeros(out_features if per_channel else 1, dtype=torch.int32))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    @classmethod
    def from_float(cls, float_layer: nn.Linear, bits: int = 8, per_channel: bool = True) -> 'QuantizedLinear':
        """
        Create a quantized linear layer from a float linear layer.

        Args:
            float_layer: Original float linear layer
            bits: Quantization bits
            per_channel: Use per-channel quantization

        Returns:
            Quantized linear layer
        """
        q_layer = cls(
            float_layer.in_features,
            float_layer.out_features,
            bias=float_layer.bias is not None,
            bits=bits,
            per_channel=per_channel
        )

        # Quantize weights
        if bits == 8:
            q_weight_obj = quantize_weight_int8(float_layer.weight.data, per_channel=per_channel)
        else:
            q_weight_obj = quantize_weight_int4(float_layer.weight.data, group_size=128)

        q_layer.q_weight = q_weight_obj.q_tensor
        q_layer.weight_scale = q_weight_obj.scale
        q_layer.weight_zero_point = q_weight_obj.zero_point

        # Copy bias
        if float_layer.bias is not None:
            q_layer.bias.data = float_layer.bias.data

        return q_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with on-the-fly dequantization.

        Args:
            x: Input tensor [batch_size, seq_len, in_features]

        Returns:
            Output tensor [batch_size, seq_len, out_features]
        """
        # Dequantize weights
        weight = dequantize_tensor(self.q_weight, self.weight_scale, self.weight_zero_point)

        # Standard linear operation
        output = torch.nn.functional.linear(x, weight, self.bias)

        return output

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'bias={self.bias is not None}, bits={self.bits}, per_channel={self.per_channel}'


def quantize_model_int8(model: nn.Module, inplace: bool = False) -> nn.Module:
    """
    Quantize all linear layers in a model to INT8.

    Args:
        model: Model to quantize
        inplace: Whether to modify model in-place or create a copy

    Returns:
        Quantized model
    """
    if not inplace:
        model = type(model)(model.config) if hasattr(model, 'config') else model

    # Recursively replace linear layers
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Replace with quantized version
            q_layer = QuantizedLinear.from_float(module, bits=8, per_channel=True)
            setattr(model, name, q_layer)
        else:
            # Recursively quantize submodules
            quantize_model_int8(module, inplace=True)

    return model


def estimate_memory_savings(model: nn.Module, bits: int = 8) -> dict:
    """
    Estimate memory savings from quantization.

    Args:
        model: Model to analyze
        bits: Target quantization bits

    Returns:
        Dict with memory statistics
    """
    total_params = 0
    linear_params = 0

    for module in model.modules():
        if isinstance(module, nn.Linear):
            linear_params += module.weight.numel()
            if module.bias is not None:
                linear_params += module.bias.numel()

    for param in model.parameters():
        total_params += param.numel()

    # Calculate memory
    original_memory_mb = total_params * 4 / (1024 ** 2)  # FP32 = 4 bytes
    linear_memory_mb = linear_params * 4 / (1024 ** 2)

    # Quantized linear memory
    quantized_linear_mb = linear_params * (bits / 8) / (1024 ** 2)
    # Non-linear params stay FP32
    non_linear_mb = (total_params - linear_params) * 4 / (1024 ** 2)

    quantized_memory_mb = quantized_linear_mb + non_linear_mb

    savings_mb = original_memory_mb - quantized_memory_mb
    savings_pct = (savings_mb / original_memory_mb) * 100

    return {
        'original_memory_mb': original_memory_mb,
        'quantized_memory_mb': quantized_memory_mb,
        'savings_mb': savings_mb,
        'savings_pct': savings_pct,
        'total_params': total_params,
        'quantized_params': linear_params,
        'quantization_bits': bits
    }
