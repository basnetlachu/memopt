"""
MemOpt quantization module

W8A8 (Weight 8-bit, Activation 8-bit) quantization for maximum performance
"""
from .w8a8_quantizer import W8A8Quantizer, quantize_model

__all__ = ['W8A8Quantizer', 'quantize_model']