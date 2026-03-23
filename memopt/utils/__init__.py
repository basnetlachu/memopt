# memopt/utils — utility helpers for loading and preparing models
from .model_loader import load_large_model, _estimate_model_gb
from .input_handler import (
    detect_input_format,
    forward,
    forward_with_inputs,
    extract_tensor,
    get_sequence_length,
    get_batch_size,
    InputFormat,
)
from .hardware_detector import detect_hardware, HardwareProfile
from .gpu_info import get_cuda_info, CudaInfo

__all__ = [
    "load_large_model",
    "_estimate_model_gb",
    "detect_input_format",
    "forward",
    "forward_with_inputs",
    "extract_tensor",
    "get_sequence_length",
    "get_batch_size",
    "InputFormat",
    "detect_hardware",
    "HardwareProfile",
    "get_cuda_info",
    "CudaInfo",
    "detect_model_type",
    "get_applicable_optimizations",
    "MODEL_FAMILY_OPTIMIZATIONS",
]
