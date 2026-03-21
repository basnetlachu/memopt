# memopt/utils — utility helpers for loading and preparing models
#
# Torch-dependent modules (model_loader, input_handler) are imported
# lazily so that non-GPU code paths (portability_layer.detect_hardware,
# gpu_info) can import this package without requiring torch.
from .hardware_detector import detect_hardware, HardwareProfile
from .gpu_info import get_cuda_info, CudaInfo


def __getattr__(name):
    """Lazy imports for torch-dependent symbols."""
    _model_loader_names = {"load_large_model", "_estimate_model_gb"}
    _input_handler_names = {
        "detect_input_format", "forward", "forward_with_inputs",
        "extract_tensor", "get_sequence_length", "get_batch_size",
        "InputFormat",
    }
    if name in _model_loader_names:
        from . import model_loader
        return getattr(model_loader, name)
    if name in _input_handler_names:
        from . import input_handler
        return getattr(input_handler, name)
    raise AttributeError(f"module 'memopt.utils' has no attribute {name!r}")


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
