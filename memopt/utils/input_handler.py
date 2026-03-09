"""
Universal input handler for memopt.
Detects and normalizes ANY PyTorch model input format.
Called once at agent startup — result cached for the session.
"""
import torch
import torch.nn as nn
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class InputFormat:
    """
    Describes how to call a model's forward pass.
    Detected once, reused everywhere.
    """
    style: str          # "kwargs" | "positional" | "args_tuple"
    inputs: Dict        # normalized inputs ready to use
    model_family: str   # "transformer" | "cnn" | "custom" | "unknown"
    input_keys: list    # actual keys used (for logging)
    sample_output: Any  # reference output shape for validation


def detect_input_format(
    model: nn.Module,
    sample: Any,
    device: str = "cuda",
) -> "InputFormat":
    """
    Given ANY model and ANY sample batch, figure out how to call it.

    sample can be:
    - dict (HuggingFace models)
    - tensor (CNNs, custom)
    - tuple/list (multi-input models)
    - HuggingFace BatchEncoding (tokenizer output)
    - anything the user's dataloader produces

    Tries every known format in order of likelihood.
    Returns the first one that works without error.
    Raises clear error if nothing works.
    """
    # Idempotent: already detected
    if isinstance(sample, InputFormat):
        return sample

    device = device if torch.cuda.is_available() else "cpu"
    sample = _to_device(sample, device)

    # Strategy 1: sample is already a working dict
    if isinstance(sample, dict):
        result = _try_kwargs(model, sample)
        if result is not None:
            family = _detect_family(model, sample)
            log.info(
                "Input format: kwargs | keys=%s | family=%s",
                list(sample.keys()), family,
            )
            return InputFormat(
                style="kwargs",
                inputs=sample,
                model_family=family,
                input_keys=list(sample.keys()),
                sample_output=result,
            )

        # Dict given but fails — try subsets (some models ignore extra keys)
        for key in list(sample.keys()):
            subset = {key: sample[key]}
            result = _try_kwargs(model, subset)
            if result is not None:
                log.info("Input format: kwargs (subset) | key=%s", key)
                return InputFormat(
                    style="kwargs",
                    inputs=subset,
                    model_family=_detect_family(model, subset),
                    input_keys=[key],
                    sample_output=result,
                )

    # Strategy 1b: encoder-decoder model — synthesize decoder_input_ids
    # Tried when the dict was given but the forward call failed, which
    # typically means the model needs both encoder and decoder inputs.
    if isinstance(sample, dict) and is_encoder_decoder(model):
        augmented = synthesize_decoder_input(model, sample, device)
        result = _try_kwargs(model, augmented)
        if result is not None:
            log.info(
                "Input format: encoder-decoder kwargs | keys=%s",
                list(augmented.keys()),
            )
            return InputFormat(
                style="kwargs",
                inputs=augmented,
                model_family="encoder_decoder",
                input_keys=list(augmented.keys()),
                sample_output=result,
            )

    # Strategy 2: sample is a tensor — try all known key names then positional
    if isinstance(sample, torch.Tensor):
        KEY_NAMES = [
            # Transformer keys
            "input_ids", "inputs_embeds", "hidden_states",
            # Vision keys
            "pixel_values", "input_features", "images",
            # Generic keys
            "x", "input", "inputs", "data", "features",
        ]
        for key in KEY_NAMES:
            result = _try_kwargs(model, {key: sample})
            if result is not None:
                log.info("Input format: kwargs | auto-key='%s'", key)
                return InputFormat(
                    style="kwargs",
                    inputs={key: sample},
                    model_family=_detect_family(model, {key: sample}),
                    input_keys=[key],
                    sample_output=result,
                )

        # Try positional
        result = _try_positional(model, sample)
        if result is not None:
            log.info("Input format: positional (single tensor)")
            return InputFormat(
                style="positional",
                inputs={"__tensor__": sample},
                model_family="custom",
                input_keys=["__tensor__"],
                sample_output=result,
            )

    # Strategy 3: sample is tuple/list
    if isinstance(sample, (tuple, list)):
        # Try as *args
        result = _try_args(model, sample)
        if result is not None:
            log.info("Input format: *args tuple (len=%d)", len(sample))
            return InputFormat(
                style="args_tuple",
                inputs={"__args__": list(sample)},
                model_family="custom",
                input_keys=[f"arg_{i}" for i in range(len(sample))],
                sample_output=result,
            )

        # Try first element as single tensor
        if len(sample) > 0 and isinstance(sample[0], torch.Tensor):
            result = _try_positional(model, sample[0])
            if result is not None:
                log.info("Input format: positional (first element of tuple)")
                return InputFormat(
                    style="positional",
                    inputs={"__tensor__": sample[0]},
                    model_family="custom",
                    input_keys=["__tensor__"],
                    sample_output=result,
                )

    # Strategy 4: HuggingFace BatchEncoding (tokenizer output)
    if hasattr(sample, "input_ids") or hasattr(sample, "__iter__"):
        try:
            d = dict(sample)
            d = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in d.items()
            }
            result = _try_kwargs(model, d)
            if result is not None:
                log.info(
                    "Input format: HuggingFace BatchEncoding | keys=%s",
                    list(d.keys()),
                )
                return InputFormat(
                    style="kwargs",
                    inputs=d,
                    model_family="transformer",
                    input_keys=list(d.keys()),
                    sample_output=result,
                )
        except Exception:
            pass

    # Nothing worked
    raise ValueError(
        f"Cannot auto-detect input format for {type(model).__name__}.\n"
        f"Sample type received: {type(sample)}\n"
        f"Fix: pass one real batch from your dataloader as sample.\n"
        f"Example:\n"
        f"  sample = next(iter(your_dataloader))\n"
        f"  report = agent.run(model, sample)"
    )


def forward(model: nn.Module, fmt: "InputFormat") -> Any:
    """
    Universal forward call using detected format.
    Use this everywhere instead of model(**inputs).
    """
    if fmt.style == "kwargs":
        return model(**fmt.inputs)
    elif fmt.style == "positional":
        return model(fmt.inputs["__tensor__"])
    elif fmt.style == "args_tuple":
        return model(*fmt.inputs["__args__"])
    else:
        raise ValueError(f"Unknown input style: {fmt.style}")


def forward_with_inputs(
    model: nn.Module,
    fmt: "InputFormat",
    inputs: Dict,
) -> Any:
    """
    Forward with custom inputs (same format, different data).
    Used during calibration loops and benchmarking with modified inputs.
    """
    if fmt.style == "kwargs":
        return model(**inputs)
    elif fmt.style == "positional":
        return model(inputs.get("__tensor__", list(inputs.values())[0]))
    elif fmt.style == "args_tuple":
        return model(*inputs.get("__args__", list(inputs.values())))
    else:
        raise ValueError(f"Unknown input style: {fmt.style}")


def extract_tensor(output: Any) -> Optional[torch.Tensor]:
    """
    Extract first tensor from ANY model output format.
    Handles: Tensor, tuple, list, dict, HuggingFace ModelOutput.
    """
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, (tuple, list)):
        for item in output:
            result = extract_tensor(item)
            if result is not None:
                return result

    if isinstance(output, dict):
        for v in output.values():
            if isinstance(v, torch.Tensor):
                return v

    # HuggingFace ModelOutput — try iterating
    if hasattr(output, "__iter__"):
        try:
            for item in output:
                if isinstance(item, torch.Tensor):
                    return item
        except Exception:
            pass

    # Last resort: check all attributes
    if hasattr(output, "__dict__"):
        for v in output.__dict__.values():
            if isinstance(v, torch.Tensor):
                return v

    return None


def get_sequence_length(fmt: "InputFormat") -> Optional[int]:
    """
    Extract sequence length from input format.
    Works for transformers (input_ids) and CNNs (spatial dims).
    Returns None if cannot determine.
    """
    inputs = fmt.inputs

    # Transformer: sequence length is last dim of input_ids / inputs_embeds
    for key in ("input_ids", "inputs_embeds"):
        if key in inputs and isinstance(inputs[key], torch.Tensor):
            return inputs[key].shape[-1]

    # Vision: return spatial size (H*W)
    for key in ("pixel_values", "images"):
        if key in inputs and isinstance(inputs[key], torch.Tensor):
            t = inputs[key]
            if t.dim() == 4:  # B, C, H, W
                return t.shape[2] * t.shape[3]
            return t.shape[-1]

    # Generic named tensor — use last dim
    for key in ("x", "input", "inputs", "data", "features", "hidden_states"):
        if key in inputs and isinstance(inputs[key], torch.Tensor):
            t = inputs[key]
            return t.shape[-1] if t.dim() >= 2 else None

    # Positional tensor
    if "__tensor__" in inputs:
        t = inputs["__tensor__"]
        return t.shape[-1] if t.dim() >= 2 else None

    # args_tuple — use last dim of first tensor
    if "__args__" in inputs:
        for item in inputs["__args__"]:
            if isinstance(item, torch.Tensor) and item.dim() >= 2:
                return item.shape[-1]

    return None


def get_batch_size(fmt: "InputFormat") -> Optional[int]:
    """Extract batch size from any input format."""
    for v in fmt.inputs.values():
        if isinstance(v, torch.Tensor):
            return v.shape[0]
        if isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            return v[0].shape[0]
    return None


# ── Encoder-decoder helpers ───────────────────────────────────────────────────

def is_encoder_decoder(model: nn.Module) -> bool:
    """
    Return True when *model* is an encoder-decoder (seq2seq) architecture.

    Three independent structural checks — no model-name matching:
      1. HuggingFace config: model.config.is_encoder_decoder == True
      2. Named children: both 'encoder' and 'decoder' exist as direct children
      3. Object __dict__: model has both 'encoder' and 'decoder' attributes

    A model passes if ANY single check is True.
    """
    # Check 1: HuggingFace config attribute
    try:
        if getattr(getattr(model, 'config', None), 'is_encoder_decoder', False):
            return True
    except Exception:
        pass

    # Check 2: named_children() contains both 'encoder' and 'decoder'
    try:
        child_names = {name for name, _ in model.named_children()}
        if 'encoder' in child_names and 'decoder' in child_names:
            return True
    except Exception:
        pass

    # Check 3: __dict__ has both 'encoder' and 'decoder' attributes
    try:
        d = model.__dict__
        if 'encoder' in d and 'decoder' in d:
            return True
    except Exception:
        pass

    return False


def synthesize_decoder_input(
    model: nn.Module,
    encoder_input: Dict,
    device: str = "cuda",
) -> Dict:
    """
    Augment *encoder_input* with a minimal ``decoder_input_ids`` tensor.

    Called when detect_input_format() detects an encoder-decoder model and
    the caller provided only encoder-side inputs. Produces the
    ``decoder_input_ids`` key required for a full seq2seq forward pass.

    The decoder start token is taken from (priority order):
      1. model.config.decoder_start_token_id
      2. model.config.bos_token_id
      3. model.config.pad_token_id
      4. 0  (final fallback)

    Args:
        model:          nn.Module with encoder-decoder architecture
        encoder_input:  dict of encoder-side tensors (e.g. {'input_ids': ...})
        device:         target device string

    Returns:
        New dict with all encoder_input keys plus 'decoder_input_ids'
        (shape: batch_size × 1).
    """
    cfg = getattr(model, 'config', None)
    start_token_id = int(
        getattr(cfg, 'decoder_start_token_id', None)
        or getattr(cfg, 'bos_token_id', None)
        or getattr(cfg, 'pad_token_id', None)
        or 0
    )

    batch_size = 1
    for v in encoder_input.values():
        if isinstance(v, torch.Tensor):
            batch_size = v.shape[0]
            break

    decoder_input_ids = torch.full(
        (batch_size, 1), start_token_id, dtype=torch.long, device=device
    )

    augmented = dict(encoder_input)
    augmented['decoder_input_ids'] = decoder_input_ids
    return augmented


# ── Private helpers ────────────────────────────────────────────────────────────

def _try_kwargs(model: nn.Module, inputs: dict) -> Optional[Any]:
    try:
        with torch.no_grad():
            out = model(**inputs)
        return out
    except Exception:
        return None


def _try_positional(model: nn.Module, tensor: torch.Tensor) -> Optional[Any]:
    try:
        with torch.no_grad():
            out = model(tensor)
        return out
    except Exception:
        return None


def _try_args(model: nn.Module, args) -> Optional[Any]:
    try:
        with torch.no_grad():
            out = model(*args)
        return out
    except Exception:
        return None


def _to_device(sample: Any, device: str) -> Any:
    """Recursively move sample to device."""
    if isinstance(sample, torch.Tensor):
        return sample.to(device)
    elif isinstance(sample, dict):
        return {k: _to_device(v, device) for k, v in sample.items()}
    elif isinstance(sample, (list, tuple)):
        converted = [_to_device(x, device) for x in sample]
        return type(sample)(converted)
    return sample  # non-tensor passthrough (int labels, etc.)


def _detect_family(model: nn.Module, inputs: Any) -> str:
    """Detect model family from architecture and input keys."""
    keys = set(inputs.keys()) if isinstance(inputs, dict) else set()

    # Check input keys first
    if "input_ids" in keys:
        return "transformer"
    if "pixel_values" in keys or "images" in keys:
        return "vision_transformer"
    if "input_features" in keys:
        return "audio"

    # Check model architecture
    module_names = {name for name, _ in model.named_modules()}

    if any("attention" in n or "attn" in n for n in module_names):
        return "transformer"
    if any("conv" in n for n in module_names):
        return "cnn"

    return "custom"


# ── Unified multi-modal interface ────────────────────────────────────────────
# Use these instead of model(**inputs) anywhere in the codebase
from memopt.utils.multimodal import forward as universal_forward  # noqa: E402
from memopt.utils.multimodal import detect_modality               # noqa: E402
