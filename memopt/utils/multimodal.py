"""
Universal Multi-Modal Input Handler.

Extends memopt beyond LLM-only inference to:
  - Vision models (ResNet, ViT, CLIP)
  - Audio models (Whisper)
  - Video models
  - Multimodal models (LLaVA, Flamingo)
  - Graph neural networks / custom formats

Same scan, same optimize, same CLI — any model type.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from enum import Enum

logger = logging.getLogger("memopt.multimodal")


class ModalityType(Enum):
    TEXT          = "text"
    VISION        = "vision"
    AUDIO         = "audio"
    VIDEO         = "video"
    MULTIMODAL    = "multimodal"
    GRAPH         = "graph"
    TABULAR       = "tabular"
    UNKNOWN       = "unknown"


@dataclass
class ModalityProfile:
    modality:           ModalityType
    input_shape_hint:   Tuple           # e.g. (batch, seq_len, hidden)
    sample_input:       Dict[str, Any]  # ready-to-use sample input
    forward_kwargs:     Dict[str, Any]  # kwargs to pass to model.forward()
    memory_estimate_mb: float           # estimated VRAM for this input
    recommended_batch:  int             # suggested batch size for this modality


class MultiModalDetector:
    """
    Detects what kind of model is being optimized and generates
    correct sample inputs for profiling without running a forward pass.
    """

    def detect(self, model: Any,
               user_hints: Dict = None) -> ModalityProfile:
        """
        Auto-detect model modality from class name, then structure.
        Falls back to user_hints if auto-detection fails.
        """
        user_hints = user_hints or {}
        model_class = type(model).__name__.lower()

        # ── TEXT / LLM ────────────────────────────────────────────────
        if any(k in model_class for k in [
            "llama", "mistral", "gpt", "falcon", "bloom",
            "opt", "gemma", "qwen", "phi", "causallm",
            "seq2seq", "t5", "bart", "bert", "roberta",
        ]):
            return self._text_profile(model)

        # ── VISION ────────────────────────────────────────────────────
        if any(k in model_class for k in [
            "resnet", "vit", "convnext", "efficientnet",
            "clip", "dino", "swin", "regnet", "mobilenet",
        ]):
            return self._vision_profile(model)

        # ── AUDIO ─────────────────────────────────────────────────────
        if any(k in model_class for k in [
            "whisper", "wav2vec", "hubert", "encodec",
            "musicgen", "audioldm",
        ]):
            return self._audio_profile(model)

        # ── VIDEO ─────────────────────────────────────────────────────
        if any(k in model_class for k in [
            "videomae", "timesformer", "vivit", "video",
        ]):
            return self._video_profile(model)

        # ── MULTIMODAL ────────────────────────────────────────────────
        if any(k in model_class for k in [
            "llava", "flamingo", "blip", "idefics", "paligemma",
        ]):
            return self._multimodal_profile(model)

        # ── FALLBACK: Inspect model structure ─────────────────────────
        return self._detect_from_structure(model, user_hints)

    def _text_profile(self, model: Any) -> ModalityProfile:
        try:
            import torch
            batch, seq = 1, 512
            input_ids = torch.ones(batch, seq, dtype=torch.long)
            return ModalityProfile(
                modality=ModalityType.TEXT,
                input_shape_hint=(batch, seq),
                sample_input={"input_ids": input_ids},
                forward_kwargs={"input_ids": input_ids},
                memory_estimate_mb=self._estimate_activation_mb(batch, seq, 4096),
                recommended_batch=32,
            )
        except ImportError:
            return self._unknown_profile()

    def _vision_profile(self, model: Any) -> ModalityProfile:
        try:
            import torch
            batch, c, h, w = 1, 3, 224, 224
            pixel_values = torch.randn(batch, c, h, w)
            return ModalityProfile(
                modality=ModalityType.VISION,
                input_shape_hint=(batch, c, h, w),
                sample_input={"pixel_values": pixel_values},
                forward_kwargs={"pixel_values": pixel_values},
                memory_estimate_mb=batch * c * h * w * 4 / (1024 ** 2) * 10,
                recommended_batch=64,
            )
        except ImportError:
            return self._unknown_profile()

    def _audio_profile(self, model: Any) -> ModalityProfile:
        try:
            import torch
            # 30 seconds of audio at 16kHz → Whisper mel spec (80 bins, 3000 frames)
            input_features = torch.randn(1, 80, 3000)
            return ModalityProfile(
                modality=ModalityType.AUDIO,
                input_shape_hint=(1, 80, 3000),
                sample_input={"input_features": input_features},
                forward_kwargs={"input_features": input_features},
                memory_estimate_mb=500,
                recommended_batch=8,
            )
        except ImportError:
            return self._unknown_profile()

    def _video_profile(self, model: Any) -> ModalityProfile:
        try:
            import torch
            batch, frames, c, h, w = 1, 16, 3, 224, 224
            pixel_values = torch.randn(batch, frames, c, h, w)
            return ModalityProfile(
                modality=ModalityType.VIDEO,
                input_shape_hint=(batch, frames, c, h, w),
                sample_input={"pixel_values": pixel_values},
                forward_kwargs={"pixel_values": pixel_values},
                memory_estimate_mb=batch * frames * c * h * w * 4 / (1024 ** 2) * 5,
                recommended_batch=4,
            )
        except ImportError:
            return self._unknown_profile()

    def _multimodal_profile(self, model: Any) -> ModalityProfile:
        try:
            import torch
            batch, seq = 1, 256
            input_ids    = torch.ones(batch, seq, dtype=torch.long)
            pixel_values = torch.randn(batch, 3, 336, 336)
            return ModalityProfile(
                modality=ModalityType.MULTIMODAL,
                input_shape_hint=(batch, seq),
                sample_input={
                    "input_ids":    input_ids,
                    "pixel_values": pixel_values,
                },
                forward_kwargs={
                    "input_ids":    input_ids,
                    "pixel_values": pixel_values,
                },
                memory_estimate_mb=8000,
                recommended_batch=4,
            )
        except ImportError:
            return self._unknown_profile()

    def _detect_from_structure(self,
                                model: Any,
                                hints: Dict) -> ModalityProfile:
        """Inspect model structure to guess modality."""
        try:
            import torch.nn as nn

            has_embedding = any(
                isinstance(m, nn.Embedding) for m in model.modules()
            )
            has_conv = any(
                isinstance(m, (nn.Conv2d, nn.Conv1d)) for m in model.modules()
            )

            if has_embedding and not has_conv:
                logger.info("Detected TEXT model from structure")
                return self._text_profile(model)
            elif has_conv and not has_embedding:
                logger.info("Detected VISION model from structure")
                return self._vision_profile(model)
            elif has_conv and has_embedding:
                logger.info("Detected MULTIMODAL model from structure")
                return self._multimodal_profile(model)

        except Exception as e:
            logger.debug("Structure detection failed: %s", e)

        return self._unknown_profile()

    def _unknown_profile(self) -> ModalityProfile:
        return ModalityProfile(
            modality=ModalityType.UNKNOWN,
            input_shape_hint=(1, 512),
            sample_input={},
            forward_kwargs={},
            memory_estimate_mb=1000,
            recommended_batch=1,
        )

    def _estimate_activation_mb(self, batch: int, seq: int, hidden: int) -> float:
        # float16 activations: batch * seq * hidden * 2 bytes * ~8 layers overhead
        return batch * seq * hidden * 2 * 8 / (1024 ** 2)


class UniversalForward:
    """
    Executes model forward pass correctly for any modality.
    Replaces scattered model(**sample_input) calls.
    Caches detected modality profile per model instance.
    """

    def __init__(self):
        self.detector = MultiModalDetector()
        self._profile_cache: Dict[int, ModalityProfile] = {}

    def forward(self, model: Any,
                inputs: Optional[Dict] = None,
                hints: Dict = None) -> Any:
        """
        Run model forward pass with correct inputs for its modality.

        Args:
            model:  any nn.Module
            inputs: optional override inputs (use detected profile if None)
            hints:  modality hints if auto-detection fails
        """
        model_id = id(model)

        if model_id not in self._profile_cache:
            profile = self.detector.detect(model, hints or {})
            self._profile_cache[model_id] = profile
            logger.info(
                "Detected modality: %s for %s",
                profile.modality.value, type(model).__name__,
            )
        else:
            profile = self._profile_cache[model_id]

        forward_inputs = inputs if inputs is not None else profile.forward_kwargs

        try:
            # Move inputs to model device
            device = next(model.parameters()).device
            forward_inputs = {
                k: v.to(device) if hasattr(v, 'to') else v
                for k, v in forward_inputs.items()
            }
            return model(**forward_inputs)

        except TypeError as e:
            # Some models don't accept all kwargs — try positional
            logger.warning("Keyword forward failed (%s), trying positional args", e)
            values = list(forward_inputs.values())
            return model(*values)

        except Exception as e:
            logger.error("Forward pass failed: %s", e, exc_info=True)
            raise

    def get_profile(self, model: Any) -> ModalityProfile:
        model_id = id(model)
        if model_id not in self._profile_cache:
            self._profile_cache[model_id] = self.detector.detect(model)
        return self._profile_cache[model_id]


# ── Module-level singleton ───────────────────────────────────────────────────

_universal_forward = UniversalForward()


def forward(model: Any, inputs: Dict = None, hints: Dict = None) -> Any:
    """Drop-in replacement for model(**sample_input) anywhere in codebase."""
    return _universal_forward.forward(model, inputs, hints)


def detect_modality(model: Any, hints: Dict = None) -> ModalityProfile:
    """Detect model modality without running forward pass."""
    return _universal_forward.detector.detect(model, hints or {})
