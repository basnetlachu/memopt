"""
Tests for the Universal Multi-Modal Input Handler.

All tests are pure-Python — no GPU required.
torch.nn CPU modules are used for structure detection only.
"""

import pytest
import torch.nn as nn

from memopt.utils.multimodal import (
    MultiModalDetector,
    ModalityType,
    forward,
    detect_modality,
)


# ── Test 1: Text model detected from structure ────────────────────────────────

def test_text_model_detected_from_structure():
    """
    A model with nn.Embedding but no Conv layers must be detected as TEXT.
    """
    model = nn.Sequential(nn.Embedding(32000, 512), nn.Linear(512, 512))
    detector = MultiModalDetector()
    profile = detector.detect(model)

    assert profile.modality == ModalityType.TEXT


# ── Test 2: Vision model detected from structure ──────────────────────────────

def test_vision_model_detected_from_structure():
    """
    A model with Conv2d but no Embedding must be detected as VISION.
    """
    model = nn.Sequential(nn.Conv2d(3, 64, 3), nn.Linear(64, 10))
    detector = MultiModalDetector()
    profile = detector._detect_from_structure(model, {})

    assert profile.modality == ModalityType.VISION


# ── Test 3: Unknown model returns UNKNOWN profile ─────────────────────────────

def test_unknown_model_returns_unknown_profile():
    """
    A model with no recognizable layers (e.g. bare ReLU) must return UNKNOWN
    rather than raising an exception.
    """
    model = nn.ReLU()
    detector = MultiModalDetector()
    profile = detector._detect_from_structure(model, {})

    assert profile.modality == ModalityType.UNKNOWN


# ── Test 4: detect() on embedding model returns TEXT or UNKNOWN ───────────────

def test_universal_forward_runs_text_model():
    """
    detect() on a model with an Embedding layer must return TEXT or UNKNOWN —
    never raise an exception.
    """
    model = nn.Sequential(nn.Embedding(32000, 64), nn.Linear(64, 64))
    profile = MultiModalDetector().detect(model)

    assert profile.modality in [ModalityType.TEXT, ModalityType.UNKNOWN]


# ── Test 5: Module-level forward and detect_modality are callable ─────────────

def test_forward_function_importable():
    """
    The module-level forward() and detect_modality() singletons must be
    importable and callable without a GPU.
    """
    assert callable(forward)
    assert callable(detect_modality)
