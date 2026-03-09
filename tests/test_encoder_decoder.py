"""
Tests for encoder-decoder profiling in input_handler.py.

Inline model classes only — no hardcoded external model names.
Covers is_encoder_decoder(), synthesize_decoder_input(), and
the detect_input_format() encoder-decoder strategy.
"""
import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

from memopt.utils.input_handler import (
    is_encoder_decoder,
    synthesize_decoder_input,
    detect_input_format,
    InputFormat,
)


# ── Inline model definitions ─────────────────────────────────────────────────

class TinyEncoder(nn.Module):
    def __init__(self, vocab_size: int = 100, d_model: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.fc(self.embed(input_ids))


class TinyDecoder(nn.Module):
    def __init__(self, vocab_size: int = 100, d_model: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        x = self.embed(decoder_input_ids)
        # pool over sequence dim to match batch shape
        enc = encoder_hidden_states.mean(dim=1, keepdim=True)
        return self.fc(x + enc)


class TinySeq2Seq(nn.Module):
    """Minimal encoder-decoder with 'encoder' and 'decoder' named children."""

    def __init__(self, vocab_size: int = 100, d_model: int = 32):
        super().__init__()
        self.encoder = TinyEncoder(vocab_size, d_model)
        self.decoder = TinyDecoder(vocab_size, d_model)
        # Simulate a HuggingFace-style config
        self.config = SimpleNamespace(
            is_encoder_decoder=True,
            decoder_start_token_id=1,
            bos_token_id=1,
            pad_token_id=0,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        enc_out = self.encoder(input_ids)
        return self.decoder(decoder_input_ids, enc_out)


class TinyDecoderOnly(nn.Module):
    """Decoder-only (GPT-style) — should NOT be flagged as encoder-decoder."""

    def __init__(self, vocab_size: int = 100, d_model: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.fc(self.embed(input_ids))


class StructuralSeq2Seq(nn.Module):
    """Has 'encoder'/'decoder' children but no config attribute."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(16, 16)
        self.decoder = nn.Linear(16, 16)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x) + y)


# ── is_encoder_decoder ────────────────────────────────────────────────────────

def test_is_encoder_decoder_with_config_flag():
    """HuggingFace-style config.is_encoder_decoder=True → True."""
    m = TinySeq2Seq()
    assert is_encoder_decoder(m)


def test_is_encoder_decoder_with_children():
    """Model with 'encoder' and 'decoder' named children → True."""
    m = StructuralSeq2Seq()
    assert is_encoder_decoder(m)


def test_is_encoder_decoder_decoder_only_false():
    """Decoder-only model → False."""
    m = TinyDecoderOnly()
    assert not is_encoder_decoder(m)


def test_is_encoder_decoder_plain_linear_false():
    """A plain Linear is not encoder-decoder."""
    m = nn.Linear(16, 16)
    assert not is_encoder_decoder(m)


def test_is_encoder_decoder_only_one_child_false():
    """Only encoder child (no decoder) → False."""

    class EncoderOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(8, 8)

        def forward(self, x):
            return self.encoder(x)

    assert not is_encoder_decoder(EncoderOnly())


# ── synthesize_decoder_input ──────────────────────────────────────────────────

def test_synthesize_adds_decoder_input_ids():
    """decoder_input_ids must be present in the augmented dict."""
    m = TinySeq2Seq()
    enc_inp = {'input_ids': torch.randint(0, 100, (2, 10))}
    aug = synthesize_decoder_input(m, enc_inp, device='cpu')
    assert 'decoder_input_ids' in aug
    assert 'input_ids' in aug  # original keys preserved


def test_synthesize_batch_size_matches():
    """decoder_input_ids batch dim matches the encoder input."""
    m = TinySeq2Seq()
    B = 4
    enc_inp = {'input_ids': torch.randint(0, 100, (B, 8))}
    aug = synthesize_decoder_input(m, enc_inp, device='cpu')
    assert aug['decoder_input_ids'].shape[0] == B


def test_synthesize_sequence_length_one():
    """Synthesized decoder_input_ids has seq_len=1."""
    m = TinySeq2Seq()
    enc_inp = {'input_ids': torch.randint(0, 100, (2, 8))}
    aug = synthesize_decoder_input(m, enc_inp, device='cpu')
    assert aug['decoder_input_ids'].shape == (2, 1)


def test_synthesize_uses_config_start_token():
    """Start token comes from config.decoder_start_token_id."""
    m = TinySeq2Seq()
    m.config.decoder_start_token_id = 42
    enc_inp = {'input_ids': torch.randint(0, 100, (1, 6))}
    aug = synthesize_decoder_input(m, enc_inp, device='cpu')
    assert aug['decoder_input_ids'].item() == 42


def test_synthesize_fallback_token_zero():
    """When config has no token IDs, fallback token is 0."""
    class NoConfig(nn.Module):
        def forward(self, x):
            return x

    m = NoConfig()
    enc_inp = {'input_ids': torch.randint(0, 10, (1, 4))}
    aug = synthesize_decoder_input(m, enc_inp, device='cpu')
    assert aug['decoder_input_ids'].item() == 0


def test_synthesize_does_not_mutate_input():
    """Original encoder_input dict must not be modified."""
    m = TinySeq2Seq()
    enc_inp = {'input_ids': torch.randint(0, 100, (2, 8))}
    original_keys = set(enc_inp.keys())
    synthesize_decoder_input(m, enc_inp, device='cpu')
    assert set(enc_inp.keys()) == original_keys


# ── detect_input_format encoder-decoder integration ──────────────────────────

def test_detect_format_seq2seq_with_encoder_input_only():
    """
    detect_input_format must succeed for a seq2seq model when only
    encoder-side inputs are provided (decoder_input_ids synthesized).
    """
    m = TinySeq2Seq()
    m.eval()
    # Only encoder input — no decoder_input_ids
    sample = {'input_ids': torch.randint(0, 100, (1, 10))}
    fmt = detect_input_format(m, sample, device='cpu')
    assert isinstance(fmt, InputFormat)
    assert fmt.style == "kwargs"
    assert 'decoder_input_ids' in fmt.inputs


def test_detect_format_seq2seq_with_full_input():
    """
    When both input_ids and decoder_input_ids are provided, detect_input_format
    should succeed using Strategy 1 (no synthesis needed).
    """
    m = TinySeq2Seq()
    m.eval()
    sample = {
        'input_ids': torch.randint(0, 100, (1, 10)),
        'decoder_input_ids': torch.randint(0, 100, (1, 1)),
    }
    fmt = detect_input_format(m, sample, device='cpu')
    assert isinstance(fmt, InputFormat)
    assert fmt.style == "kwargs"
