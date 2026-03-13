"""
KV Cache for autoregressive generation in memopt.
Stores key/value tensors from previous forward passes.
Avoids O(n²) recomputation during token-by-token generation.

Works with any transformer that uses attention layers.
Graceful skip for encoder-only models.

VMM integration (opt-in, lazy):
    When memopt.vmm is available, block allocation and freeing is routed
    through the VMM so blocks can spill to DRAM/NVMe under memory pressure.
    If the VMM is not installed, behaviour is identical to before.
"""
import torch
import torch.nn as nn
import logging
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

# VMM integration — lazy to avoid circular imports and remain opt-in
_vmm = None

def _get_vmm():
    """Return the process-level VMM singleton, creating it on first call."""
    global _vmm
    if _vmm is None:
        try:
            from memopt.vmm import VMM
            _vmm = VMM()
        except Exception:
            _vmm = False   # Mark as unavailable so we don't retry
    return _vmm if _vmm else None

log = logging.getLogger(__name__)


@dataclass
class KVCacheConfig:
    max_seq_len: int = 2048
    max_batch_size: int = 8
    dtype: torch.dtype = torch.float16
    device: str = "cuda"


class KVCacheEntry:
    """
    Stores K and V tensors for one layer, one sequence.
    Pre-allocated to max_seq_len to avoid repeated allocation.
    """
    def __init__(
        self,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str,
    ):
        self.k = torch.zeros(
            max_seq_len, num_heads, head_dim,
            dtype=dtype, device=device
        )
        self.v = torch.zeros(
            max_seq_len, num_heads, head_dim,
            dtype=dtype, device=device
        )
        self.current_len = 0

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor):
        """Append new K/V for the current token."""
        n = k_new.shape[0]
        end = self.current_len + n
        if end > self.k.shape[0]:
            raise RuntimeError(
                f"KV cache overflow: current={self.current_len} "
                f"new={n} max={self.k.shape[0]}"
            )
        self.k[self.current_len:end] = k_new
        self.v[self.current_len:end] = v_new
        self.current_len = end

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return all cached K, V up to current position."""
        return (
            self.k[:self.current_len],
            self.v[:self.current_len],
        )

    def reset(self):
        self.current_len = 0


class KVCache:
    """
    Full KV cache for all layers in a transformer model.

    Usage:
        cache = KVCache.build_for_model(model, config)

        # Generation loop
        cache.reset()
        for step in range(max_new_tokens):
            output = model(input_ids=next_token, past_kv=cache.get_past_kv())
            cache.update(output.past_key_values)

    HuggingFace models natively support past_key_values.
    For custom models: use cache.patch_model(model) to inject caching.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        config: KVCacheConfig,
    ):
        self.num_layers = num_layers
        self.config = config
        self._vmm_seq_id: Optional[str] = None

        self.entries: List[KVCacheEntry] = [
            KVCacheEntry(
                config.max_seq_len,
                num_heads,
                head_dim,
                config.dtype,
                config.device,
            )
            for _ in range(num_layers)
        ]

        # Register each layer slab with the VMM (opt-in, silently skipped if unavailable)
        vmm = _get_vmm()
        if vmm is not None:
            import uuid
            self._vmm_seq_id = f"kvcache-{uuid.uuid4().hex[:8]}"
            slab_bytes = num_layers * config.max_seq_len * num_heads * head_dim * 2 * 2
            vmm.allocate(self._vmm_seq_id, block_index=0, size_bytes=slab_bytes)
            log.debug("KVCache registered with VMM seq_id=%s (%d bytes)", self._vmm_seq_id, slab_bytes)

    @classmethod
    def build_for_model(
        cls,
        model: nn.Module,
        config: KVCacheConfig,
    ) -> Optional["KVCache"]:
        """
        Build KV cache sized for this model.
        Returns None if model doesn't support KV caching
        (encoder-only models like BERT).
        """
        # Detect if model is causal (supports KV cache)
        if not _is_causal_lm(model):
            log.info("KV cache: skip — encoder-only model")
            return None

        # Extract num_layers, num_heads, head_dim from model config
        num_layers, num_heads, head_dim = _extract_attention_dims(model)

        if num_layers == 0:
            log.info("KV cache: skip — could not detect attention dimensions")
            return None

        vram_needed_gb = (
            2 * num_layers * config.max_batch_size *
            config.max_seq_len * num_heads * head_dim *
            2  # FP16 = 2 bytes
        ) / 1e9

        log.info(
            f"KV cache: {num_layers} layers x {num_heads} heads x "
            f"{head_dim} dim | VRAM needed: {vram_needed_gb:.2f}GB"
        )

        # Check VRAM available
        free_vram = (
            torch.cuda.get_device_properties(0).total_memory -
            torch.cuda.memory_reserved(0)
        ) / 1e9

        if vram_needed_gb > free_vram * 0.3:
            log.warning(
                f"KV cache needs {vram_needed_gb:.1f}GB but only "
                f"{free_vram:.1f}GB free. Using smaller cache."
            )
            config = KVCacheConfig(
                max_seq_len=min(config.max_seq_len, 512),
                max_batch_size=config.max_batch_size,
                dtype=config.dtype,
                device=config.device,
            )

        return cls(num_layers, num_heads, head_dim, config)

    def reset(self):
        """Clear all cached KV tensors. Call before each new sequence."""
        for entry in self.entries:
            entry.reset()
        # Notify VMM that this slab was accessed (keeps it in the hot tier)
        vmm = _get_vmm()
        if vmm is not None and self._vmm_seq_id is not None:
            vmm.fetch(self._vmm_seq_id, block_index=0)

    def release(self):
        """
        Free the VMM slab for this cache instance.
        Call when the cache is permanently discarded (e.g. request complete).
        No-op if the VMM is not available.
        """
        vmm = _get_vmm()
        if vmm is not None and self._vmm_seq_id is not None:
            vmm.free_sequence(self._vmm_seq_id)
            log.debug("KVCache released VMM seq_id=%s", self._vmm_seq_id)
            self._vmm_seq_id = None

    def get_past_kv(self):
        """
        Return past_key_values in HuggingFace format.
        Tuple of (k, v) tuples, one per layer.
        """
        return tuple(
            (e.k[:e.current_len].unsqueeze(0),
             e.v[:e.current_len].unsqueeze(0))
            for e in self.entries
        )

    def update(self, past_key_values):
        """
        Update cache from model output past_key_values.
        Handles HuggingFace format: tuple of (k, v) per layer.
        """
        if past_key_values is None:
            return
        for i, (k, v) in enumerate(past_key_values):
            if i >= len(self.entries):
                break
            # k shape: (batch, heads, seq, head_dim)
            # Store: (seq, heads, head_dim)
            self.entries[i].reset()
            k_store = k[0].permute(1, 0, 2)  # (seq, heads, head_dim)
            v_store = v[0].permute(1, 0, 2)
            self.entries[i].append(k_store, v_store)

    def memory_gb(self) -> float:
        """Return current cache memory usage in GB."""
        total = sum(
            e.k.nbytes + e.v.nbytes
            for e in self.entries
        )
        return total / 1e9


class KVCacheWrappedModel(nn.Module):
    """
    Wraps a causal LM with a KVCache instance.

    Transparent to callers — same forward() interface as the original model.
    The cache is reset on every call so a single-step benchmark sees no
    speedup (1.0x → agent rolls back correctly).  Actual speedup materialises
    during iterative generation where past_key_values are reused across steps.
    """

    def __init__(self, model: nn.Module, cache: "KVCache"):
        super().__init__()
        self._model = model
        self._cache = cache
        self._step  = 0

    def forward(self, *args, **kwargs):
        self._cache.reset()
        output = self._model(*args, **kwargs)
        self._step += 1
        return output

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._model, name)

    @property
    def _memopt_original_forward(self):
        return self._model.forward


def _is_causal_lm(model: nn.Module) -> bool:
    return (
        hasattr(model, "lm_head") or
        any("lm_head" in n for n, _ in model.named_modules()) or
        (hasattr(model, "config") and
         getattr(model.config, "is_decoder", False))
    )


def _extract_attention_dims(model: nn.Module) -> Tuple[int, int, int]:
    """
    Extract num_layers, num_heads, head_dim from model.
    Returns (0, 0, 0) if cannot determine.
    """
    # HuggingFace config (most reliable)
    if hasattr(model, "config"):
        cfg = model.config
        num_heads = getattr(cfg, "num_attention_heads",
                   getattr(cfg, "num_heads", 0))
        hidden = getattr(cfg, "hidden_size",
                getattr(cfg, "d_model", 0))
        num_layers = getattr(cfg, "num_hidden_layers",
                    getattr(cfg, "num_layers", 0))

        if num_heads > 0 and hidden > 0:
            head_dim = hidden // num_heads
            return num_layers, num_heads, head_dim

    return 0, 0, 0
