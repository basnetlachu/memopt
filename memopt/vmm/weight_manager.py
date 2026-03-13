"""
Weight manager — streams transformer layer weights through the VMM.

For models too large to fit in HBM, this manager:
  1. Registers each transformer layer as a VMM block at load time.
  2. Keeps only the current + next N layers in HBM.
  3. Uses the prefetch engine to pre-load the next layer while the
     current layer is executing — hiding NVMe→HBM transfer latency.
"""
from __future__ import annotations
import contextlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_PREFETCH_LOOKAHEAD = 2   # layers to keep pre-loaded beyond the current one


class WeightManager:

    def __init__(self, vmm, model=None):
        self.vmm   = vmm
        self.model = model
        self._num_layers: int = 0
        self._layer_block_size: dict[int, int] = {}
        self._registered = False

    def prepare(self, num_layers: int, layer_size_bytes: Optional[int] = None):
        """Register all transformer layers with the VMM."""
        self._num_layers = num_layers

        if layer_size_bytes is None:
            layer_size_bytes = self._estimate_layer_size(num_layers)

        logger.info(
            "WeightManager: registering %d layers × %.2f GB each",
            num_layers, layer_size_bytes / 1e9,
        )

        for layer_idx in range(num_layers):
            try:
                self.vmm.allocate(
                    sequence_id=f"weights_layer_{layer_idx}",
                    block_index=0,
                    size_bytes=layer_size_bytes,
                )
                self._layer_block_size[layer_idx] = layer_size_bytes
            except Exception as e:
                logger.warning("Layer %d allocation failed: %s", layer_idx, e)

        self._registered = True
        logger.info("WeightManager: all layers registered in VMM")

    @contextlib.contextmanager
    def layer_context(self, layer_idx: int):
        """
        Guarantees layer weights are in HBM before yielding.
        Records the access so the prefetch engine learns execution order.
        """
        if not self._registered:
            yield
            return

        seq_id = f"weights_layer_{layer_idx}"

        try:
            self.vmm.fetch(seq_id, block_index=0)
        except KeyError:
            logger.warning("Layer %d not in VMM — running without promotion", layer_idx)
            yield
            return

        # Hint the next N layers to the prefetch engine
        for ahead in range(1, _PREFETCH_LOOKAHEAD + 1):
            next_idx = layer_idx + ahead
            if next_idx < self._num_layers:
                self.vmm.prefetch.record_access(f"weights_layer_{next_idx}", block_index=0)

        yield

        self.vmm.prefetch.record_access(seq_id, block_index=0)

    def evict_layer(self, layer_idx: int):
        """Manually evict a layer from HBM to free space."""
        try:
            self.vmm.free_sequence(f"weights_layer_{layer_idx}")
        except Exception as e:
            logger.debug("Evict layer %d: %s", layer_idx, e)

    def stats(self) -> dict:
        return {
            "num_layers":         self._num_layers,
            "registered":         self._registered,
            "prefetch_lookahead": _PREFETCH_LOOKAHEAD,
            "vmm_stats":          self.vmm.stats(),
        }

    def _estimate_layer_size(self, num_layers: int) -> int:
        if self.model is None:
            default = 1 * 1024 ** 3   # 1 GB fallback
            logger.info("No model provided — assuming 1 GB per layer")
            return default
        try:
            import torch  # noqa: F401
            total_bytes = sum(
                p.numel() * p.element_size()
                for p in self.model.parameters()
            )
            return total_bytes // num_layers
        except Exception:
            return 1 * 1024 ** 3
