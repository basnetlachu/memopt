"""
PrefetchLoader: drop-in DataLoader wrapper for async GPU prefetch.

Uses a dedicated CUDA stream to transfer the next batch to GPU while the
current batch is being processed on the main stream.  This hides CPU→GPU
PCIe transfer latency (typically 0.5–3 ms per batch) behind compute.

Usage:
    from memopt.training.prefetch_loader import PrefetchLoader

    loader = DataLoader(dataset, batch_size=64)
    fast_loader = PrefetchLoader(loader, device='cuda')

    for batch in fast_loader:
        loss = model(batch)   # next batch already on GPU
        loss.backward()
        optimizer.step()

Notes:
- Batches that are already on the target device are left untouched (no
  redundant copy).
- Handles Tensors, lists/tuples of Tensors, and dicts of Tensors.
- Falls back to plain iteration when CUDA is not available.
"""

from __future__ import annotations

from typing import Any, Iterator

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _move_to(batch: Any, device: Any, non_blocking: bool) -> Any:
    """Recursively move tensors to *device*; leave non-tensors untouched."""
    if not HAS_TORCH:
        return batch

    if isinstance(batch, torch.Tensor):
        # Skip if already on the target device
        if batch.device == (device if isinstance(device, torch.device)
                            else torch.device(device)):
            return batch
        return batch.to(device, non_blocking=non_blocking)

    if isinstance(batch, dict):
        return {k: _move_to(v, device, non_blocking) for k, v in batch.items()}

    if isinstance(batch, (list, tuple)):
        moved = [_move_to(item, device, non_blocking) for item in batch]
        return type(batch)(moved)

    return batch


class PrefetchLoader:
    """
    Drop-in replacement for a PyTorch DataLoader that pre-fetches the next
    batch to the GPU while the current batch is being processed.

    Args:
        loader: Any iterable that yields batches (DataLoader, list, …).
        device: Target device.  Defaults to 'cuda' when available.
    """

    def __init__(self, loader: Any, device: str = 'cuda'):
        self.loader = loader
        self.device = device

        self._use_cuda = (
            HAS_TORCH and
            torch.cuda.is_available() and
            device.startswith('cuda')
        )

        if self._use_cuda:
            self._stream = torch.cuda.Stream(device=device)

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Any]:
        if not self._use_cuda:
            # No CUDA: just yield as-is
            yield from self.loader
            return

        loader_iter = iter(self.loader)

        # Pre-fetch the first batch
        try:
            next_batch = next(loader_iter)
        except StopIteration:
            return

        with torch.cuda.stream(self._stream):
            next_batch = _move_to(next_batch, self.device, non_blocking=True)

        while True:
            # Current batch is ready; start fetching the one after it
            current_batch = next_batch

            try:
                next_batch = next(loader_iter)
                with torch.cuda.stream(self._stream):
                    next_batch = _move_to(next_batch, self.device, non_blocking=True)
            except StopIteration:
                next_batch = None

            # Wait for the current batch's transfer to finish before yielding
            torch.cuda.current_stream().wait_stream(self._stream)
            yield current_batch

            if next_batch is None:
                break

    def __len__(self) -> int:
        return len(self.loader)

    # Pass-through to the underlying loader so callers can access
    # attributes like .dataset, .batch_size, etc.
    def __getattr__(self, name: str) -> Any:
        return getattr(self.loader, name)
