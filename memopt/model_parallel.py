"""
Stage 6: Tensor Parallelism for Multi-GPU Inference

Splits model weights across multiple GPUs to enable:
1. Models larger than single GPU memory
2. Faster inference through parallel computation

Key techniques:
- Column-wise sharding of linear layers
- NCCL all-reduce for result aggregation
- Overlap communication with computation
"""

import torch
import torch.distributed as dist
from typing import Optional, Tuple
import os


class TensorParallel:
    """
    Tensor parallelism implementation for splitting models across GPUs.

    Example usage:
        tp = TensorParallel(world_size=2, rank=0)
        model = tp.parallelize_model(model)
        output = model(input)  # Automatically uses both GPUs
    """

    def __init__(
        self,
        world_size: int = 1,
        rank: int = 0,
        backend: str = "nccl"
    ):
        """
        Initialize tensor parallelism.

        Args:
            world_size: Total number of GPUs
            rank: Current GPU rank (0 to world_size-1)
            backend: Communication backend ('nccl' for NVIDIA GPUs, 'gloo' for CPU)
        """
        self.world_size = world_size
        self.rank = rank
        self.backend = backend

        # Initialize distributed if not already initialized
        if world_size > 1 and not dist.is_initialized():
            self._initialize_distributed()

        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    def _initialize_distributed(self):
        """Initialize PyTorch distributed for multi-GPU communication."""
        # Setup environment variables if not set
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"

        # Initialize process group
        dist.init_process_group(
            backend=self.backend,
            world_size=self.world_size,
            rank=self.rank
        )

        print(f"[GPU {self.rank}] Initialized distributed training")

    def shard_tensor(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """
        Split tensor along dimension across all GPUs.

        Args:
            tensor: Tensor to shard [... , D, ...]
            dim: Dimension to split along

        Returns:
            Shard for this GPU [... , D/world_size, ...]
        """
        if self.world_size == 1:
            return tensor

        # Calculate shard size
        total_size = tensor.shape[dim]
        shard_size = total_size // self.world_size

        # Get this GPU's shard
        start_idx = self.rank * shard_size
        end_idx = start_idx + shard_size

        # Use torch.narrow to avoid copying
        shard = torch.narrow(tensor, dim, start_idx, shard_size)

        return shard.contiguous()

    def all_reduce(self, tensor: torch.Tensor, async_op: bool = False) -> torch.Tensor:
        """
        Sum tensors from all GPUs and broadcast result.

        Args:
            tensor: Local tensor to reduce
            async_op: If True, return immediately without waiting

        Returns:
            Reduced tensor (same shape as input)
        """
        if self.world_size == 1:
            return tensor

        # All-reduce sums tensors from all GPUs
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=async_op)

        return tensor

    def gather_tensor(self, tensor: torch.Tensor, dim: int = 0) -> Optional[torch.Tensor]:
        """
        Gather tensors from all GPUs to rank 0.

        Args:
            tensor: Local tensor shard
            dim: Dimension to concatenate along

        Returns:
            Full tensor on rank 0, None on other ranks
        """
        if self.world_size == 1:
            return tensor

        # Gather to rank 0
        if self.rank == 0:
            gathered = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        else:
            gathered = None

        dist.gather(tensor, gathered, dst=0)

        # Concatenate on rank 0
        if self.rank == 0:
            return torch.cat(gathered, dim=dim)
        return None

    def parallelize_linear(self, layer: torch.nn.Linear, column_parallel: bool = True) -> torch.nn.Linear:
        """
        Convert linear layer to tensor parallel version.

        Args:
            layer: Original linear layer
            column_parallel: If True, split columns; else split rows

        Returns:
            Sharded linear layer
        """
        if self.world_size == 1:
            return layer

        # Get original weights
        weight = layer.weight.data  # [out_features, in_features]
        bias = layer.bias.data if layer.bias is not None else None

        if column_parallel:
            # Split output dimension (columns)
            # Each GPU gets: [out_features/N, in_features]
            shard_weight = self.shard_tensor(weight, dim=0)
            shard_bias = self.shard_tensor(bias, dim=0) if bias is not None else None

            out_features = shard_weight.shape[0]
            in_features = shard_weight.shape[1]
        else:
            # Split input dimension (rows)
            # Each GPU gets: [out_features, in_features/N]
            shard_weight = self.shard_tensor(weight, dim=1)
            shard_bias = bias  # Bias not sharded for row-parallel

            out_features = shard_weight.shape[0]
            in_features = shard_weight.shape[1]

        # Create new layer with sharded weights
        new_layer = torch.nn.Linear(
            in_features,
            out_features,
            bias=(bias is not None)
        ).to(self.device)

        new_layer.weight.data = shard_weight.to(self.device)
        if shard_bias is not None:
            new_layer.bias.data = shard_bias.to(self.device)

        # Mark layer for all-reduce after forward pass
        new_layer.column_parallel = column_parallel
        new_layer.needs_all_reduce = column_parallel

        return new_layer

    def parallelize_embedding(self, layer: torch.nn.Embedding) -> torch.nn.Embedding:
        """
        Convert embedding layer to tensor parallel version.

        Args:
            layer: Original embedding layer

        Returns:
            Sharded embedding layer
        """
        if self.world_size == 1:
            return layer

        # Split embedding dimension (vocab size)
        weight = layer.weight.data  # [vocab_size, embedding_dim]
        shard_weight = self.shard_tensor(weight, dim=0)

        vocab_size_shard = shard_weight.shape[0]
        embedding_dim = shard_weight.shape[1]

        # Create new embedding with sharded vocab
        new_layer = torch.nn.Embedding(
            vocab_size_shard,
            embedding_dim,
            padding_idx=layer.padding_idx
        ).to(self.device)

        new_layer.weight.data = shard_weight.to(self.device)
        new_layer.needs_all_reduce = True

        return new_layer

    def forward_hook_all_reduce(self, module, input, output):
        """
        Hook to perform all-reduce after forward pass.

        Automatically aggregates results from all GPUs.
        """
        if hasattr(module, 'needs_all_reduce') and module.needs_all_reduce:
            return self.all_reduce(output)
        return output

    def parallelize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Automatically parallelize all linear and embedding layers in model.

        Args:
            model: Original model

        Returns:
            Parallelized model
        """
        if self.world_size == 1:
            return model.to(self.device)

        print(f"[GPU {self.rank}] Parallelizing model across {self.world_size} GPUs...")

        # Recursively parallelize all layers
        for name, module in model.named_children():
            if isinstance(module, torch.nn.Linear):
                # Parallelize linear layers (column-parallel by default)
                setattr(model, name, self.parallelize_linear(module, column_parallel=True))
            elif isinstance(module, torch.nn.Embedding):
                # Parallelize embeddings
                setattr(model, name, self.parallelize_embedding(module))
            elif len(list(module.children())) > 0:
                # Recursively parallelize submodules
                setattr(model, name, self.parallelize_model(module))

        # Register forward hooks for all-reduce
        for module in model.modules():
            if hasattr(module, 'needs_all_reduce'):
                module.register_forward_hook(self.forward_hook_all_reduce)

        # Move model to this GPU
        model = model.to(self.device)

        print(f"[GPU {self.rank}] Model parallelization complete")

        return model

    def get_memory_stats(self) -> dict:
        """Get memory usage statistics for this GPU."""
        if not torch.cuda.is_available():
            return {}

        return {
            'rank': self.rank,
            'allocated_gb': torch.cuda.memory_allocated(self.device) / (1024**3),
            'reserved_gb': torch.cuda.memory_reserved(self.device) / (1024**3),
            'max_allocated_gb': torch.cuda.max_memory_allocated(self.device) / (1024**3)
        }

    def synchronize(self):
        """Wait for all GPUs to reach this point."""
        if self.world_size > 1:
            dist.barrier()

    def cleanup(self):
        """Cleanup distributed training."""
        if self.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
            print(f"[GPU {self.rank}] Cleanup complete")


def init_model_parallel(world_size: int = None, rank: int = None) -> TensorParallel:
    """
    Initialize model parallelism.

    Args:
        world_size: Number of GPUs (auto-detect if None)
        rank: Current GPU rank (auto-detect from environment if None)

    Returns:
        TensorParallel instance
    """
    # Auto-detect world size
    if world_size is None:
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Auto-detect rank from environment
    if rank is None:
        rank = int(os.environ.get("LOCAL_RANK", 0))

    return TensorParallel(world_size=world_size, rank=rank)
