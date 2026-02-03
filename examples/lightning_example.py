#!/usr/bin/env python3
"""
Example: Automatic optimization with PyTorch Lightning.

This demonstrates how memopt integrates with PyTorch Lightning
using its callback system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import Callback
except ImportError:
    print("PyTorch Lightning not installed. Run: pip install pytorch-lightning")
    exit(1)

from torch.utils.data import DataLoader, TensorDataset

from memopt.training import TrainingOptimizer, TrainingConfig


class SimpleModel(pl.LightningModule):
    """Simple PyTorch Lightning model."""

    def __init__(self, input_size=512, hidden_size=1024, output_size=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class MemoptCallback(Callback):
    """
    PyTorch Lightning callback for memopt integration.

    This callback:
    1. Attaches memopt at training start
    2. Monitors training progress
    3. Applies optimizations between epochs
    4. Detaches at training end
    """

    def __init__(self, config: TrainingConfig = None, sample_input: torch.Tensor = None):
        super().__init__()
        self.config = config or TrainingConfig(verbose=True)
        self.sample_input = sample_input
        self.optimizer = None

    def on_fit_start(self, trainer: Trainer, pl_module: pl.LightningModule):
        """Attach memopt optimizer at training start."""
        self.optimizer = TrainingOptimizer(self.config)
        self.optimizer.attach(pl_module, self.sample_input)
        print("MemOpt: Attached to Lightning module")

    def on_fit_end(self, trainer: Trainer, pl_module: pl.LightningModule):
        """Detach memopt at training end."""
        if self.optimizer:
            self.optimizer.detach()
            print("MemOpt: Detached from Lightning module")
            print(f"MemOpt Stats: {self.optimizer.get_stats()}")

    def on_train_epoch_start(self, trainer: Trainer, pl_module: pl.LightningModule):
        """Notify epoch start."""
        if self.optimizer and self.optimizer._hook:
            self.optimizer._hook.notify_epoch_start(trainer.current_epoch)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: pl.LightningModule):
        """Notify epoch end - triggers optimization."""
        if self.optimizer and self.optimizer._hook:
            # Get validation loss if available
            val_loss = trainer.callback_metrics.get("val_loss")
            val_loss_val = val_loss.item() if val_loss is not None else None
            self.optimizer._hook.notify_epoch_end(trainer.current_epoch, val_loss_val)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Track loss for convergence monitoring."""
        if self.optimizer and self.optimizer._hook:
            loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
            if loss is not None:
                self.optimizer._hook.notify_loss(loss)


def create_dataloaders(num_samples=1000, input_size=512, output_size=10, batch_size=32):
    """Create train and val dataloaders."""
    # Training data
    X_train = torch.randn(num_samples, input_size)
    y_train = torch.randint(0, output_size, (num_samples,))
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Validation data
    X_val = torch.randn(num_samples // 5, input_size)
    y_val = torch.randint(0, output_size, (num_samples // 5,))
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader


def main():
    print("=" * 60)
    print("MEMOPT + PyTorch Lightning Example")
    print("=" * 60)

    # Create model and data
    model = SimpleModel()
    train_loader, val_loader = create_dataloaders()

    # Create sample input for profiling
    sample_input = torch.randn(32, 512)
    if torch.cuda.is_available():
        sample_input = sample_input.cuda()

    # Create memopt callback
    config = TrainingConfig(
        profile_batches=20,
        optimize_after_epoch=1,
        verbose=True,
    )
    memopt_callback = MemoptCallback(config=config, sample_input=sample_input)

    # Create trainer with memopt callback
    trainer = Trainer(
        max_epochs=5,
        accelerator="auto",
        callbacks=[memopt_callback],
        enable_progress_bar=True,
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
