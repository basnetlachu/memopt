#!/usr/bin/env python3
"""
Example: Automatic optimization with pure PyTorch training loop.

This demonstrates how to use memopt to automatically optimize
a training loop with zero code changes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from memopt import optimize_training, auto_optimize
from memopt.training import TrainingConfig


# Simple model for demonstration
class SimpleModel(nn.Module):
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


def create_dummy_data(num_samples=1000, input_size=512, output_size=10):
    """Create dummy training data."""
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, output_size, (num_samples,))
    return TensorDataset(X, y)


# =============================================================================
# Method 1: Decorator style
# =============================================================================

@optimize_training(verbose=True)
def train_with_decorator():
    """Training with automatic optimization via decorator."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SimpleModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    dataset = create_dummy_data()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    num_epochs = 5

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return model


# =============================================================================
# Method 2: Context manager style
# =============================================================================

def train_with_context_manager():
    """Training with automatic optimization via context manager."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SimpleModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    dataset = create_dummy_data()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    sample_input = torch.randn(32, 512).to(device)
    num_epochs = 5

    # Use context manager for automatic optimization
    with auto_optimize(model=model, sample_input=sample_input, verbose=True):
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0

            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return model


# =============================================================================
# Method 3: Manual control with TrainingOptimizer
# =============================================================================

def train_with_manual_control():
    """Training with manual optimization control."""
    from memopt.training import TrainingOptimizer, TrainingConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SimpleModel().to(device)
    optimizer_torch = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    dataset = create_dummy_data()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    sample_input = torch.randn(32, 512).to(device)
    num_epochs = 5

    # Create optimizer with custom config
    config = TrainingConfig(
        profile_batches=20,
        optimize_after_epoch=1,
        min_improvement=0.02,
        verbose=True,
    )

    memopt_optimizer = TrainingOptimizer(config)
    memopt_optimizer.attach(model, sample_input)

    try:
        for epoch in range(num_epochs):
            # Notify epoch start (for hook-based tracking)
            if memopt_optimizer._hook:
                memopt_optimizer._hook.notify_epoch_start(epoch)

            model.train()
            total_loss = 0

            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)

                optimizer_torch.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer_torch.step()

                total_loss += loss.item()

                # Notify loss (for convergence monitoring)
                if memopt_optimizer._hook:
                    memopt_optimizer._hook.notify_loss(loss)

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

            # Notify epoch end (triggers optimization if configured)
            if memopt_optimizer._hook:
                memopt_optimizer._hook.notify_epoch_end(epoch, avg_loss)

            # Check for divergence and rollback if needed
            memopt_optimizer.check_and_rollback()

        # Print stats
        print("\nOptimization stats:")
        print(memopt_optimizer.get_stats())

    finally:
        memopt_optimizer.detach()

    return model


if __name__ == "__main__":
    print("=" * 60)
    print("MEMOPT Training Optimization Example")
    print("=" * 60)

    print("\n1. Training with decorator...")
    print("-" * 40)
    train_with_decorator()

    print("\n2. Training with context manager...")
    print("-" * 40)
    train_with_context_manager()

    print("\n3. Training with manual control...")
    print("-" * 40)
    train_with_manual_control()

    print("\nDone!")
