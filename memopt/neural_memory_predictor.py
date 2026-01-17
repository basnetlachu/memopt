"""
Neural Memory Predictor

A lightweight neural network that predicts GPU memory usage more accurately
than static formulas by learning from real production traces.

Architecture:
- Input: [batch_size, avg_seq_len, hidden_size, num_layers, quantize_kv, use_flash]
- Hidden: 2 layers (64 → 32 neurons)
- Output: predicted_memory_mb

Expected Improvement: +5-10% better memory utilization (fewer OOMs)
Training Time: ~30 minutes on CPU, ~5 minutes on GPU
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple, List
import os


class MemoryPredictorNet(nn.Module):
    """
    Neural network for predicting GPU memory usage.

    Input features (6 total):
    - batch_size: Number of sequences
    - avg_seq_len: Average sequence length
    - hidden_size: Model hidden dimension
    - num_layers: Number of transformer layers
    - quantize_kv: KV cache quantization (0 or 1)
    - use_flash_attention: Flash Attention enabled (0 or 1)

    Output:
    - predicted_memory_mb: Estimated GPU memory in MB
    """

    def __init__(
        self,
        input_size: int = 6,
        hidden_size1: int = 64,
        hidden_size2: int = 32,
        dropout: float = 0.1
    ):
        """
        Args:
            input_size: Number of input features
            hidden_size1: First hidden layer size
            hidden_size2: Second hidden layer size
            dropout: Dropout rate for regularization
        """
        super().__init__()

        self.network = nn.Sequential(
            # Input layer
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Hidden layer
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Output layer
            nn.Linear(hidden_size2, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, 6]

        Returns:
            Predicted memory in MB [batch, 1]
        """
        return self.network(x)


class NeuralMemoryPredictor:
    """
    Wrapper for trained neural memory predictor.

    Usage:
        # Training
        predictor = NeuralMemoryPredictor()
        predictor.train(traces_csv='memory_traces.csv', epochs=100)
        predictor.save('memory_predictor.pth')

        # Inference
        predictor = NeuralMemoryPredictor.load('memory_predictor.pth')
        memory_mb = predictor.predict(
            batch_size=8,
            avg_seq_len=512,
            model_config={'hidden_size': 4096, 'num_layers': 32},
            quantize_kv=True,
            use_flash_attention=True
        )
    """

    def __init__(
        self,
        model: Optional[MemoryPredictorNet] = None,
        device: str = "cpu"
    ):
        """
        Args:
            model: Pre-trained model (None to create new)
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model = model or MemoryPredictorNet()
        self.device = torch.device(device)
        self.model.to(self.device)

        # Feature normalization parameters (learned from training data)
        self.feature_mean: Optional[torch.Tensor] = None
        self.feature_std: Optional[torch.Tensor] = None
        self.target_mean: float = 0.0
        self.target_std: float = 1.0

    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize input features using learned statistics."""
        if self.feature_mean is not None and self.feature_std is not None:
            return (features - self.feature_mean) / (self.feature_std + 1e-8)
        return features

    def _denormalize_output(self, output: torch.Tensor) -> torch.Tensor:
        """Denormalize model output to original scale."""
        return output * self.target_std + self.target_mean

    def predict(
        self,
        batch_size: int,
        avg_seq_len: float,
        model_config: Dict,
        quantize_kv: bool = False,
        use_flash_attention: bool = True
    ) -> float:
        """
        Predict memory usage for given configuration.

        Args:
            batch_size: Number of sequences
            avg_seq_len: Average sequence length
            model_config: Model configuration dict with 'hidden_size' and 'num_layers'
            quantize_kv: Whether KV cache is quantized
            use_flash_attention: Whether Flash Attention is used

        Returns:
            Predicted memory in MB
        """
        # Extract model features
        hidden_size = model_config.get('hidden_size', 4096)
        num_layers = model_config.get('num_layers', 32)

        # Prepare input tensor
        features = torch.tensor([
            float(batch_size),
            float(avg_seq_len),
            float(hidden_size),
            float(num_layers),
            1.0 if quantize_kv else 0.0,
            1.0 if use_flash_attention else 0.0
        ], dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, 6]

        # Normalize
        features = self._normalize_features(features)

        # Predict
        self.model.eval()
        with torch.no_grad():
            output = self.model(features)
            memory_mb = self._denormalize_output(output).item()

        # Ensure non-negative
        memory_mb = max(0.0, memory_mb)

        return memory_mb

    def train_from_csv(
        self,
        csv_path: str,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train model from CSV traces.

        Args:
            csv_path: Path to memory traces CSV
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            validation_split: Fraction of data for validation
            verbose: Print training progress

        Returns:
            Training history (train_loss, val_loss per epoch)
        """
        import pandas as pd
        from torch.utils.data import DataLoader, TensorDataset

        # Load data
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Traces file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        if verbose:
            print(f"Loaded {len(df)} traces from {csv_path}")

        # Prepare features and targets
        # Convert columns to numeric, handling booleans properly
        feature_cols = ['batch_size', 'avg_seq_len', 'hidden_size', 'num_layers', 'quantize_kv', 'use_flash_attention']
        feature_df = df[feature_cols].copy()

        # Convert boolean columns to int
        for col in ['quantize_kv', 'use_flash_attention']:
            if col in feature_df.columns:
                feature_df[col] = feature_df[col].astype(float)

        # Convert all to float
        feature_df = feature_df.astype(float)

        features = torch.tensor(feature_df.values, dtype=torch.float32)

        targets = torch.tensor(
            df['gpu_memory_allocated_mb'].values,
            dtype=torch.float32
        ).unsqueeze(1)  # [N, 1]

        # Compute normalization statistics
        self.feature_mean = features.mean(dim=0)
        self.feature_std = features.std(dim=0)
        self.target_mean = targets.mean().item()
        self.target_std = targets.std().item()

        # Normalize
        features = self._normalize_features(features)
        targets_norm = (targets - self.target_mean) / (self.target_std + 1e-8)

        # Split train/val
        n = len(features)
        n_val = int(n * validation_split)
        n_train = n - n_val

        indices = torch.randperm(n)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        train_dataset = TensorDataset(features[train_indices], targets_norm[train_indices])
        val_dataset = TensorDataset(features[val_indices], targets_norm[val_indices])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Training setup
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        history = {'train_loss': [], 'val_loss': []}

        # Training loop
        for epoch in range(epochs):
            # Train
            train_loss = 0.0
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(batch_features)
                loss = criterion(predictions, batch_targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validate
            val_loss = 0.0
            self.model.eval()
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)

                    predictions = self.model(batch_features)
                    loss = criterion(predictions, batch_targets)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            self.model.train()

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if verbose:
            print(f"\nTraining complete!")
            print(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
            print(f"Final Val Loss: {history['val_loss'][-1]:.4f}")

        return history

    def save(self, path: str = "memory_predictor.pth"):
        """Save trained model to disk."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std,
            'target_mean': self.target_mean,
            'target_std': self.target_std
        }, path)

        print(f"✓ Model saved to {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> 'NeuralMemoryPredictor':
        """Load trained model from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        checkpoint = torch.load(path, map_location=device)

        model = MemoryPredictorNet()
        model.load_state_dict(checkpoint['model_state_dict'])

        predictor = cls(model=model, device=device)
        predictor.feature_mean = checkpoint['feature_mean']
        predictor.feature_std = checkpoint['feature_std']
        predictor.target_mean = checkpoint['target_mean']
        predictor.target_std = checkpoint['target_std']

        print(f"✓ Model loaded from {path}")

        return predictor

    def evaluate(
        self,
        csv_path: str,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Args:
            csv_path: Path to test traces CSV
            verbose: Print results

        Returns:
            Evaluation metrics (MSE, MAE, R²)
        """
        import pandas as pd

        df = pd.read_csv(csv_path)

        # Convert columns to numeric, handling booleans properly
        feature_cols = ['batch_size', 'avg_seq_len', 'hidden_size', 'num_layers', 'quantize_kv', 'use_flash_attention']
        feature_df = df[feature_cols].copy()

        # Convert boolean columns to float
        for col in ['quantize_kv', 'use_flash_attention']:
            if col in feature_df.columns:
                feature_df[col] = feature_df[col].astype(float)

        # Convert all to float
        feature_df = feature_df.astype(float)

        features = torch.tensor(feature_df.values, dtype=torch.float32)

        targets = torch.tensor(
            df['gpu_memory_allocated_mb'].astype(float).values,
            dtype=torch.float32
        ).unsqueeze(1)

        # Normalize features
        features = self._normalize_features(features).to(self.device)
        targets = targets.to(self.device)

        # Predict
        self.model.eval()
        with torch.no_grad():
            predictions_norm = self.model(features)
            predictions = self._denormalize_output(predictions_norm)

        # Metrics
        mse = ((predictions - targets) ** 2).mean().item()
        mae = (predictions - targets).abs().mean().item()

        # R² score
        ss_res = ((targets - predictions) ** 2).sum().item()
        ss_tot = ((targets - targets.mean()) ** 2).sum().item()
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        metrics = {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': np.sqrt(mse)
        }

        if verbose:
            print("\n" + "="*70)
            print("EVALUATION RESULTS")
            print("="*70)
            print(f"Test samples: {len(targets)}")
            print(f"MSE: {mse:.2f} MB²")
            print(f"MAE: {mae:.2f} MB")
            print(f"RMSE: {metrics['rmse']:.2f} MB")
            print(f"R² score: {r2:.4f}")
            print("="*70)

        return metrics
