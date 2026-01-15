#!/usr/bin/env python3
"""
Training Script for Neural Memory Predictor

This script trains a neural network to predict GPU memory usage based on
batch characteristics and model configuration.

Training Time: ~30 minutes on CPU, ~5 minutes on GPU
Expected Improvement: +5-10% better memory utilization

Usage:
    # Option 1: Train on real production traces
    python3 train_memory_predictor.py --traces memory_traces.csv

    # Option 2: Train on synthetic data (for testing)
    python3 train_memory_predictor.py --synthetic --num-traces 5000

    # Option 3: With custom parameters
    python3 train_memory_predictor.py \
        --traces memory_traces.csv \
        --epochs 200 \
        --batch-size 64 \
        --learning-rate 0.001 \
        --device cuda
"""

import argparse
import sys
import os
import matplotlib.pyplot as plt


def train_memory_predictor(
    traces_csv: str,
    save_path: str = "memory_predictor.pth",
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    validation_split: float = 0.2,
    device: str = "cpu",
    plot: bool = True
):
    """
    Train neural memory predictor.

    Args:
        traces_csv: Path to memory traces CSV
        save_path: Where to save trained model
        epochs: Training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        validation_split: Validation fraction
        device: 'cpu' or 'cuda'
        plot: Plot training curves
    """
    from memopt.neural_memory_predictor import NeuralMemoryPredictor

    print("="*70)
    print("NEURAL MEMORY PREDICTOR TRAINING")
    print("="*70)
    print(f"Traces: {traces_csv}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Device: {device}")
    print("="*70)

    # Create predictor
    predictor = NeuralMemoryPredictor(device=device)

    # Train
    print("\nTraining...")
    history = predictor.train_from_csv(
        csv_path=traces_csv,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_split=validation_split,
        verbose=True
    )

    # Save model
    predictor.save(save_path)

    # Evaluate on validation set
    print("\nEvaluating...")
    metrics = predictor.evaluate(traces_csv, verbose=True)

    # Plot training curves
    if plot:
        try:
            plt.figure(figsize=(10, 5))

            plt.subplot(1, 2, 1)
            plt.plot(history['train_loss'], label='Train Loss')
            plt.plot(history['val_loss'], label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('MSE Loss')
            plt.title('Training History')
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(history['val_loss'])
            plt.xlabel('Epoch')
            plt.ylabel('Validation Loss')
            plt.title('Validation Loss')
            plt.grid(True)

            plot_path = save_path.replace('.pth', '_training.png')
            plt.tight_layout()
            plt.savefig(plot_path)
            print(f"\n✓ Training plot saved to {plot_path}")

        except Exception as e:
            print(f"⚠ Could not create plot: {e}")

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Model saved: {save_path}")
    print(f"MAE: {metrics['mae']:.2f} MB")
    print(f"RMSE: {metrics['rmse']:.2f} MB")
    print(f"R² score: {metrics['r2']:.4f}")
    print("\nTo use in production:")
    print(f"  predictor = NeuralMemoryPredictor.load('{save_path}')")
    print("="*70)

    return predictor, metrics


def generate_synthetic_data(
    num_traces: int = 5000,
    output_path: str = "memory_traces_synthetic.csv"
):
    """Generate synthetic training data for testing."""
    from memopt.memory_tracer import create_synthetic_traces
    import pandas as pd
    from dataclasses import asdict

    print(f"\nGenerating {num_traces} synthetic traces...")

    traces = create_synthetic_traces(num_traces=num_traces)

    # Convert to DataFrame and save
    df = pd.DataFrame([asdict(trace) for trace in traces])
    df.to_csv(output_path, index=False)

    print(f"✓ Synthetic traces saved to {output_path}")
    print(f"  Traces: {len(traces)}")
    print(f"  Features: {df.shape[1]}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Train neural memory predictor for LLM inference"
    )

    # Data source
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--traces",
        type=str,
        help="Path to memory traces CSV (from production)"
    )
    data_group.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic training data"
    )

    # Synthetic data options
    parser.add_argument(
        "--num-traces",
        type=int,
        default=5000,
        help="Number of synthetic traces to generate (default: 5000)"
    )

    # Training options
    parser.add_argument(
        "--save-path",
        type=str,
        default="memory_predictor.pth",
        help="Where to save trained model (default: memory_predictor.pth)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs (default: 100)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Validation split fraction (default: 0.2)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to train on (default: cpu)"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting training curves"
    )

    args = parser.parse_args()

    # Generate synthetic data if requested
    if args.synthetic:
        traces_csv = generate_synthetic_data(
            num_traces=args.num_traces,
            output_path="memory_traces_synthetic.csv"
        )
    else:
        traces_csv = args.traces

        # Check if file exists
        if not os.path.exists(traces_csv):
            print(f"Error: Traces file not found: {traces_csv}")
            print("\nOptions:")
            print("1. Collect real traces first using MemoryTracer")
            print("2. Use --synthetic to generate test data")
            sys.exit(1)

    # Train model
    try:
        predictor, metrics = train_memory_predictor(
            traces_csv=traces_csv,
            save_path=args.save_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            validation_split=args.validation_split,
            device=args.device,
            plot=not args.no_plot
        )

        # Success
        sys.exit(0)

    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
