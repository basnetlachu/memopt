"""
Basic Bandwidth Profiling Example

Demonstrates how to use the bandwidth profiler to analyze a simple model.
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memopt.bandwidth_analyzer import BandwidthAnalyzer, profile_model
from memopt.bottleneck_detector import BottleneckDetector


class SimpleTransformer(nn.Module):
    """Simple transformer model for demonstration."""

    def __init__(self, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(10000, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 10000)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x)
        return x


def main():
    """Run bandwidth profiling on a simple model."""

    print("\n" + "="*70)
    print("MEMOPT - GPU Memory Bandwidth Profiling")
    print("="*70)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\n⚠️  CUDA not available. Running on CPU (profiling will be limited).")
        device = "cpu"
    else:
        device = "cuda"
        print(f"\n✅ Using GPU: {torch.cuda.get_device_name(0)}")

    # Create model
    print("\n📦 Creating model...")
    model = SimpleTransformer(d_model=512, nhead=8, num_layers=6)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model: SimpleTransformer")
    print(f"   Parameters: {total_params:,}")

    # Create input generator
    def input_generator():
        return torch.randint(0, 10000, (8, 128), device=device)  # batch_size=8, seq_len=128

    # Profile the model
    print("\n🔍 Profiling model...")
    analyzer = BandwidthAnalyzer(model=model, device=device)

    stats = analyzer.profile_inference(
        input_generator=input_generator,
        num_iterations=20,
        warmup_iterations=5,
        optimization_name="baseline",
    )

    # Print bandwidth stats
    print("\n📊 BANDWIDTH PROFILING RESULTS:")
    analyzer.profiler.print_stats()

    # Analyze bottlenecks
    print("\n🔍 Analyzing bottlenecks...")
    detector = BottleneckDetector()
    bottlenecks = detector.detect_bottlenecks(stats)
    detector.print_bottlenecks(bottlenecks)

    # Generate optimization plan
    if bottlenecks:
        print("\n💡 OPTIMIZATION PLAN:")
        plan = detector.generate_optimization_plan(bottlenecks, max_optimizations=3)
        for i, (name, description, speedup) in enumerate(plan, 1):
            print(f"   {i}. {description}")
            if speedup > 1.1:
                print(f"      Estimated speedup: {speedup:.2f}x")

    # Analysis
    print("\n📈 ANALYSIS:")
    analyzer.print_analysis(stats)

    # Save report
    output_file = "bandwidth_report.json"
    analyzer.generate_report(output_file)

    print("\n✅ Profiling complete!")
    print(f"   Results saved to: {output_file}")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
