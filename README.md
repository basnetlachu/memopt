# Memopt: GPU Memory Optimization Platform

**Professional GPU memory optimization for AI workloads.**

## The Problem

AI workloads waste **60-80% of GPU time** on memory stalls. This costs companies hundreds of thousands annually in underutilized hardware.

## The Solution

Memopt profiles memory bandwidth usage in AI workloads and applies hardware-validated optimizations.

## Features

| Feature | Description |
|---------|-------------|
| **Profiler** | 3-step pipeline: profile → attribute → optimize |
| **Daemon** | Background GPU monitoring service |
| **Training** | Zero-code training optimization |
| **Dashboard** | Real-time fleet monitoring |

## Quick Start

### Installation

```bash
pip install memopt
```

### Basic Usage

```python
from memopt.profiler import api

model = YourModel().cuda()
sample = torch.randn(8, 512, 1024).cuda()

# Optimize
optimized_model, session = api.optimize(model, sample)
print(f"Speedup: {session.total_speedup:.2f}x")
```

### Daemon Mode

```bash
# Start background monitoring
memopt daemon start

# Check status
memopt daemon status

# Stop daemon
memopt daemon stop
```

### Training Optimization

```python
from memopt import optimize_training

@optimize_training(check_interval=100)
def train(model, dataloader, optimizer):
    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
```

### Dashboard

```bash
cd dashboard
docker-compose up -d
# Access at http://localhost:3000
```

## Documentation

See [MEMOPT_EXPLAINED.md](MEMOPT_EXPLAINED.md) for complete documentation.

## Project Structure

```
memopt/
├── memopt/                # Main package
│   ├── profiler/          # Core optimization
│   ├── daemon/            # Background service
│   ├── training/          # Training wrappers
│   ├── measurement/       # Bandwidth tracking
│   └── validation/        # Validation tools
├── dashboard/             # Web monitoring
├── tests/                 # Test suite
├── examples/              # Usage examples
└── config/                # Configuration
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+

## License

Proprietary - Contact for licensing.
