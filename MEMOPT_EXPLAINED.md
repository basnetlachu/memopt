# memopt - Complete Guide

## What is memopt?

**memopt** is a GPU memory optimization platform for PyTorch models. It automatically finds and fixes memory bottlenecks in your neural networks to make them run faster.

### The Problem It Solves

When you run a neural network on a GPU, there are two types of operations:
1. **Compute operations** - Math like matrix multiplication
2. **Memory operations** - Moving data between GPU memory and compute units

Modern GPUs are so fast at math that they often wait for data to arrive from memory. This is called being **"memory-bound"**. memopt identifies where your model is memory-bound and applies optimizations to reduce memory traffic.

### Simple Analogy

Think of a GPU like a super-fast chef (compute) with a slow delivery person (memory). The chef can cook instantly, but has to wait for ingredients to arrive. memopt is like hiring a better delivery service - it:
- Pre-fetches ingredients before needed
- Packs multiple deliveries together
- Keeps frequently-used ingredients on the counter (cache)

---

## Features Overview

memopt provides four main components:

| Component | Purpose |
|-----------|---------|
| **Profiler** | Core optimization pipeline (profile → attribute → optimize) |
| **Daemon** | Background GPU monitoring service |
| **Training Wrapper** | Automatic training optimization with zero code changes |
| **Dashboard** | Real-time fleet monitoring web interface |

---

## Project Structure

```
memopt/
├── memopt/                     # Main package
│   ├── __init__.py            # Package entry point
│   ├── cli.py                 # CLI commands
│   ├── profiler/              # Core optimization logic
│   │   ├── continuous_profiler.py   # GPU profiling
│   │   ├── traffic_attribution.py   # Bottleneck analysis
│   │   ├── adaptive_optimizer.py    # Apply optimizations
│   │   ├── config.py                # Configuration
│   │   ├── gpu_profiles.py          # GPU-specific settings
│   │   ├── hardware_metrics.py      # Hardware counters
│   │   ├── session_persistence.py   # Save/load sessions
│   │   ├── multi_gpu.py             # Multi-GPU support
│   │   └── api.py                   # Public API
│   ├── daemon/                # Background monitoring
│   │   ├── daemon_service.py        # Main daemon
│   │   ├── process_monitor.py       # GPU process detection
│   │   ├── scheduler.py             # Safe optimization scheduler
│   │   └── reporter.py              # Dashboard integration
│   ├── training/              # Training optimization
│   │   ├── wrapper.py               # Decorator & context manager
│   │   ├── hooks.py                 # Framework hooks
│   │   ├── gradient_validator.py    # Gradient validation
│   │   └── convergence_monitor.py   # Training monitoring
│   ├── measurement/           # Bandwidth tracking
│   └── validation/            # Validation tools
├── dashboard/                 # Web monitoring dashboard
│   ├── backend/               # FastAPI backend
│   └── frontend/              # React frontend
├── tests/                     # Test suite
├── examples/                  # Usage examples
├── config/                    # Configuration files
└── docker/                    # Docker deployment
```

---

## 1. Core Profiler - The 3-Step Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   1. PROFILE    │ ──► │  2. ATTRIBUTE   │ ──► │   3. OPTIMIZE   │
│                 │     │                 │     │                 │
│ Measure memory  │     │ Find which ops  │     │ Apply fixes and │
│ traffic & time  │     │ cause problems  │     │ verify speedup  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Step 1: Profile
- Runs your model and measures GPU time
- Tracks memory allocations and access patterns
- Identifies if model is memory-bound or compute-bound

### Step 2: Attribute
- Analyzes which layers/operations cause memory bottlenecks
- Looks for patterns like:
  - Cache thrashing (data too big for L2 cache)
  - Redundant memory fetches (same data loaded multiple times)
  - Poor memory layout (non-contiguous tensors)

### Step 3: Optimize
- Applies potential fixes one by one
- Measures if each fix actually helps
- Keeps good optimizations, rolls back bad ones
- Uses statistical validation (not just one measurement)

### Basic Usage

```python
import torch
from memopt.profiler import api

# Your model
model = MyModel().cuda()
sample = torch.randn(8, 512, 1024).cuda()

# Optimize it
optimized_model, session = api.optimize(
    model=model,
    sample_input=sample,
    verbose=True
)

print(f"Speedup: {session.total_speedup:.2f}x")
```

---

## 2. Daemon Mode - Background GPU Monitoring

The daemon runs as a background service that continuously monitors GPU workloads and applies safe optimizations.

### Features
- **Process Detection**: Monitors GPU processes using NVML
- **Non-invasive Profiling**: Collects GPU stats without injecting into processes
- **Safe Optimization Scheduling**: Only optimizes when GPU utilization allows
- **Dashboard Integration**: Reports metrics to centralized dashboard

### CLI Commands

```bash
# Start daemon
memopt daemon start

# Start in foreground (for debugging)
memopt daemon start --foreground

# Check status
memopt daemon status

# Stop daemon
memopt daemon stop

# View logs
memopt daemon logs
memopt daemon logs --follow  # Stream logs
```

### Configuration

**~/.memopt/daemon_config.yaml:**
```yaml
poll_interval: 5.0          # Seconds between checks
stability_threshold: 3      # Checks before profiling
auto_optimize: false        # Auto-apply optimizations
log_level: INFO
dashboard_url: http://dashboard:8000  # Dashboard API
dashboard_report_interval: 10.0
```

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MemoptDaemon                             │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ProcessMonitor│  │  Scheduler   │  │DashboardReporter │  │
│  │              │  │              │  │                  │  │
│  │ - GPU states │  │ - Safe times │  │ - HTTP client    │  │
│  │ - Processes  │  │ - Throttling │  │ - Heartbeats     │  │
│  │ - Stability  │  │              │  │ - Alerts         │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Training Wrapper - Zero-Code Training Optimization

Automatic training optimization with gradient validation and convergence monitoring.

### Decorator Usage

```python
from memopt import optimize_training

@optimize_training(
    check_interval=100,      # Steps between checks
    rollback_on_divergence=True,
    min_improvement=0.05,    # 5% minimum speedup
)
def train(model, dataloader, optimizer):
    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
```

### Context Manager Usage

```python
from memopt import auto_optimize

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters())

with auto_optimize(model, check_interval=100) as ctx:
    for epoch in range(10):
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            optimizer.step()

            # Report loss for convergence monitoring
            ctx.report_loss(loss.item())
```

### Framework Hooks

Automatic integration with popular frameworks:

```python
from memopt.training.hooks import (
    PyTorchHook,
    LightningHook,
    HuggingFaceHook,
    AccelerateHook,
)

# PyTorch Lightning
hook = LightningHook(check_interval=100)
trainer = pl.Trainer(callbacks=[hook.as_callback()])

# HuggingFace Trainer
hook = HuggingFaceHook(check_interval=100)
trainer = Trainer(callbacks=[hook.as_callback()])

# Accelerate
hook = AccelerateHook(accelerator, check_interval=100)
```

### Features

| Feature | Description |
|---------|-------------|
| **Gradient Validation** | Ensures gradients flow correctly after optimization |
| **Convergence Monitoring** | Detects divergence and automatically rolls back |
| **Automatic Rollback** | Reverts bad optimizations to preserve training |
| **Framework Support** | PyTorch, Lightning, HuggingFace, Accelerate |

---

## 4. Dashboard - Fleet Monitoring

Real-time GPU optimization monitoring across your entire fleet.

### Quick Start

```bash
cd dashboard
docker-compose up -d
```

Access at http://localhost:3000

### Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   GPU Server 1  │     │   GPU Server 2  │     │   GPU Server N  │
│  memopt daemon  │     │  memopt daemon  │     │  memopt daemon  │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │    Dashboard Backend    │
                    │       (FastAPI)         │
                    │    - REST API           │
                    │    - WebSocket          │
                    │    - SQLite DB          │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Dashboard Frontend    │
                    │       (React)           │
                    │    - Fleet Overview     │
                    │    - Server Details     │
                    │    - Training Runs      │
                    │    - Session History    │
                    │    - Alerts             │
                    └─────────────────────────┘
```

### Connecting Daemons

Configure each GPU server's daemon to report to the dashboard:

```yaml
# ~/.memopt/daemon_config.yaml
dashboard_url: http://dashboard-server:8000
dashboard_report_interval: 10.0
```

Or via environment variable:
```bash
export MEMOPT_DASHBOARD_URL=http://dashboard-server:8000
memopt daemon start
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/fleet/overview` | GET | Aggregate fleet statistics |
| `/api/servers` | GET/POST | List/register servers |
| `/api/servers/{hostname}/heartbeat` | POST | Update server metrics |
| `/api/sessions` | GET/POST | List/create sessions |
| `/api/training` | GET/POST | Training runs |
| `/api/alerts` | GET/POST | Alerts |
| `/ws` | WebSocket | Real-time updates |

### Frontend Pages

1. **Fleet Overview** - Dashboard home with aggregate stats
2. **Servers** - Detailed view of each GPU server
3. **Training Runs** - Active and recent training progress
4. **Session History** - Searchable optimization log
5. **Alerts** - System alerts and notifications

### Tech Stack

- **Backend:** FastAPI, SQLAlchemy, WebSockets
- **Frontend:** React, Recharts, TailwindCSS, Vite
- **Database:** SQLite (can be swapped for PostgreSQL)
- **Container:** Docker, Docker Compose

---

## Core Files Explained

### continuous_profiler.py - The Profiler

**Purpose:** Measures what your model is doing on the GPU.

```python
profiler = ContinuousProfiler()
profiler.start()

with profiler.profile_region("forward"):
    output = model(input)

profiler.stop()
snapshot = profiler.snapshot()

# snapshot contains:
# - total_gpu_time_ms: How long GPU spent working
# - memory_bound_pct: % of time waiting for memory
# - total_dram_bytes: How much memory was accessed
```

### traffic_attribution.py - The Analyzer

**Purpose:** Figures out WHICH parts of your model cause memory problems.

```python
attributor = TrafficAttributor()
attributor.analyze_model(model, sample_input)

candidates = attributor.get_optimization_candidates()
# Returns optimization suggestions
```

**Optimization Types:**

| Type | What It Fixes |
|------|---------------|
| `CACHE_RESIDENCY` | Keep hot data in cache |
| `KERNEL_FUSION` | Combine operations to reduce memory reads |
| `LAYOUT_TRANSFORM` | Make memory layout more efficient |
| `TILING` | Process data in cache-sized chunks |
| `PREFETCH_INJECTION` | Load data before it's needed |

### adaptive_optimizer.py - The Optimizer

**Purpose:** Applies optimizations and verifies they work.

```python
optimizer = AdaptiveOptimizer()

session = optimizer.optimize(
    model=model,
    candidates=candidates,
    input_fn=lambda: torch.randn(8, 512).cuda(),
    num_warmup=5,
    num_measure=20,
)

# session contains:
# - total_speedup: e.g., 1.73x faster
# - committed_count: How many optimizations worked
# - rollback_count: How many were reverted
```

**The Test-Measure-Commit Loop:**

```
For each optimization candidate:
    1. SAVE current model state (checkpoint)
    2. MEASURE baseline performance (20 runs)
    3. APPLY the optimization
    4. VERIFY semantics (output still matches)
    5. MEASURE new performance (20 runs)
    6. DECIDE:
       - If faster AND significant → COMMIT
       - If slower OR breaks output → ROLLBACK
```

---

## GPU-Specific Settings

Different GPUs have different optimal settings:

```python
_GPU_PROFILES = {
    "A100": GPUProfile(
        l2_cache_mb=40,
        memory_bandwidth_gbps=2039,
        enable_tf32=True,
        preferred_compile_mode="reduce-overhead",
    ),
    "H100": GPUProfile(
        l2_cache_mb=50,
        memory_bandwidth_gbps=3350,
        ...
    ),
    "T4": GPUProfile(
        l2_cache_mb=4,
        enable_tf32=False,  # Turing doesn't have TF32
        ...
    ),
}

# Auto-detect and apply
profile = get_gpu_profile()
apply_gpu_profile(profile)
```

---

## What Optimizations Does It Apply?

### 1. cuDNN Benchmark + TF32
```python
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
```
**Effect:** 10-30% speedup on Ampere+ GPUs

### 2. torch.compile
```python
model = torch.compile(model, mode="reduce-overhead")
```
**Effect:** Fuses kernels, 20-100% speedup

### 3. Memory Layout
```python
for param in model.parameters():
    param.data = param.data.contiguous()
model = model.to(memory_format=torch.channels_last)
```
**Effect:** Removes strided access overhead

### 4. Gradient Checkpointing
```python
model.gradient_checkpointing_enable()
```
**Effect:** Trades compute for memory

---

## Key Metrics

### Coefficient of Variation (CV)
```
CV = (standard_deviation / mean) × 100%
```
- **< 5%:** Excellent measurement stability
- **5-10%:** Good
- **> 10%:** Noisy, results may be unreliable

### Speedup
```
Speedup = baseline_time / optimized_time
```
- **1.0x:** No improvement
- **1.5x:** 50% faster
- **2.0x:** Twice as fast

---

## Installation

```bash
pip install memopt
```

Or from source:
```bash
git clone https://github.com/your-org/memopt.git
cd memopt
pip install -e .
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (for GPU profiling)
- pynvml (for NVML access)

Optional:
- httpx (for dashboard integration)
- PyYAML (for config files)

---

## Summary

**memopt** is a comprehensive GPU memory optimization platform:

1. **Profiler** → Measure, analyze, and optimize memory traffic
2. **Daemon** → Background monitoring and safe optimization
3. **Training Wrapper** → Zero-code training optimization
4. **Dashboard** → Fleet-wide monitoring and visibility

It uses:
- Statistical validation (not single measurements)
- Semantic verification (outputs must match)
- Automatic rollback (bad optimizations are reverted)
- GPU-specific tuning (A100 vs T4 vs H100)
- Session persistence (results are saved)

**Best for:** Large models (> 10M params) on memory-bound workloads.

**Typical speedup:** 1.1x - 3.8x depending on model and GPU.

---

*Document generated for memopt v0.4.0*
