# memopt Distribution Guide

This document explains how to build and distribute memopt privately to customers.

---

## Quick Start

```bash
# Build a private wheel
python build_private.py

# Output: dist/memopt-0.4.0-delivery.tar.gz
```

Send `memopt-0.4.0-delivery.tar.gz` to your customer.

---

## Option 1: Private Wheel (Recommended)

Best for: Quick delivery, most customers.

### Build

```bash
# Standard wheel (source included but private)
python build_private.py

# OR with Cython compilation (better protection, slower build)
python build_private.py --cython
```

### What Gets Created

```
dist/
├── memopt-0.4.0-py3-none-any.whl    # The package
├── INSTALL.md                        # Customer instructions
├── LICENSE                           # Proprietary license
└── memopt-0.4.0-delivery.tar.gz     # Ready-to-send package
```

### Customer Installation

```bash
# Customer extracts and installs
tar -xzf memopt-0.4.0-delivery.tar.gz
cd memopt-0.4.0-delivery
pip install memopt-0.4.0-py3-none-any.whl

# Verify
python -c "from memopt.profiler import api; print(api.__version__)"
```

---

## Option 2: Docker Container

Best for: Enterprise customers (G42), air-gapped environments.

### Build

```bash
# First build the wheel
python build_private.py

# Then build Docker image
docker build -t memopt:0.4.0 .

# Export for delivery
docker save memopt:0.4.0 | gzip > memopt-0.4.0-docker.tar.gz
```

### Customer Usage

```bash
# Load the image
docker load < memopt-0.4.0-docker.tar.gz

# Run with GPU access
docker run --gpus all memopt:0.4.0

# Mount their code
docker run --gpus all -v /path/to/their/code:/workspace memopt:0.4.0 \
    python /workspace/optimize_model.py
```

---

## Option 3: Private GitHub

Best for: Ongoing collaboration, updates.

### Setup

1. Create private GitHub repo: `github.com/yourorg/memopt-private`
2. Push code
3. Add customer as collaborator when they sign

### Customer Installation

```bash
pip install git+https://github.com/yourorg/memopt-private.git@v0.4.0
```

---

## Source Protection Levels

| Method | Protection | Build Time | Notes |
|--------|------------|------------|-------|
| Basic Wheel | Low | Fast | Source visible in .whl |
| Cython Wheel | High | Slow | Compiled to .so binaries |
| Docker | High | Medium | Source not accessible |
| Private GitHub | Low | N/A | Requires access control |

### Cython Compilation

Compiles Python to C, then to binary `.so` files. Customer gets binaries, not source.

```bash
python build_private.py --cython
```

**Requirements:** `pip install cython`

**What happens:**
```
memopt/profiler/adaptive_optimizer.py  →  adaptive_optimizer.cpython-310-x86_64-linux-gnu.so
```

Customer sees only compiled binaries, not Python source.

---

## Delivery Checklist

- [ ] Build the wheel: `python build_private.py`
- [ ] Test installation on clean environment
- [ ] Verify GPU functionality
- [ ] Package with license and instructions
- [ ] Upload to secure location (Google Drive, Dropbox, S3)
- [ ] Send download link to customer
- [ ] Provide support email

---

## Versioning

Update version in these files when releasing:
- `pyproject.toml` → `version = "X.Y.Z"`
- `setup.py` → `version="X.Y.Z"`
- `memopt/profiler/api.py` → `__version__ = "X.Y.Z"`

---

## Customer Onboarding Template

**Email to customer:**

```
Subject: memopt v0.4.0 - Download Link

Hi [Customer],

Your memopt license is ready. Download link:
[SECURE LINK]

Installation:
1. Extract: tar -xzf memopt-0.4.0-delivery.tar.gz
2. Install: pip install memopt-0.4.0-*.whl
3. Verify: python -c "from memopt.profiler import api; print(api.__version__)"

Quick start:
```python
from memopt.profiler import api
model, session = api.optimize(model, sample, verbose=True)
```

Support: hello@memopt.ai

Best,
MemOpt Team
```

---

## Troubleshooting

### Customer: "Import error"

```bash
# Check installation
pip show memopt

# Reinstall
pip uninstall memopt
pip install memopt-0.4.0-*.whl
```

### Customer: "CUDA not found"

```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# memopt requires CUDA-capable GPU
```

### Customer: "No speedup"

Small models (< 10M params) may not see improvement. memopt works best on larger models.

---

## Security Notes

1. **Never publish to PyPI** - This is private software
2. **Use secure transfer** - HTTPS links only
3. **Track downloads** - Know who has access
4. **Revoke access** - Can rebuild with different obfuscation
5. **License enforcement** - Consider adding license key validation (future)

---

## Future: License Key Validation

For stronger protection, add license key validation:

```python
# In memopt/__init__.py
def validate_license(key: str) -> bool:
    # Validate against your server
    # Expire after trial period
    # Tie to specific GPU hardware ID
    pass
```

This prevents redistribution even if wheel is shared.

---

*Document for memopt v0.4.0*
