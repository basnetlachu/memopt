# memopt Demo

## Quick start

```bash
# Complete demo — all six pillars in order
bash demo/memopt_demo.sh

# Open the live dashboard in your browser
open demo/dashboard.html

# Generate and open the notebook
python3 demo/create_notebook.py
jupyter notebook demo/memopt_demo.ipynb
```

## Before a demo call (10 minutes before)

```bash
# 1. Rent any CUDA GPU (Lambda Labs A10 ~$0.75/hr)
# 2. SSH in and run:
git clone <your-repo> memopt && cd memopt
pip install -e . -q
bash demo/memopt_demo.sh   # verify everything works
# 3. Share your screen on the call
# 4. Run bash demo/memopt_demo.sh live
```

## What works without GPU (Mac preparation)

- P2 LCP benchmark — real numbers, no GPU needed
- P4 ledger + certificate — real numbers, no GPU needed
- P6 silicon certification — runs on CPU, shows cert structure
- P1 VMM — shows pre-recorded Blackwell numbers
- P3 kernels — shows synthesis pipeline, pre-recorded speedup
- Dashboard — shows static proof numbers in demo mode

## What needs GPU for live numbers

- P1 VMM tier usage (HBM/DRAM/NVMe split)
- P3 actual kernel synthesis (requires ANTHROPIC_API_KEY)
- P6 throughput benchmark (memory bandwidth %)
