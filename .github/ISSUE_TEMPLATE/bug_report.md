---
name: Bug report
about: Report a defect in memopt
labels: bug
---

**Environment**

- memopt version: `python -c "import memopt; print(memopt.__version__)"` →
- OS:
- Python version:
- GPU (if applicable):
- Installed via:  `pip install memopt` / source / editable

**Reproduction**

Exact pytest invocation or minimal code example.

```bash
PYTHONPATH=. python -m pytest tests/<file>.py::<test> -q
```

**Expected behavior**

What you expected to happen.

**Actual behavior**

What actually happened. Paste the last 30 lines of pytest output, or
the full traceback.

**Layer / pillar**

Which layer or pillar is affected? (Layer 1 substrate, Layer 2
orchestrator, pillar 1-8, integrations, CLI, …)

**Additional context**

Anything else relevant. CHANGELOG.md entry that introduced the
regression, hardware specifics, env vars set, etc.
