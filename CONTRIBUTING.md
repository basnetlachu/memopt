# Contributing to memopt

Thanks for contributing. This document describes how to file issues,
open pull requests, and work with the codebase without breaking the
shipped contracts.

## Quick start

```bash
git clone https://github.com/<your-fork>/memopt.git
cd memopt
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,daemon,api]"

# Baseline regression — must pass before AND after any change.
PYTHONPATH=. python3 -m pytest tests/ memopt/ -q -p no:cacheprovider \
  -k 'not gpu and not cuda' \
  --ignore=memopt/vmm/tests/test_cuda_vmm_smoke.py \
  --ignore=memopt/vmm/tests/test_torch_allocator_sanity.py
```

The repo ships with a known baseline (currently **876 PASSED, 34
SKIPPED, 15 DESELECTED, 0 FAILED** on Mac without CUDA). Your PR must
preserve PASSED ≥ baseline-PASSED, and not flip any SKIPPED to FAILED.

## License

By contributing, you agree that your contributions will be licensed
under the [Apache License 2.0](LICENSE).

## Architecture rules

memopt is built in layers. The top layers have **20-year contracts**
(`docs/orchestrator_v1_implementation_prompt.md` §H.4). Do not edit
files under these paths without a full design redesign:

  - `memopt/substrate/` — Layer 1 memory substrate (v1.1)
  - `memopt/orchestrator/` — Layer 2 orchestrator (v1.2.0)
  - `tests/test_orchestrator_legacy_parity.py` — the bridge contract

Specifically:

  - The seven substrate primitives are pinned: `reserve_va`,
    `create_physical`, `map`, `set_access`, `unmap`,
    `release_physical`, `free_va`. New backends must implement all
    seven.
  - The five event kinds are pinned: `("alloc", "free", "evict",
    "promote", "migrate")`. Adding a sixth kind requires a substrate
    redesign.
  - The orchestrator's `Predictor` protocol (`observe`, `predict`,
    `forget_tenant`, `stats`) is the Layer-3 plug-in interface. New
    predictors implement the same shape.
  - `Predictor` keys are always `(tenant, tag)` tuples (DECISION 6,
    enforced by `test_predictor_keys_are_tenant_aware`). Cross-tenant
    pollution is a security bug.

Layer-1 isolation is verified by an automated AST scan in
`memopt/orchestrator/tests/test_decision_d1_greenfield.py`. New
orchestrator code must NOT import from `memopt.vmm.*` (legacy).

## Pull request process

1. **One change per PR.** Don't bundle a refactor with a bug fix.
2. **Write the test first.** Every new module gets a test under
   `<package>/tests/`. Tests live next to the code they test.
3. **Cite the design doc.** PRs that touch the substrate or
   orchestrator must reference the design doc section that describes
   the change.
4. **Run the regression locally** before opening the PR. Paste the
   final line of pytest output into the PR description.
5. **Don't `--no-verify`.** Pre-commit hooks exist for a reason. If a
   hook fails, fix the root cause.
6. **Don't `git push --force` to `main`.** Force-pushes to topic
   branches are fine; `main` is protected.
7. **Update the docs** if you change behavior. Stale docs are worse
   than no docs.

## Issue process

When reporting a bug, include:

  - The exact pytest invocation that failed.
  - The last 30 lines of pytest output.
  - Output of `python -c "import memopt; print(memopt.__version__)"`.
  - Hardware / OS (Mac/Linux, CPU/GPU model).

When proposing a feature, describe:

  - What the user would call (the public API).
  - What invariant the feature preserves (which design-doc decision).
  - Whether it's a Layer 1, Layer 2, or Layer 3+ change.

## Layer 3 (Prefetch Oracle) contributions

Layer 3 is the ML-driven prefetch oracle (see
`docs/oracle_v1_design.md` Phase 1). It plugs into Layer 2's
`Predictor` protocol. To contribute a new predictor:

1. Read `docs/oracle_v1_design.md` end-to-end.
2. Implement the four-method protocol (`observe`, `predict`,
   `forget_tenant`, `stats`).
3. Honor the `(tenant, tag)` key invariant.
4. Add a fallback to the classical Layer-2 predictor when confidence
   is low — the classical predictor at `memopt/orchestrator/predict.py`
   is the long-term safety net and must never be deleted.

## Code style

  - Python 3.10+ syntax. `from __future__ import annotations` at the
    top of every new module.
  - Type-annotate the public surface. Internal helpers are flexible.
  - No emojis in source code (per project convention).
  - No comments unless the *why* is non-obvious.

## Security

If you find a security vulnerability, **do not** open a public issue.
Email `basnetlawservices@gmail.com` with subject line
`memopt-security: <one-line summary>`. We will acknowledge within 5
business days. (For first-OSS-release we use a personal email; this
will move to `security@<future-domain>` when the project gains a
dedicated org.)

## Questions

Open a discussion at https://github.com/basnetlachu/memopt/discussions
or file an issue at https://github.com/basnetlachu/memopt/issues.
