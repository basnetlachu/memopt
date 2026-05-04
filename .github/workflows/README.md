# GitHub Actions workflows

## Active

- **`ci.yml`** — canonical CI matrix (Linux + Mac × Python 3.10/3.11/3.12).
  Runs the regression command from `CONTRIBUTING.md`, plus a public-API
  smoke import, plus the `@perf` microbench job and the
  `pip-licenses` audit. This is the only workflow that runs on every
  push to `main` and every PR to `main`.

## Disabled (kept for reference)

These workflows predate the Layer 1 substrate + Layer 2 orchestrator
work and the v1.3.0 pillar revival. They were failing on the first
push because their assumptions no longer hold (different test scope,
CUDA-only build, deleted helper scripts). They are renamed to
`.yml.disabled` so GitHub Actions ignores them; original content is
preserved verbatim.

- **`test.yml.disabled`** — Old `Tests` workflow. Ran `pytest` without
  the `-k 'not gpu and not cuda'` filter, so it tried to execute
  `@gpu` tests on a CPU runner. It also called
  `scripts/audit_wiring.py` which references modules that were
  removed in commit `c5705a3` ("optimizations deleted……"). To
  re-enable, fix the audit script and align the pytest invocation
  with the regression command in `ci.yml`, then rename back to
  `test.yml`.

- **`build_wheels.yml.disabled`** — Old `Build Wheels` workflow.
  Required CUDA toolkit on Linux runners (`Jimver/cuda-toolkit`),
  built per-CUDA-version wheels (12.4, 12.6). The CUDA install step
  is fragile on hosted runners and the cython/C++ build path is not
  required for the pure-Python alpha release. To re-enable, switch to
  pure-Python wheel builds first (drop the `[cpp]` extra) and add
  CUDA-extension wheels behind a separate workflow that only runs on
  release tags.

- **`build_golden_image.yml.disabled`** — Old `Build Golden Image`
  workflow. Built a Docker image baked with the model + memopt
  binaries. Out of scope for the alpha OSS release; deferred to a
  future packaging pass.

To re-enable any of these, rename the file (drop the `.disabled`
suffix) AND make sure it passes locally first (e.g. via `act`).
