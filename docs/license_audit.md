# License audit — v1.3.0

| Field | Value |
| --- | --- |
| Date | 2026-05-04 |
| Tool | `pip-licenses --format=plain --order=license` |
| Target | All transitive runtime dependencies of `memopt[dev,daemon,api]` |
| memopt license | Apache-2.0 (`pyproject.toml:10`, `LICENSE`) |
| Verdict | **OSS-distribution-compatible** with one LGPL callout below |

This audit documents every transitive dependency license under
`memopt`'s declared install profile and flags anything that requires
attention before public distribution.

## License classes (compatible with Apache-2.0 distribution)

  - **Apache-2.0** — fully compatible. Includes our own license.
    Examples: `accelerate`, `huggingface-hub`, `requests`,
    `safetensors`, `tokenizers`, `transformers`, `regex`,
    `prometheus_client`, `hf-xet`.
  - **BSD-2-Clause / BSD-3-Clause / 0BSD** — permissive; compatible
    with Apache-2.0 redistribution provided the BSD copyright line
    is preserved (already handled by the upstream wheels).
    Examples: `numpy`, `torch`, `MarkupSafe`, `psutil`, `pycparser`,
    `starlette`, `Pygments`, `Jinja2`, `httpx`, `httpcore`, `idna`,
    `networkx`, `fsspec`, `nbformat`, `cairocffi`, `cssselect2`,
    `tinycss2`, `webencodings`, `mpmath`, `sympy`, `traitlets`,
    `fastjsonschema`.
  - **MIT** — fully compatible. Examples: `fastapi`, `pydantic`,
    `pydantic_core`, `pytest`, `pluggy`, `iniconfig`, `attrs`,
    `cffi`, `charset-normalizer`, `urllib3`, `jsonschema`,
    `jsonschema-specifications`, `referencing`, `rpds-py`,
    `platformdirs`, `anyio`, `annotated-doc`, `typing-inspection`,
    `PyYAML`, `annotated-types`, `h11`, `pillow` (MIT-CMU
    variant).
  - **PSF-2.0** (Python Software Foundation) — compatible.
    `typing_extensions`.
  - **Unlicense** — public-domain dedication; permissive.
    `filelock`.

## Mixed / dual licenses (compatible)

  - `numpy` — `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0`.
    Compatible.
  - `prometheus_client` — `Apache-2.0 AND BSD-2-Clause`. Compatible.
  - `regex` — `Apache-2.0 AND CNRI-Python`. CNRI-Python is the same
    family as PSF-2.0 → compatible.
  - `tqdm` — `MIT License; Mozilla Public License 2.0`. MIT branch
    governs runtime usage. Compatible.
  - `packaging` — `Apache Software License; BSD License`. Compatible.

## Callouts

### MPL-2.0 (file-level copyleft) — review path

  - `certifi` (Mozilla Public License 2.0) — file-level copyleft.
    Compatible with Apache-2.0 redistribution provided modifications
    to MPL-licensed source files are released under MPL. memopt does
    NOT modify `certifi`; we consume the wheel as-is. **Verdict: OK.**

### LGPL-3.0-or-later (weak copyleft) — flag in NOTICE

  - `CairoSVG 2.8.2` (LGPL-3.0-or-later) — pulled in transitively
    (via the PDF / SVG export chain that `reportlab` may use). LGPL
    is compatible with Apache-2.0 distribution when dynamically
    linked / imported as a runtime dependency. memopt does NOT bundle
    or statically link CairoSVG sources. **Verdict: OK with NOTICE
    attribution.** NOTICE has been updated.

### Optional dependencies that are NOT installed by default

The audit ran with `[dev,daemon,api]` extras. The `cpp` extra
(`pybind11`, `scikit-build-core`, `cmake`, `ninja`) was not measured
here; all four are individually Apache-2.0 or BSD-3-Clause.

## Reproduction

```bash
pip install -e ".[dev,daemon,api]"
pip install pip-licenses
pip-licenses --format=plain --order=license
```

The CI job `license-audit` in `.github/workflows/ci.yml` runs the
same command on every push / PR.

## Action items

  - [x] Apache-2.0 declared in `pyproject.toml`.
  - [x] `LICENSE` file at repo root.
  - [x] `NOTICE` file at repo root listing third-party deps.
  - [x] CI license-audit job added.
  - [x] LGPL CairoSVG flagged in NOTICE.
  - [ ] **Before public release**: re-run with `--with-license-file`
        and bundle the upstream `LICENSE` files alongside any
        statically-linked binaries (currently zero — the engine ships
        as pure Python wheels).
