"""DECISION 1 verification (orchestrator v1 Commit 9; design §2.1
DECISION 1, §3.1.9). Static `ast` import scans assert the orchestrator
package does NOT depend on the legacy VMM stack."""
from __future__ import annotations

import ast
import pathlib

import memopt.orchestrator as orch_pkg


_ORCH_ROOT = pathlib.Path(orch_pkg.__file__).parent


def _orchestrator_source_files():
    files = []
    for p in _ORCH_ROOT.rglob("*.py"):
        if "/tests/" in str(p) or p.name.startswith("test_"):
            continue
        files.append(p)
    return files


def _imports_in(path: pathlib.Path):
    tree = ast.parse(path.read_text())
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.append(node.module)
    return out


def test_orchestrator_does_not_import_tier_manager():
    forbidden = {"memopt.vmm.tier_manager", "memopt.vmm"}
    for f in _orchestrator_source_files():
        for mod in _imports_in(f):
            assert "tier_manager" not in mod, (f, mod)


def test_orchestrator_does_not_import_elastic_allocator():
    for f in _orchestrator_source_files():
        for mod in _imports_in(f):
            assert "elastic_allocator" not in mod, (f, mod)


def test_orchestrator_does_not_import_memory_governor():
    for f in _orchestrator_source_files():
        for mod in _imports_in(f):
            assert "memory_governor" not in mod, (f, mod)


def test_orchestrator_does_not_import_prefetch_engine():
    for f in _orchestrator_source_files():
        for mod in _imports_in(f):
            assert "prefetch_engine" not in mod, (f, mod)


def test_orchestrator_does_not_call_memopt_evict_to_target():
    # Static grep of orchestrator source for the C ABI symbol or any
    # ctypes binding that touches it.
    for f in _orchestrator_source_files():
        src = f.read_text()
        assert "memopt_evict_to_target" not in src, f
        # Defensive: the orchestrator must not import ctypes either.
        assert "import ctypes" not in src, f
        assert "from ctypes" not in src, f
