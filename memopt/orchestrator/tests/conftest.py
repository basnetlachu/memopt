"""Auto-skip @pytest.mark.perf tests in the default regression run.

`pytest -m perf …` runs them; the bare regression command (per
docs/orchestrator_v1_implementation_prompt.md §A.2) does not."""
import pytest


def pytest_collection_modifyitems(config, items):
    expr = config.getoption("-m") or ""
    if "perf" in expr:
        return
    skip_perf = pytest.mark.skip(reason="@perf — run with `pytest -m perf`")
    for item in items:
        # Scope only to orchestrator's microbench file so we don't disturb
        # other @perf-marked tests (e.g. substrate's test_events.py).
        if "memopt/orchestrator/tests/test_perf_microbench.py" not in item.nodeid:
            continue
        if any(m.name == "perf" for m in item.iter_markers()):
            item.add_marker(skip_perf)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "perf: orchestrator perf microbench (skipped by default)"
    )
