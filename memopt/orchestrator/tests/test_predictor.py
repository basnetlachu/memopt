"""Tests for Predictor (orchestrator v1 Commit 3; design §2.3.2,
DECISION 6). 13 tests per §3.1.2."""
from __future__ import annotations

import sys
import threading

from memopt.orchestrator.predict import Predictor, Prediction


def test_observe_inserts_transition():
    p = Predictor()
    p.observe("alice", "a")
    p.observe("alice", "b")
    s = p.stats()
    assert s["transition_rows"] == 1
    assert s["transition_cells"] == 1


def test_observe_increments_existing_transition():
    p = Predictor()
    for _ in range(3):
        p.observe("alice", "a")
        p.observe("alice", "b")
        p.observe("alice", "a")
    # Transitions: a->b (3 times), b->a (3 times — wait, actually 2x b->a,
    # because the very first 'a' has no predecessor).
    s = p.stats()
    # Two distinct transition rows: from 'a' and from 'b'.
    assert s["transition_rows"] == 2


def test_predict_returns_top_k_sorted_by_confidence():
    p = Predictor(min_confidence=0.0)
    # Strong a->b association.
    for _ in range(10):
        p.observe("alice", "a")
        p.observe("alice", "b")
    out = p.predict("alice", "a", top_k=3)
    assert len(out) <= 3
    assert out[0].tag == "b"
    confs = [pr.confidence for pr in out]
    assert confs == sorted(confs, reverse=True)


def test_predict_includes_sequential_source():
    p = Predictor(min_confidence=0.0)
    out = p.predict("alice", "block7")
    sources = {pr.source for pr in out}
    assert "sequential" in sources
    tags = {pr.tag for pr in out}
    assert "block8" in tags
    assert "block9" in tags


def test_predict_includes_recency_source():
    p = Predictor(min_confidence=0.0)
    # Build recency for tenant alice with a non-numeric tag base (no
    # sequential source for "y" / "z"), so recency entries are not shadowed.
    p.observe("alice", "y")
    p.observe("alice", "z")
    p.observe("alice", "x")
    out = p.predict("alice", "x")
    sources = {pr.source for pr in out}
    assert "recency" in sources


def test_predict_fallback_when_cold_start():
    p = Predictor()
    out = p.predict("alice", "fresh")
    # No observations exist; fallback (or sequential if numeric) must be returned.
    assert len(out) >= 1
    sources = {pr.source for pr in out}
    assert "fallback" in sources


def test_min_confidence_filters_predictions():
    # Sequential source emits 0.4 for delta=2. Setting min_confidence
    # above 0.4 must drop that prediction.
    p = Predictor(min_confidence=0.5)
    out = p.predict("alice", "block1")
    confs = [pr.confidence for pr in out]
    assert all(c >= 0.5 for c in confs), confs


def test_max_transitions_bounded():
    p = Predictor(max_transitions=10)
    # Observe a long sequence; many distinct (prev, next) pairs.
    for i in range(100):
        p.observe("alice", f"k{i}")
    s = p.stats()
    assert s["transition_rows"] <= 10


def test_predictor_keys_are_tenant_aware():
    # DECISION 6: alice and bob observing the same sequence must
    # produce disjoint transitions.
    p = Predictor(min_confidence=0.0)
    for _ in range(5):
        p.observe("alice", "a")
        p.observe("alice", "b")
    bob_pred = p.predict("bob", "a")
    bob_sources = {pr.source for pr in bob_pred}
    assert "transition" not in bob_sources
    alice_pred = p.predict("alice", "a")
    assert any(pr.source == "transition" and pr.tag == "b" for pr in alice_pred)


def test_forget_tenant_removes_all_transitions():
    p = Predictor(min_confidence=0.0)
    p.observe("alice", "a")
    p.observe("alice", "b")
    p.observe("bob", "a")
    p.observe("bob", "b")
    p.forget_tenant("alice")
    s = p.stats()
    # All remaining keys belong to bob.
    # There should be at least one bob row, no alice row.
    pred_alice = p.predict("alice", "a")
    assert all(pr.source != "transition" for pr in pred_alice)
    # bob still has its transition.
    pred_bob = p.predict("bob", "a")
    assert any(pr.source == "transition" for pr in pred_bob)


def test_predictor_thread_safe_under_rlock():
    p = Predictor(min_confidence=0.0)

    def writer(start: int) -> None:
        for i in range(200):
            p.observe(f"t{start}", f"k{i}")

    def reader() -> None:
        for _ in range(200):
            p.predict("t0", "k0")

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(3)]
    threads.append(threading.Thread(target=reader))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # No exception, state coherent.
    s = p.stats()
    assert s["transition_rows"] >= 0  # liveness
    assert s["tenants_tracked"] >= 3


def test_predictor_stats_shape():
    p = Predictor(max_transitions=42, min_confidence=0.25)
    s = p.stats()
    assert set(s.keys()) >= {
        "transition_rows",
        "transition_cells",
        "recency_size",
        "tenants_tracked",
        "max_transitions",
        "min_confidence",
    }
    assert s["max_transitions"] == 42
    assert s["min_confidence"] == 0.25


def test_predictor_does_not_share_state_with_memory_oracle():
    # DECISION 6 / §2.4.5: orchestrator's predictor is greenfield.
    # Verified two ways:
    #  (a) static — predict.py imports neither _oracle_py nor oracle
    #  (b) runtime — modifying a fresh MemoryOracle does not touch
    #      a Predictor instance's stats.
    import ast
    import memopt.orchestrator.predict as predict_mod

    tree = ast.parse(open(predict_mod.__file__).read())
    forbidden = {"memopt.vmm._oracle_py", "memopt.vmm.oracle"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in forbidden, alias.name
        elif isinstance(node, ast.ImportFrom):
            assert node.module not in forbidden, node.module

    p = Predictor()
    before = p.stats()
    try:
        from memopt.vmm._oracle_py import MemoryOracle
        oracle = MemoryOracle()
        oracle.observe("seq", 42)
    except Exception:
        pass
    after = p.stats()
    assert before == after
