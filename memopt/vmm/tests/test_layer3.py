"""
Tests for Layer 3: federation, elastic_allocator, memory_governor.
All pass on CPU — no torch, no numpy, no external deps.
"""
import socket
import time

import pytest

from memopt.vmm.oracle import MemoryOracle
from memopt.vmm.universal_profile import MemoryTier, UniversalMemoryProfile
from memopt.vmm.federation import (
    FederationManager, FederationStats, GossipBatch, TransitionGossip,
    _send_frame,
)
from memopt.vmm.elastic_allocator import (
    AllocationDecision, AllocatorStats, ElasticAllocator, NodeMemoryState,
)
from memopt.vmm.memory_governor import (
    GovernorStats, MemoryGovernor,
    PRESSURE_NORMAL, PRESSURE_ELEVATED, PRESSURE_CRITICAL,
)


# ── Helpers ───────────────────────────────────────────────────────────────


def _free_port():
    """Find a free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _make_hw_profile(hbm_gb=80.0, dram_gb=512.0):
    """Build a synthetic UniversalMemoryProfile for testing."""
    tiers = []
    if hbm_gb > 0:
        tiers.append(MemoryTier(
            name="hbm", capacity_gb=hbm_gb, bandwidth_gbs=3350.0,
            latency_us=1.0, is_available=True, device_path="",
        ))
    tiers.append(MemoryTier(
        name="dram", capacity_gb=dram_gb, bandwidth_gbs=50.0,
        latency_us=80.0, is_available=True, device_path="",
    ))
    return UniversalMemoryProfile(
        tiers=tiers, total_capacity_gb=hbm_gb + dram_gb,
        architecture="cpu", device_name="test",
        detected_at=time.time(),
    )


# ══════════════════════════════════════════════════════════════════════════
#  Federation Tests (10)
# ══════════════════════════════════════════════════════════════════════════


def test_federation_manager_instantiates_with_defaults():
    oracle = MemoryOracle()
    fm = FederationManager(oracle, node_id="n1")
    assert fm._node_id == "n1"
    assert fm._gossip_interval == 5.0


def test_gossip_batch_serialization_roundtrip():
    oracle = MemoryOracle()
    fm = FederationManager(oracle, node_id="n1")
    batch = GossipBatch(
        node_id="n1", epoch=1,
        transitions=[TransitionGossip(0, 1, 5), TransitionGossip(1, 2, 3)],
    )
    raw = fm._serialize_batch(batch)
    restored = fm._deserialize_batch(raw)
    assert restored.node_id == "n1"
    assert restored.epoch == 1
    assert len(restored.transitions) == 2
    assert restored.transitions[0].count == 5


def test_build_batch_reads_oracle_transitions():
    oracle = MemoryOracle()
    oracle.observe("s1", 0, step=1)
    oracle.observe("s1", 1, step=2)
    oracle.observe("s1", 2, step=3)
    fm = FederationManager(oracle, node_id="n1")
    batch = fm._build_batch()
    assert len(batch.transitions) >= 2


def test_merge_batch_updates_oracle_transitions():
    oracle = MemoryOracle()
    fm = FederationManager(oracle, node_id="n1")
    batch = GossipBatch(
        node_id="n2", epoch=1,
        transitions=[TransitionGossip(10, 11, 5)],
    )
    fm._merge_batch(batch)
    assert oracle._transitions[10][11] == 5


def test_merge_batch_ignores_self():
    oracle = MemoryOracle()
    fm = FederationManager(oracle, node_id="n1")
    batch = GossipBatch(
        node_id="n1", epoch=1,
        transitions=[TransitionGossip(10, 11, 5)],
    )
    fm._merge_batch(batch)
    assert 10 not in oracle._transitions


def test_federation_starts_and_stops():
    oracle = MemoryOracle()
    port = _free_port()
    fm = FederationManager(
        oracle, node_id="n1", gossip_port=port, gossip_interval_s=0.1,
    )
    fm.start()
    time.sleep(0.3)
    fm.stop()


def test_federation_stats_fields():
    oracle = MemoryOracle()
    fm = FederationManager(oracle, node_id="n1", peers=["host1:9000"])
    s = fm.stats()
    assert isinstance(s, FederationStats)
    assert s.node_id == "n1"
    assert s.peers_known == 1
    assert s.batches_sent == 0
    assert s.batches_received == 0


def test_transition_gossip_is_frozen():
    t = TransitionGossip(0, 1, 5)
    with pytest.raises(AttributeError):
        t.count = 10


def test_gossip_batch_is_frozen():
    batch = GossipBatch(node_id="n1", epoch=1, transitions=[])
    with pytest.raises(AttributeError):
        batch.epoch = 2


def test_federation_tcp_gossip_integration():
    """Full TCP path: receiver accepts a batch and merges it."""
    oracle1 = MemoryOracle()
    oracle1.observe("s1", 0, step=1)
    oracle1.observe("s1", 1, step=2)

    oracle2 = MemoryOracle()
    port = _free_port()

    fm1 = FederationManager(oracle1, node_id="n1", gossip_port=port)
    fm2 = FederationManager(oracle2, node_id="n2", gossip_port=port)

    fm2.start()
    time.sleep(0.3)

    batch = fm1._build_batch()
    payload = fm1._serialize_batch(batch)
    sock = socket.create_connection(("127.0.0.1", port), timeout=2.0)
    _send_frame(sock, payload)
    sock.close()

    time.sleep(0.5)
    fm2.stop()

    s = fm2.stats()
    assert s.batches_received >= 1
    assert s.transitions_merged >= 1


# ══════════════════════════════════════════════════════════════════════════
#  Allocator Tests (9)
# ══════════════════════════════════════════════════════════════════════════


def test_allocator_instantiates_with_defaults():
    profile = _make_hw_profile()
    alloc = ElasticAllocator(profile, node_id="local")
    s = alloc.stats()
    assert isinstance(s, AllocatorStats)
    assert s.decisions_made == 0


def test_high_urgency_selects_local_hbm():
    profile = _make_hw_profile(hbm_gb=80.0)
    alloc = ElasticAllocator(profile, node_id="local")
    decision = alloc.decide(confidence=0.9)
    assert decision.target_tier == "hbm"
    assert decision.target_node == "local"


def test_medium_urgency_selects_local_dram():
    profile = _make_hw_profile(hbm_gb=0.0)
    alloc = ElasticAllocator(profile, node_id="local")
    decision = alloc.decide(confidence=0.6)
    assert decision.target_tier == "dram"
    assert decision.target_node == "local"


def test_low_urgency_falls_back_to_dram():
    profile = _make_hw_profile(hbm_gb=0.0)
    alloc = ElasticAllocator(profile, node_id="local")
    decision = alloc.decide(confidence=0.3)
    assert decision.target_tier == "dram"


def test_no_space_falls_back_to_nvme():
    profile = _make_hw_profile(hbm_gb=0.0, dram_gb=0.0)
    alloc = ElasticAllocator(profile, node_id="local")
    decision = alloc.decide(confidence=0.9)
    assert decision.target_tier == "nvme"


def test_remote_hbm_when_local_full():
    profile = _make_hw_profile(hbm_gb=0.0)
    remote = NodeMemoryState(
        node_id="remote_1", hbm_used_gb=0.0, hbm_total_gb=80.0,
        dram_used_gb=0.0, dram_total_gb=512.0, is_local=False,
    )
    alloc = ElasticAllocator(profile, node_id="local", remote_nodes=[remote])
    decision = alloc.decide(confidence=0.8)
    assert decision.target_tier == "remote_hbm"
    assert decision.target_node == "remote_1"


def test_update_node_state_changes_decisions():
    profile = _make_hw_profile(hbm_gb=0.0)
    alloc = ElasticAllocator(profile, node_id="local")
    d1 = alloc.decide(confidence=0.8)
    assert d1.target_tier == "dram"

    remote = NodeMemoryState(
        node_id="r1", hbm_used_gb=0.0, hbm_total_gb=80.0,
        dram_used_gb=0.0, dram_total_gb=512.0, is_local=False,
    )
    alloc.update_node_state(remote)
    d2 = alloc.decide(confidence=0.8)
    assert d2.target_tier == "remote_hbm"


def test_allocator_stats_count_correctly():
    profile = _make_hw_profile(hbm_gb=80.0)
    alloc = ElasticAllocator(profile, node_id="local")
    alloc.decide(confidence=0.9)
    alloc.decide(confidence=0.9)
    alloc.decide(confidence=0.9)
    s = alloc.stats()
    assert s.decisions_made == 3
    assert s.local_hbm == 3


def test_decision_is_frozen():
    profile = _make_hw_profile(hbm_gb=80.0)
    alloc = ElasticAllocator(profile, node_id="local")
    decision = alloc.decide(confidence=0.9)
    with pytest.raises(AttributeError):
        decision.target_tier = "nvme"


# ══════════════════════════════════════════════════════════════════════════
#  Governor Tests (6)
# ══════════════════════════════════════════════════════════════════════════


def test_governor_instantiates_with_defaults():
    oracle = MemoryOracle(horizon=50)
    profile = _make_hw_profile(hbm_gb=80.0)
    gov = MemoryGovernor(oracle, profile)
    assert gov.pressure_level() == PRESSURE_NORMAL


def test_pressure_normal_when_low_utilization():
    oracle = MemoryOracle(horizon=50)
    profile = _make_hw_profile(hbm_gb=80.0)
    gov = MemoryGovernor(oracle, profile, hbm_used_gb_override=10.0)
    gov._poll_once()
    assert gov.pressure_level() == PRESSURE_NORMAL
    assert oracle._horizon == 50


def test_pressure_elevated_when_medium_utilization():
    oracle = MemoryOracle(horizon=50)
    profile = _make_hw_profile(hbm_gb=80.0)
    gov = MemoryGovernor(oracle, profile, hbm_used_gb_override=60.0)
    gov._poll_once()
    assert gov.pressure_level() == PRESSURE_ELEVATED
    assert oracle._horizon == 25


def test_pressure_critical_when_high_utilization():
    oracle = MemoryOracle(horizon=50)
    profile = _make_hw_profile(hbm_gb=80.0)
    gov = MemoryGovernor(oracle, profile, hbm_used_gb_override=75.0)
    gov._poll_once()
    assert gov.pressure_level() == PRESSURE_CRITICAL
    assert oracle._horizon == 12


def test_horizon_adjusted_on_pressure_change():
    oracle = MemoryOracle(horizon=100)
    profile = _make_hw_profile(hbm_gb=80.0)
    gov = MemoryGovernor(oracle, profile, hbm_used_gb_override=10.0)

    gov._poll_once()
    assert oracle._horizon == 100

    gov.set_hbm_used_gb(60.0)
    gov._poll_once()
    assert oracle._horizon == 50

    gov.set_hbm_used_gb(10.0)
    gov._poll_once()
    assert oracle._horizon == 100

    s = gov.stats()
    assert s.adjustments_made >= 2


def test_governor_stats_fields():
    oracle = MemoryOracle(horizon=50)
    profile = _make_hw_profile(hbm_gb=80.0)
    gov = MemoryGovernor(oracle, profile)
    s = gov.stats()
    assert isinstance(s, GovernorStats)
    for field_name in (
        "pressure_level", "hbm_utilization_pct", "current_horizon",
        "original_horizon", "adjustments_made", "drift_detected",
    ):
        assert hasattr(s, field_name)


# ══════════════════════════════════════════════════════════════════════════
#  Federation — bounded fan-out + delta gossip
# ══════════════════════════════════════════════════════════════════════════

def test_bounded_fanout_limits_peers():
    """Gossip sends to at most FANOUT peers, not all."""
    from unittest.mock import patch, MagicMock

    oracle = MemoryOracle()
    oracle.observe("s1", 0, step=1)
    oracle.observe("s1", 1, step=2)

    fm = FederationManager(
        oracle, node_id="n1",
        peers=["p1:18600", "p2:18600", "p3:18600",
               "p4:18600", "p5:18600"],
    )
    # Force fanout=3
    fm._fanout = 3

    with patch.object(fm, '_send_to_peer') as mock_send:
        fm._do_gossip_round()
        # Must send to exactly 3 peers, not 5
        assert mock_send.call_count == 3


def test_bounded_fanout_does_not_exceed_available():
    """If fewer peers than fanout, send to all available."""
    from unittest.mock import patch

    oracle = MemoryOracle()
    oracle.observe("s1", 0, step=1)
    oracle.observe("s1", 1, step=2)

    fm = FederationManager(
        oracle, node_id="n1",
        peers=["p1:18600", "p2:18600"],
    )
    fm._fanout = 10  # more than available

    with patch.object(fm, '_send_to_peer') as mock_send:
        fm._do_gossip_round()
        assert mock_send.call_count == 2  # only 2 available


def test_delta_gossip_skips_unchanged():
    """No gossip sent when transitions have not changed."""
    from unittest.mock import patch

    oracle = MemoryOracle()
    oracle.observe("s1", 0, step=1)
    oracle.observe("s1", 1, step=2)

    fm = FederationManager(
        oracle, node_id="n1", peers=["p1:18600"])

    # First round: should send (everything is new)
    with patch.object(fm, '_send_to_peer') as mock_send:
        fm._do_gossip_round()
        assert mock_send.call_count == 1

    # Second round: nothing changed — should NOT send
    with patch.object(fm, '_send_to_peer') as mock_send:
        fm._do_gossip_round()
        mock_send.assert_not_called()


def test_delta_gossip_sends_after_new_observation():
    """Delta gossip sends when new transitions are observed."""
    from unittest.mock import patch

    oracle = MemoryOracle()
    oracle.observe("s1", 0, step=1)
    oracle.observe("s1", 1, step=2)

    fm = FederationManager(
        oracle, node_id="n1", peers=["p1:18600"])

    # First round: flush initial transitions
    with patch.object(fm, '_send_to_peer'):
        fm._do_gossip_round()

    # Observe new transition
    oracle.observe("s1", 2, step=3)

    # Second round: should send the new transition
    with patch.object(fm, '_send_to_peer') as mock_send:
        fm._do_gossip_round()
        assert mock_send.call_count == 1


def test_peer_discovery_falls_back_to_static():
    """Static MEMOPT_NODE_HOSTS used when Redis absent."""
    oracle = MemoryOracle()
    fm = FederationManager(
        oracle, node_id="n1",
        peers=["static_peer:18600"])
    fm._redis_client = None  # no Redis
    peers = fm._discover_peers()
    assert peers == ["static_peer:18600"]


def test_peer_discovery_returns_empty_without_config():
    """No peers when neither Redis nor MEMOPT_NODE_HOSTS set."""
    oracle = MemoryOracle()
    fm = FederationManager(oracle, node_id="n1", peers=[])
    fm._redis_client = None
    peers = fm._discover_peers()
    assert peers == []


def test_do_gossip_round_no_peers_no_crash():
    """Gossip round with no peers doesn't crash."""
    oracle = MemoryOracle()
    oracle.observe("s1", 0, step=1)
    oracle.observe("s1", 1, step=2)

    fm = FederationManager(oracle, node_id="n1", peers=[])
    fm._redis_client = None
    fm._do_gossip_round()  # must not crash
