"""
Tests for canary monitoring artifacts:

  - Prometheus alert rules (alert_rules.yml)
  - Prometheus config update (rule_files wiring)
  - Grafana canary rollout dashboard JSON
  - Stage-to-int mapping contract (memopt/canary/metrics.py)

These tests only validate structure and the enum↔integer contract —
they do not require a running Prometheus or Grafana instance.
"""
import json
import os

import yaml

from memopt.canary.metrics import (
    NO_ACTIVE_ROLLOUT,
    STAGE_TO_INT,
    stage_to_int,
    update_rollout_metrics,
)
from memopt.canary.models import RolloutStage


_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))


# ══════════════════════════════════════════════════════════════════════
#  Prometheus alert rules
# ══════════════════════════════════════════════════════════════════════


def test_alert_rules_valid_yaml():
    path = os.path.join(
        _REPO_ROOT, "deploy/prometheus/alert_rules.yml")
    with open(path) as f:
        doc = yaml.safe_load(f)

    assert "groups" in doc
    assert len(doc["groups"]) >= 1
    rules = doc["groups"][0]["rules"]
    assert len(rules) >= 5

    for rule in rules:
        assert "alert" in rule
        assert "expr" in rule
        assert "annotations" in rule
        assert "summary" in rule["annotations"]


def test_alert_rules_include_rollout_alerts():
    """Both PAUSED and FAILED alerts must be present."""
    path = os.path.join(
        _REPO_ROOT, "deploy/prometheus/alert_rules.yml")
    with open(path) as f:
        doc = yaml.safe_load(f)
    rules = doc["groups"][0]["rules"]
    names = {r["alert"] for r in rules}
    assert "MemoptRolloutPaused" in names
    assert "MemoptRolloutFailed" in names
    assert "MemoptCertFailureHigh" in names
    assert "MemoptNodeUnhealthy" in names


def test_alert_rules_use_stage_integers():
    """PAUSED alert uses 6, FAILED uses 7 — matches metrics mapping."""
    path = os.path.join(
        _REPO_ROOT, "deploy/prometheus/alert_rules.yml")
    with open(path) as f:
        doc = yaml.safe_load(f)
    rules = {r["alert"]: r for r in doc["groups"][0]["rules"]}
    assert "memopt_rollout_stage == 6" in rules["MemoptRolloutPaused"]["expr"]
    assert "memopt_rollout_stage == 7" in rules["MemoptRolloutFailed"]["expr"]


# ══════════════════════════════════════════════════════════════════════
#  Prometheus config
# ══════════════════════════════════════════════════════════════════════


def test_prometheus_config_has_rules():
    path = os.path.join(
        _REPO_ROOT, "deploy/prometheus/prometheus.yml")
    with open(path) as f:
        doc = yaml.safe_load(f)
    assert "rule_files" in doc
    assert len(doc["rule_files"]) >= 1
    assert "alert_rules.yml" in doc["rule_files"]


def test_prometheus_config_has_alerting_stub():
    path = os.path.join(
        _REPO_ROOT, "deploy/prometheus/prometheus.yml")
    with open(path) as f:
        doc = yaml.safe_load(f)
    assert "alerting" in doc
    assert "alertmanagers" in doc["alerting"]


def test_prometheus_config_scrape_jobs_preserved():
    """Existing scrape jobs must not be removed by the edit."""
    path = os.path.join(
        _REPO_ROOT, "deploy/prometheus/prometheus.yml")
    with open(path) as f:
        doc = yaml.safe_load(f)
    jobs = {j["job_name"] for j in doc["scrape_configs"]}
    assert "memopt-serving" in jobs
    assert "memopt-control-plane" in jobs


# ══════════════════════════════════════════════════════════════════════
#  Grafana dashboard
# ══════════════════════════════════════════════════════════════════════


def test_grafana_dashboard_valid_json():
    path = os.path.join(
        _REPO_ROOT,
        "deploy/grafana/dashboards/canary_rollout.json")
    with open(path) as f:
        doc = json.load(f)

    assert "panels" in doc
    assert len(doc["panels"]) >= 6

    titles = [p.get("title", "") for p in doc["panels"]]
    assert any("Stage" in t or "stage" in t for t in titles)
    assert any("Error" in t or "error" in t for t in titles)


def test_grafana_dashboard_has_rollout_query():
    path = os.path.join(
        _REPO_ROOT,
        "deploy/grafana/dashboards/canary_rollout.json")
    with open(path) as f:
        content = f.read()
    assert "memopt_rollout_stage" in content
    assert "memopt_rollout_nodes_updated" in content
    assert "memopt_rollout_nodes_total" in content


def test_grafana_dashboard_panels_have_targets():
    path = os.path.join(
        _REPO_ROOT,
        "deploy/grafana/dashboards/canary_rollout.json")
    with open(path) as f:
        doc = json.load(f)
    for panel in doc["panels"]:
        assert "targets" in panel, \
            f"panel {panel.get('title')} missing targets"
        assert len(panel["targets"]) >= 1
        for t in panel["targets"]:
            assert "expr" in t
            assert t["expr"]


def test_grafana_dashboard_has_stage_mappings():
    """The Rollout Stage panel must encode all 9 (-1..7) mappings."""
    path = os.path.join(
        _REPO_ROOT,
        "deploy/grafana/dashboards/canary_rollout.json")
    with open(path) as f:
        content = f.read()
    # Each expected integer should show up as a mapping key
    for key in ("-1", "0", "1", "2", "3", "4", "5", "6", "7"):
        assert f'"{key}"' in content, \
            f"Stage mapping for {key} missing"


def test_grafana_dashboard_refresh_and_uid():
    path = os.path.join(
        _REPO_ROOT,
        "deploy/grafana/dashboards/canary_rollout.json")
    with open(path) as f:
        doc = json.load(f)
    assert doc.get("refresh")  # non-empty string
    assert doc.get("uid") == "memopt-canary"


# ══════════════════════════════════════════════════════════════════════
#  Stage-to-int contract (metrics module)
# ══════════════════════════════════════════════════════════════════════


def test_rollout_stage_metric_values():
    """Mapping matches the contract used by Grafana and Prometheus."""
    assert STAGE_TO_INT[RolloutStage.PENDING]   == 0
    assert STAGE_TO_INT[RolloutStage.STAGE_1]   == 1
    assert STAGE_TO_INT[RolloutStage.STAGE_2]   == 2
    assert STAGE_TO_INT[RolloutStage.STAGE_3]   == 3
    assert STAGE_TO_INT[RolloutStage.STAGE_ALL] == 4
    assert STAGE_TO_INT[RolloutStage.COMPLETED] == 5
    assert STAGE_TO_INT[RolloutStage.PAUSED]    == 6
    assert STAGE_TO_INT[RolloutStage.FAILED]    == 7


def test_stage_to_int_none_is_sentinel():
    assert stage_to_int(None) == NO_ACTIVE_ROLLOUT
    assert NO_ACTIVE_ROLLOUT == -1


def test_stage_to_int_all_enum_values_covered():
    """Every RolloutStage must have a mapping — no missing enum."""
    for stage in RolloutStage:
        assert stage in STAGE_TO_INT, f"Missing: {stage}"


def test_update_rollout_metrics_tolerates_none():
    """update_rollout_metrics(None) never raises."""
    update_rollout_metrics(None)  # must not raise


def test_update_rollout_metrics_tolerates_dict():
    """update_rollout_metrics({...}) never raises on well-formed input."""
    update_rollout_metrics({
        "current_stage": "stage_1",
        "nodes_updated": 2,
        "nodes_total":   10,
    })


def test_update_rollout_metrics_tolerates_unknown_stage():
    """Bogus stage strings don't crash the updater."""
    update_rollout_metrics({
        "current_stage": "nonexistent-stage",
        "nodes_updated": 0,
        "nodes_total":   0,
    })
