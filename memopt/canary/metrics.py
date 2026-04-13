"""
Canary rollout Prometheus metrics contract.

The stage→int mapping below is the single source of truth shared by:

  - The canary controller (writes the gauge)
  - Prometheus alert rules (deploy/prometheus/alert_rules.yml)
  - The Grafana dashboard (deploy/grafana/dashboards/canary_rollout.json)

If you change the mapping here, update both Prometheus and Grafana
in the same change — the dashboard's `value` mappings are hand-keyed
to these integers.

NOTE: Exposition to Prometheus via `prometheus_client` is optional.
If the package is present, `update_rollout_metrics()` writes to a
module-level gauge registry. If absent, the function is a no-op so
this module remains importable on any deployment.
"""
from __future__ import annotations

import logging
from typing import Optional

from memopt.canary.models import RolloutStage

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Stage → int contract
# ─────────────────────────────────────────────

STAGE_TO_INT = {
    RolloutStage.PENDING:   0,
    RolloutStage.STAGE_1:   1,
    RolloutStage.STAGE_2:   2,
    RolloutStage.STAGE_3:   3,
    RolloutStage.STAGE_ALL: 4,
    RolloutStage.COMPLETED: 5,
    RolloutStage.PAUSED:    6,
    RolloutStage.FAILED:    7,
}

# When there is no active rollout, the gauge reads -1 so Grafana's
# "No Active Rollout" green mapping activates.
NO_ACTIVE_ROLLOUT = -1


def stage_to_int(stage: Optional[RolloutStage]) -> int:
    """Return the integer encoding for a rollout stage, or -1 if None."""
    if stage is None:
        return NO_ACTIVE_ROLLOUT
    return STAGE_TO_INT.get(stage, NO_ACTIVE_ROLLOUT)


# ─────────────────────────────────────────────
# Optional prometheus_client gauges
# ─────────────────────────────────────────────

try:
    from prometheus_client import Gauge
    _HAVE_PROM = True
except ImportError:
    _HAVE_PROM = False
    Gauge = None  # type: ignore

if _HAVE_PROM:
    _rollout_stage_g = Gauge(
        "memopt_rollout_stage",
        "Current rollout stage as integer "
        "(-1=none, 0=pending, 1..4=stages, "
        "5=completed, 6=paused, 7=failed)",
    )
    _rollout_nodes_updated_g = Gauge(
        "memopt_rollout_nodes_updated",
        "Number of nodes updated in the active rollout",
    )
    _rollout_nodes_total_g = Gauge(
        "memopt_rollout_nodes_total",
        "Total nodes in scope for the active rollout",
    )
else:
    _rollout_stage_g = None
    _rollout_nodes_updated_g = None
    _rollout_nodes_total_g = None


def update_rollout_metrics(status: Optional[dict]) -> None:
    """
    Update Prometheus gauges from a canary status dict.

    `status` is the result of `CanaryController.get_status()`.
    None means "no active rollout" — gauges are set to sentinel values.
    """
    if not _HAVE_PROM:
        return

    try:
        if status is None:
            _rollout_stage_g.set(NO_ACTIVE_ROLLOUT)
            _rollout_nodes_updated_g.set(0)
            _rollout_nodes_total_g.set(0)
            return

        stage_str = status.get("current_stage", "")
        # Convert back to enum for mapping; tolerate unknown strings.
        for enum_val in RolloutStage:
            if enum_val.value == stage_str:
                _rollout_stage_g.set(STAGE_TO_INT[enum_val])
                break
        else:
            _rollout_stage_g.set(NO_ACTIVE_ROLLOUT)

        _rollout_nodes_updated_g.set(
            int(status.get("nodes_updated", 0)))
        _rollout_nodes_total_g.set(
            int(status.get("nodes_total", 0)))
    except Exception as e:
        logger.debug("update_rollout_metrics failed: %s", e)
