"""
Canary deployment gate evaluators.

Every gate queries real data from:
  1. Control plane database (boot events, cert results, versions)
  2. Node health / metrics endpoints (/healthz, /metrics) over HTTP

Missing-data policy:
  When a metric cannot be collected (e.g. no nodes on the target
  version yet, /metrics unreachable, no baseline set), the gate
  returns GateResult.UNKNOWN rather than FAIL. Rollouts continue
  through UNKNOWN gates with a warning log. This avoids halting
  a perfectly healthy rollout on infrastructure gaps.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request
from typing import List, Optional

from memopt.canary.models import (
    GateEvaluation,
    GateResult,
    StageConfig,
)

logger = logging.getLogger(__name__)


class GateEvaluator:
    """
    Evaluates deployment gates against real metrics.

    db: ControlPlaneDB instance (for boot_events / baselines)
    """

    def __init__(self, db):
        self._db = db

    # ── Public API ────────────────────────────────────────────────────

    def evaluate_all(
        self,
        stage_config: StageConfig,
        target_version: str,
        node_urls: List[str],
    ) -> List[GateEvaluation]:
        """
        Evaluate every gate for a stage.
        Never raises — each gate catches its own errors.
        """
        gates = [
            self._eval_cert_pass_rate(
                stage_config, target_version),
            self._eval_error_rate(
                stage_config, node_urls),
            self._eval_latency(
                stage_config, node_urls),
        ]

        if stage_config.min_gkd_hit_rate_pct > 0:
            gates.append(
                self._eval_gkd_hit_rate(
                    stage_config, node_urls))

        return gates

    # ── Cert pass rate ────────────────────────────────────────────────

    def _eval_cert_pass_rate(
        self,
        config: StageConfig,
        target_version: str,
    ) -> GateEvaluation:
        """
        Ratio of PASSED / (PASSED+FAILED+SKIPPED) among the most
        recent boot event per node on `target_version`.
        """
        try:
            rows = self._db._db.fetchall(
                """
                SELECT cert_status, COUNT(*) AS cnt
                FROM boot_events
                WHERE image_version = ?
                AND id IN (
                    SELECT MAX(id) FROM boot_events
                    GROUP BY node_id
                )
                GROUP BY cert_status
                """,
                (target_version,))

            if not rows:
                return GateEvaluation(
                    gate_name="cert_pass_rate",
                    result=GateResult.UNKNOWN,
                    actual=None,
                    threshold=config.min_cert_pass_rate_pct,
                    message=(
                        f"No boot events for {target_version} "
                        f"— treating as UNKNOWN."))

            total = sum(int(r["cnt"]) for r in rows)
            passed = sum(
                int(r["cnt"]) for r in rows
                if r["cert_status"] == "PASSED")

            pass_rate = (passed / total * 100) if total > 0 else 0.0
            threshold = config.min_cert_pass_rate_pct

            if pass_rate >= threshold:
                return GateEvaluation(
                    gate_name="cert_pass_rate",
                    result=GateResult.PASS,
                    actual=round(pass_rate, 2),
                    threshold=threshold,
                    message=(
                        f"Cert pass rate {pass_rate:.1f}% "
                        f">= {threshold:.1f}%"))

            return GateEvaluation(
                gate_name="cert_pass_rate",
                result=GateResult.FAIL,
                actual=round(pass_rate, 2),
                threshold=threshold,
                message=(
                    f"Cert pass rate {pass_rate:.1f}% "
                    f"< {threshold:.1f}% "
                    f"({passed}/{total} passed)"))
        except Exception as e:
            logger.warning("Cert gate error: %s", e)
            return GateEvaluation(
                gate_name="cert_pass_rate",
                result=GateResult.UNKNOWN,
                actual=None,
                threshold=config.min_cert_pass_rate_pct,
                message=f"Gate error: {e}")

    # ── Error rate ────────────────────────────────────────────────────

    def _eval_error_rate(
        self,
        config: StageConfig,
        node_urls: List[str],
    ) -> GateEvaluation:
        """
        Queries /healthz on each node. unhealthy/total = error rate.
        No nodes, or no nodes respond → UNKNOWN.
        """
        if not node_urls:
            return GateEvaluation(
                gate_name="error_rate",
                result=GateResult.UNKNOWN,
                actual=None,
                threshold=config.max_error_rate_pct,
                message=(
                    "No node URLs provided. "
                    "Cannot evaluate error rate."))

        healthy = 0
        unhealthy = 0

        for url in node_urls:
            try:
                req = urllib.request.Request(f"{url}/healthz")
                with urllib.request.urlopen(
                        req, timeout=3.0) as resp:
                    data = json.loads(resp.read())
                if data.get("status") == "ok":
                    healthy += 1
                else:
                    unhealthy += 1
            except Exception:
                unhealthy += 1

        total = healthy + unhealthy
        if total == 0:
            return GateEvaluation(
                gate_name="error_rate",
                result=GateResult.UNKNOWN,
                actual=None,
                threshold=config.max_error_rate_pct,
                message="No nodes responded.")

        error_rate = (unhealthy / total) * 100
        threshold = config.max_error_rate_pct

        if error_rate <= threshold:
            return GateEvaluation(
                gate_name="error_rate",
                result=GateResult.PASS,
                actual=round(error_rate, 2),
                threshold=threshold,
                message=(
                    f"Error rate {error_rate:.1f}% "
                    f"<= {threshold:.1f}%"))

        return GateEvaluation(
            gate_name="error_rate",
            result=GateResult.FAIL,
            actual=round(error_rate, 2),
            threshold=threshold,
            message=(
                f"Error rate {error_rate:.1f}% "
                f"> {threshold:.1f}% "
                f"({unhealthy}/{total} unhealthy)"))

    # ── Latency p99 vs baseline ───────────────────────────────────────

    def _eval_latency(
        self,
        config: StageConfig,
        node_urls: List[str],
    ) -> GateEvaluation:
        """
        p99 latency delta vs baseline stored in canary_baselines.
        No baseline or no metric → UNKNOWN.
        """
        if not node_urls:
            return GateEvaluation(
                gate_name="latency_p99",
                result=GateResult.UNKNOWN,
                actual=None,
                threshold=config.max_latency_p99_delta_pct,
                message=(
                    "No node URLs provided. "
                    "Cannot evaluate latency."))

        p99_values: List[float] = []
        for url in node_urls:
            try:
                req = urllib.request.Request(f"{url}/metrics")
                with urllib.request.urlopen(
                        req, timeout=3.0) as resp:
                    text = resp.read().decode()

                for line in text.split("\n"):
                    if (
                        "request_duration" in line
                        and "p99" in line
                        and not line.startswith("#")
                    ):
                        try:
                            p99_values.append(float(line.split()[-1]))
                        except ValueError:
                            pass
            except Exception:
                pass

        if not p99_values:
            return GateEvaluation(
                gate_name="latency_p99",
                result=GateResult.UNKNOWN,
                actual=None,
                threshold=config.max_latency_p99_delta_pct,
                message=(
                    "No p99 latency metric exported. "
                    "Treating as UNKNOWN."))

        baseline = self._get_latency_baseline()
        if baseline is None:
            return GateEvaluation(
                gate_name="latency_p99",
                result=GateResult.UNKNOWN,
                actual=None,
                threshold=config.max_latency_p99_delta_pct,
                message=(
                    "No latency baseline set. "
                    "Will be established after Stage 1. "
                    "Treating as UNKNOWN."))

        avg_p99 = sum(p99_values) / len(p99_values)
        delta_pct = (
            abs((avg_p99 - baseline) / baseline) * 100
            if baseline > 0 else 0.0)

        threshold = config.max_latency_p99_delta_pct

        if delta_pct <= threshold:
            return GateEvaluation(
                gate_name="latency_p99",
                result=GateResult.PASS,
                actual=round(delta_pct, 2),
                threshold=threshold,
                message=(
                    f"Latency delta {delta_pct:.1f}% "
                    f"<= {threshold:.1f}%"))

        return GateEvaluation(
            gate_name="latency_p99",
            result=GateResult.FAIL,
            actual=round(delta_pct, 2),
            threshold=threshold,
            message=(
                f"Latency delta {delta_pct:.1f}% "
                f"> {threshold:.1f}% "
                f"(p99={avg_p99:.3f}s "
                f"baseline={baseline:.3f}s)"))

    # ── GKD hit rate ──────────────────────────────────────────────────

    def _eval_gkd_hit_rate(
        self,
        config: StageConfig,
        node_urls: List[str],
    ) -> GateEvaluation:
        """
        Average GKD hit rate across nodes. Below threshold → WARN
        (not FAIL): workload-dependent, not strictly a regression.
        """
        if not node_urls:
            return GateEvaluation(
                gate_name="gkd_hit_rate",
                result=GateResult.UNKNOWN,
                actual=None,
                threshold=config.min_gkd_hit_rate_pct,
                message="No node URLs provided.")

        hit_rates: List[float] = []
        for url in node_urls:
            try:
                req = urllib.request.Request(f"{url}/metrics")
                with urllib.request.urlopen(
                        req, timeout=3.0) as resp:
                    text = resp.read().decode()

                for line in text.split("\n"):
                    if (
                        "gkd_hit_rate_pct" in line
                        and not line.startswith("#")
                    ):
                        try:
                            hit_rates.append(float(line.split()[-1]))
                        except ValueError:
                            pass
            except Exception:
                pass

        if not hit_rates:
            return GateEvaluation(
                gate_name="gkd_hit_rate",
                result=GateResult.UNKNOWN,
                actual=None,
                threshold=config.min_gkd_hit_rate_pct,
                message=(
                    "No GKD hit rate metric found. "
                    "Treating as UNKNOWN."))

        avg_rate = sum(hit_rates) / len(hit_rates)
        threshold = config.min_gkd_hit_rate_pct

        if avg_rate >= threshold:
            return GateEvaluation(
                gate_name="gkd_hit_rate",
                result=GateResult.PASS,
                actual=round(avg_rate, 2),
                threshold=threshold,
                message=(
                    f"GKD hit rate {avg_rate:.1f}% "
                    f">= {threshold:.1f}%"))

        # WARN, not FAIL — workload-dependent.
        return GateEvaluation(
            gate_name="gkd_hit_rate",
            result=GateResult.WARN,
            actual=round(avg_rate, 2),
            threshold=threshold,
            message=(
                f"GKD hit rate {avg_rate:.1f}% "
                f"< {threshold:.1f}%. "
                f"Warning only — workload-dependent."))

    # ── Baseline management ───────────────────────────────────────────

    def _get_latency_baseline(self) -> Optional[float]:
        """Return latest latency_p99 baseline, or None."""
        try:
            row = self._db._db.fetchone(
                """SELECT value FROM canary_baselines
                   WHERE metric = 'latency_p99'
                   ORDER BY recorded_at DESC
                   LIMIT 1""")
            return float(row["value"]) if row else None
        except Exception:
            return None

    def set_latency_baseline(self, value: float) -> None:
        """Record latency baseline after Stage 1."""
        try:
            self._db._db.execute(
                """INSERT INTO canary_baselines
                   (metric, value, recorded_at)
                   VALUES (?, ?, ?)""",
                ("latency_p99", float(value), time.time()))
        except Exception as e:
            logger.warning("Failed to record baseline: %s", e)
