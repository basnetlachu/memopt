"""
Hardware drift detector for Pillar 6.

Tracks achieved bandwidth percentage over time per node.
When the rolling average drops by more than DRIFT_THRESHOLD_PCT
compared to the established baseline, the node is flagged as drifted.

Baseline: average of the first BASELINE_WINDOW measurements.
Rolling average: last ROLLING_WINDOW measurements.
Drift: (baseline_avg - rolling_avg) / baseline_avg > threshold.

Lifecycle:
  Created by certify_daemon.py (or standalone).
  record(bw_pct) called after each certification run.
  is_drifted() checked before writing node status.
  reset() called when hardware is replaced or driver updated.

Storage: measurements written to ~/.memopt/drift_history.json
Atomic write: tmp file + rename — safe on crash.

Environment variables:
  MEMOPT_DRIFT_THRESHOLD_PCT  float  default 5.0
  MEMOPT_DRIFT_BASELINE_N     int    default 7   (measurements)
  MEMOPT_DRIFT_ROLLING_N      int    default 3   (measurements)
  MEMOPT_DRIFT_HISTORY_PATH   path   default ~/.memopt/drift_history.json
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_THRESHOLD_PCT  = float(os.environ.get("MEMOPT_DRIFT_THRESHOLD_PCT", "5.0"))
_BASELINE_N     = int(os.environ.get("MEMOPT_DRIFT_BASELINE_N",     "7"))
_ROLLING_N      = int(os.environ.get("MEMOPT_DRIFT_ROLLING_N",      "3"))
_HISTORY_PATH   = Path(os.path.expanduser(
    os.environ.get(
        "MEMOPT_DRIFT_HISTORY_PATH",
        "~/.memopt/drift_history.json"
    )
))


class DriftDetector:
    """
    Tracks bandwidth percentage measurements and detects drift.

    A node is considered drifted when its recent performance
    has dropped more than DRIFT_THRESHOLD_PCT below its baseline.

    All measurements are in percentage of theoretical peak bandwidth
    as reported by SiliconCertificate.throughput_tests[].pct_of_peak.
    """

    def __init__(
        self,
        node_id:        str,
        threshold_pct:  float = _THRESHOLD_PCT,
        baseline_n:     int   = _BASELINE_N,
        rolling_n:      int   = _ROLLING_N,
        history_path:   Path  = _HISTORY_PATH,
    ) -> None:
        self._node_id       = node_id
        self._threshold_pct = threshold_pct
        self._baseline_n    = baseline_n
        self._rolling_n     = rolling_n
        self._history_path  = history_path
        self._measurements: List[dict] = []   # {bw_pct, timestamp}
        self._load()

    # ── Public API ────────────────────────────────────────────────

    def record(self, bw_pct: float) -> None:
        """
        Record one bandwidth percentage measurement.
        Called after each certification run.
        Persists to disk atomically.
        """
        if bw_pct is None or bw_pct <= 0:
            logger.debug(
                "DriftDetector: skipping invalid bw_pct=%s", bw_pct
            )
            return

        entry = {"bw_pct": round(bw_pct, 2), "timestamp": time.time()}
        self._measurements.append(entry)
        self._save()

        logger.info(
            "DriftDetector: recorded bw_pct=%.1f%% "
            "(n=%d baseline=%.1f%% rolling=%.1f%%)",
            bw_pct,
            len(self._measurements),
            self.baseline_avg() or 0,
            self.rolling_avg() or 0,
        )

    def is_drifted(self) -> bool:
        """
        True when rolling average has dropped more than
        threshold_pct below the baseline average.

        Returns False when not enough measurements to decide.
        """
        baseline = self.baseline_avg()
        rolling  = self.rolling_avg()

        if baseline is None or rolling is None:
            return False   # not enough data yet

        drop_pct = (baseline - rolling) / baseline * 100
        drifted  = drop_pct > self._threshold_pct

        if drifted:
            logger.error(
                "DriftDetector: DRIFT DETECTED node=%s "
                "baseline=%.1f%% rolling=%.1f%% drop=%.1f%% "
                "threshold=%.1f%%",
                self._node_id, baseline, rolling,
                drop_pct, self._threshold_pct,
            )
        return drifted

    def drift_pct(self) -> Optional[float]:
        """
        Current drop from baseline in percentage points.
        None when not enough data.
        Positive = degraded. Negative = improved.
        """
        baseline = self.baseline_avg()
        rolling  = self.rolling_avg()
        if baseline is None or rolling is None:
            return None
        return round((baseline - rolling) / baseline * 100, 2)

    def baseline_avg(self) -> Optional[float]:
        """Average of first baseline_n measurements. None if < baseline_n."""
        if len(self._measurements) < self._baseline_n:
            return None
        values = [m["bw_pct"] for m in self._measurements[:self._baseline_n]]
        return round(sum(values) / len(values), 2)

    def rolling_avg(self) -> Optional[float]:
        """Average of last rolling_n measurements. None if < rolling_n."""
        if len(self._measurements) < self._rolling_n:
            return None
        values = [m["bw_pct"] for m in self._measurements[-self._rolling_n:]]
        return round(sum(values) / len(values), 2)

    def reset(self) -> None:
        """
        Clear all measurements.
        Call when hardware is replaced or driver is updated.
        The baseline will be re-established from the next
        baseline_n measurements.
        """
        self._measurements = []
        self._save()
        logger.info(
            "DriftDetector: history reset for node=%s", self._node_id
        )

    def stats(self) -> dict:
        return {
            "node_id":        self._node_id,
            "n_measurements": len(self._measurements),
            "baseline_avg":   self.baseline_avg(),
            "rolling_avg":    self.rolling_avg(),
            "drift_pct":      self.drift_pct(),
            "is_drifted":     self.is_drifted(),
            "threshold_pct":  self._threshold_pct,
            "baseline_n":     self._baseline_n,
            "rolling_n":      self._rolling_n,
        }

    # ── Persistence ───────────────────────────────────────────────

    def _save(self) -> None:
        """Atomic write to drift_history.json."""
        try:
            self._history_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._history_path.with_suffix(".tmp")
            payload = {
                "node_id":      self._node_id,
                "measurements": self._measurements,
            }
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.rename(self._history_path)
        except Exception as exc:
            logger.debug("DriftDetector: save failed: %s", exc)

    def _load(self) -> None:
        """Load existing measurements on startup."""
        try:
            if self._history_path.exists():
                data = json.loads(self._history_path.read_text())
                if data.get("node_id") == self._node_id:
                    self._measurements = data.get("measurements", [])
                    logger.info(
                        "DriftDetector: loaded %d measurements for %s",
                        len(self._measurements), self._node_id
                    )
        except Exception as exc:
            logger.debug("DriftDetector: load failed: %s", exc)
            self._measurements = []
