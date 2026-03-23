"""
Continuous certification daemon for Pillar 6.

Runs run_certification() on a configurable schedule.
Feeds bandwidth results into DriftDetector after each run.
Writes node status atomically to disk.
Alerts when certification fails OR hardware drift is detected.

Environment variables:
  MEMOPT_CERTIFY_INTERVAL_H   float  hours between runs (default 24.0)
  MEMOPT_CERTIFY_ON_STARTUP   "1"    run immediately on start (default "1")
  MEMOPT_NODE_STATUS_PATH     path   default ~/.memopt/node_status.json
  MEMOPT_NODE_ID              str    node identifier
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_INTERVAL_H  = float(os.environ.get("MEMOPT_CERTIFY_INTERVAL_H", "24.0"))
_ON_STARTUP  = os.environ.get("MEMOPT_CERTIFY_ON_STARTUP", "1") == "1"
_STATUS_PATH = Path(os.path.expanduser(
    os.environ.get(
        "MEMOPT_NODE_STATUS_PATH",
        "~/.memopt/node_status.json"
    )
))


class CertifyDaemon:
    """
    Background daemon: runs certification on schedule,
    detects drift, writes signed node status to disk.
    """

    def __init__(
        self,
        interval_hours: float = _INTERVAL_H,
        on_startup:     bool  = _ON_STARTUP,
        alert_callback: Optional[Callable[[dict], None]] = None,
        status_path:    Path  = _STATUS_PATH,
        node_id:        str   = "",
    ) -> None:
        import socket
        self._interval   = interval_hours * 3600
        self._on_startup = on_startup
        self._alert      = alert_callback
        self._status     = status_path
        self._node_id    = node_id or os.environ.get(
            "MEMOPT_NODE_ID", socket.gethostname()
        )
        self._stop       = threading.Event()
        self._lock       = threading.RLock()
        self._thread:    Optional[threading.Thread] = None
        self._last:      Optional[dict] = None
        self._certified: bool = False

        # Drift detector — persists measurements across restarts
        from memopt.kernels.drift_detector import DriftDetector
        self._drift = DriftDetector(node_id=self._node_id)

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._loop, daemon=True,
                name="memopt-certify-daemon"
            )
            self._thread.start()
        logger.info(
            "CertifyDaemon: started interval=%.1fh node=%s",
            self._interval / 3600, self._node_id
        )

    def stop(self) -> None:
        self._stop.set()

    def last_result(self) -> Optional[dict]:
        with self._lock:
            return dict(self._last) if self._last else None

    def is_certified(self) -> bool:
        with self._lock:
            return self._certified

    def drift_stats(self) -> dict:
        return self._drift.stats()

    def _loop(self) -> None:
        if self._on_startup:
            self._run_once()
        while not self._stop.wait(timeout=self._interval):
            self._run_once()

    def _run_once(self) -> None:
        logger.info("CertifyDaemon: running certification...")
        try:
            from memopt.kernels.certification import run_certification
            from dataclasses import asdict
            cert   = run_certification(node_id=self._node_id)
            result = asdict(cert)
            passed = cert.all_passed

            # Feed bandwidth into drift detector
            for t in cert.throughput_tests:
                pct = t.pct_of_peak if hasattr(t, "pct_of_peak") else None
                if pct is not None and pct > 0:
                    self._drift.record(pct)
                    break   # use first throughput result

            drifted = self._drift.is_drifted()
            healthy = passed and not drifted

            with self._lock:
                self._last      = result
                self._certified = healthy

            self._write_status(healthy, passed, drifted, result)

            if not passed:
                failed_count = sum(
                    1 for t in cert.correctness_tests if not t.passed
                )
                logger.error(
                    "CertifyDaemon: FAIL — %d test(s) failed",
                    failed_count
                )
                self._fire_alert(result)
            elif drifted:
                drift_pct = self._drift.drift_pct()
                logger.error(
                    "CertifyDaemon: DRIFT — bandwidth dropped "
                    "%.1f%% below baseline on node=%s",
                    drift_pct or 0, self._node_id
                )
                result["drift_detected"] = True
                result["drift_pct"]      = drift_pct
                self._fire_alert(result)
            else:
                logger.info(
                    "CertifyDaemon: PASS drift=ok node=%s",
                    self._node_id
                )

        except Exception as exc:
            logger.error(
                "CertifyDaemon: exception: %s", exc, exc_info=True
            )
            with self._lock:
                self._certified = False
            self._write_status(False, False, False, {"error": str(exc)})

    def _fire_alert(self, result: dict) -> None:
        if self._alert:
            try:
                self._alert(result)
            except Exception as exc:
                logger.debug("CertifyDaemon: alert callback: %s", exc)

    def _write_status(self, healthy: bool, certified: bool,
                      drifted: bool, detail: dict) -> None:
        try:
            self._status.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._status.with_suffix(".tmp")
            payload = {
                "node_id":    self._node_id,
                "healthy":    healthy,
                "certified":  certified,
                "drifted":    drifted,
                "drift":      self._drift.stats(),
                "checked_at": time.time(),
                "detail":     detail,
            }
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.rename(self._status)
        except Exception as exc:
            logger.debug("CertifyDaemon: write status failed: %s", exc)
