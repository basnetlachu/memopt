"""
Continuous silicon certification daemon.

Runs run_certification() on a configurable schedule.
On failure: logs at ERROR, writes node status to disk,
fires an optional alert callback.

The existing one-shot `memopt certify` CLI is unchanged.
This daemon runs alongside it — they share the same
run_certification() function and produce identical certificates.

Environment variables:
  MEMOPT_CERTIFY_INTERVAL_H   float  hours between runs  (default 24.0)
  MEMOPT_CERTIFY_ON_STARTUP   "1"    run immediately on start (default "1")
  MEMOPT_NODE_STATUS_PATH     path   node status JSON file
                                     default: ~/.memopt/node_status.json

Usage:
  from memopt.kernels.certify_daemon import CertifyDaemon
  daemon = CertifyDaemon()
  daemon.start()                          # non-blocking daemon thread
  daemon.stop()                           # graceful shutdown
  daemon.last_result()                    # most recent certificate or None
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_INTERVAL_H  = float(os.environ.get("MEMOPT_CERTIFY_INTERVAL_H", "24.0"))
_ON_STARTUP  = os.environ.get("MEMOPT_CERTIFY_ON_STARTUP", "1") == "1"
_STATUS_PATH = Path(
    os.environ.get(
        "MEMOPT_NODE_STATUS_PATH",
        os.path.expanduser("~/.memopt/node_status.json")
    )
)


class CertifyDaemon:
    """
    Background daemon that re-runs the silicon certification suite
    on a fixed schedule and maintains a node status file.

    Thread safety: all state protected by threading.RLock.
    The run loop uses threading.Event for clean shutdown.
    """

    def __init__(
        self,
        interval_hours:  float                                = _INTERVAL_H,
        on_startup:      bool                                 = _ON_STARTUP,
        alert_callback:  Optional[Callable[[dict], None]]     = None,
        status_path:     Path                                 = _STATUS_PATH,
        node_id:         str                                  = "",
    ) -> None:
        self._interval      = interval_hours * 3600
        self._on_startup    = on_startup
        self._alert         = alert_callback
        self._status_path   = status_path
        self._node_id       = node_id or os.environ.get(
            "MEMOPT_NODE_ID", ""
        )
        self._lock          = threading.RLock()
        self._stop_event    = threading.Event()
        self._thread:       Optional[threading.Thread] = None
        self._last_result:  Optional[dict]             = None
        self._certified:    bool                       = False

    def start(self) -> None:
        """Start the daemon thread. Non-blocking."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                daemon=True,
                name="memopt-certify-daemon",
            )
            self._thread.start()
            logger.info(
                "CertifyDaemon: started (interval=%.1fh node=%s)",
                self._interval / 3600, self._node_id or "hostname"
            )

    def stop(self) -> None:
        """Signal the daemon to stop. Does not block."""
        self._stop_event.set()
        logger.info("CertifyDaemon: stop requested")

    def last_result(self) -> Optional[dict]:
        with self._lock:
            return dict(self._last_result) if self._last_result else None

    def is_certified(self) -> bool:
        with self._lock:
            return self._certified

    def _run_loop(self) -> None:
        if self._on_startup:
            self._run_once()

        while not self._stop_event.wait(timeout=self._interval):
            self._run_once()

    def _run_once(self) -> None:
        logger.info("CertifyDaemon: running certification suite...")
        try:
            from memopt.kernels.certification import run_certification
            cert    = run_certification(node_id=self._node_id)
            result  = asdict(cert)
            passed  = cert.all_passed

            with self._lock:
                self._last_result = result
                self._certified   = passed

            self._write_status(passed, result)

            if passed:
                logger.info(
                    "CertifyDaemon: PASS (node=%s signature=%s)",
                    self._node_id, cert.signature_status
                )
            else:
                failed_count = sum(
                    1 for t in cert.correctness_tests if not t.passed
                )
                logger.error(
                    "CertifyDaemon: FAIL — %d test(s) failed on %s. "
                    "Node marked as uncertified.",
                    failed_count, self._node_id
                )
                if self._alert:
                    try:
                        self._alert(result)
                    except Exception as exc:
                        logger.debug(
                            "CertifyDaemon: alert callback raised: %s", exc
                        )

        except Exception as exc:
            logger.error(
                "CertifyDaemon: certification raised an exception: %s",
                exc, exc_info=True
            )
            with self._lock:
                self._certified = False
            self._write_status(False, {"error": str(exc)})

    def _write_status(self, certified: bool, detail: dict) -> None:
        """
        Write node status to disk.
        Other processes (load balancers, monitoring) read this file.
        Write is atomic: write to .tmp then rename.
        """
        try:
            self._status_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._status_path.with_suffix(".tmp")
            payload = {
                "node_id":    self._node_id,
                "certified":  certified,
                "checked_at": time.time(),
                "detail":     detail,
            }
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.rename(self._status_path)
        except Exception as exc:
            logger.debug(
                "CertifyDaemon: could not write status: %s", exc
            )
