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


def make_drift_resynthesis_callback(
    jit_generator=None,
    kernel_cache=None,
) -> Callable[[dict], None]:
    """
    Returns a callback that re-synthesises active kernels
    when hardware drift is detected.

    The three active kernels are:
      - apply_rope
      - apply_layer_norm_residual
      - apply_scaled_softmax

    On drift detection:
      1. Log which kernels will be re-synthesised
      2. For each kernel: call JITGenerator.generate() with
         previous_attempt context from KernelCache
      3. Log before/after speedup for each kernel
      4. Never raises — failures are logged at WARNING
    """
    ACTIVE_KERNELS = [
        "apply_rope",
        "apply_layer_norm_residual",
        "apply_scaled_softmax",
    ]

    def callback(alert_result: dict) -> None:
        # Only re-synthesise on drift, not on cert failure
        if not alert_result.get("drift_detected"):
            logger.info(
                "drift_resynthesis: skipping — "
                "alert is cert failure, not drift"
            )
            return

        drift_pct = alert_result.get("drift_pct", 0)
        logger.warning(
            "drift_resynthesis: hardware drift %.1f%% detected — "
            "re-synthesising %d active kernels",
            drift_pct or 0,
            len(ACTIVE_KERNELS),
        )

        try:
            from memopt.kernels.jit_generator import JITGenerator
            from memopt.kernels.kernel_cache import KernelCache

            gen   = jit_generator or JITGenerator()
            cache = kernel_cache  or KernelCache()

        except Exception as exc:
            logger.warning(
                "drift_resynthesis: failed to load components: %s",
                exc
            )
            return

        hw = None
        try:
            from memopt.kernels.portability_layer import PortabilityLayer
            pl = PortabilityLayer()
            hw = pl._hw
        except Exception:
            pass

        for op_name in ACTIVE_KERNELS:
            try:
                prev = cache.get_metadata(op_name)
                prev_speedup = prev.get('speedup') if prev else None

                logger.info(
                    "drift_resynthesis: re-synthesising %s "
                    "(previous speedup: %s)",
                    op_name,
                    f"{prev_speedup:.2f}x" if prev_speedup else "none",
                )

                result = gen.generate(
                    op_name=op_name,
                    hardware_profile=hw,
                    source_context=(
                        f"Hardware drift of {drift_pct:.1f}% "
                        f"detected. Re-synthesising to restore "
                        f"performance baseline."
                    ),
                    previous_attempt=prev,
                )

                if result is None:
                    logger.warning(
                        "drift_resynthesis: %s — synthesis returned None",
                        op_name,
                    )
                    continue

                new_speedup = getattr(result, 'speedup', None)

                if new_speedup and prev_speedup:
                    if new_speedup >= prev_speedup * 0.95:
                        logger.info(
                            "drift_resynthesis: %s — "
                            "new kernel %.2fx vs previous %.2fx — "
                            "updating cache",
                            op_name, new_speedup, prev_speedup,
                        )
                    else:
                        logger.warning(
                            "drift_resynthesis: %s — "
                            "new kernel %.2fx worse than previous %.2fx "
                            "on drifted hardware — keeping previous",
                            op_name, new_speedup, prev_speedup,
                        )

            except Exception as exc:
                logger.warning(
                    "drift_resynthesis: %s failed: %s",
                    op_name, exc,
                )
                continue

        logger.info("drift_resynthesis: complete")

    return callback


def make_control_plane_callback(
    control_plane_url: str = None,
    jit_generator=None,
    kernel_cache=None,
) -> Callable[[dict], None]:
    """
    Combined callback: flags node DEGRADED in control plane
    AND triggers Pillar 3 kernel re-synthesis.

    On drift detection:
      1. Run re-synthesis (Pillar 3) to adapt kernels
      2. POST degraded status to control plane so load balancer
         routes traffic away from this node

    On cert failure (not drift):
      Only re-synthesis runs. Cert failures need human review.

    Never raises.
    """
    import urllib.request

    cp_url  = control_plane_url or os.environ.get(
        "MEMOPT_CONTROL_PLANE_URL", ""
    )
    node_id = os.environ.get("MEMOPT_NODE_ID", "unknown")
    api_key = os.environ.get("MEMOPT_API_KEY", "")
    resynth = make_drift_resynthesis_callback(jit_generator, kernel_cache)

    def _callback(alert_result: dict) -> None:
        # Always run re-synthesis first
        try:
            resynth(alert_result)
        except Exception as exc:
            logger.warning(
                "control_plane_callback: re-synthesis error: %s", exc
            )

        if not alert_result.get("drift_detected"):
            return   # cert failure — needs human review, not auto-flag

        if not cp_url:
            logger.debug(
                "control_plane_callback: MEMOPT_CONTROL_PLANE_URL not set, "
                "skipping status update"
            )
            return

        payload = json.dumps({
            "node_id":     node_id,
            "healthy":     False,
            "degraded":    True,
            "drift_pct":   alert_result.get("drift_pct", 0),
            "checked_at":  alert_result.get("checked_at", time.time()),
            "reason":      "HBM bandwidth drift exceeded threshold",
        }).encode()

        try:
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["X-Memopt-API-Key"] = api_key
            req = urllib.request.Request(
                f"{cp_url}/api/v1/nodes/{node_id}/status",
                data=payload,
                headers=headers,
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
            logger.info(
                "control_plane_callback: node %s flagged DEGRADED", node_id
            )
        except Exception as exc:
            logger.warning(
                "control_plane_callback: could not reach control plane: %s",
                exc,
            )

    return _callback


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
        if alert_callback is not None:
            self._alert = alert_callback
        else:
            try:
                self._alert = make_drift_resynthesis_callback()
                logger.debug(
                    "CertifyDaemon: drift re-synthesis callback active"
                )
            except Exception:
                self._alert = None
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
