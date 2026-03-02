"""
Alert notifier for memopt drift detection.

Delivers DriftAlert notifications through multiple channels:
  - Python logging (always, cannot be disabled)
  - Prometheus textfile metrics (always, if metrics dir is writable)
  - Webhook POST (optional — set MEMOPT_ALERT_WEBHOOK_URL)
  - Email (optional — set MEMOPT_ALERT_SMTP_* env vars)

Design principle: notify() NEVER raises an exception.
A failed notification channel logs a warning and continues.
The daemon must not crash because Slack is down.
"""

import json
import logging
import os
import smtplib
import socket
import time
import urllib.error
import urllib.request
from email.message import EmailMessage
from pathlib import Path
from typing import Optional

from memopt.alerts.alert_store import DriftAlert

log = logging.getLogger(__name__)

# Prometheus textfile written by node_exporter --collector.textfile.directory
_DEFAULT_METRICS_DIR = Path.home() / ".memopt" / "metrics"
_DRIFT_PROM_FILE = "drift_alerts.prom"

# Environment variable names
_ENV_WEBHOOK = "MEMOPT_ALERT_WEBHOOK_URL"
_ENV_SMTP_HOST = "MEMOPT_ALERT_SMTP_HOST"
_ENV_SMTP_PORT = "MEMOPT_ALERT_SMTP_PORT"
_ENV_SMTP_USER = "MEMOPT_ALERT_SMTP_USER"
_ENV_SMTP_PASS = "MEMOPT_ALERT_SMTP_PASS"
_ENV_ALERT_FROM = "MEMOPT_ALERT_EMAIL_FROM"
_ENV_ALERT_TO = "MEMOPT_ALERT_EMAIL_TO"


class AlertNotifier:
    """
    Fan-out notifier: log → Prometheus textfile → webhook → email.

    Usage (inside ZeroTouchDaemon):
        notifier = AlertNotifier()
        for alert in detector.check_all(processes):
            notifier.notify(alert)

    All channels are optional except logging.
    Channels that are not configured are silently skipped.
    Channels that fail are logged as warnings, never as exceptions.
    """

    def __init__(
        self,
        metrics_dir: Optional[Path] = None,
        webhook_url: Optional[str] = None,
        smtp_host: Optional[str] = None,
        smtp_port: int = 587,
        smtp_user: Optional[str] = None,
        smtp_pass: Optional[str] = None,
        alert_from: Optional[str] = None,
        alert_to: Optional[str] = None,
    ):
        self._metrics_dir = metrics_dir or _DEFAULT_METRICS_DIR
        # Constructor args take priority; fall back to env vars
        self._webhook_url = webhook_url or os.environ.get(_ENV_WEBHOOK)
        self._smtp_host = smtp_host or os.environ.get(_ENV_SMTP_HOST)
        self._smtp_port = int(os.environ.get(_ENV_SMTP_PORT, smtp_port))
        self._smtp_user = smtp_user or os.environ.get(_ENV_SMTP_USER)
        self._smtp_pass = smtp_pass or os.environ.get(_ENV_SMTP_PASS)
        self._alert_from = alert_from or os.environ.get(_ENV_ALERT_FROM)
        self._alert_to = alert_to or os.environ.get(_ENV_ALERT_TO)

        # In-memory counters for textfile generation
        self._counters: dict = {"info": 0, "warning": 0, "critical": 0}
        # Active alert gauges: (node_name, model_family, severity) → 0/1
        self._active_gauges: dict = {}

        self._metrics_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Public API                                                           #
    # ------------------------------------------------------------------ #

    def notify(self, alert: DriftAlert) -> None:
        """
        Deliver alert through all configured channels.
        Never raises — failures are logged at WARNING level.
        """
        try:
            self._log_alert(alert)
        except Exception as exc:  # pragma: no cover
            log.warning("Log channel failed: %s", exc)

        try:
            self._update_prometheus(alert)
        except Exception as exc:
            log.warning("Prometheus textfile update failed: %s", exc)

        if self._webhook_url:
            try:
                self._send_webhook(alert)
            except Exception as exc:
                log.warning("Webhook notification failed: %s", exc)

        if self._smtp_host and self._alert_to:
            try:
                self._send_email(alert)
            except Exception as exc:
                log.warning("Email notification failed: %s", exc)

    def mark_resolved(self, node_name: str, model_family: str, severity: str) -> None:
        """
        Clear the active gauge for a resolved alert.
        Call this after AlertStore.resolve_alert().
        """
        key = (node_name, model_family, severity)
        self._active_gauges.pop(key, None)
        self._write_prom_file()

    # ------------------------------------------------------------------ #
    #  Channel implementations                                             #
    # ------------------------------------------------------------------ #

    def _log_alert(self, alert: DriftAlert) -> None:
        """Always log — at WARNING or CRITICAL level depending on severity."""
        level = logging.CRITICAL if alert.severity == "critical" else logging.WARNING
        log.log(
            level,
            "DRIFT ALERT [%s] PID=%d model=%s node=%s "
            "util %.1f%% → %.1f%% (drop %.1f%%) | %s",
            alert.severity.upper(),
            alert.pid,
            alert.model_family,
            alert.node_name,
            alert.baseline_util_pct,
            alert.current_util_pct,
            alert.util_drop_pct,
            alert.recommended_action,
        )

    def _update_prometheus(self, alert: DriftAlert) -> None:
        """
        Update in-memory counters + gauges, then rewrite the .prom file.
        Counters monotonically increase (persist across reloads via the file).
        Gauges reflect currently-active (unresolved) alerts.
        """
        sev = alert.severity if alert.severity in self._counters else "info"
        self._counters[sev] += 1

        key = (alert.node_name, alert.model_family, sev)
        self._active_gauges[key] = 1

        self._write_prom_file()

    def _write_prom_file(self) -> None:
        """Atomically write Prometheus textfile metrics."""
        prom_path = self._metrics_dir / _DRIFT_PROM_FILE
        tmp_path = prom_path.with_suffix(".prom.tmp")

        lines = [
            "# HELP memopt_drift_alerts_total "
            "Total drift alerts fired since daemon start, by severity.",
            "# TYPE memopt_drift_alerts_total counter",
        ]
        for sev, count in self._counters.items():
            lines.append(f'memopt_drift_alerts_total{{severity="{sev}"}} {count}')

        lines += [
            "",
            "# HELP memopt_drift_alert_active "
            "1 if a drift alert is currently active (unresolved) for this target.",
            "# TYPE memopt_drift_alert_active gauge",
        ]
        for (node, model, sev), val in self._active_gauges.items():
            lines.append(
                f'memopt_drift_alert_active{{node="{node}",model="{model}",'
                f'severity="{sev}"}} {val}'
            )

        content = "\n".join(lines) + "\n"
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(prom_path)  # atomic on POSIX
        log.debug("Prometheus drift metrics written to %s", prom_path)

    def _send_webhook(self, alert: DriftAlert) -> None:
        """
        POST a JSON payload to the configured webhook URL.
        Timeout: 5 seconds.  Failure is non-fatal.
        """
        payload = {
            "event": "memopt_drift_alert",
            "severity": alert.severity,
            "pid": alert.pid,
            "node": alert.node_name,
            "model": alert.model_family,
            "gpu_ids": alert.gpu_ids,
            "baseline_util_pct": alert.baseline_util_pct,
            "current_util_pct": alert.current_util_pct,
            "util_drop_pct": alert.util_drop_pct,
            "speedup_range": [alert.original_speedup_min, alert.original_speedup_max],
            "optimizations": alert.optimizations_originally_applied,
            "recommended_action": alert.recommended_action,
            "detection_ts": alert.detection_timestamp,
        }
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            self._webhook_url,
            data=body,
            headers={"Content-Type": "application/json", "User-Agent": "memopt-alerter/1.0"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            status = resp.status
        if status >= 400:
            log.warning("Webhook returned HTTP %d", status)
        else:
            log.debug("Webhook delivered (HTTP %d)", status)

    def _send_email(self, alert: DriftAlert) -> None:
        """
        Send alert via SMTP/TLS.  Requires SMTP_HOST, SMTP_USER, SMTP_PASS,
        ALERT_FROM, ALERT_TO to all be set.
        """
        hostname = socket.gethostname()
        subject = (
            f"[memopt {alert.severity.upper()}] "
            f"Drift on {alert.node_name} — {alert.model_family} PID {alert.pid}"
        )
        body = (
            f"MemOpt Drift Alert\n"
            f"{'=' * 60}\n"
            f"Severity   : {alert.severity.upper()}\n"
            f"Host       : {alert.node_name} (reported from {hostname})\n"
            f"Model      : {alert.model_family}\n"
            f"PID        : {alert.pid}\n"
            f"GPUs       : {alert.gpu_ids}\n\n"
            f"GPU Utilisation\n"
            f"  Baseline : {alert.baseline_util_pct:.1f}%\n"
            f"  Current  : {alert.current_util_pct:.1f}%\n"
            f"  Drop     : {alert.util_drop_pct:.1f}%\n\n"
            f"Original speedup: {alert.original_speedup_min:.1f}x – "
            f"{alert.original_speedup_max:.1f}x\n"
            f"Optimisations applied: {', '.join(alert.optimizations_originally_applied)}\n\n"
            f"Recommended action:\n  {alert.recommended_action}\n\n"
            f"This is a proxy-metric alert. Verify before acting.\n"
        )

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = self._alert_from
        msg["To"] = self._alert_to
        msg.set_content(body)

        with smtplib.SMTP(self._smtp_host, self._smtp_port, timeout=10) as smtp:
            smtp.starttls()
            if self._smtp_user and self._smtp_pass:
                smtp.login(self._smtp_user, self._smtp_pass)
            smtp.send_message(msg)

        log.info("Alert email sent to %s", self._alert_to)
