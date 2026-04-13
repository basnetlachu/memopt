#!/usr/bin/env python3
"""
memopt Kubernetes Operator — entry point.

Runs the operator controller loop.
Requires: kubernetes Python package and Kubernetes cluster access.

Usage:
  python -m memopt.operator.main

Or via CLI:
  memopt operator start
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import threading


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s")


def main() -> None:
    _configure_logging()
    logger = logging.getLogger(__name__)

    from memopt.operator.controller import (
        MemoptOperator, K8S_AVAILABLE)

    if not K8S_AVAILABLE:
        logger.error(
            "kubernetes package required: pip install kubernetes")
        sys.exit(1)

    namespace = os.getenv("MEMOPT_NAMESPACE", "memopt")
    interval = float(
        os.getenv("MEMOPT_RECONCILE_INTERVAL_S", "30"))
    dry_run = os.getenv(
        "MEMOPT_OPERATOR_DRY_RUN", "false").lower() == "true"

    operator = MemoptOperator(
        namespace=namespace,
        reconcile_interval_s=interval,
        dry_run=dry_run,
    )

    stop_event = threading.Event()

    def handle_signal(sig, frame):
        logger.info("Signal %s received, stopping", sig)
        stop_event.set()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    success = operator.start()
    if not success:
        logger.error("Operator failed to start")
        sys.exit(1)

    logger.info(
        "Operator running: namespace=%s interval=%ss dry_run=%s",
        namespace, interval, dry_run)

    stop_event.wait()
    operator.stop()
    logger.info("Operator stopped cleanly")


if __name__ == "__main__":
    main()
