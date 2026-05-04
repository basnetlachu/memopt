"""Bridge: substrate Event stream -> OptimizationLedger.

Subscribes to substrate `evict` / `promote` / `migrate` events via
`memopt.observe(kind, callback)` and converts each into a
`LedgerEntry`. The substrate's dispatcher delivers events on its own
thread (one per kind), so the ledger writes are off the alloc warm
path.

Usage:
    from memopt.integrations import attach_ledger_to_substrate, detach_ledger
    from memopt.observability.ledger import OptimizationLedger

    ledger = OptimizationLedger()
    handles = attach_ledger_to_substrate(ledger)
    # ... workload runs; events stream into the ledger ...
    detach_ledger(handles)
"""
from __future__ import annotations

import logging
from typing import List

import memopt
from memopt.observability.ledger import OptimizationLedger
from memopt.substrate.events import Event, SubscriptionHandle


logger = logging.getLogger("memopt.integrations.ledger_bridge")


def _make_callback(ledger: OptimizationLedger):
    def _on_event(event: Event) -> None:
        try:
            # OptimizationLedger.record() pins a strict kwargs signature
            # (per memopt/observability/ledger.py:355-366). We embed the
            # event placement in `node_id` for grep-ability; the byte
            # size becomes hbm_saved_bytes (negative for evicts) so the
            # ledger's totals() reflects substrate-side movement.
            tagged_node = (
                f"substrate:{event.kind}:{event.from_placement or '-'}->"
                f"{event.to_placement or '-'}"
            )
            sign = -1 if event.kind == "evict" else 1
            ledger.record(
                tokens=0,
                node_id=tagged_node,
                tenant_id=event.tenant or "_default",
                hbm_saved_bytes=float(sign * (event.size_bytes or 0)),
                energy_source="unmeasured",
            )
        except Exception:
            logger.warning("ledger.record raised in bridge callback", exc_info=True)
    return _on_event


def attach_ledger_to_substrate(
    ledger: OptimizationLedger,
    kinds=("evict", "promote", "migrate"),
) -> List[SubscriptionHandle]:
    """Subscribe `ledger` to substrate events. Returns the subscription
    handles so the caller can detach later."""
    handles: List[SubscriptionHandle] = []
    cb = _make_callback(ledger)
    for kind in kinds:
        handles.append(memopt.observe(kind, cb))
    return handles


def detach_ledger(handles) -> None:
    for h in handles or ():
        try:
            h.unsubscribe()
        except Exception:
            logger.warning("unsubscribe raised", exc_info=True)
