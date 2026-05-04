"""Assemble a Production Trust Receipt from Layer 1/2 + pillar state.

Constructs a ReceiptBuilder seeded with whichever pillar instances are
available, then calls `build_for_request` for one request id. Pure
consumer; never edits Layer 1 or Layer 2 source.

Usage:
    from memopt.integrations import assemble_production_receipt
    receipt = assemble_production_receipt(
        request_id="req_abc",
        tenant_id="alice",
        tokens=42,
        ledger=my_ledger,
        finops=my_finops_tracker,
    )
"""
from __future__ import annotations

import logging
from typing import Optional

import memopt
import memopt.orchestrator as orchestrator


logger = logging.getLogger("memopt.integrations.receipt_assembler")


def assemble_production_receipt(
    *,
    request_id: str = "ad-hoc",
    tenant_id: str = "_default",
    tokens: int = 0,
    cert: Optional[dict] = None,
    ledger=None,
    finops=None,
    gkd=None,
    kernel_cache=None,
    vmm=None,
    cache_hit: bool = False,
    joules_per_token: float = 0.0,
    energy_source: str = "unmeasured",
):
    """Build a `ProductionReceipt` snapshotting current memopt state.

    All pillar instances are optional. Missing pillars surface as
    zero/unknown placeholder fields in the receipt (the underlying
    ReceiptBuilder emits a UserWarning if every pillar is None — that
    is fine for ad-hoc inspection but should not be relied on for
    production audits).

    The receipt is signed by `ReceiptBuilder.build_for_request()`
    using `MEMOPT_SIGNING_KEY`.
    """
    from memopt.trust.receipt import ReceiptBuilder
    builder = ReceiptBuilder(
        cert=cert,
        ledger=ledger,
        finops=finops,
        gkd=gkd,
        kernel_cache=kernel_cache,
        vmm=vmm,
    )
    return builder.build_for_request(
        request_id=request_id,
        tenant_id=tenant_id,
        tokens=tokens,
        cache_hit=cache_hit,
        joules_per_token=joules_per_token,
        energy_source=energy_source,
    )


def snapshot_layer_state(*, tenants=("_default",)) -> dict:
    """Snapshot Layer 1 + Layer 2 state into a plain dict (for logs).

    Independent helper from `assemble_production_receipt` — useful for
    debug dashboards that want the same data without wrapping it in
    a signed receipt."""
    out = {"substrate": {}, "orchestrator": {}}
    for tenant in tenants:
        try:
            out["substrate"][tenant] = memopt.stats(tenant=tenant)
        except Exception:
            logger.warning("memopt.stats(%r) raised", tenant, exc_info=True)
            out["substrate"][tenant] = {}
    try:
        out["orchestrator"] = orchestrator.stats()
    except Exception:
        logger.warning("orchestrator.stats() raised", exc_info=True)
        out["orchestrator"] = {}
    return out
