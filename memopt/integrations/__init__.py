"""memopt integrations — Layer 1/2 wiring for the revived pillars.

This package contains small adapter modules that glue the pillars
(observability ledger, FinOps tracker, production receipt) to the
shipped Layer 1 substrate (`memopt.observe`, `memopt.stats`,
`memopt.peek_handle`) and Layer 2 orchestrator
(`memopt.orchestrator.start/stats/register_policy`).

Design principle (per CONTRIBUTING.md): NEVER edit Layer 1 or Layer 2
source. Pillars consume Layer 1/2 through the public API only.
"""
from .ledger_bridge   import attach_ledger_to_substrate, detach_ledger
from .finops_poller   import FinOpsPoller
from .receipt_assembler import assemble_production_receipt

__all__ = [
    "attach_ledger_to_substrate",
    "detach_ledger",
    "FinOpsPoller",
    "assemble_production_receipt",
]
