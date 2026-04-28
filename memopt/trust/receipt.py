"""
memopt Production Trust Receipt.

The unifying artifact across all 7 pillars.
One signed JSON per request that proves
every claim about the AI inference.

This is what gets enterprises out of
pilot purgatory — they can finally answer
"did our AI work correctly today?" with
millions of cryptographically verifiable
receipts instead of "probably."
"""
import os
import json
import hashlib
import hmac as _hmac
import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict
from datetime import datetime, timezone


@dataclass
class HardwareProof:
    """From P6 — Silicon Certification."""
    gpu_name: str = ""
    cert_status: str = "UNKNOWN"
    cert_hash: str = ""
    bandwidth_pct_of_peak: Optional[float] = None
    cert_timestamp: str = ""


@dataclass
class MemoryProof:
    """From P1 — VMM."""
    pages_hbm: int = 0
    pages_dram: int = 0
    bytes_evicted_gb: float = 0.0
    eviction_count: int = 0


@dataclass
class CacheProof:
    """From P2 — GKD Dedup."""
    cache_hit: bool = False
    workflow_id: str = ""
    matched_tokens: int = 0
    compute_saved_pct: float = 0.0


@dataclass
class KernelProof:
    """From P3 — Kernel Library."""
    kernels_used: list = field(default_factory=list)
    library_size: int = 0
    library_hash: str = ""


@dataclass
class EnergyProof:
    """From P4 — Compliance Ledger."""
    joules_total: float = 0.0
    joules_per_token: float = 0.0
    energy_source: str = "unmeasured"
    tokens_generated: int = 0


@dataclass
class CostProof:
    """From P7 — FinOps."""
    gpu_seconds: float = 0.0
    cost_usd: float = 0.0
    cost_per_token_usd: float = 0.0
    savings_usd: float = 0.0


@dataclass
class HardwareBackendProof:
    """From P8 — HAL."""
    backend: str = "unknown"
    tiers_available: list = field(default_factory=list)


@dataclass
class ProductionReceipt:
    """
    The signed proof of one inference request.

    This is the artifact that EU AI Act auditors,
    enterprise CFOs, and compliance teams need
    to trust AI in production.
    """
    # Request identity
    request_id: str = ""
    tenant_id:  str = ""
    timestamp:  str = ""

    # Per-pillar proofs
    hardware: HardwareProof        = field(default_factory=HardwareProof)
    memory:   MemoryProof          = field(default_factory=MemoryProof)
    cache:    CacheProof           = field(default_factory=CacheProof)
    kernel:   KernelProof          = field(default_factory=KernelProof)
    energy:   EnergyProof          = field(default_factory=EnergyProof)
    cost:     CostProof            = field(default_factory=CostProof)
    backend:  HardwareBackendProof = field(default_factory=HardwareBackendProof)

    # Audit chain
    receipt_hash: str = ""
    signature:    str = ""


class ReceiptBuilder:
    """
    Assembles a ProductionReceipt by querying
    each pillar component for its current state.

    Usage:
        builder = ReceiptBuilder(
            cert=current_silicon_cert,
            ledger=optimization_ledger,
            finops=finops_tracker,
            gkd=gkd_store,
            hal=hal,
            kernel_cache=kernel_cache,
        )

        receipt = builder.build_for_request(
            request_id="req_abc",
            tenant_id="customer-1",
            tokens=42,
            workflow_id="wf_xyz",
            cache_hit=True,
            joules_per_token=0.84,
        )
    """

    def __init__(
        self,
        cert: Optional[Dict] = None,
        ledger=None,
        finops=None,
        gkd=None,
        hal=None,
        kernel_cache=None,
        vmm=None,
    ):
        self.cert         = cert or {}
        self.ledger       = ledger
        self.finops       = finops
        self.gkd          = gkd
        self.hal          = hal
        self.kernel_cache = kernel_cache
        self.vmm          = vmm

        if not any([cert, ledger, finops, gkd, hal, kernel_cache, vmm]):
            warnings.warn(
                "ReceiptBuilder created with no pillar instances. "
                "Receipts will contain only zero/unknown placeholder "
                "fields. This is fine for testing but unsafe for "
                "production audits.",
                UserWarning,
                stacklevel=2,
            )

    def _build_hardware_proof(self) -> HardwareProof:
        if not self.cert:
            return HardwareProof()
        return HardwareProof(
            gpu_name=self.cert.get("gpu", "unknown"),
            cert_status=self.cert.get("certificate_status", "UNKNOWN"),
            cert_hash=self.cert.get("certificate_hash", "")[:32],
            bandwidth_pct_of_peak=self.cert.get("bandwidth_pct_of_peak"),
            cert_timestamp=self.cert.get("timestamp", ""),
        )

    def _build_memory_proof(self) -> MemoryProof:
        """Read memory stats from whichever VMM-shaped object the caller
        provided. CUDAVMMAllocator.stats() exposes {pages_hbm, pages_dram,
        bytes_evicted_total (bytes), eviction_count}. The high-level
        VMM.stats() exposes the tier_manager dict and lacks page counters,
        so handle both shapes and fall back gracefully."""
        if not self.vmm:
            return MemoryProof()
        try:
            stats = self.vmm.stats()

            pages_hbm = stats.get("pages_hbm")
            if pages_hbm is None:
                pages_hbm = stats.get("hbm_pages")
            if pages_hbm is None:
                pages_hbm = (
                    stats.get("pages_total", 0)
                    - stats.get("pages_dram", 0)
                )

            pages_dram = stats.get("pages_dram", 0)

            # Convert bytes_evicted_total (bytes from CUDAVMMAllocator)
            # to GB. If a caller already passed GB-shaped stats, accept
            # that too. Threshold guards against double-conversion.
            bytes_evicted = stats.get("bytes_evicted_total", 0)
            if bytes_evicted and bytes_evicted > 1e6:
                bytes_evicted_gb = bytes_evicted / 1e9
            else:
                bytes_evicted_gb = stats.get("bytes_evicted_gb", 0.0)

            return MemoryProof(
                pages_hbm=int(pages_hbm or 0),
                pages_dram=int(pages_dram or 0),
                bytes_evicted_gb=float(bytes_evicted_gb or 0.0),
                eviction_count=int(stats.get("eviction_count", 0) or 0),
            )
        except Exception:
            return MemoryProof()

    def _build_cache_proof(
        self,
        cache_hit: bool = False,
        workflow_id: str = "",
        matched_tokens: int = 0,
        context_tokens: int = 0,
    ) -> CacheProof:
        """compute_saved_pct = matched_tokens / context_tokens * 100,
        clamped to [0, 100]. Falls back to 0 on a miss or when the caller
        couldn't supply context_tokens — never invents a savings number."""
        if not cache_hit or context_tokens <= 0:
            compute_saved = 0.0
        else:
            compute_saved = min(
                100.0, (matched_tokens / context_tokens) * 100.0,
            )

        return CacheProof(
            cache_hit=cache_hit,
            workflow_id=workflow_id,
            matched_tokens=matched_tokens,
            compute_saved_pct=round(compute_saved, 1),
        )

    def _build_kernel_proof(self) -> KernelProof:
        if not self.kernel_cache:
            return KernelProof()
        try:
            catalog = self.kernel_cache.catalog()
            return KernelProof(
                kernels_used=[e.get("op_name", "") for e in catalog[:5]],
                library_size=len(catalog),
                library_hash=hashlib.sha256(
                    json.dumps(
                        sorted([e.get("op_name", "") for e in catalog])
                    ).encode()
                ).hexdigest()[:16],
            )
        except Exception:
            return KernelProof()

    def _build_energy_proof(
        self,
        tokens: int = 0,
        joules_per_token: float = 0.0,
        energy_source: str = "unmeasured",
    ) -> EnergyProof:
        return EnergyProof(
            joules_total=round(tokens * joules_per_token, 4),
            joules_per_token=joules_per_token,
            energy_source=energy_source,
            tokens_generated=tokens,
        )

    def _build_cost_proof(
        self,
        gpu_seconds: float = 0.0,
        tokens: int = 0,
    ) -> CostProof:
        cost_per_hour = 2.00
        savings = 0.0

        if self.finops:
            try:
                cost_per_hour = self.finops._gpu_cost_per_hour
                # Allocate this request's slice of the tracker's
                # current savings estimate by cost ratio. This is an
                # approximation — FinOps tracks workload-level savings,
                # so we apportion by per-request GPU cost. With sign=False
                # we avoid the HMAC overhead since we only need the
                # numbers.
                report = self.finops.get_report(sign=False)
                total_cost = report.total_cost_usd
                if total_cost > 0:
                    this_cost = (gpu_seconds / 3600.0) * cost_per_hour
                    proportion = this_cost / total_cost
                    savings = report.estimated_savings_usd * proportion
            except Exception:
                pass

        cost = (gpu_seconds / 3600.0) * cost_per_hour
        cost_per_tok = cost / tokens if tokens > 0 else 0.0

        return CostProof(
            gpu_seconds=round(gpu_seconds, 4),
            cost_usd=round(cost, 6),
            cost_per_token_usd=round(cost_per_tok, 8),
            savings_usd=round(savings, 6),
        )

    def _build_backend_proof(self) -> HardwareBackendProof:
        if not self.hal:
            return HardwareBackendProof()
        try:
            return HardwareBackendProof(
                backend=str(self.hal.backend),
                tiers_available=list(self.hal.tier_names),
            )
        except Exception:
            return HardwareBackendProof()

    def _compute_receipt_hash(self, receipt: ProductionReceipt) -> str:
        """Hash everything except hash + signature."""
        d = asdict(receipt)
        d.pop("receipt_hash", None)
        d.pop("signature", None)
        canonical = json.dumps(d, sort_keys=True).encode()
        return hashlib.sha256(canonical).hexdigest()

    def _sign_receipt(self, receipt: ProductionReceipt) -> str:
        """HMAC-SHA256 sign the receipt hash."""
        key = os.environ.get("MEMOPT_SIGNING_KEY", "").encode()
        if not key:
            return "unsigned"
        sig = _hmac.new(
            key, receipt.receipt_hash.encode(), hashlib.sha256,
        ).hexdigest()
        return f"hmac-sha256:{sig}"

    def build_for_request(
        self,
        request_id: str,
        tenant_id: str,
        tokens: int = 0,
        gpu_seconds: float = 0.0,
        joules_per_token: float = 0.0,
        energy_source: str = "unmeasured",
        cache_hit: bool = False,
        workflow_id: str = "",
        matched_tokens: int = 0,
        context_tokens: int = 0,
    ) -> ProductionReceipt:
        """
        Assemble a full receipt for one request.
        Returns signed ProductionReceipt.
        """
        receipt = ProductionReceipt(
            request_id=request_id,
            tenant_id=tenant_id,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            hardware=self._build_hardware_proof(),
            memory=self._build_memory_proof(),
            cache=self._build_cache_proof(
                cache_hit=cache_hit,
                workflow_id=workflow_id,
                matched_tokens=matched_tokens,
                context_tokens=context_tokens,
            ),
            kernel=self._build_kernel_proof(),
            energy=self._build_energy_proof(
                tokens=tokens,
                joules_per_token=joules_per_token,
                energy_source=energy_source,
            ),
            cost=self._build_cost_proof(
                gpu_seconds=gpu_seconds,
                tokens=tokens,
            ),
            backend=self._build_backend_proof(),
        )

        receipt.receipt_hash = self._compute_receipt_hash(receipt)
        receipt.signature    = self._sign_receipt(receipt)
        return receipt


def verify_receipt(receipt_dict: dict, signing_key: str) -> bool:
    """
    Verify a ProductionReceipt is unmodified.
    Returns True only if signature matches the
    reconstructed receipt hash.
    """
    sig = receipt_dict.get("signature", "")
    if not sig.startswith("hmac-sha256:"):
        return False
    provided = sig.split(":", 1)[1]

    d = dict(receipt_dict)
    d.pop("receipt_hash", None)
    d.pop("signature", None)
    canonical = json.dumps(d, sort_keys=True).encode()
    expected_hash = hashlib.sha256(canonical).hexdigest()

    if receipt_dict.get("receipt_hash", "") != expected_hash:
        return False

    expected_sig = _hmac.new(
        signing_key.encode(),
        expected_hash.encode(),
        hashlib.sha256,
    ).hexdigest()

    return _hmac.compare_digest(provided, expected_sig)
