"""
Billing and invoice generation
Calculate charges, generate invoices, usage queries
"""
from datetime import datetime, timedelta
from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import func
import structlog

from .database import Tenant, UsageEvent, Invoice, TenantTier, InvoiceStatus
from .config import settings

logger = structlog.get_logger()


class BillingService:
    """
    Billing service for invoice generation and usage queries
    """

    @staticmethod
    def calculate_revenue_share_charge(total_tokens: int) -> float:
        """
        Calculate charge for revenue_share tier
        Amount = total_tokens * (price_per_1k_tokens / 1000)
        """
        price_per_token = settings.PRICE_PER_1K_TOKENS / 1000
        return round(total_tokens * price_per_token, 2)

    @staticmethod
    def calculate_enterprise_charge() -> float:
        """
        Calculate monthly charge for enterprise_flat tier
        Fixed monthly fee: $800k/year = $66,666.67/month
        """
        return settings.ENTERPRISE_FLAT_MONTHLY

    @staticmethod
    def get_usage_summary(
        db: Session,
        tenant_id: int,
        start: datetime,
        end: datetime,
    ) -> dict:
        """
        Get usage summary for a tenant in a time period
        Returns total tokens, requests, latency stats
        """
        query = db.query(
            func.count(UsageEvent.id).label("total_requests"),
            func.sum(UsageEvent.total_tokens).label("total_tokens"),
            func.sum(UsageEvent.prompt_tokens).label("prompt_tokens"),
            func.sum(UsageEvent.completion_tokens).label("completion_tokens"),
            func.avg(UsageEvent.latency_ms).label("avg_latency_ms"),
            func.count(func.nullif(UsageEvent.success, True)).label("failed_requests"),
        ).filter(
            UsageEvent.tenant_id == tenant_id,
            UsageEvent.timestamp >= start,
            UsageEvent.timestamp < end,
        )

        result = query.one()

        return {
            "total_requests": result.total_requests or 0,
            "total_tokens": result.total_tokens or 0,
            "prompt_tokens": result.prompt_tokens or 0,
            "completion_tokens": result.completion_tokens or 0,
            "avg_latency_ms": round(result.avg_latency_ms, 2) if result.avg_latency_ms else 0,
            "failed_requests": result.failed_requests or 0,
            "success_rate": (
                round((result.total_requests - (result.failed_requests or 0)) / result.total_requests * 100, 2)
                if result.total_requests > 0
                else 100.0
            ),
        }

    @staticmethod
    def generate_monthly_invoice(
        db: Session,
        tenant_id: int,
        year: int,
        month: int,
    ) -> Invoice:
        """
        Generate monthly invoice for a tenant
        Calculates usage and charges based on tier
        """
        # Get tenant
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).one()

        # Calculate period boundaries
        period_start = datetime(year, month, 1)
        if month == 12:
            period_end = datetime(year + 1, 1, 1)
        else:
            period_end = datetime(year, month + 1, 1)

        # Check if invoice already exists
        existing = (
            db.query(Invoice)
            .filter(
                Invoice.tenant_id == tenant_id,
                Invoice.period_start == period_start,
                Invoice.period_end == period_end,
            )
            .first()
        )

        if existing:
            logger.warning(
                "invoice_already_exists",
                tenant_id=tenant_id,
                invoice_id=existing.id,
                period=f"{year}-{month:02d}",
            )
            return existing

        # Get usage summary
        usage = BillingService.get_usage_summary(db, tenant_id, period_start, period_end)

        # Calculate charge based on tier
        if tenant.tier == TenantTier.ENTERPRISE_FLAT:
            amount_usd = BillingService.calculate_enterprise_charge()
        else:  # revenue_share
            amount_usd = BillingService.calculate_revenue_share_charge(usage["total_tokens"])

        # Create invoice
        invoice = Invoice(
            tenant_id=tenant_id,
            period_start=period_start,
            period_end=period_end,
            tier=tenant.tier,
            total_tokens=usage["total_tokens"],
            total_requests=usage["total_requests"],
            amount_usd=amount_usd,
            status=InvoiceStatus.DRAFT,
            notes=f"Usage: {usage['total_requests']:,} requests, {usage['total_tokens']:,} tokens",
        )

        db.add(invoice)
        db.commit()
        db.refresh(invoice)

        logger.info(
            "invoice_generated",
            invoice_id=invoice.id,
            tenant_id=tenant_id,
            period=f"{year}-{month:02d}",
            amount_usd=amount_usd,
            total_tokens=usage["total_tokens"],
        )

        return invoice

    @staticmethod
    def issue_invoice(db: Session, invoice_id: int) -> Invoice:
        """
        Mark invoice as issued
        Sets issued_at timestamp and changes status
        """
        invoice = db.query(Invoice).filter(Invoice.id == invoice_id).one()

        if invoice.status != InvoiceStatus.DRAFT:
            raise ValueError(f"Cannot issue invoice in status {invoice.status}")

        invoice.status = InvoiceStatus.ISSUED
        invoice.issued_at = datetime.utcnow()

        db.commit()
        db.refresh(invoice)

        logger.info("invoice_issued", invoice_id=invoice.id, tenant_id=invoice.tenant_id)

        return invoice

    @staticmethod
    def mark_invoice_paid(db: Session, invoice_id: int) -> Invoice:
        """
        Mark invoice as paid
        Sets paid_at timestamp and changes status
        """
        invoice = db.query(Invoice).filter(Invoice.id == invoice_id).one()

        if invoice.status not in [InvoiceStatus.ISSUED, InvoiceStatus.OVERDUE]:
            raise ValueError(f"Cannot mark invoice as paid in status {invoice.status}")

        invoice.status = InvoiceStatus.PAID
        invoice.paid_at = datetime.utcnow()

        db.commit()
        db.refresh(invoice)

        logger.info("invoice_paid", invoice_id=invoice.id, tenant_id=invoice.tenant_id)

        return invoice

    @staticmethod
    def get_current_month_usage(db: Session, tenant_id: int) -> dict:
        """
        Get usage for current month (for quota checks)
        """
        now = datetime.utcnow()
        month_start = datetime(now.year, now.month, 1)

        return BillingService.get_usage_summary(db, tenant_id, month_start, now)

    @staticmethod
    def get_usage_by_model(
        db: Session,
        tenant_id: int,
        start: datetime,
        end: datetime,
    ) -> List[dict]:
        """
        Get usage breakdown by model
        """
        query = (
            db.query(
                UsageEvent.model,
                func.count(UsageEvent.id).label("requests"),
                func.sum(UsageEvent.total_tokens).label("tokens"),
            )
            .filter(
                UsageEvent.tenant_id == tenant_id,
                UsageEvent.timestamp >= start,
                UsageEvent.timestamp < end,
                UsageEvent.success == True,
            )
            .group_by(UsageEvent.model)
            .order_by(func.sum(UsageEvent.total_tokens).desc())
        )

        results = query.all()

        return [
            {
                "model": r.model,
                "requests": r.requests,
                "tokens": r.tokens,
            }
            for r in results
        ]
