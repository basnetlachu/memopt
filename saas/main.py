"""
MemOpt SaaS API
FastAPI application with auth, metering, billing, and inference
"""
import sys
import os
import uuid
import time
from datetime import datetime, timedelta
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import redis.asyncio as redis
import structlog

# Add parent directory to path to import memopt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memopt import OptimizedLLM

from .config import settings
from .database import (
    get_db,
    create_tables,
    Tenant,
    APIKey,
    UsageEvent,
    Invoice,
    TenantTier,
    TenantStatus,
    InvoiceStatus,
)
from .auth import (
    get_tenant_from_api_key,
    require_admin,
    generate_api_key,
    hash_api_key,
    get_key_prefix,
    TenantContext,
)
from .rate_limiter import init_rate_limiter, get_rate_limiter
from .billing import BillingService

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer() if settings.LOG_JSON else structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Global resources
redis_client: Optional[redis.Redis] = None
model_cache = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("starting_memopt_saas", env=settings.ENV)

    # Initialize Redis (optional for now)
    global redis_client
    if settings.REDIS_URL:
        try:
            redis_client = redis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                max_connections=settings.REDIS_MAX_CONNECTIONS,
            )
            await redis_client.ping()
            logger.info("redis_connected", url=settings.REDIS_URL)

            # Initialize rate limiter only if Redis is available
            init_rate_limiter(redis_client)
        except Exception as e:
            logger.warning("redis_unavailable", error=str(e), message="Rate limiting disabled")
            redis_client = None
    else:
        logger.warning("redis_not_configured", message="Rate limiting disabled")
        redis_client = None

    # Create tables (in development only, use migrations in production)
    if settings.ENV == "development":
        create_tables()

    logger.info("memopt_saas_started", version=settings.API_VERSION)

    yield

    # Shutdown
    logger.info("shutting_down_memopt_saas")
    if redis_client:
        await redis_client.close()
    logger.info("memopt_saas_stopped")


# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="Memory-optimized LLM inference as a service",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================


class InferenceRequest(BaseModel):
    """Inference request model"""

    prompt: str = Field(..., description="Input text prompt", min_length=1, max_length=10000)
    max_tokens: int = Field(default=256, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(default=1.0, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling")
    model: str = Field(default="gpt2", description="Model name")
    optimization_level: str = Field(
        default="high",
        pattern="^(conservative|balanced|high|maximum|aggressive)$",
    )


class InferenceResponse(BaseModel):
    """Inference response model"""

    request_id: str
    generated_text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    model: str


class UsageSummary(BaseModel):
    """Usage summary response"""

    total_requests: int
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    avg_latency_ms: float
    failed_requests: int
    success_rate: float


class InvoiceResponse(BaseModel):
    """Invoice response model"""

    id: int
    tenant_id: int
    period_start: datetime
    period_end: datetime
    tier: str
    total_tokens: int
    total_requests: int
    amount_usd: float
    status: str
    created_at: datetime
    issued_at: Optional[datetime]
    paid_at: Optional[datetime]
    notes: Optional[str]


class CreateTenantRequest(BaseModel):
    """Create tenant request (admin only)"""

    name: str = Field(..., min_length=1, max_length=255)
    tier: TenantTier
    allowed_models: Optional[str] = Field(None, description="Comma-separated model names")
    max_context_length: Optional[int] = Field(4096, ge=512, le=32768)
    priority_level: int = Field(default=1, ge=1, le=10)


class CreateTenantResponse(BaseModel):
    """Create tenant response"""

    tenant_id: int
    name: str
    tier: str
    api_key: str  # Only returned on creation!


class CreateAPIKeyResponse(BaseModel):
    """Create API key response"""

    api_key_id: int
    tenant_id: int
    api_key: str  # Only returned on creation!
    prefix: str


# ============================================================================
# PUBLIC ENDPOINTS
# ============================================================================


@app.get("/", tags=["Public"])
async def root():
    """API root"""
    return {
        "name": settings.API_TITLE,
        "version": settings.API_VERSION,
        "env": settings.ENV,
        "endpoints": {
            "health": "/health",
            "inference": "/v1/infer (POST, requires auth)",
            "usage": "/v1/usage (GET, requires auth)",
            "invoices": "/v1/invoices (GET, requires auth)",
            "docs": "/docs",
        },
    }


@app.get("/health", tags=["Public"])
async def health(db: Session = Depends(get_db)):
    """Health check endpoint"""
    # Check database
    try:
        db.execute("SELECT 1")
        db_status = "healthy"
    except Exception as e:
        logger.error("database_unhealthy", error=str(e))
        db_status = "unhealthy"

    # Check Redis
    try:
        await redis_client.ping()
        redis_status = "healthy"
    except Exception as e:
        logger.error("redis_unhealthy", error=str(e))
        redis_status = "unhealthy"

    overall_status = "healthy" if db_status == "healthy" and redis_status == "healthy" else "unhealthy"

    return {
        "status": overall_status,
        "database": db_status,
        "redis": redis_status,
        "version": settings.API_VERSION,
        "env": settings.ENV,
    }


# ============================================================================
# INFERENCE ENDPOINT (Authenticated)
# ============================================================================


@app.post("/v1/infer", response_model=InferenceResponse, tags=["Inference"])
async def infer(
    request: InferenceRequest,
    ctx: TenantContext = Depends(get_tenant_from_api_key),
    db: Session = Depends(get_db),
):
    """
    Run inference with rate limiting, quota checking, and usage metering
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info(
        "inference_request_received",
        request_id=request_id,
        tenant_id=ctx.tenant_id,
        model=request.model,
        max_tokens=request.max_tokens,
    )

    rate_limiter = get_rate_limiter()

    try:
        # 1. Check rate limits (skip if Redis unavailable)
        if rate_limiter:
            await rate_limiter.check_request_limit(ctx)
            await rate_limiter.check_concurrent_requests(ctx)

        # 2. Check monthly quota (works without Redis)
        if rate_limiter:
            await rate_limiter.check_monthly_token_quota(ctx, db)

        # 3. Check model allowed
        if not ctx.is_model_allowed(request.model):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Model '{request.model}' not allowed for your tier",
            )

        # 4. Acquire concurrent slot (skip if Redis unavailable)
        if rate_limiter:
            await rate_limiter.acquire_request_slot(ctx)

        try:
            # 5. Run inference (use real MemOpt model)
            model_key = f"{request.model}_{request.optimization_level}"

            if model_key not in model_cache:
                logger.info("loading_model", model=request.model, optimization_level=request.optimization_level)
                model_cache[model_key] = OptimizedLLM(
                    model=request.model,
                    optimization_level=request.optimization_level,
                    enable_profiling=True,
                )

            model = model_cache[model_key]

            # Generate text
            generated_text = model.generate(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=(request.temperature > 0),
            )

            # Calculate tokens (rough estimate - use real tokenizer in production)
            prompt_tokens = len(request.prompt.split())
            completion_tokens = len(generated_text.split())
            total_tokens = prompt_tokens + completion_tokens

            latency_ms = (time.time() - start_time) * 1000

            # 6. Check token rate limit (after we know token count, skip if Redis unavailable)
            if rate_limiter:
                await rate_limiter.check_token_limit_per_minute(ctx, total_tokens)

            # 7. Record usage event
            usage_event = UsageEvent(
                tenant_id=ctx.tenant_id,
                request_id=request_id,
                timestamp=datetime.utcnow(),
                model=request.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                success=True,
            )
            db.add(usage_event)
            db.commit()

            logger.info(
                "inference_success",
                request_id=request_id,
                tenant_id=ctx.tenant_id,
                tokens=total_tokens,
                latency_ms=round(latency_ms, 2),
            )

            return InferenceResponse(
                request_id=request_id,
                generated_text=generated_text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=round(latency_ms, 2),
                model=request.model,
            )

        finally:
            # Always release concurrent slot (skip if Redis unavailable)
            if rate_limiter:
                await rate_limiter.release_request_slot(ctx)

    except HTTPException:
        # Re-raise HTTP exceptions
        raise

    except Exception as e:
        # Log and record failure
        logger.error(
            "inference_failed",
            request_id=request_id,
            tenant_id=ctx.tenant_id,
            error=str(e),
            exc_info=True,
        )

        usage_event = UsageEvent(
            tenant_id=ctx.tenant_id,
            request_id=request_id,
            timestamp=datetime.utcnow(),
            model=request.model,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            latency_ms=(time.time() - start_time) * 1000,
            success=False,
            error_code="internal_error",
            error_message=str(e)[:500],
        )
        db.add(usage_event)
        db.commit()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}",
        )


# ============================================================================
# USAGE & BILLING ENDPOINTS (Authenticated)
# ============================================================================


@app.get("/v1/usage", response_model=UsageSummary, tags=["Usage"])
async def get_usage(
    start: Optional[datetime] = Query(None, description="Start time (UTC)"),
    end: Optional[datetime] = Query(None, description="End time (UTC)"),
    ctx: TenantContext = Depends(get_tenant_from_api_key),
    db: Session = Depends(get_db),
):
    """
    Get usage summary for authenticated tenant
    Defaults to current month if no dates provided
    """
    if not start:
        now = datetime.utcnow()
        start = datetime(now.year, now.month, 1)

    if not end:
        end = datetime.utcnow()

    usage = BillingService.get_usage_summary(db, ctx.tenant_id, start, end)

    logger.info(
        "usage_query",
        tenant_id=ctx.tenant_id,
        start=start.isoformat(),
        end=end.isoformat(),
        total_tokens=usage["total_tokens"],
    )

    return UsageSummary(**usage)


@app.get("/v1/invoices", response_model=List[InvoiceResponse], tags=["Billing"])
async def list_invoices(
    ctx: TenantContext = Depends(get_tenant_from_api_key),
    db: Session = Depends(get_db),
):
    """List all invoices for authenticated tenant"""
    invoices = (
        db.query(Invoice)
        .filter(Invoice.tenant_id == ctx.tenant_id)
        .order_by(Invoice.period_start.desc())
        .all()
    )

    return [
        InvoiceResponse(
            id=inv.id,
            tenant_id=inv.tenant_id,
            period_start=inv.period_start,
            period_end=inv.period_end,
            tier=inv.tier.value,
            total_tokens=inv.total_tokens,
            total_requests=inv.total_requests,
            amount_usd=inv.amount_usd,
            status=inv.status.value,
            created_at=inv.created_at,
            issued_at=inv.issued_at,
            paid_at=inv.paid_at,
            notes=inv.notes,
        )
        for inv in invoices
    ]


@app.get("/v1/invoices/{invoice_id}", response_model=InvoiceResponse, tags=["Billing"])
async def get_invoice(
    invoice_id: int,
    ctx: TenantContext = Depends(get_tenant_from_api_key),
    db: Session = Depends(get_db),
):
    """Get specific invoice (must belong to authenticated tenant)"""
    invoice = (
        db.query(Invoice)
        .filter(Invoice.id == invoice_id, Invoice.tenant_id == ctx.tenant_id)
        .first()
    )

    if not invoice:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invoice not found")

    return InvoiceResponse(
        id=invoice.id,
        tenant_id=invoice.tenant_id,
        period_start=invoice.period_start,
        period_end=invoice.period_end,
        tier=invoice.tier.value,
        total_tokens=invoice.total_tokens,
        total_requests=invoice.total_requests,
        amount_usd=invoice.amount_usd,
        status=invoice.status.value,
        created_at=invoice.created_at,
        issued_at=invoice.issued_at,
        paid_at=invoice.paid_at,
        notes=invoice.notes,
    )


# ============================================================================
# ADMIN ENDPOINTS (Require admin auth)
# ============================================================================


@app.post("/admin/tenants", response_model=CreateTenantResponse, tags=["Admin"])
async def create_tenant(
    request: CreateTenantRequest,
    admin: bool = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Create a new tenant with API key (admin only)"""
    # Create tenant
    tenant = Tenant(
        name=request.name,
        tier=request.tier,
        status=TenantStatus.ACTIVE,
        allowed_models=request.allowed_models,
        max_context_length=request.max_context_length,
        priority_level=request.priority_level,
    )
    db.add(tenant)
    db.flush()  # Get tenant ID

    # Generate API key
    api_key = generate_api_key()
    key_hash = hash_api_key(api_key)
    key_prefix = get_key_prefix(api_key)

    api_key_record = APIKey(
        tenant_id=tenant.id,
        key_hash=key_hash,
        key_prefix=key_prefix,
        name="Default API Key",
    )
    db.add(api_key_record)
    db.commit()
    db.refresh(tenant)

    logger.info("tenant_created", tenant_id=tenant.id, name=tenant.name, tier=tenant.tier.value)

    return CreateTenantResponse(
        tenant_id=tenant.id,
        name=tenant.name,
        tier=tenant.tier.value,
        api_key=api_key,  # Only returned once!
    )


@app.post("/admin/tenants/{tenant_id}/suspend", tags=["Admin"])
async def suspend_tenant(
    tenant_id: int,
    admin: bool = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Suspend a tenant (admin only)"""
    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    if not tenant:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tenant not found")

    tenant.status = TenantStatus.SUSPENDED
    db.commit()

    logger.info("tenant_suspended", tenant_id=tenant_id)

    return {"status": "suspended", "tenant_id": tenant_id}


@app.post("/admin/tenants/{tenant_id}/activate", tags=["Admin"])
async def activate_tenant(
    tenant_id: int,
    admin: bool = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Activate a suspended tenant (admin only)"""
    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    if not tenant:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tenant not found")

    tenant.status = TenantStatus.ACTIVE
    db.commit()

    logger.info("tenant_activated", tenant_id=tenant_id)

    return {"status": "active", "tenant_id": tenant_id}


@app.post("/admin/tenants/{tenant_id}/api-keys", response_model=CreateAPIKeyResponse, tags=["Admin"])
async def create_api_key(
    tenant_id: int,
    name: Optional[str] = None,
    admin: bool = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Generate new API key for tenant (admin only)"""
    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    if not tenant:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tenant not found")

    # Generate API key
    api_key = generate_api_key()
    key_hash = hash_api_key(api_key)
    key_prefix = get_key_prefix(api_key)

    api_key_record = APIKey(
        tenant_id=tenant_id,
        key_hash=key_hash,
        key_prefix=key_prefix,
        name=name or "API Key",
    )
    db.add(api_key_record)
    db.commit()
    db.refresh(api_key_record)

    logger.info("api_key_created", api_key_id=api_key_record.id, tenant_id=tenant_id)

    return CreateAPIKeyResponse(
        api_key_id=api_key_record.id,
        tenant_id=tenant_id,
        api_key=api_key,  # Only returned once!
        prefix=key_prefix,
    )


@app.delete("/admin/api-keys/{api_key_id}", tags=["Admin"])
async def revoke_api_key(
    api_key_id: int,
    admin: bool = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Revoke an API key (admin only)"""
    api_key = db.query(APIKey).filter(APIKey.id == api_key_id).first()
    if not api_key:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API key not found")

    api_key.revoked_at = datetime.utcnow()
    db.commit()

    logger.info("api_key_revoked", api_key_id=api_key_id, tenant_id=api_key.tenant_id)

    return {"status": "revoked", "api_key_id": api_key_id}


@app.post("/admin/invoices/generate", tags=["Admin"])
async def generate_invoices(
    year: int = Query(..., ge=2020, le=2100),
    month: int = Query(..., ge=1, le=12),
    admin: bool = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Generate invoices for all active tenants for a given month (admin only)"""
    tenants = db.query(Tenant).filter(Tenant.status == TenantStatus.ACTIVE).all()

    invoices_created = []
    for tenant in tenants:
        try:
            invoice = BillingService.generate_monthly_invoice(db, tenant.id, year, month)
            invoices_created.append({"tenant_id": tenant.id, "invoice_id": invoice.id})
        except Exception as e:
            logger.error("invoice_generation_failed", tenant_id=tenant.id, error=str(e))

    logger.info("bulk_invoices_generated", count=len(invoices_created), period=f"{year}-{month:02d}")

    return {"invoices_created": invoices_created, "count": len(invoices_created)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        log_level=settings.LOG_LEVEL.lower(),
    )
