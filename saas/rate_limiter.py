"""
Rate limiting and quota enforcement
Token-based rate limiting using Redis
"""
import time
from typing import Optional
from datetime import datetime, timedelta
import redis.asyncio as redis
from fastapi import HTTPException, status
import structlog

from .config import settings
from .auth import TenantContext

logger = structlog.get_logger()


class RateLimiter:
    """
    Token bucket rate limiter using Redis
    Enforces per-tenant rate limits for requests and tokens
    """

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    async def check_request_limit(self, ctx: TenantContext) -> None:
        """
        Check if tenant has exceeded requests per minute limit
        Raises HTTPException if limit exceeded
        """
        # Get limit based on tier
        if ctx.is_enterprise:
            limit = settings.RATE_LIMIT_ENTERPRISE
        else:
            limit = settings.RATE_LIMIT_REVENUE_SHARE

        # Redis key for request count (sliding window: 1 minute)
        key = f"rate_limit:requests:{ctx.tenant_id}:minute"

        # Increment and get count
        count = await self.redis.incr(key)

        # Set expiry on first increment
        if count == 1:
            await self.redis.expire(key, 60)  # 60 seconds TTL

        if count > limit:
            logger.warning(
                "request_rate_limit_exceeded",
                tenant_id=ctx.tenant_id,
                count=count,
                limit=limit,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {limit} requests per minute",
                headers={"Retry-After": "60"},
            )

        logger.debug("request_rate_check", tenant_id=ctx.tenant_id, count=count, limit=limit)

    async def check_token_limit_per_minute(self, ctx: TenantContext, tokens: int) -> None:
        """
        Check if tenant has exceeded tokens per minute limit
        Raises HTTPException if limit exceeded
        """
        # Get limit based on tier
        if ctx.is_enterprise:
            limit = settings.TOKEN_LIMIT_ENTERPRISE_PER_MIN
        else:
            limit = settings.TOKEN_LIMIT_REVENUE_SHARE_PER_MIN

        # Redis key for token count (sliding window: 1 minute)
        key = f"rate_limit:tokens:{ctx.tenant_id}:minute"

        # Add tokens and get total
        total = await self.redis.incrby(key, tokens)

        # Set expiry on first increment
        if total == tokens:
            await self.redis.expire(key, 60)  # 60 seconds TTL

        if total > limit:
            logger.warning(
                "token_rate_limit_exceeded",
                tenant_id=ctx.tenant_id,
                total=total,
                limit=limit,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Token rate limit exceeded: {limit} tokens per minute",
                headers={"Retry-After": "60"},
            )

        logger.debug("token_rate_check", tenant_id=ctx.tenant_id, total=total, limit=limit)

    async def check_monthly_token_quota(self, ctx: TenantContext, db_session) -> None:
        """
        Check if tenant has exceeded monthly token quota
        Queries database for current month usage
        """
        # Get limit based on tier
        if ctx.is_enterprise:
            limit = settings.TOKEN_LIMIT_ENTERPRISE_PER_MONTH
        else:
            limit = settings.TOKEN_LIMIT_REVENUE_SHARE_PER_MONTH

        # Calculate current month boundaries
        now = datetime.utcnow()
        month_start = datetime(now.year, now.month, 1)

        # Query total tokens this month from database
        from .database import UsageEvent
        from sqlalchemy import func

        total_tokens = (
            db_session.query(func.sum(UsageEvent.total_tokens))
            .filter(
                UsageEvent.tenant_id == ctx.tenant_id,
                UsageEvent.timestamp >= month_start,
                UsageEvent.success == True,
            )
            .scalar()
        ) or 0

        if total_tokens >= limit:
            logger.warning(
                "monthly_quota_exceeded",
                tenant_id=ctx.tenant_id,
                total=total_tokens,
                limit=limit,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Monthly quota exceeded: {limit:,} tokens per month. Used: {total_tokens:,}",
            )

        logger.debug("monthly_quota_check", tenant_id=ctx.tenant_id, used=total_tokens, limit=limit)

    async def check_concurrent_requests(self, ctx: TenantContext) -> None:
        """
        Check if tenant has exceeded max concurrent requests
        Uses Redis counter with short TTL
        """
        # Get limit based on tier
        if ctx.is_enterprise:
            limit = settings.MAX_CONCURRENT_ENTERPRISE
        else:
            limit = settings.MAX_CONCURRENT_REVENUE_SHARE

        # Redis key for concurrent request tracking
        key = f"concurrent:requests:{ctx.tenant_id}"

        # Get current count
        count = await self.redis.get(key)
        current = int(count) if count else 0

        if current >= limit:
            logger.warning(
                "concurrent_limit_exceeded",
                tenant_id=ctx.tenant_id,
                current=current,
                limit=limit,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Too many concurrent requests: max {limit}",
                headers={"Retry-After": "5"},
            )

        logger.debug("concurrent_check", tenant_id=ctx.tenant_id, current=current, limit=limit)

    async def acquire_request_slot(self, ctx: TenantContext, ttl: int = 300) -> str:
        """
        Acquire a concurrent request slot
        Returns request ID that must be released after request completes
        """
        key = f"concurrent:requests:{ctx.tenant_id}"
        request_id = f"req_{int(time.time() * 1000)}"

        # Increment counter
        await self.redis.incr(key)
        await self.redis.expire(key, ttl)  # Auto-release if not explicitly released

        return request_id

    async def release_request_slot(self, ctx: TenantContext) -> None:
        """Release concurrent request slot"""
        key = f"concurrent:requests:{ctx.tenant_id}"
        current = await self.redis.get(key)

        if current and int(current) > 0:
            await self.redis.decr(key)


# Global rate limiter instance (initialized in main app)
_rate_limiter: Optional[RateLimiter] = None


def init_rate_limiter(redis_client: redis.Redis):
    """Initialize global rate limiter"""
    global _rate_limiter
    _rate_limiter = RateLimiter(redis_client)
    logger.info("rate_limiter_initialized")


def get_rate_limiter() -> Optional[RateLimiter]:
    """Get global rate limiter instance (None if Redis unavailable)"""
    return _rate_limiter
