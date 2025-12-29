"""
Authentication and authorization
API key validation, tenant context, admin auth
"""
import secrets
import hashlib
from typing import Optional
from fastapi import Depends, HTTPException, Header, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from passlib.context import CryptContext
import structlog

from .database import get_db, APIKey, Tenant, TenantStatus
from .config import settings

logger = structlog.get_logger()

# Password/key hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security scheme for OpenAPI
security = HTTPBearer()


def generate_api_key() -> str:
    """
    Generate a secure random API key
    Format: sk_memopt_<32_random_chars>
    """
    random_part = secrets.token_urlsafe(32)[:32]  # 32 chars
    return f"sk_memopt_{random_part}"


def hash_api_key(api_key: str) -> str:
    """Hash API key using bcrypt"""
    return pwd_context.hash(api_key)


def verify_api_key(api_key: str, key_hash: str) -> bool:
    """Verify API key against stored hash"""
    return pwd_context.verify(api_key, key_hash)


def get_key_prefix(api_key: str) -> str:
    """Extract key prefix for database lookup (first 12 chars)"""
    return api_key[:12] if len(api_key) >= 12 else api_key


class TenantContext:
    """
    Tenant context for authenticated requests
    Contains tenant, API key, and authorization info
    """

    def __init__(self, tenant: Tenant, api_key: APIKey):
        self.tenant = tenant
        self.api_key = api_key

    @property
    def tenant_id(self) -> int:
        return self.tenant.id

    @property
    def tenant_name(self) -> str:
        return self.tenant.name

    @property
    def tier(self) -> str:
        return self.tenant.tier.value

    @property
    def is_active(self) -> bool:
        return self.tenant.status == TenantStatus.ACTIVE

    @property
    def is_enterprise(self) -> bool:
        return self.tier == "enterprise_flat"

    @property
    def is_revenue_share(self) -> bool:
        return self.tier == "revenue_share"

    def get_allowed_models(self) -> Optional[list[str]]:
        """Get list of allowed models (None = all models)"""
        if not self.tenant.allowed_models:
            return None
        return [m.strip() for m in self.tenant.allowed_models.split(",")]

    def is_model_allowed(self, model: str) -> bool:
        """Check if model is allowed for this tenant"""
        allowed = self.get_allowed_models()
        if allowed is None:
            return True  # All models allowed
        return model in allowed

    def __repr__(self):
        return f"<TenantContext(id={self.tenant_id}, name={self.tenant_name}, tier={self.tier})>"


async def get_tenant_from_api_key(
    x_api_key: Optional[str] = Header(None, description="API key"),
    authorization: Optional[str] = Header(None, description="Bearer token"),
    db: Session = Depends(get_db),
) -> TenantContext:
    """
    Extract and validate API key from headers
    Supports both X-API-Key header and Authorization: Bearer header
    """
    # Try X-API-Key header first
    api_key = x_api_key

    # Fallback to Authorization header (Bearer token)
    if not api_key and authorization:
        if authorization.startswith("Bearer "):
            api_key = authorization[7:]  # Remove "Bearer " prefix

    if not api_key:
        logger.warning("missing_api_key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header or Authorization: Bearer <key>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Validate API key format
    if not api_key.startswith("sk_memopt_"):
        logger.warning("invalid_api_key_format", prefix=api_key[:12])
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key format",
        )

    # Look up API key by prefix (fast index scan)
    prefix = get_key_prefix(api_key)
    api_key_record = db.query(APIKey).filter(
        APIKey.key_prefix == prefix,
        APIKey.revoked_at.is_(None),  # Only active keys
    ).first()

    if not api_key_record:
        logger.warning("api_key_not_found", prefix=prefix)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    # Verify key hash
    if not verify_api_key(api_key, api_key_record.key_hash):
        logger.warning("api_key_hash_mismatch", key_id=api_key_record.id)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    # Load tenant
    tenant = db.query(Tenant).filter(Tenant.id == api_key_record.tenant_id).first()
    if not tenant:
        logger.error("tenant_not_found", tenant_id=api_key_record.tenant_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Tenant not found",
        )

    # Check tenant status
    if tenant.status != TenantStatus.ACTIVE:
        logger.warning("tenant_suspended", tenant_id=tenant.id, status=tenant.status)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Account suspended. Contact support.",
        )

    # Update last_used_at timestamp (async, don't block)
    api_key_record.last_used_at = datetime.utcnow()
    db.commit()

    logger.info(
        "authenticated",
        tenant_id=tenant.id,
        tenant_name=tenant.name,
        tier=tenant.tier.value,
        api_key_id=api_key_record.id,
    )

    return TenantContext(tenant=tenant, api_key=api_key_record)


async def require_admin(
    x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key"),
    authorization: Optional[str] = Header(None),
) -> bool:
    """
    Require admin API key
    Used for privileged operations (create tenant, etc.)
    """
    admin_key = x_admin_key

    # Fallback to Authorization header
    if not admin_key and authorization:
        if authorization.startswith("Bearer "):
            admin_key = authorization[7:]

    if not admin_key:
        logger.warning("admin_auth_missing")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing admin API key. Provide X-Admin-Key header.",
        )

    # Simple constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(admin_key, settings.ADMIN_API_KEY):
        logger.warning("admin_auth_failed", key_prefix=admin_key[:8])
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid admin API key",
        )

    logger.info("admin_authenticated")
    return True


# Fix import
from datetime import datetime
