"""
Database models and session management
SQLAlchemy models for tenants, API keys, usage events, and invoices
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    Text,
    BigInteger,
    Enum as SQLEnum,
    ForeignKey,
    Index,
    CheckConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.pool import QueuePool
import enum

from .config import settings

# Base class for models
Base = declarative_base()


class TenantTier(str, enum.Enum):
    """Tenant pricing tiers"""
    ENTERPRISE_FLAT = "enterprise_flat"
    REVENUE_SHARE = "revenue_share"


class TenantStatus(str, enum.Enum):
    """Tenant account status"""
    ACTIVE = "active"
    SUSPENDED = "suspended"


class InvoiceStatus(str, enum.Enum):
    """Invoice payment status"""
    DRAFT = "draft"
    ISSUED = "issued"
    PAID = "paid"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"


class Tenant(Base):
    """Tenant/Customer model"""
    __tablename__ = "tenants"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    tier = Column(SQLEnum(TenantTier), nullable=False, index=True)
    status = Column(SQLEnum(TenantStatus), nullable=False, default=TenantStatus.ACTIVE, index=True)

    # Feature flags (JSON stored as text, parse as needed)
    allowed_models = Column(Text, nullable=True, comment="Comma-separated list of allowed models")
    max_context_length = Column(Integer, nullable=True, default=4096)
    priority_level = Column(Integer, nullable=False, default=1, comment="1=low, 10=high")

    # Metadata
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    notes = Column(Text, nullable=True)

    # Relationships
    api_keys = relationship("APIKey", back_populates="tenant", cascade="all, delete-orphan")
    usage_events = relationship("UsageEvent", back_populates="tenant", cascade="all, delete-orphan")
    invoices = relationship("Invoice", back_populates="tenant", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Tenant(id={self.id}, name={self.name}, tier={self.tier}, status={self.status})>"


class APIKey(Base):
    """API key model - stores hashed keys only"""
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)

    # Key storage (never store plaintext!)
    key_hash = Column(String(255), nullable=False, unique=True, index=True, comment="bcrypt hash of API key")
    key_prefix = Column(String(16), nullable=False, index=True, comment="First 8-12 chars for lookup")

    # Key metadata
    name = Column(String(255), nullable=True, comment="Human-readable key name")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    last_used_at = Column(DateTime, nullable=True, index=True)
    revoked_at = Column(DateTime, nullable=True, index=True)

    # Relationships
    tenant = relationship("Tenant", back_populates="api_keys")

    @property
    def is_active(self) -> bool:
        """Check if key is active (not revoked)"""
        return self.revoked_at is None

    def __repr__(self):
        return f"<APIKey(id={self.id}, prefix={self.key_prefix}, active={self.is_active})>"


class UsageEvent(Base):
    """Usage event model - one record per inference request"""
    __tablename__ = "usage_events"

    id = Column(BigInteger, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)

    # Request tracking
    request_id = Column(String(64), nullable=False, unique=True, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Model and usage
    model = Column(String(255), nullable=False, index=True)
    prompt_tokens = Column(Integer, nullable=False, default=0)
    completion_tokens = Column(Integer, nullable=False, default=0)
    total_tokens = Column(Integer, nullable=False, default=0, index=True)

    # Performance
    latency_ms = Column(Float, nullable=True)

    # Status
    success = Column(Boolean, nullable=False, default=True, index=True)
    error_code = Column(String(64), nullable=True, index=True)
    error_message = Column(Text, nullable=True)

    # Relationships
    tenant = relationship("Tenant", back_populates="usage_events")

    # Indexes for fast queries
    __table_args__ = (
        Index("idx_usage_tenant_timestamp", "tenant_id", "timestamp"),
        Index("idx_usage_timestamp_success", "timestamp", "success"),
    )

    def __repr__(self):
        return f"<UsageEvent(id={self.id}, tenant_id={self.tenant_id}, tokens={self.total_tokens}, success={self.success})>"


class Invoice(Base):
    """Invoice model - monthly billing records"""
    __tablename__ = "invoices"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)

    # Billing period
    period_start = Column(DateTime, nullable=False, index=True)
    period_end = Column(DateTime, nullable=False, index=True)

    # Tier at time of invoice
    tier = Column(SQLEnum(TenantTier), nullable=False)

    # Usage summary
    total_tokens = Column(BigInteger, nullable=False, default=0)
    total_requests = Column(Integer, nullable=False, default=0)

    # Billing
    amount_usd = Column(Float, nullable=False)
    status = Column(SQLEnum(InvoiceStatus), nullable=False, default=InvoiceStatus.DRAFT, index=True)

    # Metadata
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    issued_at = Column(DateTime, nullable=True)
    paid_at = Column(DateTime, nullable=True)
    notes = Column(Text, nullable=True)

    # Relationships
    tenant = relationship("Tenant", back_populates="invoices")

    # Constraints
    __table_args__ = (
        CheckConstraint("amount_usd >= 0", name="check_amount_positive"),
        Index("idx_invoice_tenant_period", "tenant_id", "period_start", "period_end"),
    )

    def __repr__(self):
        return f"<Invoice(id={self.id}, tenant_id={self.tenant_id}, amount=${self.amount_usd}, status={self.status})>"


# Database engine and session
engine = create_engine(
    settings.DATABASE_URL,
    poolclass=QueuePool,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_pre_ping=True,  # Verify connections before using
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Session:
    """
    Dependency for FastAPI to get database session
    Usage: db: Session = Depends(get_db)
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Create all tables (for development/testing)"""
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created")


def drop_tables():
    """Drop all tables (dangerous!)"""
    Base.metadata.drop_all(bind=engine)
    print("⚠️  Database tables dropped")
