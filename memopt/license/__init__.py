"""
memopt license validation package.

Validates license keys against Keygen.sh.
Enforces GPU entitlements.
Provides graceful fail-open behavior during Keygen outages.
"""

from memopt.license.validator import (
    LicenseStatus,
    validate_license,
    check_gpu_limit,
    get_license_key,
    require_license,
    KEYGEN_ACCOUNT_ID,
    CACHE_TTL_SECONDS,
    GRACE_PERIOD_SECONDS,
)

__all__ = [
    "LicenseStatus",
    "validate_license",
    "check_gpu_limit",
    "get_license_key",
    "require_license",
    "KEYGEN_ACCOUNT_ID",
    "CACHE_TTL_SECONDS",
    "GRACE_PERIOD_SECONDS",
]
