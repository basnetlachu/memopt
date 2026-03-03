"""
License validation via Keygen.sh.

Every memopt feature that requires a paid license calls check_license()
before proceeding.

License entitlements stored in metadata on Keygen:
  gpu_limit: int or "unlimited"
  tier: "pro" | "enterprise" | "cluster"

Validation is cached for 1 hour — does not call Keygen on every GPU
scan cycle.

If Keygen is unreachable: fail open with warning for up to 24 hours
(grace period). After 24 hours: fail closed to free tier.
This prevents memopt from breaking if Keygen has downtime.

Never fails in ways that corrupt customer data.
Never deletes optimizations if license check fails.
"""
import os
import json
import time
import logging
import urllib.request
import urllib.error
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

# Keygen account — replace with your actual account ID
KEYGEN_ACCOUNT_ID = os.getenv(
    "KEYGEN_ACCOUNT_ID", "a44cf991-48c9-435b-bad6-9ed7dd32cc3e"
)
KEYGEN_VALIDATE_URL = (
    f"https://api.keygen.sh/v1/accounts/{KEYGEN_ACCOUNT_ID}"
    f"/licenses/actions/validate-key"
)

# Cache validation result for 1 hour
CACHE_TTL_SECONDS = 3600
# Fail open for 24 hours if Keygen unreachable
GRACE_PERIOD_SECONDS = 86400

LICENSE_CACHE_FILE = Path.home() / ".memopt" / "license_cache.json"


@dataclass
class LicenseStatus:
    valid: bool
    tier: str                    # free | pro | enterprise | cluster
    gpu_limit: int               # -1 = unlimited
    expiry: Optional[str]        # ISO date or None
    cached_at: float             # unix timestamp
    error: Optional[str]         # None if valid


# In-memory cache
_cache: Optional[LicenseStatus] = None


def validate_license(license_key: str) -> LicenseStatus:
    """
    Validate license key against Keygen.sh.
    Returns cached result if within TTL.
    Fails open with warning if Keygen unreachable.
    """
    global _cache

    # Return in-memory cache if fresh
    if _cache and (time.time() - _cache.cached_at) < CACHE_TTL_SECONDS:
        return _cache

    # Try to load disk cache
    disk_cache = _load_disk_cache()
    if disk_cache and (time.time() - disk_cache.cached_at) < CACHE_TTL_SECONDS:
        _cache = disk_cache
        return _cache

    # Call Keygen
    status = _call_keygen(license_key)

    # Cache result
    _cache = status
    _save_disk_cache(status)

    return status


def _call_keygen(license_key: str) -> LicenseStatus:
    """Make API call to Keygen.sh."""
    try:
        payload = json.dumps({"meta": {"key": license_key}}).encode()
        req = urllib.request.Request(
            KEYGEN_VALIDATE_URL,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        meta = data.get("meta", {})
        lic_data = data.get("data", {})
        attrs = lic_data.get("attributes", {}) if lic_data else {}
        metadata = attrs.get("metadata", {}) if attrs else {}

        if not meta.get("valid"):
            reason = meta.get("detail", "invalid key")
            log.warning("License invalid: %s", reason)
            return LicenseStatus(
                valid=False,
                tier="free",
                gpu_limit=4,
                expiry=None,
                cached_at=time.time(),
                error=reason,
            )

        gpu_limit_raw = metadata.get("gpu_limit", "4")
        gpu_limit = (
            -1 if gpu_limit_raw == "unlimited"
            else int(gpu_limit_raw)
        )

        return LicenseStatus(
            valid=True,
            tier=metadata.get("tier", "pro"),
            gpu_limit=gpu_limit,
            expiry=attrs.get("expiry") if attrs else None,
            cached_at=time.time(),
            error=None,
        )

    except urllib.error.URLError as e:
        log.warning("Keygen unreachable: %s", e)
        return _fail_open(str(e))
    except Exception as e:
        log.warning("License check failed: %s", e)
        return _fail_open(str(e))


def _fail_open(error: str) -> LicenseStatus:
    """
    Keygen unreachable — use disk cache if within grace period.
    If no disk cache or grace period expired — fail closed to free tier.
    """
    disk_cache = _load_disk_cache()

    if disk_cache and disk_cache.valid:
        age = time.time() - disk_cache.cached_at
        if age < GRACE_PERIOD_SECONDS:
            log.warning(
                "Keygen unreachable — using cached license "
                "(grace period: %dh remaining)",
                int((GRACE_PERIOD_SECONDS - age) / 3600),
            )
            # Return cached but note the error
            disk_cache.error = f"Keygen unreachable: {error}"
            return disk_cache
        else:
            log.error(
                "License validation grace period expired. "
                "Connect to internet and restart memopt."
            )

    # No valid cache — fail closed to free tier
    return LicenseStatus(
        valid=False,
        tier="free",
        gpu_limit=4,
        expiry=None,
        cached_at=time.time(),
        error=f"Cannot validate license: {error}",
    )


def check_gpu_limit(license_status: LicenseStatus, requested_gpus: int) -> bool:
    """
    Check if requested GPU count is within license entitlement.
    Returns True if allowed, False if exceeds license.
    gpu_limit == -1 means unlimited.
    """
    if license_status.gpu_limit == -1:
        return True  # unlimited
    return requested_gpus <= license_status.gpu_limit


def get_license_key() -> str:
    """
    Get license key from env or file.
    Returns empty string if not configured.
    Priority: MEMOPT_LICENSE_KEY env var > ~/.memopt/license_key file.
    """
    env_key = os.getenv("MEMOPT_LICENSE_KEY", "")
    if env_key:
        return env_key.strip()

    key_file = Path.home() / ".memopt" / "license_key"
    if key_file.exists():
        return key_file.read_text().strip()

    return ""


def _load_disk_cache() -> Optional[LicenseStatus]:
    """Load license status from disk cache. Returns None on any error."""
    try:
        if not LICENSE_CACHE_FILE.exists():
            return None
        data = json.loads(LICENSE_CACHE_FILE.read_text())
        return LicenseStatus(**data)
    except Exception:
        return None


def _save_disk_cache(status: LicenseStatus) -> None:
    """Persist license status to disk. Swallows exceptions (non-critical)."""
    try:
        LICENSE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        LICENSE_CACHE_FILE.write_text(json.dumps({
            "valid": status.valid,
            "tier": status.tier,
            "gpu_limit": status.gpu_limit,
            "expiry": status.expiry,
            "cached_at": status.cached_at,
            "error": status.error,
        }))
        LICENSE_CACHE_FILE.chmod(0o600)
    except Exception as e:
        log.warning("Failed to cache license: %s", e)


def require_license(feature: str = ""):
    """
    Decorator for features requiring a valid paid license.
    Raises PermissionError if license key is absent or invalid.

    Usage:
        @require_license("cluster_dashboard")
        def start_control_plane():
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            key = get_license_key()
            if not key:
                raise PermissionError(
                    f"Feature '{feature}' requires a paid license. "
                    f"Get yours at memopt.ai"
                )
            status = validate_license(key)
            if not status.valid:
                raise PermissionError(
                    f"License invalid: {status.error}. "
                    f"Contact support@memopt.ai"
                )
            return func(*args, **kwargs)
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator
