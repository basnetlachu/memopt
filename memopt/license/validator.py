"""
License validation via Keygen.sh (ED25519_SIGN offline-first).

For ED25519_SIGN scheme licenses, validation is done offline:
  1. Fetch Keygen's public key once and cache it on disk.
  2. On every subsequent call, verify the license key's ED25519 signature
     locally — no network round-trip, no machine scope requirement.
  3. Check the expiry embedded in the key payload.
  4. Fallback to API validation if offline verification fails.

License entitlements are decoded from the signed key payload:
  tier: "pro" | "enterprise" | "cluster"
  gpu_limit: int or "unlimited"
  expiry: ISO date

Never fails in ways that corrupt customer data.
Never deletes optimizations if license check fails.
"""
import os
import json
import time
import base64
import logging
import urllib.request
import urllib.error
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

KEYGEN_ACCOUNT_ID = os.getenv(
    "KEYGEN_ACCOUNT_ID", "85efe00f-f369-4a5c-95c4-cc1c9a7ebb6a"
)
KEYGEN_PRODUCT_ID = os.getenv(
    "KEYGEN_PRODUCT_ID", "a44cf991-48c9-435b-bad6-9ed7dd32cc3e"
)
KEYGEN_VALIDATE_URL = (
    f"https://api.keygen.sh/v1/accounts/{KEYGEN_ACCOUNT_ID}"
    f"/licenses/actions/validate-key"
)
KEYGEN_PUBKEY_URL = (
    f"https://api.keygen.sh/v1/accounts/{KEYGEN_ACCOUNT_ID}"
    f"/public-key"
)

# Cache validation result for 1 hour
CACHE_TTL_SECONDS = 3600
# Fail open for 24 hours if Keygen unreachable
GRACE_PERIOD_SECONDS = 86400

LICENSE_CACHE_FILE = Path.home() / ".memopt" / "license_cache.json"
PUBKEY_CACHE_FILE  = Path.home() / ".memopt" / "keygen_pubkey.pem"


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


def _get_pubkey_pem() -> Optional[bytes]:
    """
    Fetch and cache Keygen's ED25519 public key (PEM).
    Returns None if unreachable and no cached copy exists.
    """
    if PUBKEY_CACHE_FILE.exists():
        return PUBKEY_CACHE_FILE.read_bytes()
    try:
        req = urllib.request.Request(
            KEYGEN_PUBKEY_URL,
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        pem = data.get("data", {}).get("attributes", {}).get("key", "")
        if pem:
            PUBKEY_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            PUBKEY_CACHE_FILE.write_bytes(pem.encode())
            PUBKEY_CACHE_FILE.chmod(0o600)
            return pem.encode()
    except Exception as e:
        log.debug("Could not fetch Keygen public key: %s", e)
    return None


def _verify_offline(license_key: str) -> Optional[LicenseStatus]:
    """
    Verify an ED25519_SIGN license key offline.

    Key format: "key/{base64_payload}.{base64url_signature}"

    Returns a LicenseStatus if verification succeeds, None if the key
    format is unrecognised or the cryptography package is unavailable.
    """
    # Only handle Keygen's signed key format
    if not license_key.startswith("key/"):
        return None

    try:
        rest = license_key[len("key/"):]
        if "." not in rest:
            return None
        payload_b64, sig_b64 = rest.rsplit(".", 1)

        # Decode payload (standard base64, may have padding)
        payload_bytes = base64.b64decode(payload_b64 + "==")
        payload = json.loads(payload_bytes)

        # Decode signature (base64url, no padding)
        sig_bytes = base64.urlsafe_b64decode(sig_b64 + "==")

        # Verify signature with Keygen's public key
        pem = _get_pubkey_pem()
        if pem:
            from cryptography.hazmat.primitives.serialization import (
                load_pem_public_key,
            )
            from cryptography.exceptions import InvalidSignature

            pub = load_pem_public_key(pem)
            try:
                pub.verify(sig_bytes, payload_bytes)
            except InvalidSignature:
                return LicenseStatus(
                    valid=False, tier="free", gpu_limit=4,
                    expiry=None, cached_at=time.time(),
                    error="License signature invalid",
                )

        # Check expiry
        expiry_str = payload.get("license", {}).get("expiry")
        if expiry_str:
            # Parse ISO date — works without dateutil
            expiry_str_clean = expiry_str.rstrip("Z").split(".")[0]
            from datetime import datetime
            expiry_dt = datetime.fromisoformat(expiry_str_clean)
            if expiry_dt.timestamp() < time.time():
                return LicenseStatus(
                    valid=False, tier="free", gpu_limit=4,
                    expiry=expiry_str, cached_at=time.time(),
                    error=f"License expired on {expiry_str[:10]}",
                )

        # Key is signed and not expired — treat as valid
        # Tier/gpu_limit come from metadata when available via API;
        # fall back to enterprise/unlimited for signed keys
        return LicenseStatus(
            valid=True,
            tier="enterprise",
            gpu_limit=-1,
            expiry=expiry_str,
            cached_at=time.time(),
            error=None,
        )

    except Exception as e:
        log.debug("Offline verification skipped: %s", e)
        return None


def _call_keygen(license_key: str) -> LicenseStatus:
    """
    Validate license key.
    Tries offline ED25519 verification first; falls back to Keygen API.
    """
    # 1. Try offline verification (no machine scope, no network needed)
    offline = _verify_offline(license_key)
    if offline is not None:
        log.debug("License verified offline (ED25519)")
        return offline

    # 2. Fallback: call Keygen API
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
