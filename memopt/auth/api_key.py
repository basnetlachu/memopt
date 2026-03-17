"""
API key management — multi-tenant.

Key format:  memopt_{tenant_id}_{secret_hex}
  tenant_id: alphanumeric, max 32 chars, URL-safe
  secret:    32 bytes of cryptographic randomness (64 hex chars)

Storage: ~/.memopt/keys/{tenant_id}.key  (mode 0600 each)
Admin key: ~/.memopt/keys/_admin.key     (special tenant_id "_admin")

Environment overrides (checked before disk):
  MEMOPT_API_KEY          single-key mode (backwards compatible)
  MEMOPT_ADMIN_KEY        admin key override

Single-key backwards compatibility:
  If MEMOPT_API_KEY is set, it is accepted for any request and
  assigned to the "_default" tenant. This preserves existing
  deployments that use the old single-key model.
"""
from __future__ import annotations
import os
import re
import stat
import hmac
import secrets
import logging
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)

_KEY_DIR   = os.path.expanduser(os.environ.get(
    "MEMOPT_KEY_DIR", "~/.memopt/keys"
))
_MODE_600  = stat.S_IRUSR | stat.S_IWUSR
_TENANT_RE = re.compile(r"^[a-zA-Z0-9_-]{1,32}$")

# Reserved tenant IDs
_ADMIN_TENANT   = "_admin"
_DEFAULT_TENANT = "_default"

# Backwards-compat constants
KEY_PREFIX  = "sk-memopt-"
KEY_ENV_VAR = "MEMOPT_API_KEY"


# ── Key format helpers ─────────────────────────────────────────────────

def _make_key(tenant_id: str) -> str:
    """Generate a new key for a tenant."""
    return f"memopt_{tenant_id}_{secrets.token_hex(32)}"


def _parse_key(key: str) -> Optional[Tuple[str, str]]:
    """
    Parse a key into (tenant_id, secret).
    Returns None if the key does not match the expected format.
    """
    parts = key.split("_", 2)
    if len(parts) != 3 or parts[0] != "memopt":
        return None
    return parts[1], parts[2]


# ── Secure file helpers ────────────────────────────────────────────────

def _key_path(tenant_id: str) -> str:
    return os.path.join(_KEY_DIR, f"{tenant_id}.key")


def _secure_write(path: str, content: str) -> None:
    """Write content to path with mode 0600 using atomic rename."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    fd  = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _MODE_600)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    os.rename(tmp, path)
    os.chmod(path, _MODE_600)


def _secure_read(path: str) -> Optional[str]:
    """Read key from path, warn if permissions are wrong."""
    if not os.path.exists(path):
        return None
    mode = os.stat(path).st_mode & 0o777
    if mode != 0o600:
        logger.warning(
            f"Key file {path} has permissions {oct(mode)} "
            f"— should be 0600. Run: chmod 600 {path}"
        )
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError as e:
        logger.warning(f"Cannot read key file {path}: {e}")
        return None


# ── Tenant management ──────────────────────────────────────────────────

def create_tenant(tenant_id: str) -> str:
    """
    Create a new tenant and return its API key.
    Raises ValueError if tenant_id is invalid or already exists.
    """
    if not _TENANT_RE.match(tenant_id):
        raise ValueError(
            f"Invalid tenant_id '{tenant_id}'. "
            f"Must match [a-zA-Z0-9_-]{{1,32}}"
        )
    if tenant_id in (_ADMIN_TENANT, _DEFAULT_TENANT):
        raise ValueError(f"Cannot create reserved tenant '{tenant_id}'")
    path = _key_path(tenant_id)
    if os.path.exists(path):
        raise ValueError(f"Tenant '{tenant_id}' already exists")
    key = _make_key(tenant_id)
    _secure_write(path, key)
    logger.info(f"Created tenant '{tenant_id}'")
    return key


def revoke_tenant(tenant_id: str) -> bool:
    """
    Revoke a tenant's API key. Returns True if key was removed.
    Does not delete the tenant's ledger entries — those are append-only.
    """
    if tenant_id in (_ADMIN_TENANT,):
        raise ValueError(f"Cannot revoke reserved tenant '{tenant_id}'")
    path = _key_path(tenant_id)
    if not os.path.exists(path):
        return False
    os.remove(path)
    logger.info(f"Revoked tenant '{tenant_id}'")
    return True


def list_tenants() -> list:
    """Return a list of all tenant IDs with active keys."""
    try:
        return [
            f[:-4] for f in os.listdir(_KEY_DIR)
            if f.endswith(".key") and not f.startswith(".")
        ]
    except OSError:
        return []


def authenticate(provided_key: Optional[str]) -> Optional[str]:
    """
    Verify a key and return the tenant_id if valid, else None.

    Priority:
    1. MEMOPT_API_KEY env var → tenant "_default" (backwards compat)
    2. Parse tenant_id from key format → load and compare stored key
    3. Scan all tenant keys (for keys created before format change)
    """
    if not provided_key:
        return None

    # Backwards compatibility: single env var key
    env_key = os.environ.get("MEMOPT_API_KEY", "").strip()
    if env_key and hmac.compare_digest(
        provided_key.encode(), env_key.encode()
    ):
        return _DEFAULT_TENANT

    # Parse tenant from key format
    parsed = _parse_key(provided_key)
    if parsed:
        tenant_id, _ = parsed
        stored = _secure_read(_key_path(tenant_id))
        if stored and hmac.compare_digest(
            provided_key.encode(), stored.encode()
        ):
            return tenant_id

    return None


def is_admin(tenant_id: str) -> bool:
    """Return True if the tenant has admin privileges."""
    return tenant_id in (_ADMIN_TENANT, _DEFAULT_TENANT)


# ── Backwards-compatible single-key API ────────────────────────────────

def generate_key() -> str:
    """
    Generate a key for the default single-tenant deployment.
    Backwards compatible with the old single-key model.
    """
    if not os.environ.get("MEMOPT_API_KEY"):
        os.makedirs(_KEY_DIR, exist_ok=True)
        key = _make_key(_DEFAULT_TENANT)
        _secure_write(_key_path(_DEFAULT_TENANT), key)
        return key
    return os.environ["MEMOPT_API_KEY"]


def load_key(path=None) -> Optional[str]:
    """Load the default key. Checks env var first."""
    env_key = os.environ.get("MEMOPT_API_KEY", "").strip()
    if env_key:
        return env_key
    return _secure_read(_key_path(_DEFAULT_TENANT))


def verify_key(provided: Optional[str], stored: Optional[str]) -> bool:
    """Constant-time comparison. Backwards compatible."""
    if not provided or not stored:
        return False
    return hmac.compare_digest(
        provided.encode("utf-8"),
        stored.encode("utf-8"),
    )


def save_key(key: str, path=None) -> str:
    """Backwards compat shim — writes key for the default tenant."""
    dest = _key_path(_DEFAULT_TENANT)
    _secure_write(dest, key)
    logger.info("API key saved to %s", dest)
    return dest


def get_or_create_key(path=None) -> str:
    """
    Return existing key if present, otherwise generate and persist a new one.
    Backwards compat shim for server startup.
    """
    existing = load_key()
    if existing:
        return existing
    return generate_key()


def mask_key(key: Optional[str]) -> str:
    """Return a safe-to-log representation of key."""
    if not key:
        return "<none>"
    visible = key[:18]
    return f"{visible}…[redacted]"
