"""
API key management for memopt.

Key format  : sk-memopt-<64 hex chars>   (32 random bytes)
Storage     : ~/.memopt/api_key           (chmod 600)
Env override: MEMOPT_API_KEY             (takes priority over file)

Usage
-----
Server startup::

    from memopt.auth.api_key import get_or_create_key, verify_key
    _API_KEY = get_or_create_key()

FastAPI dependency::

    from fastapi import Header, HTTPException, Security
    from fastapi.security.api_key import APIKeyHeader

    _api_key_header = APIKeyHeader(name="X-Memopt-API-Key", auto_error=False)

    def verify_api_key(key: str = Security(_api_key_header)):
        if not verify_key(key, _API_KEY):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

Client usage::

    import urllib.request
    from memopt.auth.api_key import load_key
    req = urllib.request.Request(url, headers={"X-Memopt-API-Key": load_key() or ""})
"""
import hmac
import logging
import os
import secrets
import stat
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

KEY_PREFIX = "sk-memopt-"
KEY_ENV_VAR = "MEMOPT_API_KEY"
_DEFAULT_KEY_PATH = Path.home() / ".memopt" / "api_key"
_MODE_600 = stat.S_IRUSR | stat.S_IWUSR   # owner read+write only


def _secure_write(path: Path, content: str) -> None:
    """
    Write content to path with mode 0600 using an atomic rename.

    Protocol:
      1. Open path.tmp with O_CREAT | O_TRUNC and mode 0600
      2. Write content
      3. Rename path.tmp → path  (atomic on POSIX)
      4. Harden final file permissions

    A crash between steps 1 and 3 leaves a .tmp file with no damage
    to the original. A crash between steps 3 and 4 leaves the final
    file readable — chmod is idempotent.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    fd  = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _MODE_600)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    tmp.rename(path)
    path.chmod(_MODE_600)   # harden in case rename preserved wrong perms


def generate_key() -> str:
    """Return a fresh key: ``sk-memopt-<64 hex chars>``."""
    return KEY_PREFIX + secrets.token_hex(32)


def save_key(key: str, path: Path = None) -> Path:
    """
    Write *key* to *path* (default ``~/.memopt/api_key``) with permissions 600.

    Uses an atomic rename so a crash mid-write cannot corrupt the file.
    Creates parent directories as needed.
    Returns the path actually written.
    """
    dest = path or _DEFAULT_KEY_PATH
    _secure_write(dest, key)
    log.info("API key saved to %s", dest)
    return dest


def load_key(path: Path = None) -> Optional[str]:
    """
    Return the stored API key, or *None* if no key file exists.

    Priority: MEMOPT_API_KEY env var > ~/.memopt/api_key file.
    Env var is checked BEFORE any filesystem access.
    Never logs the actual key — uses :func:`mask_key` for any log lines.
    """
    # Env var takes priority — no disk read needed
    env_key = os.environ.get(KEY_ENV_VAR)
    if env_key:
        log.debug("API key loaded from env var %s (%s)", KEY_ENV_VAR, mask_key(env_key))
        return env_key.strip()

    src = path or _DEFAULT_KEY_PATH
    if not src.exists():
        return None

    # Warn if file has wrong permissions
    file_mode = src.stat().st_mode & 0o777
    if file_mode != 0o600:
        log.warning(
            "API key file %s has permissions %s — should be 0600. "
            "Run: chmod 600 %s",
            src, oct(file_mode), src,
        )

    try:
        key = src.read_text().strip()
        log.debug("API key loaded from %s (%s)", src, mask_key(key))
        return key
    except OSError as e:
        log.warning("Cannot read API key file: %s", e)
        return None


def get_or_create_key(path: Path = None) -> str:
    """
    Return existing key if present, otherwise generate and persist a new one.

    This is the canonical function to call at server startup.
    """
    existing = load_key(path)
    if existing:
        return existing
    key = generate_key()
    save_key(key, path)
    log.info("Generated new memopt API key (%s) — store this securely", mask_key(key))
    return key


def verify_key(provided: Optional[str], stored: Optional[str]) -> bool:
    """
    Constant-time comparison of *provided* vs *stored* API key.

    Always uses ``hmac.compare_digest`` — never ``==`` — to prevent
    timing-based key enumeration attacks.

    Returns ``False`` (rather than raising) for any falsy input so callers
    can use a simple ``if not verify_key(...): raise HTTPException(401)``.
    """
    if not provided or not stored:
        return False
    return hmac.compare_digest(
        provided.encode("utf-8"),
        stored.encode("utf-8"),
    )


def mask_key(key: Optional[str]) -> str:
    """
    Return a safe-to-log representation of *key*.

    Example: ``sk-memopt-ab12cd…[redacted]``
    """
    if not key:
        return "<none>"
    visible = key[:18]  # "sk-memopt-" + 8 hex chars
    return f"{visible}…[redacted]"
