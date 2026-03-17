"""
Optimization certificate — signs a ledger entry and produces a
verifiable JSON document.

The certificate proves:
  - Which node generated these tokens
  - How much energy, CO₂, and cost was saved
  - That the figures have not been tampered with since signing

Signing algorithm: HMAC-SHA256 over the canonical JSON of the
LedgerEntry fields (sorted keys, no whitespace).

Verification: anyone with the signing key can re-derive the HMAC
and compare it to the signature field.

Environment variables:
  MEMOPT_SIGNING_KEY   str   HMAC signing key (hex or plain text).
                              If not set, certificates are produced
                              without a signature and marked
                              signature_status: "unsigned".
                              Unsigned certificates are valid for
                              internal reporting but not for external
                              audit submission.
"""
from __future__ import annotations
import os
import json
import time
import hmac
import hashlib
import logging
from dataclasses import asdict

logger = logging.getLogger(__name__)

# MEMOPT_SIGNING_KEY must be set via environment variable only.
# The application never writes this key to disk.
# In production: inject via Kubernetes secret, AWS SSM, or Vault.
_SIGNING_KEY  = os.environ.get("MEMOPT_SIGNING_KEY", "")
_CERT_VERSION = "1.0"


def _canonical_json(data: dict) -> bytes:
    """
    Produce a canonical (deterministic) JSON encoding for signing.
    Sorted keys, no extra whitespace, UTF-8 encoded.
    """
    return json.dumps(data, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8")


def sign_entry(entry) -> dict:
    """
    Produce a signed Optimization Certificate from a LedgerEntry.

    Returns a dict with:
      version            str    certificate format version
      issued_at          float  Unix timestamp of signing
      payload            dict   the full ledger entry
      signature          str    HMAC-SHA256 hex digest (or None)
      signature_status   str    "signed" | "unsigned"
      signing_algorithm  str    "HMAC-SHA256" | "none"

    The signature covers: canonical_json(payload + issued_at + version).
    """
    issued_at = time.time()
    payload   = asdict(entry) if hasattr(entry, "__dataclass_fields__") \
                else dict(entry)

    sign_target = {
        "version":   _CERT_VERSION,
        "issued_at": issued_at,
        "payload":   payload,
    }
    canonical = _canonical_json(sign_target)

    if _SIGNING_KEY:
        key_bytes = _SIGNING_KEY.encode("utf-8")
        signature = hmac.new(key_bytes, canonical, hashlib.sha256).hexdigest()
        status    = "signed"
        algorithm = "HMAC-SHA256"
    else:
        signature = None
        status    = "unsigned"
        algorithm = "none"
        logger.debug(
            "MEMOPT_SIGNING_KEY not set — certificate produced unsigned. "
            "Set the env var for audit-grade certificates."
        )

    return {
        "version":           _CERT_VERSION,
        "issued_at":         issued_at,
        "payload":           payload,
        "signature":         signature,
        "signature_status":  status,
        "signing_algorithm": algorithm,
    }


def verify_certificate(cert: dict, signing_key: str) -> bool:
    """
    Verify the HMAC-SHA256 signature on a certificate dict.

    Returns True if the signature is valid.
    Returns False if the signature is invalid or missing.

    Usage:
        cert = sign_entry(entry)
        assert verify_certificate(cert, signing_key=os.environ["MEMOPT_SIGNING_KEY"])
    """
    if cert.get("signature_status") == "unsigned":
        logger.warning("Certificate is unsigned — cannot verify")
        return False

    stored_sig = cert.get("signature")
    if not stored_sig:
        return False

    sign_target = {
        "version":   cert["version"],
        "issued_at": cert["issued_at"],
        "payload":   cert["payload"],
    }
    canonical = _canonical_json(sign_target)
    key_bytes = signing_key.encode("utf-8")
    expected  = hmac.new(key_bytes, canonical, hashlib.sha256).hexdigest()

    # Constant-time comparison prevents timing attacks
    return hmac.compare_digest(stored_sig, expected)
