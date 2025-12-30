"""
MemOpt License Validation
Validates Ed25519-signed license files for on-premise deployments
"""
import os
import json
import base64
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional

try:
    from nacl.signing import VerifyKey
    from nacl.exceptions import BadSignatureError
    NACL_AVAILABLE = True
except ImportError:
    NACL_AVAILABLE = False


# Embedded public key for license verification
# Generated with: python -c "from nacl.signing import SigningKey; sk = SigningKey.generate(); print(sk.verify_key.encode().hex())"
PUBLIC_KEY_HEX = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"


class LicenseError(Exception):
    """License validation error"""
    pass


class License:
    """MemOpt license with signature verification"""

    def __init__(self, data: Dict):
        self.customer_id = data.get("customer_id", "")
        self.tier = data.get("tier", "")
        self.expires_at = data.get("expires_at", "")
        self.max_gpus = data.get("max_gpus", 0)
        self.features = data.get("features", [])
        self.telemetry_required = data.get("telemetry_required", False)
        self.signature = data.get("signature", "")
        self._raw_data = data

    def verify(self) -> None:
        """Verify license signature and validity"""
        if not NACL_AVAILABLE:
            raise LicenseError("PyNaCl not installed. Install with: pip install pynacl")

        if not self.signature:
            raise LicenseError("License signature missing")

        # Verify signature
        try:
            verify_key = VerifyKey(bytes.fromhex(PUBLIC_KEY_HEX))

            # Create payload without signature
            payload_data = {k: v for k, v in self._raw_data.items() if k != "signature"}
            payload_json = json.dumps(payload_data, sort_keys=True)
            payload_bytes = payload_json.encode('utf-8')

            # Decode and verify signature
            signature_bytes = base64.b64decode(self.signature)
            verify_key.verify(payload_bytes, signature_bytes)

        except BadSignatureError:
            raise LicenseError("Invalid license signature")
        except Exception as e:
            raise LicenseError(f"Signature verification failed: {e}")

        # Check expiration
        try:
            expires = datetime.fromisoformat(self.expires_at.replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)

            if now > expires:
                raise LicenseError(f"License expired on {self.expires_at}")
        except ValueError as e:
            raise LicenseError(f"Invalid expiration date format: {e}")

    def has_feature(self, feature: str) -> bool:
        """Check if license includes a specific feature"""
        return feature in self.features


class LicenseManager:
    """Manages license loading and validation"""

    def __init__(self):
        self.license_path = os.getenv("MEMOPT_LICENSE_PATH", "/etc/memopt/license.json")
        self.dev_mode = os.getenv("MEMOPT_DEV_MODE") == "1"
        self._license: Optional[License] = None

    def load_and_validate(self) -> License:
        """Load and validate license file"""
        # Dev mode bypasses license check
        if self.dev_mode:
            print("[MemOpt] Running in DEV MODE - license check bypassed")
            return License({
                "customer_id": "dev",
                "tier": "dev",
                "expires_at": "2099-12-31T23:59:59Z",
                "max_gpus": 999,
                "features": ["vllm", "all"],
                "telemetry_required": False,
                "signature": ""
            })

        # Check if license file exists
        license_file = Path(self.license_path)
        if not license_file.exists():
            raise LicenseError(
                f"License file not found: {self.license_path}\n"
                f"Set MEMOPT_LICENSE_PATH environment variable or\n"
                f"place license.json at /etc/memopt/license.json\n"
                f"For development, use: MEMOPT_DEV_MODE=1"
            )

        # Load license
        try:
            with open(license_file) as f:
                license_data = json.load(f)
        except json.JSONDecodeError as e:
            raise LicenseError(f"Invalid license JSON: {e}")
        except Exception as e:
            raise LicenseError(f"Failed to read license file: {e}")

        # Create and verify license
        license_obj = License(license_data)
        license_obj.verify()

        self._license = license_obj
        return license_obj

    @property
    def license(self) -> Optional[License]:
        """Get loaded license"""
        return self._license


# Global license manager instance
_license_manager = LicenseManager()


def get_license_manager() -> LicenseManager:
    """Get global license manager instance"""
    return _license_manager


def validate_license() -> License:
    """Convenience function to validate license"""
    return get_license_manager().load_and_validate()
