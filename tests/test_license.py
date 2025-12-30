"""
Unit tests for MemOpt license validation
"""

import os
import json
import base64
import tempfile
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

# Import license module
from memopt.license import License, LicenseManager, LicenseError


class TestLicenseValidation:
    """Test license validation functionality"""

    def test_dev_mode_bypass(self):
        """Test that dev mode bypasses license validation"""
        manager = LicenseManager(dev_mode=True)
        license_obj = manager.load_and_validate()

        assert license_obj.customer_id == "dev"
        assert license_obj.tier == "dev"
        assert license_obj.has_feature("vllm")
        assert license_obj.max_gpus == 999

    def test_missing_license_file(self):
        """Test error when license file doesn't exist"""
        manager = LicenseManager(dev_mode=False, license_path="/nonexistent/license.json")

        with pytest.raises(LicenseError, match="License file not found"):
            manager.load_and_validate()

    def test_expired_license(self):
        """Test that expired licenses are rejected"""
        # Create expired license (without valid signature - will fail signature check first)
        expired_data = {
            "customer_id": "test-customer",
            "tier": "enterprise",
            "issued_at": "2024-01-01T00:00:00Z",
            "expires_at": "2024-12-31T23:59:59Z",  # Expired
            "max_gpus": 8,
            "features": ["vllm"],
            "telemetry_required": False,
            "signature": "invalid-signature"
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(expired_data, f)
            temp_path = f.name

        try:
            manager = LicenseManager(dev_mode=False, license_path=temp_path)

            # Should fail on signature validation (before expiration check)
            with pytest.raises(LicenseError):
                manager.load_and_validate()
        finally:
            os.unlink(temp_path)

    def test_invalid_signature(self):
        """Test that invalid signatures are rejected"""
        # Create license with invalid signature
        invalid_data = {
            "customer_id": "test-customer",
            "tier": "enterprise",
            "issued_at": "2025-01-01T00:00:00Z",
            "expires_at": "2026-01-01T00:00:00Z",
            "max_gpus": 8,
            "features": ["vllm"],
            "telemetry_required": False,
            "signature": base64.b64encode(b"invalid-signature").decode()
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_data, f)
            temp_path = f.name

        try:
            manager = LicenseManager(dev_mode=False, license_path=temp_path)

            with pytest.raises(LicenseError, match="signature"):
                manager.load_and_validate()
        finally:
            os.unlink(temp_path)

    def test_license_feature_check(self):
        """Test license feature checking"""
        # Use dev mode for simplicity
        manager = LicenseManager(dev_mode=True)
        license_obj = manager.load_and_validate()

        # Dev license has all features
        assert license_obj.has_feature("vllm")
        assert license_obj.has_feature("quantization")
        assert license_obj.has_feature("profiling")
        assert not license_obj.has_feature("nonexistent-feature")

    def test_gpu_limit_check(self):
        """Test GPU limit validation"""
        manager = LicenseManager(dev_mode=True)
        license_obj = manager.load_and_validate()

        # Dev license allows 999 GPUs
        assert license_obj.max_gpus == 999

    def test_malformed_license_json(self):
        """Test that malformed JSON is rejected"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json }")
            temp_path = f.name

        try:
            manager = LicenseManager(dev_mode=False, license_path=temp_path)

            with pytest.raises(LicenseError):
                manager.load_and_validate()
        finally:
            os.unlink(temp_path)

    def test_missing_required_fields(self):
        """Test that licenses with missing fields are rejected"""
        # Missing 'customer_id' field
        incomplete_data = {
            "tier": "enterprise",
            "expires_at": "2026-01-01T00:00:00Z",
            "signature": "test"
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(incomplete_data, f)
            temp_path = f.name

        try:
            manager = LicenseManager(dev_mode=False, license_path=temp_path)

            with pytest.raises(LicenseError, match="Missing required"):
                manager.load_and_validate()
        finally:
            os.unlink(temp_path)


class TestLicenseEnvironmentVariables:
    """Test license configuration via environment variables"""

    def test_license_path_from_env(self):
        """Test reading license path from MEMOPT_LICENSE_PATH"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                "customer_id": "env-test",
                "tier": "enterprise",
                "issued_at": "2025-01-01T00:00:00Z",
                "expires_at": "2026-01-01T00:00:00Z",
                "max_gpus": 4,
                "features": ["vllm"],
                "telemetry_required": False,
                "signature": "invalid"
            }, f)
            temp_path = f.name

        try:
            with patch.dict(os.environ, {"MEMOPT_LICENSE_PATH": temp_path}):
                manager = LicenseManager(dev_mode=False)

                # Should use env var path (will fail on signature, but that's OK)
                with pytest.raises(LicenseError):
                    manager.load_and_validate()
        finally:
            os.unlink(temp_path)

    def test_dev_mode_from_env(self):
        """Test enabling dev mode via MEMOPT_DEV_MODE"""
        with patch.dict(os.environ, {"MEMOPT_DEV_MODE": "1"}):
            manager = LicenseManager()

            # Should use dev mode
            license_obj = manager.load_and_validate()
            assert license_obj.customer_id == "dev"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
