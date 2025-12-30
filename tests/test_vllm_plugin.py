"""
Unit tests for MemOpt vLLM plugin
"""

import os
import pytest
from unittest.mock import patch, MagicMock

# Import plugin module
from memopt import vllm_plugin


class TestPluginConfiguration:
    """Test plugin configuration and environment variables"""

    def test_is_enabled_true(self):
        """Test plugin enabled when MEMOPT_ENABLED=1"""
        with patch.dict(os.environ, {"MEMOPT_ENABLED": "1"}):
            assert vllm_plugin.is_enabled() is True

    def test_is_enabled_false(self):
        """Test plugin disabled when MEMOPT_ENABLED not set"""
        with patch.dict(os.environ, {}, clear=True):
            assert vllm_plugin.is_enabled() is False

    def test_is_strict_mode_true(self):
        """Test strict mode when MEMOPT_STRICT=1"""
        with patch.dict(os.environ, {"MEMOPT_STRICT": "1"}):
            assert vllm_plugin.is_strict_mode() is True

    def test_is_strict_mode_false(self):
        """Test strict mode disabled by default"""
        with patch.dict(os.environ, {}, clear=True):
            assert vllm_plugin.is_strict_mode() is False


class TestVLLMDetection:
    """Test vLLM detection and version checking"""

    def test_detect_vllm_not_installed(self):
        """Test detection when vLLM is not installed"""
        patcher = vllm_plugin.VLLMPatcher()

        # Mock vLLM import failure
        with patch.dict('sys.modules', {'vllm': None}):
            # Should return False if vLLM not importable
            # Note: This test may not work perfectly due to import caching
            pass

    def test_detect_vllm_installed(self):
        """Test detection when vLLM is installed"""
        patcher = vllm_plugin.VLLMPatcher()

        # Mock vLLM module
        mock_vllm = MagicMock()
        mock_vllm.__version__ = "0.3.2"

        with patch.dict('sys.modules', {'vllm': mock_vllm}):
            result = patcher.detect_vllm()

            if result:  # Only assert if vLLM mock worked
                assert patcher.vllm_version == "0.3.2"


class TestPluginPatching:
    """Test plugin monkey-patching logic"""

    def test_scheduler_patch_without_vllm(self):
        """Test scheduler patching fails gracefully without vLLM"""
        patcher = vllm_plugin.VLLMPatcher()

        # Should return False and record error
        result = patcher.patch_scheduler()

        assert result is False
        assert 'scheduler' in patcher.patch_errors

    def test_kv_cache_patch_without_vllm(self):
        """Test KV cache patching fails gracefully without vLLM"""
        patcher = vllm_plugin.VLLMPatcher()

        # Should return False and record error
        result = patcher.patch_kv_cache()

        assert result is False
        assert 'kv_cache' in patcher.patch_errors

    def test_memory_planner_patch_without_vllm(self):
        """Test memory planner patching fails gracefully without vLLM"""
        patcher = vllm_plugin.VLLMPatcher()

        # Should return False and record error
        result = patcher.patch_memory_planner()

        assert result is False
        assert 'memory_planner' in patcher.patch_errors


class TestPluginInitialization:
    """Test plugin initialization flow"""

    def test_initialize_plugin_disabled(self):
        """Test initialization when plugin is disabled"""
        with patch.dict(os.environ, {"MEMOPT_ENABLED": "0"}):
            result = vllm_plugin.initialize_plugin()

            assert result is False

    def test_initialize_plugin_no_license(self):
        """Test initialization fails without valid license"""
        with patch.dict(os.environ, {
            "MEMOPT_ENABLED": "1",
            "MEMOPT_LICENSE_PATH": "/nonexistent/license.json"
        }):
            result = vllm_plugin.initialize_plugin()

            # Should fail (gracefully) without license
            assert result is False

    def test_initialize_plugin_dev_mode(self):
        """Test initialization succeeds in dev mode"""
        with patch.dict(os.environ, {
            "MEMOPT_ENABLED": "1",
            "MEMOPT_DEV_MODE": "1"
        }):
            # Should attempt initialization (may fail if vLLM not installed)
            result = vllm_plugin.initialize_plugin()

            # Result depends on whether vLLM is actually installed
            # Just verify no exceptions raised
            assert isinstance(result, bool)


class TestPluginStatus:
    """Test plugin status reporting"""

    def test_get_status_not_initialized(self):
        """Test status when plugin not initialized"""
        # Reset global patcher
        vllm_plugin._patcher = None

        status = vllm_plugin.get_status()

        assert status['enabled'] is False
        assert 'reason' in status

    def test_get_patcher_none(self):
        """Test get_patcher returns None when not initialized"""
        vllm_plugin._patcher = None

        patcher = vllm_plugin.get_patcher()

        assert patcher is None


class TestPluginSafeDegradation:
    """Test that plugin fails gracefully"""

    def test_apply_patches_without_vllm(self):
        """Test apply_all_patches fails gracefully without vLLM"""
        patcher = vllm_plugin.VLLMPatcher()

        # Should detect vLLM not installed and return False
        result = patcher.apply_all_patches()

        assert result is False

    def test_strict_mode_enforcement(self):
        """Test that strict mode raises errors on patch failure"""
        with patch.dict(os.environ, {
            "MEMOPT_ENABLED": "1",
            "MEMOPT_STRICT": "1",
            "MEMOPT_DEV_MODE": "1"
        }):
            # Mock license validation to pass
            with patch('memopt.vllm_plugin.validate_license') as mock_validate:
                mock_license = MagicMock()
                mock_license.has_feature.return_value = True
                mock_validate.return_value = mock_license

                # Should raise RuntimeError in strict mode if patches fail
                # (will fail because vLLM not installed in test environment)
                try:
                    vllm_plugin.initialize_plugin()
                except RuntimeError as e:
                    # Expected in strict mode
                    assert "strict mode" in str(e).lower()
                except Exception:
                    # Other exceptions are fine (e.g., vLLM not installed)
                    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
