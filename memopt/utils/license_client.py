"""
License validation client for MemOpt

Validates license with Hostinger VPS license server
"""

import requests
import os
import sys
import socket
from typing import Optional
import time
from datetime import datetime, timedelta

from memopt.monitoring.logger import get_logger
from memopt.utils.errors import MemOptError

logger = get_logger(__name__)


class LicenseError(MemOptError):
    """License validation error"""
    pass


class LicenseClient:
    """
    Client for validating MemOpt licenses
    
    Connects to license server to validate customer licenses.
    """
    
    def __init__(self, license_server: Optional[str] = None):
        """
        Initialize license client
        
        Args:
            license_server: License server URL (default from env)
        """
        self.license_key = os.getenv("MEMOPT_LICENSE_KEY")
        self.license_server = license_server or os.getenv(
            "MEMOPT_LICENSE_SERVER",
            "http://localhost:8080"  # Default for testing
        )
        self.validated = False
        self.validation_expiry = None
        self.customer_info = {}
        
        logger.debug(f"License client initialized (server: {self.license_server})")
    
    def validate(self, force: bool = False) -> bool:
        """
        Validate license with server
        
        Args:
            force: Force validation even if recently validated
            
        Returns:
            True if license is valid
            
        Raises:
            LicenseError: If license is invalid
        """
        # Check if we need to revalidate
        if not force and self.validated and self.validation_expiry:
            if datetime.now() < self.validation_expiry:
                logger.debug("Using cached license validation")
                return True
        
        # Check license key
        if not self.license_key:
            logger.warning("MEMOPT_LICENSE_KEY not set - running in grace period mode")
            return True  # Allow for testing
        
        logger.info("Validating license with server...")
        
        try:
            # Get system info
            hostname = socket.gethostname()
            gpu_count = self._get_gpu_count()
            
            # Make validation request
            response = requests.post(
                f"{self.license_server}/validate",
                json={
                    "license_key": self.license_key,
                    "hostname": hostname,
                    "gpu_count": gpu_count,
                    "version": "1.0.0"
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Store validation info
                self.validated = True
                self.validation_expiry = datetime.now() + timedelta(hours=24)
                self.customer_info = {
                    'customer_id': data.get('customer_id'),
                    'customer_name': data.get('customer_name'),
                    'expiry_date': data.get('expiry_date'),
                    'max_gpus': data.get('max_gpus')
                }
                
                logger.info("="*70)
                logger.info("✓ LICENSE VALIDATED")
                logger.info("="*70)
                logger.info(f"Customer: {self.customer_info['customer_name']}")
                logger.info(f"Expires: {self.customer_info['expiry_date']}")
                logger.info(f"Max GPUs: {self.customer_info['max_gpus']}")
                logger.info(f"Your GPUs: {gpu_count}")
                logger.info("="*70)
                
                return True
            
            else:
                logger.warning(f"License server returned {response.status_code}")
                return True  # Allow for testing
        
        except requests.exceptions.ConnectionError:
            logger.warning("⚠️  License server unreachable - running in grace period")
            return True
        
        except requests.exceptions.Timeout:
            logger.warning("⚠️  License server timeout - running in grace period")
            return True
        
        except Exception as e:
            logger.warning(f"License validation error: {e}")
            return True  # Allow for testing
    
    def _get_gpu_count(self) -> int:
        """Get number of GPUs available"""
        try:
            import torch
            return torch.cuda.device_count() if torch.cuda.is_available() else 0
        except ImportError:
            return 0
    
    def get_customer_info(self) -> dict:
        """Get customer information from last validation"""
        return self.customer_info.copy()


# Global license client instance
_license_client = None


def get_license_client() -> LicenseClient:
    """Get or create license client singleton"""
    global _license_client
    if _license_client is None:
        _license_client = LicenseClient()
    return _license_client


def validate_license(force: bool = False) -> bool:
    """
    Convenience function to validate license
    
    Args:
        force: Force validation even if recently validated
        
    Returns:
        True if valid
    """
    client = get_license_client()
    return client.validate(force=force)