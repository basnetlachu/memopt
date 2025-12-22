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
    
    Connects to license server on Hostinger VPS to validate
    customer licenses before allowing MemOpt to run.
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
            "http://YOUR_HOSTINGER_IP"  # TODO: Update after deploying license server
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
            logger.error("MEMOPT_LICENSE_KEY environment variable not set")
            raise LicenseError(
                "License key not found",
                {"hint": "Set environment variable: export MEMOPT_LICENSE_KEY='your-key'"}
            )
        
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
            
            elif response.status_code == 401:
                error_detail = response.json().get('detail', 'Invalid license')
                logger.error(f"License validation failed: {error_detail}")
                raise LicenseError(f"Invalid license: {error_detail}")
            
            elif response.status_code == 403:
                error_detail = response.json().get('detail', 'License limit exceeded')
                logger.error(f"License validation failed: {error_detail}")
                raise LicenseError(f"License limit exceeded: {error_detail}")
            
            else:
                logger.error(f"License server error: {response.status_code}")
                raise LicenseError(f"License server error: {response.status_code}")
        
        except requests.exceptions.ConnectionError:
            logger.warning("⚠️  License server unreachable")
            logger.warning("   Entering grace period mode (7 days)")
            return self._check_grace_period()
        
        except requests.exceptions.Timeout:
            logger.warning("⚠️  License server timeout")
            logger.warning("   Entering grace period mode (7 days)")
            return self._check_grace_period()
        
        except LicenseError:
            raise
        
        except Exception as e:
            logger.error(f"Unexpected license validation error: {e}", exc_info=True)
            raise LicenseError(f"License validation failed: {e}")
    
    def _get_gpu_count(self) -> int:
        """Get number of GPUs available"""
        try:
            import torch
            return torch.cuda.device_count() if torch.cuda.is_available() else 0
        except ImportError:
            return 0
    
    def _check_grace_period(self) -> bool:
        """
        Check if within grace period
        
        Allows operation for 7 days if license server is unreachable.
        Stores last validation time in /tmp/.memopt_grace
        """
        grace_file = "/tmp/.memopt_grace"
        grace_period_days = 7
        
        try:
            # Check if grace file exists
            if os.path.exists(grace_file):
                with open(grace_file, 'r') as f:
                    last_validation = float(f.read().strip())
                
                # Check if grace period expired
                elapsed_days = (time.time() - last_validation) / 86400
                
                if elapsed_days < grace_period_days:
                    remaining = grace_period_days - elapsed_days
                    logger.warning(f"Grace period: {remaining:.1f} days remaining")
                    return True
                else:
                    logger.error("Grace period expired (7 days)")
                    raise LicenseError("Grace period expired. Cannot reach license server.")
            else:
                # First time - create grace file
                with open(grace_file, 'w') as f:
                    f.write(str(time.time()))
                logger.warning(f"Started grace period: {grace_period_days} days")
                return True
        
        except LicenseError:
            raise
        except Exception as e:
            logger.error(f"Grace period check failed: {e}")
            # Fail safe: deny access
            return False
    
    def get_customer_info(self) -> dict:
        """Get customer information from last validation"""
        return self.customer_info.copy()


# Global license client instance
_license_client = None
_license_lock = None


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
        
    Raises:
        LicenseError: If invalid
    """
    client = get_license_client()
    return client.validate(force=force)