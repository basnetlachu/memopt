"""
Node Identity - Multi-Node Infrastructure

Provides unique node identification for horizontal scaling.
Each replica in a multi-node deployment needs a unique identifier.

DESIGN:
- Read MEMOPT_NODE_ID from environment (set by K8s StatefulSet)
- Fallback to hostname if not set
- Used in logs, metrics, and health endpoints

PERFORMANCE IMPACT: None (initialized once at startup)
"""

import os
import socket
from dataclasses import dataclass
from typing import Optional


@dataclass
class NodeIdentity:
    """
    Unique node identifier for multi-node deployments.
    
    Use Cases:
    - Kubernetes StatefulSet: memopt-0, memopt-1, memopt-2
    - Docker Compose: node1, node2, node3
    - Development: hostname (macbook-pro.local)
    """
    node_id: str
    hostname: str
    
    @staticmethod
    def from_environment() -> 'NodeIdentity':
        """
        Create node identity from environment.
        
        Priority:
        1. MEMOPT_NODE_ID env var (explicit override)
        2. Hostname (default fallback)
        
        Returns:
            NodeIdentity instance
        """
        node_id = os.getenv("MEMOPT_NODE_ID")
        hostname = socket.gethostname()
        
        if node_id is None:
            node_id = hostname
        
        return NodeIdentity(node_id=node_id, hostname=hostname)
    
    def __str__(self) -> str:
        return self.node_id


# Global instance (initialized once at module import)
NODE_IDENTITY = NodeIdentity.from_environment()


def get_node_id() -> str:
    """Get current node ID (convenience function)."""
    return NODE_IDENTITY.node_id


def get_hostname() -> str:
    """Get current hostname (convenience function)."""
    return NODE_IDENTITY.hostname
