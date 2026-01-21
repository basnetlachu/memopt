"""
Production Router - Multi-Node Deployment

CPU-only router for distributing requests across GPU workers.
NO GPU usage, NO model loading, NO inference.

USAGE:
    # Router mode
    MEMOPT_NODE_ROLE=router \
    MEMOPT_WORKER_HOSTS=worker1:9000,worker2:9000,worker3:9000 \
    python3 production_router.py
    
    # Environment:
    MEMOPT_NODE_ID=router-0
    MEMOPT_WORKER_HOSTS=host1:port1,host2:port2  (REQUIRED)
    ROUTER_PORT=8000  (default)
    ROUTER_MAX_RETRIES=3  (default)
    ROUTER_REQUEST_TIMEOUT=300  (default)
"""

import sys
import time
import os

from memopt.router import Router
from memopt.node_identity import get_node_id


def main():
    print("=" * 80)
    print("MemOpt Production Router")
    print("=" * 80)
    
    # Load configuration
    router_port = int(os.getenv('ROUTER_PORT', '8000'))
    worker_hosts = os.getenv('MEMOPT_WORKER_HOSTS', '')
    
    print(f"\n[Config] Node ID: {get_node_id()}")
    print(f"[Config] Role: ROUTER")
    print(f"[Config] Router port: {router_port}")
    print(f"[Config] Worker hosts: {worker_hosts}")
    
    if not worker_hosts:
        print("\n❌ ERROR: MEMOPT_WORKER_HOSTS must be set")
        print("Example: MEMOPT_WORKER_HOSTS=worker1:9000,worker2:9000,worker3:9000")
        sys.exit(1)
    
    # Create router
    print(f"\n[Router] Initializing...")
    try:
        router = Router.from_environment()
    except Exception as e:
        print(f"\n❌ ERROR: Failed to create router: {e}")
        sys.exit(1)
    
    # Start router
    router.start()
    print(f"✅ Router started")
    
    print("\n" + "=" * 80)
    print("🚀 Router ready!")
    print("=" * 80)
    print(f"\nNode ID: {get_node_id()}")
    print(f"Role: ROUTER (CPU-only)")
    print(f"Endpoint: http://0.0.0.0:{router_port}/generate")
    print(f"Health: http://0.0.0.0:{router_port}/health")
    print(f"Workers: {len(router.workers)}")
    print("\nPress Ctrl+C to shutdown")
    print("=" * 80)
    
    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Shutdown] Stopping router")
        router.stop()


if __name__ == "__main__":
    main()
