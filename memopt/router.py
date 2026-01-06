"""
Request Router - Multi-Node Scaling

Lightweight HTTP router that distributes requests across worker nodes.
NO GPU usage, NO model loading, NO inference logic.

DESIGN:
- Accept client HTTP requests
- Round-robin load balance to workers
- Retry on worker failure
- Return response to client

CRITICAL: Router does NOT:
- Inspect prompts
- Modify requests
- Batch tokens
- Touch inference logic
- Use GPU

PERFORMANCE IMPACT: Minimal HTTP routing overhead only.
"""

import json
import time
import random
import requests
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import List, Optional, Dict, Any
import threading
from dataclasses import dataclass

from .node_identity import get_node_id


@dataclass
class WorkerNode:
    """Worker node configuration."""
    host: str
    port: int
    
    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    def __str__(self) -> str:
        return f"{self.host}:{self.port}"


class LoadBalancer:
    """
    Simple round-robin load balancer with health tracking.
    
    DESIGN:
    - Round-robin distribution
    - Tracks worker health (failed requests)
    - Retries on different worker if one fails
    - NO complex algorithms (keep it simple)
    """
    
    def __init__(self, workers: List[WorkerNode]):
        self.workers = workers
        self.current_index = 0
        self.lock = threading.Lock()
        self.health_status = {worker.url: True for worker in workers}
    
    def get_next_worker(self) -> Optional[WorkerNode]:
        """Get next healthy worker (round-robin)."""
        with self.lock:
            # Find next healthy worker
            attempts = 0
            while attempts < len(self.workers):
                worker = self.workers[self.current_index]
                self.current_index = (self.current_index + 1) % len(self.workers)
                
                if self.health_status.get(worker.url, True):
                    return worker
                
                attempts += 1
            
            # All workers unhealthy, return first one anyway
            return self.workers[0] if self.workers else None
    
    def mark_unhealthy(self, worker: WorkerNode):
        """Mark worker as unhealthy (after failed request)."""
        self.health_status[worker.url] = False
        print(f"[Router] Worker {worker} marked unhealthy")
    
    def mark_healthy(self, worker: WorkerNode):
        """Mark worker as healthy (after successful request)."""
        self.health_status[worker.url] = True


class RouterHandler(BaseHTTPRequestHandler):
    """HTTP handler for router requests."""
    
    # Class variables set by Router
    load_balancer: Optional[LoadBalancer] = None
    max_retries: int = 3
    request_timeout: int = 300  # 5 minutes for long inference
    
    def do_POST(self):
        """Handle POST /generate requests."""
        if self.path == '/generate' or self.path == '/infer':
            self._handle_generate()
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')
    
    def do_GET(self):
        """Handle GET /health for router health check."""
        if self.path == '/health':
            self._handle_health()
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')
    
    def _handle_health(self):
        """Router health check."""
        response = {
            'status': 'healthy',
            'node_id': get_node_id(),
            'role': 'router',
            'workers': len(self.load_balancer.workers) if self.load_balancer else 0
        }
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
    
    def _handle_generate(self):
        """
        Handle generate request by routing to worker.
        
        Request format:
        {
          "prompt": "text",
          "max_tokens": 512,
          "temperature": 1.0,
          "do_sample": false
        }
        
        Response: Pass-through from worker
        """
        try:
            # Read request
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            request_data = json.loads(body.decode('utf-8'))
            
            # Route to worker with retry
            result = self._route_to_worker(request_data)
            
            if result is None:
                # All workers failed
                self.send_response(503)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'error': 'All workers unavailable',
                    'router_node_id': get_node_id()
                }).encode())
                return
            
            # Return worker response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'error': str(e),
                'router_node_id': get_node_id()
            }).encode())
    
    def _route_to_worker(self, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Route request to worker with retry.
        
        Returns:
            Worker response dict, or None if all workers failed
        """
        for attempt in range(self.max_retries):
            worker = self.load_balancer.get_next_worker()
            if worker is None:
                return None
            
            try:
                # Send request to worker
                response = requests.post(
                    f"{worker.url}/infer",
                    json=request_data,
                    timeout=self.request_timeout
                )
                
                if response.status_code == 200:
                    # Success
                    self.load_balancer.mark_healthy(worker)
                    return response.json()
                else:
                    # Worker returned error
                    print(f"[Router] Worker {worker} returned {response.status_code}")
                    self.load_balancer.mark_unhealthy(worker)
                    
            except Exception as e:
                # Worker request failed
                print(f"[Router] Worker {worker} request failed: {e}")
                self.load_balancer.mark_unhealthy(worker)
        
        # All retries failed
        return None
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


class Router:
    """
    HTTP request router for multi-node deployment.
    
    DESIGN:
    - CPU-only (no GPU)
    - No model loading
    - No inference logic
    - Just routes requests to workers
    
    Use Case:
    - Multi-node: Router accepts external traffic, distributes to workers
    - Single-node: Disabled (direct model access)
    """
    
    def __init__(
        self,
        workers: List[WorkerNode],
        port: int = 8000,
        max_retries: int = 3,
        request_timeout: int = 300
    ):
        """
        Args:
            workers: List of worker nodes
            port: Router HTTP port
            max_retries: Maximum retry attempts per request
            request_timeout: Request timeout in seconds
        """
        self.workers = workers
        self.port = port
        self.load_balancer = LoadBalancer(workers)
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        
        # Wire to handler
        RouterHandler.load_balancer = self.load_balancer
        RouterHandler.max_retries = max_retries
        RouterHandler.request_timeout = request_timeout
    
    def start(self):
        """Start router server."""
        try:
            self.server = HTTPServer(('0.0.0.0', self.port), RouterHandler)
            self.thread = threading.Thread(target=self._run_server, daemon=True)
            self.thread.start()
            print(f"[Router] Started on port {self.port}")
            print(f"[Router] Node ID: {get_node_id()}")
            print(f"[Router] Workers: {[str(w) for w in self.workers]}")
            print(f"[Router] Endpoint: POST /generate")
        except Exception as e:
            print(f"[Router] Failed to start: {e}")
    
    def _run_server(self):
        """Run HTTP server."""
        try:
            self.server.serve_forever()
        except Exception as e:
            print(f"[Router] Server error: {e}")
    
    def stop(self):
        """Stop router server."""
        if self.server:
            self.server.shutdown()
            print("[Router] Stopped")
    
    @staticmethod
    def from_environment() -> 'Router':
        """
        Create router from environment variables.
        
        Environment:
            MEMOPT_WORKER_HOSTS: Comma-separated list (host1:port1,host2:port2)
            ROUTER_PORT: Router port (default: 8000)
            ROUTER_MAX_RETRIES: Max retries (default: 3)
            ROUTER_REQUEST_TIMEOUT: Timeout seconds (default: 300)
        
        Returns:
            Router instance
        """
        import os
        
        # Parse worker hosts
        worker_hosts_str = os.getenv('MEMOPT_WORKER_HOSTS', '')
        if not worker_hosts_str:
            raise ValueError("MEMOPT_WORKER_HOSTS must be set for router mode")
        
        workers = []
        for host_port in worker_hosts_str.split(','):
            host_port = host_port.strip()
            if ':' in host_port:
                host, port = host_port.split(':')
                workers.append(WorkerNode(host=host, port=int(port)))
            else:
                # Default to port 9000
                workers.append(WorkerNode(host=host_port, port=9000))
        
        router_port = int(os.getenv('ROUTER_PORT', '8000'))
        max_retries = int(os.getenv('ROUTER_MAX_RETRIES', '3'))
        request_timeout = int(os.getenv('ROUTER_REQUEST_TIMEOUT', '300'))
        
        return Router(
            workers=workers,
            port=router_port,
            max_retries=max_retries,
            request_timeout=request_timeout
        )
