"""
Health & Readiness HTTP Endpoints - Multi-Node Infrastructure

Lightweight HTTP server for Kubernetes liveness/readiness probes.
Must respond in <10ms with no GPU calls or blocking operations.

DESIGN:
- GET /health → Liveness probe (process alive)
- GET /ready → Readiness probe (model loaded, queue not overloaded)
- GET /metrics → Prometheus metrics (optional, delegates to exporter)

PERFORMANCE IMPACT: None on inference (runs in separate thread)
"""

import os
import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional, Callable, Dict, Any

from .node_identity import get_node_id


class HealthCheckHandler(BaseHTTPRequestHandler):
    """
    HTTP handler for health/readiness checks.
    
    CRITICAL: Must be fast (<10ms) and never block.
    """
    
    # Class variables set by HealthEndpointServer
    health_check_fn: Optional[Callable[[], Dict[str, Any]]] = None
    readiness_check_fn: Optional[Callable[[], Dict[str, Any]]] = None
    metrics_fn: Optional[Callable[[], str]] = None
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/health' or self.path == '/healthz':
            self._handle_health()
        elif self.path == '/ready' or self.path == '/readiness':
            self._handle_readiness()
        elif self.path == '/metrics':
            self._handle_metrics()
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')
    
    def _handle_health(self):
        """
        Liveness probe: Process is alive.
        
        Returns 200 if process is running.
        Kubernetes will restart if this fails.
        """
        try:
            if self.health_check_fn:
                result = self.health_check_fn()
                status = result.get('status', 'healthy')
                status_code = 200 if status == 'healthy' else 503
            else:
                # Default: just check process is alive
                result = {
                    'status': 'healthy',
                    'node_id': get_node_id(),
                    'timestamp': time.time()
                }
                status_code = 200
            
            self.send_response(status_code)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
            
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'error', 'error': str(e)}).encode())
    
    def _handle_readiness(self):
        """
        Readiness probe: Ready to accept traffic.
        
        Returns 200 if:
        - Model is loaded
        - Queue is not overloaded
        
        Kubernetes will stop sending traffic if this fails.
        """
        try:
            if self.readiness_check_fn:
                result = self.readiness_check_fn()
                ready = result.get('ready', False)
                status_code = 200 if ready else 503
            else:
                # Default: assume ready (no model reference available)
                result = {
                    'ready': True,
                    'node_id': get_node_id(),
                    'timestamp': time.time()
                }
                status_code = 200
            
            self.send_response(status_code)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
            
        except Exception as e:
            self.send_response(503)
            self.end_headers()
            self.wfile.write(json.dumps({'ready': False, 'error': str(e)}).encode())
    
    def _handle_metrics(self):
        """
        Prometheus metrics endpoint.
        
        Delegates to metrics exporter if available.
        """
        try:
            if self.metrics_fn:
                metrics_text = self.metrics_fn()
                self.send_response(200)
                self.send_header('Content-Type', 'text/plain; version=0.0.4')
                self.end_headers()
                self.wfile.write(metrics_text.encode())
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'Metrics not enabled')
                
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f'Error: {e}'.encode())
    
    def log_message(self, format, *args):
        """Suppress default logging (use custom logging instead)."""
        pass  # Suppress HTTP server logs


class HealthEndpointServer:
    """
    Lightweight HTTP server for health checks.
    
    CRITICAL: Must never block inference.
    Runs in separate daemon thread.
    """
    
    def __init__(
        self,
        port: Optional[int] = None,
        health_check_fn: Optional[Callable[[], Dict[str, Any]]] = None,
        readiness_check_fn: Optional[Callable[[], Dict[str, Any]]] = None,
        metrics_fn: Optional[Callable[[], str]] = None,
        enable: bool = True
    ):
        """
        Args:
            port: HTTP port (default: from PORT env var or 8080)
            health_check_fn: Function returning health status dict
            readiness_check_fn: Function returning readiness status dict
            metrics_fn: Function returning Prometheus metrics text
            enable: Enable server (default True)
        """
        self.enable = enable
        if not enable:
            return
        
        # Read port from environment (K8s standard)
        if port is None:
            port = int(os.getenv('PORT', '8080'))
        
        self.port = port
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        
        # Wire callbacks to handler class
        HealthCheckHandler.health_check_fn = health_check_fn
        HealthCheckHandler.readiness_check_fn = readiness_check_fn
        HealthCheckHandler.metrics_fn = metrics_fn
    
    def start(self):
        """Start HTTP server in background thread."""
        if not self.enable:
            return
        
        try:
            self.server = HTTPServer(('0.0.0.0', self.port), HealthCheckHandler)
            self.thread = threading.Thread(target=self._run_server, daemon=True)
            self.thread.start()
            print(f"[HealthEndpoints] HTTP server started on port {self.port}")
            print(f"[HealthEndpoints] Node ID: {get_node_id()}")
            print(f"[HealthEndpoints] Endpoints: /health, /ready, /metrics")
        except Exception as e:
            print(f"[HealthEndpoints] Failed to start server: {e}")
            self.enable = False
    
    def _run_server(self):
        """Run HTTP server (called in background thread)."""
        try:
            self.server.serve_forever()
        except Exception as e:
            print(f"[HealthEndpoints] Server error: {e}")
    
    def stop(self):
        """Stop HTTP server."""
        if self.server:
            self.server.shutdown()
            print("[HealthEndpoints] HTTP server stopped")
