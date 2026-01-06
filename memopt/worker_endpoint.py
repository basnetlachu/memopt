"""
Worker Inference Endpoint - Multi-Node Scaling

Exposes HTTP endpoint for router → worker communication.
This is a THIN WRAPPER around existing OptimizedLLM.generate().

CRITICAL: NO changes to inference logic.
CRITICAL: NO changes to batching.
CRITICAL: NO changes to KV cache.

DESIGN:
- POST /infer → accepts request, calls model.generate(), returns result
- Uses existing generate() method (zero modifications)
- Runs in same process as GPU worker
- No cross-node coordination

PERFORMANCE IMPACT: Minimal HTTP serialization overhead only.
"""

import json
import time
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional, Dict, Any
import threading

from .node_identity import get_node_id


class WorkerInferenceHandler(BaseHTTPRequestHandler):
    """
    HTTP handler for worker inference requests.
    
    CRITICAL: This is just HTTP I/O wrapper.
    ALL inference logic remains in OptimizedLLM.generate().
    """
    
    # Class variable set by WorkerInferenceServer
    model = None  # OptimizedLLM instance
    
    def do_POST(self):
        """Handle POST /infer requests."""
        if self.path == '/infer':
            self._handle_infer()
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')
    
    def _handle_infer(self):
        """
        Handle inference request.
        
        Request format:
        {
          "prompt": "text",
          "max_tokens": 512,
          "temperature": 1.0,
          "do_sample": false
        }
        
        Response format:
        {
          "result": "generated text",
          "node_id": "worker-0",
          "latency_ms": 123.45
        }
        """
        try:
            # Read request
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            request_data = json.loads(body.decode('utf-8'))
            
            # Extract parameters
            prompt = request_data.get('prompt', '')
            max_tokens = request_data.get('max_tokens', 512)
            temperature = request_data.get('temperature', 1.0)
            do_sample = request_data.get('do_sample', False)
            
            if not prompt:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Missing prompt'}).encode())
                return
            
            # CRITICAL: Call existing generate() method
            # NO modifications to inference logic
            start_time = time.time()
            result = self.model.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                do_sample=do_sample
            )
            latency_ms = (time.time() - start_time) * 1000
            
            # Return response
            response = {
                'result': result,
                'node_id': get_node_id(),
                'latency_ms': latency_ms
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            # Return error
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            error_response = {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'node_id': get_node_id()
            }
            self.wfile.write(json.dumps(error_response).encode())
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


class WorkerInferenceServer:
    """
    HTTP server for worker inference endpoint.
    
    DESIGN:
    - Runs in background thread (same process as model)
    - Exposes POST /infer for router requests
    - Uses existing OptimizedLLM.generate() method
    - NO changes to inference logic
    
    Use Case:
    - Multi-node deployment: router sends requests here
    - Single-node deployment: disabled (direct access to model)
    """
    
    def __init__(
        self,
        model,  # OptimizedLLM instance
        port: int = 9000,
        enable: bool = False
    ):
        """
        Args:
            model: OptimizedLLM instance (existing model)
            port: HTTP port for worker endpoint (default: 9000)
            enable: Enable worker endpoint (default: False)
        """
        self.enable = enable
        if not enable:
            return
        
        self.model = model
        self.port = port
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        
        # Wire model to handler
        WorkerInferenceHandler.model = model
    
    def start(self):
        """Start worker inference server."""
        if not self.enable:
            return
        
        try:
            self.server = HTTPServer(('0.0.0.0', self.port), WorkerInferenceHandler)
            self.thread = threading.Thread(target=self._run_server, daemon=True)
            self.thread.start()
            print(f"[WorkerEndpoint] Inference server started on port {self.port}")
            print(f"[WorkerEndpoint] Node ID: {get_node_id()}")
            print(f"[WorkerEndpoint] Endpoint: POST /infer")
        except Exception as e:
            print(f"[WorkerEndpoint] Failed to start server: {e}")
            self.enable = False
    
    def _run_server(self):
        """Run HTTP server (background thread)."""
        try:
            self.server.serve_forever()
        except Exception as e:
            print(f"[WorkerEndpoint] Server error: {e}")
    
    def stop(self):
        """Stop worker inference server."""
        if self.server:
            self.server.shutdown()
            print("[WorkerEndpoint] Inference server stopped")
