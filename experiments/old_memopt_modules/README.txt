Old Memopt Modules - Moved on 2026-01-21

These modules were only used by the old production scripts (now in
experiments/old_production_scripts/) and are no longer needed.

Modules moved:
- gpu_worker.py - replaced by production/worker_service.py
- router.py - replaced by production/load_balancer.py
- worker_endpoint.py - replaced by FastAPI in worker_service.py
- env_config.py - replaced by production/config.py
- health_endpoints.py - replaced by endpoints in worker_service.py
- metrics_exporter.py - not needed in current system
- signal_handling.py - not needed in current system
- node_identity.py - not needed in current system
- request_queue.py - replaced by dynamic batching in OptimizedLLM
- cutoff_policies.py - unused
- request_batching.py - unused
- unbounded_check.py - unused

Current working memopt modules are in memopt/ directory.
