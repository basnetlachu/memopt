Old Production Scripts - Moved on 2026-01-21

These files were part of the original production implementation but have been
replaced by the new production/ directory architecture.

Replaced by:
- production_worker.py -> production/worker_service.py
- production_router.py -> production/load_balancer.py  
- production_multinode_server.py -> production/worker_service.py + scripts/start_node.sh
- production_server_example.py -> (example code, not needed)

These old scripts used memopt modules that have also been moved to
experiments/old_memopt_modules/

Current working system:
- scripts/benchmark_production.sh
- scripts/start_worker.sh
- scripts/start_node.sh
- production/worker_service.py
- production/registry.py
- production/load_balancer.py
