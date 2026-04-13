.PHONY: build push dev dev-down dev-logs k8s-deploy k8s-status k8s-logs k8s-uninstall bootstrap verify test test-verbose audit clean install-crds deploy-operator apply-example operator-status validate

GIT_SHA := $(shell git rev-parse --short HEAD 2>/dev/null || echo "dev")

# ── Docker ────────────────────────────────────────────────────────────
build:
	docker build \
	  --build-arg CUDA_ARCHITECTURES="86;90;100" \
	  -t memopt/serving:$(GIT_SHA) \
	  -t memopt/serving:latest \
	  .

push: build
	docker push memopt/serving:latest
	docker push memopt/serving:$(GIT_SHA)

# ── Local development ────────────────────────────────────────────────
dev:
	docker-compose up -d
	@echo ""
	@echo "  Serving:       http://localhost:8080"
	@echo "  Control plane: http://localhost:8765"
	@echo "  Grafana:       http://localhost:3000"
	@echo "  Prometheus:    http://localhost:9090"
	@echo ""

dev-down:
	docker-compose down

dev-logs:
	docker-compose logs -f memopt-serving

# ── Kubernetes ────────────────────────────────────────────────────────
k8s-deploy:
	@if [ -z "$(REDIS_URL)" ]; then \
	  echo "ERROR: REDIS_URL required. Usage: make k8s-deploy REDIS_URL=redis://..."; exit 1; fi
	helm upgrade --install memopt \
	  deploy/helm/memopt/ \
	  --values deploy/helm/memopt/values.production.yaml \
	  --set redis.url=$(REDIS_URL) \
	  --set secrets.anthropicApiKey=$(ANTHROPIC_API_KEY) \
	  --namespace memopt \
	  --create-namespace \
	  --wait \
	  --timeout 5m

k8s-status:
	kubectl get all -n memopt
	kubectl get daemonset -n memopt

k8s-logs:
	kubectl logs -n memopt -l component=serving -f

k8s-uninstall:
	helm uninstall memopt -n memopt

# ── Single node bootstrap ────────────────────────────────────────────
bootstrap:
	@if [ -z "$(NODE_ID)" ]; then \
	  echo "ERROR: NODE_ID required. Usage: make bootstrap NODE_ID=node-01"; exit 1; fi
	./scripts/deploy.sh \
	  --node-id $(NODE_ID) \
	  --control-plane $(CONTROL_PLANE) \
	  --redis-url $(REDIS_URL)

verify:
	./scripts/verify_node.sh

# ── Testing ───────────────────────────────────────────────────────────
test:
	pytest --tb=short -q

test-verbose:
	pytest -v

audit:
	python3 scripts/audit_wiring.py

# ── Operator / CRDs ───────────────────────────────────────────────────
install-crds:
	kubectl apply -f deploy/crds/
	@echo "CRDs installed"

deploy-operator:
	kubectl apply -f deploy/operator/rbac.yaml
	kubectl apply -f deploy/operator/deployment.yaml
	@echo "Operator deployed"

apply-example:
	kubectl apply -f deploy/examples/single-node-cluster.yaml
	@echo "Example cluster applied"

operator-status:
	kubectl get memoptclusters -n memopt
	kubectl get memoptnodes -n memopt

validate:
	python3 scripts/validate_crds.py
	helm lint deploy/helm/memopt/
	pytest --tb=short -q

# ── Cleanup ───────────────────────────────────────────────────────────
clean:
	docker-compose down -v 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info/
