#!/bin/bash
# MemOpt SaaS Deployment Script
# Pulls latest code, rebuilds, and restarts services

set -e  # Exit on error

echo "=========================================="
echo "MemOpt SaaS Deployment"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo -e "${RED}ERROR: Do not run this script as root${NC}"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${RED}ERROR: .env file not found${NC}"
    echo "Copy .env.example to .env and configure values"
    exit 1
fi

# Source environment variables
set -a
source .env
set +a

# Validate required environment variables
if [ -z "$ADMIN_API_KEY" ] || [ "$ADMIN_API_KEY" = "CHANGE_ME_TO_SECURE_RANDOM_STRING_32_CHARS_MINIMUM" ]; then
    echo -e "${RED}ERROR: ADMIN_API_KEY not set in .env${NC}"
    exit 1
fi

if [ -z "$POSTGRES_PASSWORD" ] || [ "$POSTGRES_PASSWORD" = "changeme" ]; then
    echo -e "${YELLOW}WARNING: Using default POSTGRES_PASSWORD. Change it for production!${NC}"
fi

# Pull latest code (if using git)
if [ -d ".git" ]; then
    echo "Pulling latest code from git..."
    git pull origin main || true
fi

# Stop services
echo "Stopping services..."
docker-compose down

# Build new image
echo "Building Docker image..."
docker-compose build --no-cache api

# Start services
echo "Starting services..."
docker-compose up -d

# Wait for services to be healthy
echo "Waiting for services to start..."
sleep 10

# Check health
echo "Checking service health..."
for i in {1..30}; do
    if curl -f http://localhost/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Services are healthy${NC}"
        break
    fi
    echo "Waiting for health check... ($i/30)"
    sleep 2
done

# Show status
echo ""
echo "=========================================="
echo "Deployment Status"
echo "=========================================="
docker-compose ps

# Show logs
echo ""
echo "Recent logs (last 20 lines):"
docker-compose logs --tail=20 api

echo ""
echo -e "${GREEN}=========================================="
echo "Deployment complete!"
echo "==========================================${NC}"
echo ""
echo "API Endpoint: https://$(hostname)"
echo "Health Check: https://$(hostname)/health"
echo "API Docs: https://$(hostname)/docs"
echo ""
echo "To view logs: docker-compose logs -f api"
echo "To restart: docker-compose restart api"
echo "To stop: docker-compose down"
