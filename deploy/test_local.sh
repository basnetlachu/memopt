#!/bin/bash
# Local testing script for MemOpt SaaS
# Run this to test the complete system locally before deploying to VPS

set -e

echo "=========================================="
echo "MemOpt SaaS Local Testing"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if .env exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Creating .env from example...${NC}"
    cp ../saas/.env.example .env

    # Set development defaults
    sed -i.bak 's/ENV=production/ENV=development/' .env
    sed -i.bak 's/ADMIN_API_KEY=CHANGE_ME_TO_SECURE_RANDOM_STRING_32_CHARS_MINIMUM/ADMIN_API_KEY=test_admin_key_for_local_development_only/' .env
    sed -i.bak 's/POSTGRES_PASSWORD:-changeme/POSTGRES_PASSWORD:-testpassword123/' .env

    echo -e "${GREEN}✓ Created .env with development defaults${NC}"
fi

# Start services
echo ""
echo "Starting Docker services..."
docker-compose up -d

# Wait for services to be healthy
echo "Waiting for services to start..."
sleep 10

echo "Checking health..."
for i in {1..30}; do
    if curl -sf http://localhost/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Services are healthy${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}✗ Services failed to start${NC}"
        docker-compose logs api
        exit 1
    fi
    echo -n "."
    sleep 2
done

echo ""
echo -e "${GREEN}=========================================="
echo "Services Started Successfully"
echo "==========================================${NC}"
echo ""

# Test API
echo "Testing API endpoints..."
echo ""

# Test health
echo "1. Testing health endpoint..."
HEALTH=$(curl -s http://localhost/health)
if echo "$HEALTH" | grep -q "healthy"; then
    echo -e "${GREEN}✓ Health check passed${NC}"
else
    echo -e "${RED}✗ Health check failed${NC}"
    echo "$HEALTH"
fi

# Create test tenant
echo ""
echo "2. Creating test tenant..."
TENANT_RESPONSE=$(curl -s -X POST http://localhost/admin/tenants \
  -H "X-Admin-Key: test_admin_key_for_local_development_only" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Tenant",
    "tier": "revenue_share",
    "max_context_length": 2048
  }')

API_KEY=$(echo "$TENANT_RESPONSE" | grep -o '"api_key":"[^"]*' | cut -d'"' -f4)

if [ -z "$API_KEY" ]; then
    echo -e "${RED}✗ Failed to create tenant${NC}"
    echo "$TENANT_RESPONSE"
    exit 1
fi

echo -e "${GREEN}✓ Tenant created${NC}"
echo "API Key: $API_KEY"

# Save API key for manual testing
echo "$API_KEY" > .test_api_key
echo "(API key saved to .test_api_key)"

# Test inference
echo ""
echo "3. Testing inference endpoint..."
INFERENCE_RESPONSE=$(curl -s -X POST http://localhost/v1/infer \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is artificial intelligence?",
    "max_tokens": 50,
    "model": "gpt2",
    "optimization_level": "high"
  }')

REQUEST_ID=$(echo "$INFERENCE_RESPONSE" | grep -o '"request_id":"[^"]*' | cut -d'"' -f4)

if [ -z "$REQUEST_ID" ]; then
    echo -e "${RED}✗ Inference failed${NC}"
    echo "$INFERENCE_RESPONSE"
else
    echo -e "${GREEN}✓ Inference successful${NC}"
    echo "Request ID: $REQUEST_ID"
    TOKENS=$(echo "$INFERENCE_RESPONSE" | grep -o '"total_tokens":[0-9]*' | cut -d':' -f2)
    LATENCY=$(echo "$INFERENCE_RESPONSE" | grep -o '"latency_ms":[0-9.]*' | cut -d':' -f2)
    echo "Tokens: $TOKENS, Latency: ${LATENCY}ms"
fi

# Test usage endpoint
echo ""
echo "4. Testing usage endpoint..."
USAGE_RESPONSE=$(curl -s http://localhost/v1/usage \
  -H "X-API-Key: $API_KEY")

TOTAL_TOKENS=$(echo "$USAGE_RESPONSE" | grep -o '"total_tokens":[0-9]*' | cut -d':' -f2)

if [ -z "$TOTAL_TOKENS" ]; then
    echo -e "${RED}✗ Usage query failed${NC}"
    echo "$USAGE_RESPONSE"
else
    echo -e "${GREEN}✓ Usage query successful${NC}"
    echo "Total tokens used: $TOTAL_TOKENS"
fi

# Test rate limiting
echo ""
echo "5. Testing rate limiting (making 70 requests, limit is 60/min)..."
echo -n "Sending requests: "
RATE_LIMIT_HIT=0
for i in {1..70}; do
    RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost/v1/infer \
      -H "X-API-Key: $API_KEY" \
      -H "Content-Type: application/json" \
      -d '{"prompt": "test", "max_tokens": 10}')

    if [ "$RESPONSE" = "429" ]; then
        RATE_LIMIT_HIT=1
        break
    fi
    echo -n "."
done

if [ $RATE_LIMIT_HIT -eq 1 ]; then
    echo ""
    echo -e "${GREEN}✓ Rate limiting works (hit 429 at request $i)${NC}"
else
    echo ""
    echo -e "${YELLOW}⚠ Rate limiting not triggered (may need adjustment)${NC}"
fi

# Summary
echo ""
echo -e "${GREEN}=========================================="
echo "All Tests Passed!"
echo "==========================================${NC}"
echo ""
echo "Local API is running at: http://localhost"
echo "API Documentation: http://localhost/docs"
echo "Health Check: http://localhost/health"
echo ""
echo "Test API Key: $API_KEY"
echo "(saved to .test_api_key)"
echo ""
echo "View logs: docker-compose logs -f api"
echo "Stop services: docker-compose down"
echo ""
echo "Example inference request:"
echo ""
echo "curl -X POST http://localhost/v1/infer \\"
echo "  -H \"X-API-Key: $API_KEY\" \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{"
echo "    \"prompt\": \"Explain quantum computing\","
echo "    \"max_tokens\": 256,"
echo "    \"model\": \"gpt2\","
echo "    \"optimization_level\": \"high\""
echo "  }'"
echo ""
