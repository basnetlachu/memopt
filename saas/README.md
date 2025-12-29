# MemOpt SaaS Backend

Production-ready SaaS backend for MemOpt with licensing, metering, billing, and multi-tier pricing.

## Quick Links

- **[Deployment Guide](../deploy/vps/README.md)** - Complete VPS deployment walkthrough
- **[Usage Examples](../deploy/USAGE_EXAMPLES.md)** - API usage examples (Python, JS, curl)
- **[Deployment Summary](../deploy/DEPLOYMENT_SUMMARY.md)** - Architecture overview & quick reference

## Features

✅ **Two-Tier Pricing**
- Enterprise Flat: $800k/year ($66,666.67/month)
- Revenue Share: 35% of savings ($0.002 per 1k tokens)

✅ **Authentication & Authorization**
- API key-based authentication (bcrypt hashed)
- Tenant isolation
- Admin API for privileged operations

✅ **Rate Limiting**
- Per-tier request limits (60-10,000 RPM)
- Per-tier token limits (100k-10M tokens/min)
- Concurrent request control (5-100 concurrent)
- Monthly quotas (100M tokens for revenue_share)

✅ **Usage Metering**
- Per-request tracking with UUID
- Token counting (prompt + completion)
- Latency monitoring
- Success/failure tracking
- Error logging

✅ **Automated Billing**
- Monthly invoice generation
- Usage-based pricing (revenue_share)
- Flat-fee billing (enterprise)
- Invoice status tracking (draft, issued, paid, overdue)

✅ **Production Ready**
- Docker Compose deployment
- Nginx reverse proxy with TLS
- PostgreSQL database
- Redis rate limiting
- Health checks
- Structured logging
- Backup/restore scripts

## Architecture

```
Client → Nginx (HTTPS) → FastAPI → PostgreSQL
                                  → Redis
```

### Components

- **FastAPI**: Main application server (port 8000)
- **PostgreSQL**: Persistent storage (tenants, API keys, usage, invoices)
- **Redis**: Rate limiting & caching
- **Nginx**: Reverse proxy, TLS termination, rate limiting

### Database Schema

**tenants** - Customer accounts
- id, name, tier, status, allowed_models, max_context_length, priority_level

**api_keys** - Authentication credentials
- id, tenant_id, key_hash, key_prefix, created_at, last_used_at, revoked_at

**usage_events** - Per-request metering
- id, tenant_id, request_id, timestamp, model, prompt_tokens, completion_tokens, total_tokens, latency_ms, success, error_code

**invoices** - Monthly billing
- id, tenant_id, period_start, period_end, tier, total_tokens, total_requests, amount_usd, status

## API Endpoints

### Public
- `GET /` - API information
- `GET /health` - Health check (database, Redis status)
- `GET /docs` - Swagger UI documentation

### Authenticated (requires X-API-Key header)
- `POST /v1/infer` - Run inference (rate limited, metered)
- `GET /v1/usage?start=&end=` - Get usage summary
- `GET /v1/invoices` - List invoices
- `GET /v1/invoices/{id}` - Get specific invoice

### Admin (requires X-Admin-Key header)
- `POST /admin/tenants` - Create tenant (returns API key)
- `POST /admin/tenants/{id}/suspend` - Suspend tenant
- `POST /admin/tenants/{id}/activate` - Activate tenant
- `POST /admin/tenants/{id}/api-keys` - Generate API key
- `DELETE /admin/api-keys/{id}` - Revoke API key
- `POST /admin/invoices/generate?year={year}&month={month}` - Generate invoices

## Quick Start

### Local Development

```bash
# Navigate to deploy directory
cd ../deploy

# Copy environment file
cp ../saas/.env.example .env

# Edit configuration (set ENV=development)
nano .env

# Start services
docker-compose up -d

# Run tests
./test_local.sh

# View API docs
open http://localhost/docs
```

### Production Deployment

```bash
# Follow the complete guide
cat ../deploy/vps/README.md

# Quick version:
cd ../deploy
cp ../saas/.env.example .env
# Edit .env (set secure keys!)
./vps/deploy.sh
```

## Configuration

### Environment Variables (.env)

**Required:**
- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string
- `ADMIN_API_KEY` - Secure admin key (32+ chars in production)

**Optional (with defaults):**
- `ENV` - Environment (development, staging, production)
- `RATE_LIMIT_REVENUE_SHARE` - Requests/min for revenue_share tier (60)
- `RATE_LIMIT_ENTERPRISE` - Requests/min for enterprise tier (10000)
- `TOKEN_LIMIT_REVENUE_SHARE_PER_MIN` - Tokens/min (100000)
- `TOKEN_LIMIT_ENTERPRISE_PER_MIN` - Tokens/min (10000000)
- `MAX_CONCURRENT_REVENUE_SHARE` - Concurrent requests (5)
- `MAX_CONCURRENT_ENTERPRISE` - Concurrent requests (100)
- `PRICE_PER_1K_TOKENS` - Pricing for revenue_share ($0.002)
- `ENTERPRISE_FLAT_MONTHLY` - Monthly fee ($66,666.67)

### Rate Limits

| Tier | RPM | Tokens/Min | Concurrent | Monthly Quota |
|------|-----|------------|------------|---------------|
| Revenue Share | 60 | 100,000 | 5 | 100,000,000 |
| Enterprise | 10,000 | 10,000,000 | 100 | Unlimited |

## Usage Examples

### Create Tenant (Admin)

```bash
curl -X POST https://api.memopt.io/admin/tenants \
  -H "X-Admin-Key: YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Acme Corp",
    "tier": "enterprise_flat",
    "priority_level": 10
  }'
```

Response:
```json
{
  "tenant_id": 1,
  "name": "Acme Corp",
  "tier": "enterprise_flat",
  "api_key": "sk_memopt_abc123..."
}
```

### Run Inference

```bash
curl -X POST https://api.memopt.io/v1/infer \
  -H "X-API-Key: sk_memopt_YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing",
    "max_tokens": 256,
    "model": "gpt2",
    "optimization_level": "high"
  }'
```

Response:
```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "generated_text": "Quantum computing is...",
  "prompt_tokens": 3,
  "completion_tokens": 150,
  "total_tokens": 153,
  "latency_ms": 1250.5,
  "model": "gpt2"
}
```

### Check Usage

```bash
curl https://api.memopt.io/v1/usage \
  -H "X-API-Key: sk_memopt_YOUR_KEY"
```

Response:
```json
{
  "total_requests": 1500,
  "total_tokens": 2400000,
  "prompt_tokens": 800000,
  "completion_tokens": 1600000,
  "avg_latency_ms": 1180.5,
  "failed_requests": 5,
  "success_rate": 99.67
}
```

More examples: [USAGE_EXAMPLES.md](../deploy/USAGE_EXAMPLES.md)

## Testing

### Run Local Tests

```bash
cd ../deploy
./test_local.sh
```

This will:
1. Start all services
2. Create a test tenant
3. Run inference requests
4. Test rate limiting
5. Verify usage tracking

### Manual Testing

```bash
# Health check
curl http://localhost/health

# View API documentation
open http://localhost/docs

# Check logs
docker-compose logs -f api
```

## Deployment

### Docker Compose (Development/Staging)

```bash
docker-compose up -d
```

### Production VPS (Hostinger)

See [VPS Deployment Guide](../deploy/vps/README.md) for complete instructions.

Quick version:
```bash
./vps/deploy.sh
```

## Maintenance

### Backups

```bash
# Manual backup
./vps/backup_db.sh

# Automated (add to cron)
0 2 * * * /path/to/backup_db.sh
```

### Restore

```bash
./vps/restore_db.sh /var/backups/memopt/memopt_backup_TIMESTAMP.sql.gz
```

### Updates

```bash
git pull origin main
./vps/deploy.sh
```

### Monitoring

```bash
# View logs
docker-compose logs -f api

# Check resource usage
docker stats

# Test health
curl https://yourdomain.com/health
```

## Security

### API Key Storage
- Keys are hashed with bcrypt (never stored in plaintext)
- Only the prefix (first 12 chars) is indexed for lookup
- Keys shown only once at creation

### Database Security
- PostgreSQL not exposed publicly (internal Docker network only)
- Redis not exposed publicly
- Connection pooling with authentication

### TLS/SSL
- Nginx enforces TLS 1.2+
- Strong cipher suites
- HSTS headers
- Let's Encrypt certificates (auto-renewal)

### Rate Limiting
- Nginx: Layer 7 rate limiting (50 req/s inference, 10 req/s admin)
- Application: Redis-backed token bucket per tenant
- Per-tier limits enforced in real-time

### Secrets Management
- All secrets in `.env` file (never committed to git)
- Admin API key required (validated length in production)
- Fail-fast on missing/weak secrets in production mode

## Troubleshooting

### API Not Responding

```bash
docker-compose ps  # Check if containers are running
docker-compose logs api  # Check logs
docker-compose restart api  # Restart API
```

### Database Connection Issues

```bash
docker-compose logs postgres
docker-compose exec postgres psql -U memopt -d memopt -c "SELECT 1;"
```

### Rate Limit Errors (429)

Edit `.env` to increase limits:
```bash
RATE_LIMIT_REVENUE_SHARE=120  # was 60
TOKEN_LIMIT_REVENUE_SHARE_PER_MIN=200000  # was 100000
```

Then restart:
```bash
docker-compose restart api
```

More troubleshooting: [VPS README](../deploy/vps/README.md#troubleshooting)

## Development

### Project Structure

```
saas/
├── main.py           # FastAPI application (endpoints)
├── config.py         # Configuration & validation
├── database.py       # SQLAlchemy models
├── auth.py           # API key authentication
├── rate_limiter.py   # Redis rate limiting
├── billing.py        # Invoice generation & usage queries
├── requirements.txt  # Python dependencies
├── Dockerfile        # Container build
└── .env.example      # Environment template
```

### Adding New Endpoints

```python
# In main.py
@app.get("/v1/new-endpoint")
async def new_endpoint(
    ctx: TenantContext = Depends(get_tenant_from_api_key),
    db: Session = Depends(get_db),
):
    # Your code here
    return {"result": "success"}
```

### Database Migrations

For production, use Alembic:

```bash
# Install
pip install alembic

# Initialize
alembic init migrations

# Create migration
alembic revision --autogenerate -m "Add new table"

# Apply
alembic upgrade head
```

## Performance

### Expected Performance

- **Latency**: <2s per inference request (model dependent)
- **Throughput**: 100+ requests/second (single API instance)
- **Database**: Handles 10M+ usage events with proper indexing

### Scaling

**Horizontal Scaling** (multiple API instances):
```yaml
api:
  deploy:
    replicas: 3
```

**Database Optimization**:
- Add indexes on frequently queried columns
- Use read replicas for usage queries
- Archive old usage_events to separate table

**Redis Optimization**:
- Use Redis Cluster for high availability
- Separate rate limiting and caching Redis instances

## License

Proprietary - All Rights Reserved

## Support

- **Documentation**: See `/deploy` directory
- **API Docs**: `https://yourdomain.com/docs`
- **Health**: `https://yourdomain.com/health`
