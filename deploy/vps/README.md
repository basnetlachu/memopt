

# MemOpt SaaS VPS Deployment Guide

Complete guide for deploying MemOpt SaaS to a Hostinger VPS with Docker Compose, Nginx, and Let's Encrypt SSL.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Initial VPS Setup](#initial-vps-setup)
3. [Install Docker](#install-docker)
4. [Configure Application](#configure-application)
5. [Obtain SSL Certificate](#obtain-ssl-certificate)
6. [Deploy Application](#deploy-application)
7. [Create First Tenant](#create-first-tenant)
8. [Usage Examples](#usage-examples)
9. [Maintenance](#maintenance)
10. [Troubleshooting](#troubleshooting)
11. [Security Best Practices](#security-best-practices)

---

## Prerequisites

- Hostinger VPS (or any Ubuntu 22.04 LTS server)
- Domain name pointed to your VPS IP
- SSH access to the server
- Minimum 2GB RAM, 2 CPU cores, 20GB disk

---

## Initial VPS Setup

### 1. Connect to VPS

```bash
ssh root@your-vps-ip
```

### 2. Create Non-Root User

```bash
# Create user
adduser memopt

# Add to sudo group
usermod -aG sudo memopt

# Switch to new user
su - memopt
```

### 3. Update System

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y curl git ufw
```

### 4. Configure Firewall

```bash
# Allow SSH, HTTP, HTTPS
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Enable firewall
sudo ufw enable
sudo ufw status
```

---

## Install Docker

### 1. Install Docker Engine

```bash
# Remove old versions
sudo apt remove docker docker-engine docker.io containerd runc

# Install dependencies
sudo apt install -y apt-transport-https ca-certificates curl gnupg lsb-release

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Set up stable repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Verify installation
docker --version
```

### 2. Add User to Docker Group

```bash
sudo usermod -aG docker $USER
newgrp docker

# Test without sudo
docker ps
```

### 3. Install Docker Compose

```bash
# Download Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

# Make executable
sudo chmod +x /usr/local/bin/docker-compose

# Verify
docker-compose --version
```

---

## Configure Application

### 1. Clone Repository

```bash
cd ~
git clone https://github.com/yourusername/memopt.git
cd memopt/deploy
```

### 2. Create Environment File

```bash
# Copy example
cp ../saas/.env.example .env

# Edit configuration
nano .env
```

**Required Configuration:**

```bash
# Environment
ENV=production

# Database (CHANGE PASSWORD!)
DATABASE_URL=postgresql://memopt:YOUR_STRONG_PASSWORD_HERE@postgres:5432/memopt

# Redis
REDIS_URL=redis://redis:6379/0

# Admin API Key (CRITICAL: Generate secure key!)
# Generate with: openssl rand -base64 32
ADMIN_API_KEY=YOUR_SECURE_RANDOM_STRING_HERE_32_CHARS_MINIMUM

# Postgres password (same as in DATABASE_URL)
POSTGRES_PASSWORD=YOUR_STRONG_PASSWORD_HERE

# Rate limits (optional, defaults shown)
RATE_LIMIT_REVENUE_SHARE=60
RATE_LIMIT_ENTERPRISE=10000

# Billing (optional, defaults shown)
PRICE_PER_1K_TOKENS=0.002
ENTERPRISE_FLAT_MONTHLY=66666.67

# CORS (optional, use your domain in production)
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
```

### 3. Generate Secure Keys

```bash
# Generate admin API key
openssl rand -base64 32

# Generate database password
openssl rand -base64 24
```

### 4. Update Nginx Configuration

```bash
nano vps/nginx.conf
```

Update the following lines with your domain:

```nginx
# Line ~100
server_name your-domain.com www.your-domain.com;

# Lines ~105-106 (update paths after getting SSL cert)
ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
```

---

## Obtain SSL Certificate

### 1. Initial HTTP-Only Setup

First, temporarily modify `nginx.conf` to allow HTTP for Let's Encrypt challenge:

```nginx
# Comment out SSL server block temporarily
# Keep only the HTTP redirect server
```

### 2. Start Services (HTTP only)

```bash
cd ~/memopt/deploy
docker-compose up -d nginx
```

### 3. Obtain Certificate

```bash
# Replace YOUR_DOMAIN and YOUR_EMAIL
docker-compose run --rm certbot certonly \
  --webroot \
  --webroot-path=/var/www/certbot \
  --email your-email@example.com \
  --agree-tos \
  --no-eff-email \
  -d your-domain.com \
  -d www.your-domain.com
```

### 4. Verify Certificate

```bash
# Check if certificate was created
docker-compose exec nginx ls -la /etc/letsencrypt/live/your-domain.com/
```

### 5. Enable HTTPS

Uncomment the HTTPS server block in `nginx.conf` and reload:

```bash
docker-compose exec nginx nginx -s reload
```

---

## Deploy Application

### 1. Build and Start All Services

```bash
cd ~/memopt/deploy
./vps/deploy.sh
```

This script will:
- Pull latest code
- Build Docker images
- Start all services (Postgres, Redis, API, Nginx)
- Run health checks

### 2. Verify Deployment

```bash
# Check all services are running
docker-compose ps

# Check API health
curl https://your-domain.com/health

# Check logs
docker-compose logs -f api
```

Expected output:
```json
{
  "status": "healthy",
  "database": "healthy",
  "redis": "healthy",
  "version": "1.0.0",
  "env": "production"
}
```

### 3. Access API Documentation

Open in browser: `https://your-domain.com/docs`

---

## Create First Tenant

### 1. Create Enterprise Tenant ($800k/year)

```bash
curl -X POST https://your-domain.com/admin/tenants \
  -H "X-Admin-Key: YOUR_ADMIN_API_KEY" \
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
  "api_key": "sk_memopt_abc123xyz..."
}
```

**IMPORTANT:** Save the `api_key` - it's only shown once!

### 2. Create Revenue Share Tenant (35% of savings)

```bash
curl -X POST https://your-domain.com/admin/tenants \
  -H "X-Admin-Key: YOUR_ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Startup XYZ",
    "tier": "revenue_share",
    "allowed_models": "gpt2,gpt2-medium",
    "max_context_length": 2048,
    "priority_level": 5
  }'
```

Response:
```json
{
  "tenant_id": 2,
  "name": "Startup XYZ",
  "tier": "revenue_share",
  "api_key": "sk_memopt_def456uvw..."
}
```

---

## Usage Examples

### 1. Run Inference

```bash
curl -X POST https://your-domain.com/v1/infer \
  -H "X-API-Key: sk_memopt_abc123xyz..." \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing in simple terms",
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
  "prompt_tokens": 8,
  "completion_tokens": 150,
  "total_tokens": 158,
  "latency_ms": 1250.5,
  "model": "gpt2"
}
```

### 2. Check Usage

```bash
curl https://your-domain.com/v1/usage \
  -H "X-API-Key: sk_memopt_abc123xyz..."
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

### 3. List Invoices

```bash
curl https://your-domain.com/v1/invoices \
  -H "X-API-Key: sk_memopt_abc123xyz..."
```

Response:
```json
[
  {
    "id": 1,
    "tenant_id": 1,
    "period_start": "2024-01-01T00:00:00Z",
    "period_end": "2024-02-01T00:00:00Z",
    "tier": "enterprise_flat",
    "total_tokens": 50000000,
    "total_requests": 100000,
    "amount_usd": 66666.67,
    "status": "issued",
    "created_at": "2024-02-01T00:00:00Z"
  }
]
```

### 4. Generate Monthly Invoices (Admin)

```bash
curl -X POST "https://your-domain.com/admin/invoices/generate?year=2024&month=1" \
  -H "X-Admin-Key: YOUR_ADMIN_API_KEY"
```

---

## Maintenance

### Daily Backups

Setup automatic daily backups with cron:

```bash
# Edit crontab
crontab -e

# Add this line (backup at 2 AM daily)
0 2 * * * /home/memopt/memopt/deploy/vps/backup_db.sh >> /var/log/memopt_backup.log 2>&1
```

Manual backup:

```bash
./vps/backup_db.sh
```

### Restore Database

```bash
# List backups
ls -lh /var/backups/memopt/

# Restore from specific backup
./vps/restore_db.sh /var/backups/memopt/memopt_backup_20240115_020000.sql.gz
```

### Update Application

```bash
cd ~/memopt/deploy

# Pull latest code
git pull origin main

# Redeploy
./vps/deploy.sh
```

### View Logs

```bash
# All services
docker-compose logs -f

# API only
docker-compose logs -f api

# Last 100 lines
docker-compose logs --tail=100 api

# Follow specific service
docker-compose logs -f postgres
```

### Restart Services

```bash
# Restart all
docker-compose restart

# Restart specific service
docker-compose restart api

# Restart with rebuild
docker-compose up -d --build api
```

### Monitor Resources

```bash
# Docker stats
docker stats

# Disk usage
docker system df

# Clean up unused images/volumes
docker system prune -a --volumes
```

---

## Troubleshooting

### API Not Responding

```bash
# Check if container is running
docker-compose ps

# Check logs
docker-compose logs --tail=50 api

# Check health endpoint
curl http://localhost:8000/health

# Restart API
docker-compose restart api
```

### Database Connection Issues

```bash
# Check Postgres is running
docker-compose ps postgres

# Test database connection
docker-compose exec postgres psql -U memopt -d memopt -c "SELECT 1;"

# Check database logs
docker-compose logs postgres

# Restart database (WARNING: may cause downtime)
docker-compose restart postgres
```

### Redis Connection Issues

```bash
# Check Redis is running
docker-compose ps redis

# Test Redis connection
docker-compose exec redis redis-cli ping

# Check Redis logs
docker-compose logs redis
```

### SSL Certificate Issues

```bash
# Check certificate expiry
docker-compose exec nginx openssl x509 -in /etc/letsencrypt/live/your-domain.com/fullchain.pem -noout -dates

# Renew certificate manually
docker-compose run --rm certbot renew

# Check Nginx configuration
docker-compose exec nginx nginx -t

# Reload Nginx
docker-compose exec nginx nginx -s reload
```

### Rate Limit Errors (429)

If users are getting rate limited:

```bash
# Edit .env and increase limits
nano .env

# Update these values
RATE_LIMIT_REVENUE_SHARE=120  # was 60
TOKEN_LIMIT_REVENUE_SHARE_PER_MIN=200000  # was 100000

# Restart API
docker-compose restart api
```

### Out of Memory

```bash
# Check memory usage
free -h

# Check Docker container memory
docker stats

# Increase swap (if needed)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Disk Full

```bash
# Check disk usage
df -h

# Clean up Docker
docker system prune -a --volumes

# Clean up old logs
sudo journalctl --vacuum-time=7d

# Check large files
du -sh /var/lib/docker/* | sort -h
```

---

## Security Best Practices

### 1. Secrets Management

**Never commit secrets to git!**

```bash
# Add .env to .gitignore
echo ".env" >> .gitignore

# Store .env backup securely (encrypted)
gpg -c .env  # Creates .env.gpg
```

### 2. API Key Rotation

Rotate tenant API keys periodically:

```bash
# Create new key for tenant
curl -X POST https://your-domain.com/admin/tenants/1/api-keys \
  -H "X-Admin-Key: YOUR_ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name": "Rotated Key 2024-01"}'

# Revoke old key
curl -X DELETE https://your-domain.com/admin/api-keys/OLD_KEY_ID \
  -H "X-Admin-Key: YOUR_ADMIN_API_KEY"
```

### 3. Database Security

```bash
# Ensure Postgres is not exposed publicly
sudo ufw status  # Should only allow 22, 80, 443

# Regular security updates
sudo apt update && sudo apt upgrade -y
```

### 4. Monitor Failed Requests

```bash
# Check logs for failed auth attempts
docker-compose logs api | grep "unauthorized"

# Check for suspicious patterns
docker-compose logs api | grep "429"  # Rate limit hits
```

### 5. Enable Fail2Ban (Optional)

```bash
# Install fail2ban
sudo apt install fail2ban

# Configure for Nginx
sudo nano /etc/fail2ban/jail.local

# Add:
[nginx-http-auth]
enabled = true
port = http,https
logpath = /var/log/nginx/error.log
```

---

## Production Checklist

Before going live:

- [ ] Change all default passwords
- [ ] Generate strong ADMIN_API_KEY (32+ characters)
- [ ] Configure proper domain in Nginx
- [ ] Obtain valid SSL certificate
- [ ] Setup automated backups (cron)
- [ ] Configure firewall (only ports 22, 80, 443)
- [ ] Test failover scenarios
- [ ] Setup monitoring (optional: Prometheus, Grafana)
- [ ] Document disaster recovery procedure
- [ ] Test API with real inference requests
- [ ] Verify rate limiting works
- [ ] Test invoice generation
- [ ] Setup log rotation

---

## Support

For issues or questions:

1. Check logs: `docker-compose logs -f`
2. Review this troubleshooting guide
3. Check GitHub issues
4. Contact support (if applicable)

---

## License

Proprietary - All Rights Reserved
