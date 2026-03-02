# memopt TLS Setup

nginx reverse-proxy with TLS termination for memopt servers.
uvicorn continues to listen on plain HTTP on localhost; nginx handles all external TLS.

## Quick Start

### Option 1 — Self-signed (dev / internal)

```bash
sudo ./setup_tls.sh --mode self-signed --domain memopt.example.com
```

Generates a 4096-bit RSA cert valid 10 years under `/etc/ssl/memopt/`.
Browsers will show a warning; internal clients can use `--insecure` / `verify=False`.

### Option 2 — Let's Encrypt (production, requires public DNS)

```bash
sudo ./setup_tls.sh \
    --mode letsencrypt \
    --domain memopt.example.com \
    --email  admin@example.com
```

Requires certbot and that `memopt.example.com` resolves to this machine's public IP.
Auto-renewal cron is added automatically (`certbot renew` at 03:00 daily).

### Option 3 — Bring your own cert

```bash
sudo ./setup_tls.sh \
    --mode    existing \
    --domain  memopt.example.com \
    --cert    /path/to/fullchain.pem \
    --key     /path/to/privkey.pem
```

## What the script does

1. Generates or locates the TLS certificate and key.
2. Renders `nginx.conf.template` → `/etc/nginx/conf.d/memopt.conf`.
3. Runs `nginx -t` to validate the config.
4. Reloads nginx (via `systemctl reload nginx` or `nginx -s reload`).

## Security properties enforced

| Property | Value |
|---|---|
| Minimum TLS version | TLS 1.2 (TLS 1.3 preferred) |
| Cipher suite | Mozilla "Intermediate" — ECDHE only, no RC4/3DES/export |
| Forward secrecy | Yes (ECDHE key exchange) |
| HSTS | `max-age=31536000; includeSubDomains` |
| OCSP stapling | Enabled |
| Session tickets | Disabled |
| X-Frame-Options | DENY |
| X-Content-Type-Options | nosniff |

## Testing

```bash
# Health check (no auth required)
curl https://memopt.example.com/health

# Protected endpoint (API key required)
curl -H "X-Memopt-API-Key: $(cat ~/.memopt/api_key)" \
     https://memopt.example.com/api/v1/status

# Verify TLS version
openssl s_client -connect memopt.example.com:443 -tls1_1 2>&1 | grep "handshake failure"
# Should print: handshake failure (TLS 1.1 is rejected)
```

## Optional `--upstream`

By default the proxy forwards to `127.0.0.1:8080`.
To use a different port:

```bash
sudo ./setup_tls.sh --mode self-signed --domain memopt.internal --upstream 127.0.0.1:9090
```
