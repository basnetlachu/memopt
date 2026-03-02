#!/usr/bin/env bash
# memopt TLS setup — nginx reverse-proxy for memopt servers
#
# Usage:
#   ./setup_tls.sh --mode self-signed   --domain memopt.example.com  [--upstream 127.0.0.1:8080]
#   ./setup_tls.sh --mode letsencrypt   --domain memopt.example.com  [--email admin@example.com]
#   ./setup_tls.sh --mode existing      --domain memopt.example.com  --cert /path/cert.pem --key /path/key.pem
#
# Modes
#   self-signed   Generate a self-signed cert (dev/internal use).
#   letsencrypt   Obtain a cert from Let's Encrypt via certbot (requires public DNS).
#   existing      Use certs you already have.
#
# Requirements: nginx, openssl (self-signed), certbot (letsencrypt)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/nginx.conf.template"
NGINX_CONF_DIR="/etc/nginx/conf.d"
NGINX_CONF="${NGINX_CONF_DIR}/memopt.conf"

# ── Defaults ──────────────────────────────────────────────────────────────────
MODE=""
DOMAIN=""
EMAIL=""
CERT_PATH=""
KEY_PATH=""
UPSTREAM="127.0.0.1:8080"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)      MODE="$2";       shift 2 ;;
        --domain)    DOMAIN="$2";     shift 2 ;;
        --email)     EMAIL="$2";      shift 2 ;;
        --cert)      CERT_PATH="$2";  shift 2 ;;
        --key)       KEY_PATH="$2";   shift 2 ;;
        --upstream)  UPSTREAM="$2";   shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Validate ──────────────────────────────────────────────────────────────────
if [[ -z "$MODE" ]]; then
    echo "ERROR: --mode is required (self-signed | letsencrypt | existing)" >&2
    exit 1
fi
if [[ -z "$DOMAIN" ]]; then
    echo "ERROR: --domain is required" >&2
    exit 1
fi
if [[ ! -f "$TEMPLATE" ]]; then
    echo "ERROR: nginx.conf.template not found at $TEMPLATE" >&2
    exit 1
fi

command -v nginx >/dev/null 2>&1 || { echo "ERROR: nginx not installed" >&2; exit 1; }

# ── Mode: self-signed ─────────────────────────────────────────────────────────
if [[ "$MODE" == "self-signed" ]]; then
    command -v openssl >/dev/null 2>&1 || { echo "ERROR: openssl not installed" >&2; exit 1; }
    CERT_DIR="/etc/ssl/memopt"
    mkdir -p "$CERT_DIR"
    CERT_PATH="${CERT_DIR}/${DOMAIN}.crt"
    KEY_PATH="${CERT_DIR}/${DOMAIN}.key"

    echo "Generating self-signed certificate for ${DOMAIN}..."
    openssl req -x509 -nodes -days 3650 -newkey rsa:4096 \
        -keyout "$KEY_PATH" \
        -out    "$CERT_PATH" \
        -subj   "/CN=${DOMAIN}/O=memopt/C=US" \
        -addext "subjectAltName=DNS:${DOMAIN}"
    chmod 600 "$KEY_PATH"
    echo "  Certificate: $CERT_PATH"
    echo "  Key:         $KEY_PATH"

# ── Mode: letsencrypt ─────────────────────────────────────────────────────────
elif [[ "$MODE" == "letsencrypt" ]]; then
    command -v certbot >/dev/null 2>&1 || { echo "ERROR: certbot not installed. Install with: apt install certbot python3-certbot-nginx" >&2; exit 1; }
    if [[ -z "$EMAIL" ]]; then
        echo "ERROR: --email is required for letsencrypt mode" >&2
        exit 1
    fi

    echo "Obtaining Let's Encrypt certificate for ${DOMAIN}..."
    certbot certonly --nginx \
        --non-interactive \
        --agree-tos \
        --email "$EMAIL" \
        -d "$DOMAIN"

    CERT_PATH="/etc/letsencrypt/live/${DOMAIN}/fullchain.pem"
    KEY_PATH="/etc/letsencrypt/live/${DOMAIN}/privkey.pem"

    # Add auto-renewal cron if not already present
    CRON_LINE="0 3 * * * certbot renew --quiet && nginx -s reload"
    (crontab -l 2>/dev/null | grep -qF "certbot renew") || \
        (crontab -l 2>/dev/null; echo "$CRON_LINE") | crontab -
    echo "  Auto-renewal cron added."

# ── Mode: existing ────────────────────────────────────────────────────────────
elif [[ "$MODE" == "existing" ]]; then
    if [[ -z "$CERT_PATH" || -z "$KEY_PATH" ]]; then
        echo "ERROR: --cert and --key are required for existing mode" >&2
        exit 1
    fi
    if [[ ! -f "$CERT_PATH" ]]; then
        echo "ERROR: cert not found: $CERT_PATH" >&2
        exit 1
    fi
    if [[ ! -f "$KEY_PATH" ]]; then
        echo "ERROR: key not found: $KEY_PATH" >&2
        exit 1
    fi

else
    echo "ERROR: unknown --mode: $MODE (must be self-signed | letsencrypt | existing)" >&2
    exit 1
fi

# ── Write nginx config ────────────────────────────────────────────────────────
mkdir -p "$NGINX_CONF_DIR"
echo "Writing nginx config to ${NGINX_CONF}..."
sed \
    -e "s|\${DOMAIN}|${DOMAIN}|g" \
    -e "s|\${CERT_PATH}|${CERT_PATH}|g" \
    -e "s|\${KEY_PATH}|${KEY_PATH}|g" \
    -e "s|\${UPSTREAM}|${UPSTREAM}|g" \
    "$TEMPLATE" > "$NGINX_CONF"

# ── Test and reload nginx ─────────────────────────────────────────────────────
echo "Testing nginx configuration..."
nginx -t

echo "Reloading nginx..."
if command -v systemctl >/dev/null 2>&1; then
    systemctl reload nginx
else
    nginx -s reload
fi

echo ""
echo "Done! memopt is now available at: https://${DOMAIN}"
echo ""
echo "Next steps:"
echo "  1. Ensure your firewall allows ports 80 and 443"
echo "  2. Set MEMOPT_API_KEY in your client environment"
echo "  3. Test: curl -k -H 'X-Memopt-API-Key: <key>' https://${DOMAIN}/health"
