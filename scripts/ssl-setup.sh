#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# TradeAI — SSL Certificate Setup via Let's Encrypt
# Run after server-setup.sh and first deployment:
#   sudo bash scripts/ssl-setup.sh your-domain.com your@email.com
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

DOMAIN="${1:?Usage: ssl-setup.sh <domain> <email>}"
EMAIL="${2:?Usage: ssl-setup.sh <domain> <email>}"
APP_DIR="/opt/tradeai"

echo "Setting up SSL for $DOMAIN..."

# 1. Get certificate
docker run --rm \
    -v "$APP_DIR/nginx/certbot/conf:/etc/letsencrypt" \
    -v "$APP_DIR/nginx/certbot/www:/var/www/certbot" \
    -p 80:80 \
    certbot/certbot certonly \
    --standalone \
    --non-interactive \
    --agree-tos \
    --email "$EMAIL" \
    -d "$DOMAIN"

echo "Certificate obtained!"

# 2. Update nginx.conf — remind user
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  SSL certificate installed for $DOMAIN"
echo ""
echo "  Next steps:"
echo "    1. Edit $APP_DIR/nginx/nginx.conf"
echo "       - Comment out the HTTP-only server block"
echo "       - Uncomment the HTTPS server block"
echo "       - Uncomment the HTTP→HTTPS redirect block"
echo "       - Replace YOUR_DOMAIN.com with $DOMAIN"
echo "    2. Restart: cd $APP_DIR && docker compose -f docker-compose.prod.yml restart nginx"
echo "    3. Certbot container will auto-renew every 12 hours"
echo "══════════════════════════════════════════════════════════"
