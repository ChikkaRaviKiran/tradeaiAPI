#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# TradeAI — SSL Certificate Setup via Let's Encrypt (System Nginx)
# Run after first successful deployment:
#   sudo bash /opt/tradeai/scripts/ssl-setup.sh your@email.com
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

DOMAIN="tradeai.tavabharat.com"
EMAIL="${1:-}"

echo "Setting up SSL for $DOMAIN..."

# Get certificate using system certbot
if [ -n "$EMAIL" ]; then
    sudo certbot --nginx \
        --non-interactive \
        --agree-tos \
        --email "$EMAIL" \
        -d "$DOMAIN"
else
    sudo certbot --nginx \
        --non-interactive \
        --agree-tos \
        --register-unsafely-without-email \
        -d "$DOMAIN"
fi

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  SSL setup complete for $DOMAIN"
echo "  https://$DOMAIN is now live!"
echo "  Certbot auto-renew is handled by systemd timer."
echo "══════════════════════════════════════════════════════════"
