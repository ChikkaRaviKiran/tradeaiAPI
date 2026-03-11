#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# TradeAI — SSL Certificate Setup via Let's Encrypt
# Run after first successful deployment:
#   sudo bash /opt/tradeai/scripts/ssl-setup.sh your@email.com
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

DOMAIN="tradeai.tavabharat.com"
EMAIL="${1:?Usage: ssl-setup.sh <email>}"
APP_DIR="/opt/tradeai"

echo "Setting up SSL for $DOMAIN..."

# 1. Stop nginx temporarily so certbot can bind to port 80
cd "$APP_DIR"
docker compose -f docker-compose.prod.yml stop nginx

# 2. Get certificate
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

# 3. Switch nginx to HTTPS mode
NGINX_CONF="$APP_DIR/nginx/nginx.conf"

# Uncomment the HTTPS redirect block
sed -i '/# server {/{
    /listen 80;/{
        /tradeai.tavabharat.com/{
            N;N;N;N;N;N;N;N;N
        }
    }
}' "$NGINX_CONF"

# Simpler approach: replace the entire config with SSL-enabled version
cat > "$NGINX_CONF" << 'NGINX_EOF'
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent"';
    access_log /var/log/nginx/access.log main;

    sendfile on;
    tcp_nopush on;
    keepalive_timeout 65;
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml;

    limit_req_zone $binary_remote_addr zone=api:10m rate=30r/m;
    limit_req_zone $binary_remote_addr zone=general:10m rate=60r/m;

    upstream backend_api {
        server backend:8000;
    }

    upstream frontend_app {
        server frontend:3000;
    }

    # HTTP → HTTPS redirect
    server {
        listen 80;
        server_name tradeai.tavabharat.com;

        location /.well-known/acme-challenge/ {
            root /var/www/certbot;
        }

        location / {
            return 301 https://$host$request_uri;
        }
    }

    # HTTPS Server
    server {
        listen 443 ssl http2;
        server_name tradeai.tavabharat.com;

        ssl_certificate /etc/letsencrypt/live/tradeai.tavabharat.com/fullchain.pem;
        ssl_certificate_key /etc/letsencrypt/live/tradeai.tavabharat.com/privkey.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;
        ssl_prefer_server_ciphers on;

        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header Referrer-Policy "strict-origin-when-cross-origin" always;

        location /api/ {
            limit_req zone=api burst=10 nodelay;
            proxy_pass http://backend_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_read_timeout 60s;
        }

        location / {
            limit_req zone=general burst=20 nodelay;
            proxy_pass http://frontend_app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
    }
}
NGINX_EOF

echo "Nginx config updated for HTTPS"

# 4. Restart nginx with SSL
docker compose -f docker-compose.prod.yml up -d nginx

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  SSL setup complete for $DOMAIN"
echo "  https://$DOMAIN is now live!"
echo "  Certbot container will auto-renew every 12 hours."
echo "══════════════════════════════════════════════════════════"
