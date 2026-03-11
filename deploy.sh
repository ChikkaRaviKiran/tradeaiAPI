#!/bin/bash
# ─── TradeAI AWS Ubuntu Deployment Script ───────────────────────────────
# Run as: sudo bash deploy.sh
# Tested on: Ubuntu 22.04 / 24.04 LTS (t3.small or larger)
#
# Expected directory layout on server:
#   /opt/tradeai/
#     backend/   ← tradeai-engine repo
#     frontend/  ← tradeai-dashboard repo
set -euo pipefail

APP_DIR="/opt/tradeai"
APP_USER="tradeai"

echo "══════════════════════════════════════════════════════════"
echo "  TradeAI — AWS Ubuntu Deployment"
echo "══════════════════════════════════════════════════════════"

# 1. System packages
echo "[1/8] Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq docker.io docker-compose curl git

# 2. Start & enable Docker
echo "[2/8] Enabling Docker..."
systemctl enable docker
systemctl start docker

# 3. Create app user (non-root)
echo "[3/8] Setting up application user..."
if ! id "$APP_USER" &>/dev/null; then
    useradd -r -m -s /bin/bash "$APP_USER"
    usermod -aG docker "$APP_USER"
fi

# 4. Clone repos (skip if already present)
echo "[4/8] Cloning repositories..."
mkdir -p "$APP_DIR"
if [ ! -d "$APP_DIR/backend" ]; then
    echo "  ⚠ Clone tradeai-engine repo to $APP_DIR/backend"
    echo "    git clone https://github.com/<you>/tradeai-engine.git $APP_DIR/backend"
fi
if [ ! -d "$APP_DIR/frontend" ]; then
    echo "  ⚠ Clone tradeai-dashboard repo to $APP_DIR/frontend"
    echo "    git clone https://github.com/<you>/tradeai-dashboard.git $APP_DIR/frontend"
fi
chown -R "$APP_USER:$APP_USER" "$APP_DIR"

# 5. Set timezone to IST
echo "[5/8] Setting timezone to Asia/Kolkata..."
timedatectl set-timezone Asia/Kolkata

# 6. Create systemd service
echo "[6/8] Creating systemd service..."
cat > /etc/systemd/system/tradeai.service << 'EOF'
[Unit]
Description=TradeAI NIFTY Options Trading System
After=docker.service network-online.target
Requires=docker.service
Wants=network-online.target

[Service]
Type=simple
User=tradeai
Group=docker
WorkingDirectory=/opt/tradeai/backend
ExecStartPre=/usr/bin/docker-compose pull
ExecStart=/usr/bin/docker-compose up --build
ExecStop=/usr/bin/docker-compose down
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal

StartLimitIntervalSec=300
StartLimitBurst=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable tradeai.service

# 7. Create update script
echo "[7/8] Creating update helper..."
cat > "$APP_DIR/update.sh" << 'UPDATESCRIPT'
#!/bin/bash
# Pull latest code and restart
cd /opt/tradeai/backend && git pull
cd /opt/tradeai/frontend && git pull
sudo systemctl restart tradeai
echo "Updated and restarted"
UPDATESCRIPT
chmod +x "$APP_DIR/update.sh"

# 8. Start it
echo "[8/8] Starting TradeAI..."
systemctl start tradeai.service

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  ✓ Deployment complete!"
echo "══════════════════════════════════════════════════════════"
echo ""
echo "  Dashboard:  http://<your-server-ip>:3000"
echo "  Backend:    http://<your-server-ip>:8000/api/health"
echo ""
echo "  Commands:"
echo "    sudo systemctl status tradeai    # Check status"
echo "    sudo systemctl restart tradeai   # Restart"
echo "    sudo journalctl -u tradeai -f    # View logs"
echo "    sudo systemctl stop tradeai      # Stop"
echo ""
echo "  The system will:"
echo "    • Auto-start on server boot"
echo "    • Auto-restart if it crashes"
echo "    • Run the trading loop daily (08:45-16:00 IST)"
echo "    • Sleep overnight and auto-resume next trading day"
echo "    • Skip weekends automatically"
echo ""
