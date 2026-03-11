#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# TradeAI — AWS Ubuntu Server Initial Setup
# Run ONCE on a fresh Ubuntu 22.04/24.04 EC2 instance:
#   curl -sSL https://raw.githubusercontent.com/<owner>/TradeAI/main/scripts/server-setup.sh | sudo bash
# Or:
#   sudo bash scripts/server-setup.sh
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

APP_DIR="/opt/tradeai"
APP_USER="tradeai"

echo "══════════════════════════════════════════════════════════"
echo "  TradeAI — Server Setup (run once)"
echo "══════════════════════════════════════════════════════════"

# ── 1. System updates & dependencies ────────────────────────────────────
echo "[1/8] Updating system packages..."
apt-get update -qq
apt-get upgrade -y -qq
apt-get install -y -qq \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    ufw \
    fail2ban \
    htop \
    unzip

# ── 2. Install Docker (official repo) ──────────────────────────────────
echo "[2/8] Installing Docker..."
if ! command -v docker &>/dev/null; then
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
        gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg

    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
        tee /etc/apt/sources.list.d/docker.list > /dev/null

    apt-get update -qq
    apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin
else
    echo "  Docker already installed: $(docker --version)"
fi

systemctl enable docker
systemctl start docker

# ── 3. Create application user ─────────────────────────────────────────
echo "[3/8] Creating application user..."
if ! id "$APP_USER" &>/dev/null; then
    useradd -r -m -s /bin/bash "$APP_USER"
    usermod -aG docker "$APP_USER"
    echo "  Created user: $APP_USER"
else
    echo "  User $APP_USER already exists"
fi

# ── 4. Create application directory ────────────────────────────────────
echo "[4/8] Setting up application directory..."
mkdir -p "$APP_DIR"/{nginx,logs,data}
chown -R "$APP_USER:$APP_USER" "$APP_DIR"

# ── 5. Timezone to IST ─────────────────────────────────────────────────
echo "[5/8] Setting timezone to Asia/Kolkata (IST)..."
timedatectl set-timezone Asia/Kolkata

# ── 6. Firewall ────────────────────────────────────────────────────────
echo "[6/8] Configuring firewall..."
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp      # SSH
ufw allow 80/tcp      # HTTP (optional — if using port 80)
ufw allow 443/tcp     # HTTPS (optional — if using port 443)
ufw allow 9080/tcp    # TradeAI HTTP (default)
ufw allow 9443/tcp    # TradeAI HTTPS (default)
ufw --force enable
echo "  Firewall: SSH(22), HTTP(80/9080), HTTPS(443/9443) allowed"

# ── 7. Fail2Ban for SSH brute-force protection ─────────────────────────
echo "[7/8] Configuring fail2ban..."
systemctl enable fail2ban
systemctl start fail2ban

# ── 8. Create .env template ───────────────────────────────────────────
echo "[8/8] Creating environment file template..."
if [ ! -f "$APP_DIR/.env" ]; then
    cat > "$APP_DIR/.env" << 'ENVEOF'
# ═══════════════════════════════════════════════════════════════
# TradeAI Production Environment — EDIT ALL VALUES BELOW
# ═══════════════════════════════════════════════════════════════

# AngelOne SmartAPI
ANGELONE_API_KEY=
ANGELONE_CLIENT_ID=
ANGELONE_PASSWORD=
ANGELONE_MPIN=
ANGELONE_TOTP_SECRET=

# Database (internal Docker network — no change needed)
DATABASE_URL=postgresql+asyncpg://tradeai:CHANGE_THIS_DB_PASSWORD@postgres:5432/tradeai
REDIS_URL=redis://redis:6379/0

# Postgres credentials (must match DATABASE_URL above)
POSTGRES_DB=tradeai
POSTGRES_USER=tradeai
POSTGRES_PASSWORD=CHANGE_THIS_DB_PASSWORD

# OpenAI
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini

# Telegram Alerts (optional)
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# Email Alerts (optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD=
ALERT_EMAIL_TO=

# Ports (change if 80/443 are taken by other apps)
HTTP_PORT=9080
HTTPS_PORT=9443

# Trading Config
INITIAL_CAPITAL=100000
MAX_TRADES_PER_DAY=2
MAX_DAILY_LOSS_PCT=2.0
CONSECUTIVE_LOSS_LIMIT=3
NIFTY_LOT_SIZE=50

# System
LOG_LEVEL=INFO
PAPER_TRADING=true
ENVEOF
    chmod 600 "$APP_DIR/.env"
    chown "$APP_USER:$APP_USER" "$APP_DIR/.env"
    echo "  Created $APP_DIR/.env — EDIT IT BEFORE STARTING!"
else
    echo "  .env already exists — skipping"
fi

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Server setup complete!"
echo ""
echo "  Next steps:"
echo "    1. Edit /opt/tradeai/.env with your credentials"
echo "    2. Push code to GitHub (main branch)"
echo "    3. Add GitHub Secrets (see DEPLOYMENT.md)"
echo "    4. GitHub Actions will auto-deploy on push"
echo "══════════════════════════════════════════════════════════"
