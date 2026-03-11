# TradeAI — Complete AWS Deployment Guide

## What You Need

| Item | Details |
|------|---------|
| **AWS Account** | Free tier works, but t3.small recommended |
| **EC2 Instance** | Ubuntu 22.04 or 24.04 LTS |
| **Instance Type** | `t3.small` (2 vCPU, 2 GB RAM) — minimum. `t3.medium` recommended. |
| **Storage** | 20 GB gp3 (default 8 GB is not enough for Docker images) |
| **GitHub Account** | To host your two repos (private recommended) |
| **SSH Key Pair** | Created during EC2 launch |

---

## Step-by-Step Deployment

### STEP 1 — Push Code to GitHub

On your **local Windows machine**, create two private repos on GitHub, then push:

```bash
# Backend (tradeai-engine)
cd C:\TradeAI\backend
git remote add origin https://github.com/<your-username>/tradeai-engine.git
git branch -M main
git push -u origin main

# Frontend (tradeai-dashboard)
cd C:\TradeAI\frontend
git remote add origin https://github.com/<your-username>/tradeai-dashboard.git
git branch -M main
git push -u origin main
```

---

### STEP 2 — Launch EC2 Instance

1. Go to **AWS Console → EC2 → Launch Instance**
2. Configure:
   - **Name:** `TradeAI`
   - **AMI:** Ubuntu Server 24.04 LTS (or 22.04)
   - **Instance type:** `t3.small` (minimum) or `t3.medium` (recommended)
   - **Key pair:** Create new → download `.pem` file → keep it safe
   - **Storage:** Change from 8 GB to **20 GB gp3**
   - **Security Group:** Create new with these rules:

| Type | Port | Source | Purpose |
|------|------|--------|---------|
| SSH | 22 | Your IP | SSH access |
| Custom TCP | 3000 | 0.0.0.0/0 | Dashboard |
| Custom TCP | 8000 | 0.0.0.0/0 | Backend API |

3. Click **Launch Instance**
4. Note the **Public IPv4 address** from the instance details

---

### STEP 3 — Connect to Your Server

```bash
# From your local terminal (PowerShell / Git Bash)
ssh -i "your-key.pem" ubuntu@<your-ec2-public-ip>
```

> On Windows, if using PowerShell:
> ```powershell
> ssh -i "C:\Users\YourName\Downloads\your-key.pem" ubuntu@<your-ec2-public-ip>
> ```

---

### STEP 4 — Install Docker & Docker Compose

Run these commands on the server:

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker
sudo apt-get install -y docker.io docker-compose curl git

# Enable Docker to start on boot
sudo systemctl enable docker
sudo systemctl start docker

# Add ubuntu user to docker group (avoids needing sudo for docker)
sudo usermod -aG docker ubuntu

# Apply group change (or logout and login again)
newgrp docker

# Verify
docker --version
docker-compose --version
```

---

### STEP 5 — Set Timezone to IST

```bash
sudo timedatectl set-timezone Asia/Kolkata
timedatectl   # verify: should show Asia/Kolkata
```

---

### STEP 6 — Clone Your Repos

```bash
# Create project directory
sudo mkdir -p /opt/tradeai
sudo chown ubuntu:ubuntu /opt/tradeai

# Clone both repos
cd /opt/tradeai
git clone https://github.com/<your-username>/tradeai-engine.git backend
git clone https://github.com/<your-username>/tradeai-dashboard.git frontend
```

Your directory structure should be:
```
/opt/tradeai/
├── backend/        ← tradeai-engine (FastAPI + Trading Engine)
│   ├── docker-compose.yml
│   ├── Dockerfile
│   ├── .env
│   ├── main.py
│   └── app/
└── frontend/       ← tradeai-dashboard (React)
    ├── Dockerfile
    ├── nginx.conf
    └── src/
```

---

### STEP 7 — Verify `.env` Configuration

```bash
cat /opt/tradeai/backend/.env
```

Make sure all these are set:
- `ANGELONE_API_KEY` — your AngelOne API key
- `ANGELONE_CLIENT_ID` — your client ID
- `ANGELONE_PASSWORD` — your password
- `ANGELONE_MPIN` — your MPIN
- `ANGELONE_TOTP_SECRET` — your TOTP secret
- `OPENAI_API_KEY` — your OpenAI API key

> **Note:** DATABASE_URL and REDIS_URL in .env use `localhost` for local development.
> The `docker-compose.yml` automatically overrides these with container hostnames (`postgres`, `redis`) — no manual change needed.

If you need to edit:
```bash
nano /opt/tradeai/backend/.env
```

---

### STEP 8 — Launch with Docker Compose

```bash
cd /opt/tradeai/backend
docker-compose up --build -d
```

This will:
- Build the backend Python image
- Build the frontend React/Nginx image
- Start PostgreSQL 15
- Start Redis 7
- Start the backend (port 8000)
- Start the frontend (port 3000)

First build takes 3-5 minutes. Check progress:
```bash
docker-compose logs -f          # Live logs (Ctrl+C to exit)
docker-compose ps               # Check all containers are running
```

---

### STEP 9 — Verify Everything is Running

```bash
# Health check
curl http://localhost:8000/api/health

# Check all 4 containers are "Up"
docker-compose ps
```

Then open in your browser:
- **Dashboard:** `http://<your-ec2-public-ip>:3000`
- **Backend API:** `http://<your-ec2-public-ip>:8000/api/health`

---

### STEP 10 — Set Up Auto-Start on Boot (systemd)

```bash
# Create systemd service
sudo tee /etc/systemd/system/tradeai.service > /dev/null << 'EOF'
[Unit]
Description=TradeAI NIFTY Options Trading System
After=docker.service network-online.target
Requires=docker.service
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
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

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable tradeai.service

# Stop docker-compose (if running from step 8) and let systemd manage it
cd /opt/tradeai/backend
docker-compose down
sudo systemctl start tradeai.service
```

---

### STEP 11 — Create Update Script

```bash
cat > /opt/tradeai/update.sh << 'EOF'
#!/bin/bash
cd /opt/tradeai/backend && git pull
cd /opt/tradeai/frontend && git pull
sudo systemctl restart tradeai
echo "✓ Updated and restarted"
EOF
chmod +x /opt/tradeai/update.sh
```

---

## Daily Commands (Cheat Sheet)

| Action | Command |
|--------|---------|
| **Check status** | `sudo systemctl status tradeai` |
| **View live logs** | `sudo journalctl -u tradeai -f` |
| **View last 100 lines** | `sudo journalctl -u tradeai -n 100` |
| **Restart** | `sudo systemctl restart tradeai` |
| **Stop** | `sudo systemctl stop tradeai` |
| **Start** | `sudo systemctl start tradeai` |
| **Update code & restart** | `/opt/tradeai/update.sh` |
| **Check containers** | `cd /opt/tradeai/backend && docker-compose ps` |
| **Backend logs only** | `cd /opt/tradeai/backend && docker-compose logs -f backend` |
| **DB access** | `docker exec -it backend-postgres-1 psql -U postgres -d tradeai` |

---

## How the System Runs Automatically

Once deployed, **you don't need to do anything daily**:

1. **Server boot** → systemd starts `tradeai.service` → Docker Compose starts all 4 containers
2. **08:45 IST** → Orchestrator wakes up, fetches global market data
3. **09:15 IST** → Market opens, trading cycle begins (every 3 minutes)
4. **15:30 IST** → Market closes, generates daily report
5. **After report** → Orchestrator sleeps until next trading day 08:45
6. **Weekends** → Skipped automatically (sleeps through Saturday/Sunday)
7. **NSE Holidays** → Skipped automatically (dynamic holiday calendar from NSE)
8. **If crash** → systemd auto-restarts within 30 seconds

---

## Optional: HTTPS with Custom Domain

If you want `https://trade.yourdomain.com` instead of `http://<ip>:3000`:

```bash
# Install Nginx as reverse proxy
sudo apt-get install -y nginx certbot python3-certbot-nginx

# Create Nginx config
sudo tee /etc/nginx/sites-available/tradeai << 'EOF'
server {
    listen 80;
    server_name trade.yourdomain.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/tradeai /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Get free SSL certificate
sudo certbot --nginx -d trade.yourdomain.com
```

Then update Security Group: add port 80 and 443, and you can remove 3000/8000.

---

## Optional: Setup Swap (for t3.small with 2 GB RAM)

```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

---

## Cost Estimate

| Resource | Monthly Cost |
|----------|-------------|
| t3.small (2 vCPU, 2 GB) | ~$15/month |
| t3.medium (2 vCPU, 4 GB) | ~$30/month |
| 20 GB gp3 storage | ~$1.60/month |
| Data transfer | ~$1-5/month |
| **Total (t3.small)** | **~$18/month** |
| **Total (t3.medium)** | **~$33/month** |

> Tip: Use **Reserved Instances** or **Savings Plans** to save 30-40%.
> Tip: Use **Spot Instances** if you're okay with occasional interruptions (~70% cheaper).

---

## Troubleshooting

### Containers not starting
```bash
cd /opt/tradeai/backend
docker-compose logs               # Check error messages
docker-compose down
docker-compose up --build          # Rebuild and see live output
```

### Backend can't connect to AngelOne
- Check `.env` credentials are correct
- AngelOne API might be down — check logs for specific error

### Dashboard shows "Cannot connect to server"
- Verify backend is running: `curl http://localhost:8000/api/health`
- Check Security Group allows port 3000

### Out of memory (t3.small)
- Add swap (see above)
- Or upgrade to t3.medium

### Disk full
```bash
# Clean old Docker images
docker system prune -a
```
