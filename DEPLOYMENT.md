# TradeAI — AWS Ubuntu Deployment Guide (Separate Repos)

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     AWS EC2 (Ubuntu)                      │
│                                                          │
│  ┌─────────┐    ┌───────────┐    ┌────────────────────┐  │
│  │  Nginx   │───▶│  Frontend  │    │      Backend       │  │
│  │:9080/9443│──▶│  (React)   │    │  (FastAPI+Engine)  │  │
│  └─────────┘    │  :3000     │    │  :8000             │  │
│       │         └───────────┘    └────────┬───────────┘  │
│       │                                    │              │
│       └──── /api/ ────────────────────────┘              │
│                                                          │
│  ┌──────────────┐    ┌───────────┐                       │
│  │  PostgreSQL   │    │   Redis    │                       │
│  │  :5432        │    │   :6379    │                       │
│  └──────────────┘    └───────────┘                       │
│                                                          │
│  ┌──────────────┐                                        │
│  │   Certbot     │  (SSL auto-renewal)                    │
│  └──────────────┘                                        │
└──────────────────────────────────────────────────────────┘
```

## Repository Structure

| Repo | Contents | CI/CD |
|------|----------|-------|
| **tradeai-engine** (backend) | FastAPI app, Docker Compose, nginx config, setup scripts | Builds backend image → deploys full stack |
| **tradeai-dashboard** (frontend) | React app | Builds frontend image → restarts frontend container |

```
GitHub Push (backend)  → Build backend image → Push to GHCR → SCP infra files → SSH deploy all
GitHub Push (frontend) → Build frontend image → Push to GHCR → SSH restart frontend
```

---

## Prerequisites

- **AWS EC2**: Ubuntu 22.04/24.04 LTS, `t3.small` or larger (2 vCPU, 2 GB RAM minimum)
- **Security Group**: Ports 22 (SSH), 9080 (HTTP), 9443 (HTTPS) open
- **Two GitHub Repos**: `tradeai-engine` (backend) and `tradeai-dashboard` (frontend)
- **SSH Key Pair**: For EC2 access and GitHub Actions deployment

---

## Step 1: Push Code to GitHub

```bash
# Backend (tradeai-engine)
cd C:\TradeAI\backend
git init
git add -A
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/tradeai-engine.git
git push -u origin main

# Frontend (tradeai-dashboard)
cd C:\TradeAI\frontend
git init
git add -A
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/tradeai-dashboard.git
git push -u origin main
```

---

## Step 2: Launch EC2 Instance

1. Go to **AWS Console → EC2 → Launch Instance**
2. Settings:
   - **AMI**: Ubuntu Server 22.04 LTS (64-bit x86)
   - **Instance Type**: `t3.small` (or `t3.medium` for better performance)
   - **Key Pair**: Create or select an existing SSH key pair
   - **Storage**: 20 GB gp3 (minimum)
   - **Security Group**: Allow inbound:
     - SSH (22) from your IP
     - Custom TCP (9080) from anywhere — TradeAI HTTP
     - Custom TCP (9443) from anywhere — TradeAI HTTPS
3. Launch and note the **Public IP**

---

## Step 3: Initial Server Setup

SSH into your EC2 and run the setup script:

```bash
ssh -i your-key.pem ubuntu@YOUR_EC2_IP

# Clone backend repo temporarily for setup script
git clone https://github.com/<your-username>/tradeai-engine.git /tmp/tradeai-setup
sudo bash /tmp/tradeai-setup/scripts/server-setup.sh
rm -rf /tmp/tradeai-setup
```

This installs Docker, creates the `tradeai` user, configures firewall, and creates the `.env` template at `/opt/tradeai/.env`.

---

## Step 4: Configure Environment

```bash
sudo nano /opt/tradeai/.env
```

**Required values to set:**
- `ANGELONE_API_KEY`, `ANGELONE_CLIENT_ID`, `ANGELONE_PASSWORD`, `ANGELONE_MPIN`, `ANGELONE_TOTP_SECRET`
- `POSTGRES_PASSWORD` (change from default — must match `DATABASE_URL`)
- `OPENAI_API_KEY`

**Also set your GHCR image names:**
```
BACKEND_IMAGE=ghcr.io/<your-username>/tradeai-backend
FRONTEND_IMAGE=ghcr.io/<your-username>/tradeai-frontend
```

**Optional but recommended:**
- `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` (for trade alerts)

---

## Step 5: Configure GitHub Secrets (BOTH Repos)

Go to **each repo** → **Settings → Secrets and variables → Actions** and add these secrets to **both** repos:

| Secret Name     | Value                                          |
|-----------------|------------------------------------------------|
| `AWS_HOST`      | Your EC2 public IP or domain                   |
| `AWS_USER`      | `ubuntu` (default EC2 user)                    |
| `AWS_SSH_KEY`   | Contents of your `.pem` SSH private key         |
| `GHCR_TOKEN`    | GitHub PAT with `read:packages` scope           |

### Creating the GHCR Token

1. Go to **GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)**
2. Generate new token with scopes: `read:packages`, `write:packages`
3. Copy the token and add it as `GHCR_TOKEN` secret in **both** repos

---

## Step 6: Deploy

Push to `main` branch in either repo — GitHub Actions will automatically build and deploy.

**First deployment** — push backend first (it deploys the full stack):
```bash
cd C:\TradeAI\backend
git push origin main
```

Then push frontend:
```bash
cd C:\TradeAI\frontend
git push origin main
```

Or trigger manually: **GitHub → Actions → Run workflow**

---

## Step 7: SSL Setup (Optional)

After first successful deployment:

```bash
ssh -i your-key.pem ubuntu@YOUR_EC2_IP
cd /opt/tradeai
sudo docker compose -f docker-compose.prod.yml stop nginx
sudo bash scripts/ssl-setup.sh your-domain.com your@email.com
sudo nano /opt/tradeai/nginx/nginx.conf   # uncomment HTTPS block
sudo docker compose -f docker-compose.prod.yml up -d
```

---

## Operations

### View Logs
```bash
cd /opt/tradeai
docker compose -f docker-compose.prod.yml logs -f           # All
docker compose -f docker-compose.prod.yml logs -f backend    # Backend only
tail -f /opt/tradeai/logs/tradeai.log                        # App log
```

### Restart Services
```bash
cd /opt/tradeai
docker compose -f docker-compose.prod.yml restart            # All
docker compose -f docker-compose.prod.yml restart backend    # Backend only
```

### Database Backup
```bash
sudo bash /opt/tradeai/scripts/backup-db.sh

# Auto-backup (add to crontab):
sudo crontab -e
# Add: 0 16 * * * /opt/tradeai/scripts/backup-db.sh
```

### Database Restore
```bash
cat /opt/tradeai/backups/tradeai_TIMESTAMP.dump | \
  docker exec -i tradeai-postgres pg_restore \
    -U tradeai -d tradeai --clean --if-exists
```

### Check Status
```bash
docker compose -f docker-compose.prod.yml ps
curl http://localhost:9080/api/health
```

### Manual Deployment (without CI/CD)
```bash
cd /opt/tradeai
docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml up -d --remove-orphans
docker image prune -f
```

---

## Troubleshooting

### Backend won't start
```bash
docker compose -f docker-compose.prod.yml logs backend
cat /opt/tradeai/.env
docker exec tradeai-postgres pg_isready -U tradeai
```

### Can't reach the site
```bash
sudo ufw status
docker compose -f docker-compose.prod.yml logs nginx
docker ps
```

### Disk space
```bash
df -h
docker system prune -a --volumes  # WARNING: removes all unused data
```

---

## Cost Estimate (AWS)

| Resource       | Type        | Monthly Cost |
|----------------|-------------|--------------|
| EC2            | t3.small    | ~$15         |
| EBS Storage    | 20 GB gp3   | ~$2          |
| Data Transfer  | <1 GB/month | ~$0          |
| **Total**      |             | **~$17/mo**  |
