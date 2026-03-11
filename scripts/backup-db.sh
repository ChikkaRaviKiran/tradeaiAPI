#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# TradeAI — Database Backup Script
# Add to crontab: 0 16 * * * /opt/tradeai/scripts/backup-db.sh
# (Runs daily at 4 PM IST, after market close)
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

APP_DIR="/opt/tradeai"
BACKUP_DIR="$APP_DIR/backups"
RETENTION_DAYS=30
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$BACKUP_DIR"

echo "Backing up TradeAI database..."

docker exec tradeai-postgres pg_dump \
    -U "${POSTGRES_USER:-tradeai}" \
    -d "${POSTGRES_DB:-tradeai}" \
    --format=custom \
    --compress=9 \
    > "$BACKUP_DIR/tradeai_${TIMESTAMP}.dump"

echo "Backup saved: tradeai_${TIMESTAMP}.dump"

# Remove backups older than retention period
find "$BACKUP_DIR" -name "tradeai_*.dump" -mtime +$RETENTION_DAYS -delete
echo "Cleaned backups older than $RETENTION_DAYS days"
