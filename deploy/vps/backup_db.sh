#!/bin/bash
# Database Backup Script
# Creates timestamped PostgreSQL dumps

set -e

# Configuration
BACKUP_DIR="/var/backups/memopt"
CONTAINER_NAME="memopt-postgres"
DB_NAME="memopt"
DB_USER="memopt"
RETENTION_DAYS=30

# Colors
GREEN='\033[0;32m'
NC='\033[0m'

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Generate timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/memopt_backup_$TIMESTAMP.sql.gz"

echo "Starting database backup..."
echo "Backup file: $BACKUP_FILE"

# Create backup
docker exec -t "$CONTAINER_NAME" pg_dump -U "$DB_USER" "$DB_NAME" | gzip > "$BACKUP_FILE"

# Check if backup was successful
if [ -f "$BACKUP_FILE" ]; then
    SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
    echo -e "${GREEN}✓ Backup successful${NC}"
    echo "Size: $SIZE"
    echo "Location: $BACKUP_FILE"
else
    echo "ERROR: Backup failed"
    exit 1
fi

# Clean up old backups (older than RETENTION_DAYS)
echo ""
echo "Cleaning up old backups (older than $RETENTION_DAYS days)..."
find "$BACKUP_DIR" -name "memopt_backup_*.sql.gz" -type f -mtime +$RETENTION_DAYS -delete
echo "Old backups removed"

# List recent backups
echo ""
echo "Recent backups:"
ls -lht "$BACKUP_DIR" | head -10

echo ""
echo -e "${GREEN}Backup complete!${NC}"
