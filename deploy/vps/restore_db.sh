#!/bin/bash
# Database Restore Script
# Restores PostgreSQL database from backup

set -e

# Configuration
CONTAINER_NAME="memopt-postgres"
DB_NAME="memopt"
DB_USER="memopt"

# Colors
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

# Check arguments
if [ -z "$1" ]; then
    echo -e "${RED}ERROR: No backup file specified${NC}"
    echo "Usage: $0 <backup_file.sql.gz>"
    echo ""
    echo "Available backups:"
    ls -lht /var/backups/memopt/ | head -10
    exit 1
fi

BACKUP_FILE="$1"

# Check if backup file exists
if [ ! -f "$BACKUP_FILE" ]; then
    echo -e "${RED}ERROR: Backup file not found: $BACKUP_FILE${NC}"
    exit 1
fi

# Confirm restore
echo -e "${YELLOW}⚠️  WARNING: This will OVERWRITE the current database!${NC}"
echo "Backup file: $BACKUP_FILE"
echo ""
read -p "Are you sure you want to restore? (type 'yes' to confirm): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Restore cancelled"
    exit 0
fi

echo ""
echo "Starting database restore..."

# Stop API to prevent connections
echo "Stopping API service..."
docker-compose stop api

# Drop and recreate database
echo "Dropping existing database..."
docker exec -t "$CONTAINER_NAME" psql -U "$DB_USER" -d postgres -c "DROP DATABASE IF EXISTS $DB_NAME;"
docker exec -t "$CONTAINER_NAME" psql -U "$DB_USER" -d postgres -c "CREATE DATABASE $DB_NAME;"

# Restore backup
echo "Restoring from backup..."
gunzip < "$BACKUP_FILE" | docker exec -i "$CONTAINER_NAME" psql -U "$DB_USER" -d "$DB_NAME"

# Restart API
echo "Restarting API service..."
docker-compose start api

# Wait for API to be healthy
echo "Waiting for API to start..."
sleep 5

for i in {1..30}; do
    if curl -f http://localhost/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ API is healthy${NC}"
        break
    fi
    echo "Waiting for health check... ($i/30)"
    sleep 2
done

echo ""
echo -e "${GREEN}=========================================="
echo "Restore complete!"
echo "==========================================${NC}"
echo ""
echo "Database restored from: $BACKUP_FILE"
echo "API is running: https://$(hostname)"
