#!/bin/bash
# Remote GPU Test Runner
# Deploys and runs brutal GPU tests on remote server

set -e

REMOTE_HOST="root@135.181.8.202"
REMOTE_DIR="/root/memopt-test"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
SSH_KEY="$HOME/.ssh/id_ed25519"

echo "🚀 Deploying Memopt to remote GPU server..."
echo "📡 Host: $REMOTE_HOST"
echo "📂 Remote dir: $REMOTE_DIR"
echo "🔑 SSH key: $SSH_KEY"
echo ""

# Test SSH connection
echo "🔍 Testing SSH connection..."
if ! ssh -i "$SSH_KEY" -o ConnectTimeout=10 -o BatchMode=yes "$REMOTE_HOST" "echo 'Connection successful'" 2>/dev/null; then
    echo "❌ SSH connection failed!"
    echo ""
    echo "Please ensure:"
    echo "  1. SSH key is added to the server: ssh-copy-id -i $SSH_KEY $REMOTE_HOST"
    echo "  2. Server is reachable: ping 135.181.8.202"
    echo "  3. SSH key has correct permissions: chmod 600 $SSH_KEY"
    echo ""
    echo "Or run manually:"
    echo "  ssh $REMOTE_HOST"
    exit 1
fi
echo "✅ SSH connection successful!"
echo ""

# Create remote directory
echo "📁 Creating remote directory..."
ssh -i "$SSH_KEY" "$REMOTE_HOST" "mkdir -p $REMOTE_DIR"

# Copy entire memopt directory
echo "📤 Uploading memopt codebase..."
rsync -avz --progress \
    -e "ssh -i $SSH_KEY" \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.env' \
    --exclude='venv' \
    --exclude='*.egg-info' \
    "$LOCAL_DIR/" "$REMOTE_HOST:$REMOTE_DIR/"

echo ""
echo "✅ Upload complete!"
echo ""

# Run tests on remote server
echo "🔥 Running brutal GPU tests on remote server..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

ssh -i "$SSH_KEY" "$REMOTE_HOST" << 'ENDSSH'
cd /root/memopt-test

echo "🔍 Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo "📦 Installing dependencies..."
pip install -q -r requirements.txt
pip install -q -e .
echo ""

echo "🧪 Running brutal GPU test suite..."
python3 brutal_gpu_test.py --all --output brutal_test_results.json

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Tests complete!"
echo ""
echo "📊 Results saved to:"
ls -lh brutal_test_results.*
ENDSSH

echo ""
echo "📥 Downloading results..."
scp -i "$SSH_KEY" "$REMOTE_HOST:$REMOTE_DIR/brutal_test_results.*" "$LOCAL_DIR/"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All done!"
echo ""
echo "📊 Results available locally:"
ls -lh "$LOCAL_DIR/brutal_test_results."*
echo ""
echo "📖 View results:"
echo "   cat brutal_test_results.json | jq ."
echo "   cat brutal_test_results.csv"

