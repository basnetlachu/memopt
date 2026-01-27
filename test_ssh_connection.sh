#!/bin/bash
# Test SSH connection to remote GPU server

REMOTE_HOST="root@135.181.8.202"
SSH_KEY="$HOME/.ssh/id_ed25519"

echo "🔍 Testing SSH connection to GPU server..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Host: $REMOTE_HOST"
echo "SSH Key: $SSH_KEY"
echo ""

# Check if SSH key exists
if [ ! -f "$SSH_KEY" ]; then
    echo "❌ SSH key not found: $SSH_KEY"
    echo ""
    echo "Please generate an SSH key first:"
    echo "  ssh-keygen -t ed25519 -f $SSH_KEY"
    exit 1
fi

# Check SSH key permissions
PERMS=$(stat -f "%OLp" "$SSH_KEY" 2>/dev/null || stat -c "%a" "$SSH_KEY" 2>/dev/null)
if [ "$PERMS" != "600" ]; then
    echo "⚠️  SSH key has incorrect permissions: $PERMS"
    echo "Fixing permissions..."
    chmod 600 "$SSH_KEY"
    echo "✅ Fixed: chmod 600 $SSH_KEY"
    echo ""
fi

# Test basic connectivity
echo "Testing network connectivity..."
if ! ping -c 1 -W 5 135.181.8.202 >/dev/null 2>&1; then
    echo "❌ Cannot reach server (ping failed)"
    echo ""
    echo "Please check:"
    echo "  1. Your internet connection"
    echo "  2. Server is online"
    echo "  3. Firewall settings"
    exit 1
fi
echo "✅ Server is reachable"
echo ""

# Test SSH connection
echo "Testing SSH connection..."
if ssh -i "$SSH_KEY" -o ConnectTimeout=10 -o BatchMode=yes "$REMOTE_HOST" "echo 'SSH connection successful'" 2>/dev/null; then
    echo "✅ SSH connection successful!"
    echo ""
    
    # Get GPU info
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🎮 GPU Information:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    ssh -i "$SSH_KEY" "$REMOTE_HOST" "nvidia-smi --query-gpu=name,memory.total,driver_version,cuda_version --format=csv"
    echo ""
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "💻 System Information:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    ssh -i "$SSH_KEY" "$REMOTE_HOST" "uname -a"
    ssh -i "$SSH_KEY" "$REMOTE_HOST" "python3 --version 2>&1"
    ssh -i "$SSH_KEY" "$REMOTE_HOST" "pip --version 2>&1"
    echo ""
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ All checks passed! Ready to run tests."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Run the brutal test suite with:"
    echo "  ./run_remote_test.sh"
    echo ""
    
else
    echo "❌ SSH connection failed!"
    echo ""
    echo "This likely means your SSH key is not authorized on the server."
    echo ""
    echo "To fix this, run:"
    echo "  ssh-copy-id -i $SSH_KEY $REMOTE_HOST"
    echo ""
    echo "Or manually add your public key to the server:"
    echo "  1. Copy your public key:"
    echo "     cat $SSH_KEY.pub"
    echo ""
    echo "  2. SSH to the server with password:"
    echo "     ssh $REMOTE_HOST"
    echo ""
    echo "  3. Add the key to authorized_keys:"
    echo "     echo 'YOUR_PUBLIC_KEY' >> ~/.ssh/authorized_keys"
    echo "     chmod 600 ~/.ssh/authorized_keys"
    echo ""
    exit 1
fi
