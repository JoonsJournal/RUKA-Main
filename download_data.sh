#!/bin/bash
# ------------------------------
# CONFIGURATION
# ------------------------------
PROJECT_ID="hwajz"                   # OSF project ID
REMOTE_DIR=""                        # empty -> download entire project
LOCAL_DIR="$HOME/RUKA/ruka_data"     # absolute local path
OSF_USERNAME="${OSF_USERNAME:-}"
OSF_TOKEN="${OSF_TOKEN:-}"
CONFIG_FILE="$HOME/.osfcli.config"
# ------------------------------

# Prompt for credentials if not set
if [ -z "$OSF_USERNAME" ]; then
    read -p "Enter your OSF username (email): " OSF_USERNAME
fi
if [ -z "$OSF_TOKEN" ]; then
    read -s -p "Enter your OSF token: " OSF_TOKEN
    echo ""
fi

# Check osfclient
if ! command -v osf &> /dev/null; then
    echo "❌ osfclient not installed. Run: pip install osfclient"
    exit 1
fi

mkdir -p "$LOCAL_DIR"

# Clone entire OSF project
osf --project "$PROJECT_ID" clone "$LOCAL_DIR"

echo "✅ Download complete. Files saved under: $LOCAL_DIR/osfstorage/"
echo "📁 Sub-folders: checkpoints / examples / motor_limits / models / data"