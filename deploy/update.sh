#!/usr/bin/env bash
#
# update.sh — Pull latest code and restart the agent
# Run as: sudo bash deploy/update.sh
#
set -euo pipefail

APP_DIR="/srv/agent/app"
AGENT_USER="agent"

echo "=== Agent Update ==="

cd "$APP_DIR"

# 1. Pull latest
echo "[+] Pulling latest code"
sudo -u "$AGENT_USER" git pull --ff-only

# 2. Update deps
echo "[+] Updating dependencies"
source .venv/bin/activate
pip install -r requirements.txt --quiet

# 3. Restart
echo "[+] Restarting agent"
systemctl restart agent

echo "[+] Done — checking status…"
sleep 2
systemctl status agent --no-pager -l
