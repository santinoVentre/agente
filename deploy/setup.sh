#!/usr/bin/env bash
#
# setup.sh — First-time VPS setup for the Agent
# Run as: sudo bash deploy/setup.sh
#
set -euo pipefail

APP_DIR="/srv/agent/app"
DATA_DIRS=("/srv/agent/workspaces" "/srv/agent/logs" "/srv/agent/media")
AGENT_USER="agent"

echo "=== Agent Setup ==="

# 1. Ensure user exists
if ! id "$AGENT_USER" &>/dev/null; then
    echo "[+] Creating user $AGENT_USER"
    useradd -r -m -s /bin/bash "$AGENT_USER"
fi

# 2. Create directories
for d in "${DATA_DIRS[@]}"; do
    mkdir -p "$d"
    chown "$AGENT_USER:$AGENT_USER" "$d"
    echo "[+] Directory: $d"
done

# 3. System packages
echo "[+] Installing system packages"
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip ffmpeg git curl

# 4. Python venv
echo "[+] Setting up Python venv"
cd "$APP_DIR"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 5. Playwright browser
echo "[+] Installing Playwright Chromium"
playwright install chromium --with-deps

# 6. Docker containers
echo "[+] Starting Docker containers"
docker compose up -d

# 7. Permissions
chown -R "$AGENT_USER:$AGENT_USER" "$APP_DIR"

# 8. Systemd service
echo "[+] Installing systemd service"
cp deploy/agent.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable agent

echo ""
echo "=== Setup complete ==="
echo "Edit .env with your API keys, then run:"
echo "  sudo systemctl start agent"
echo "  journalctl -u agent -f"
