#!/usr/bin/env bash
#
# setup.sh — Complete VPS setup for the Agent (run on a fresh Ubuntu 22.04/24.04)
#
# Usage (from any user with sudo):
#   sudo bash setup.sh
#
# Or remotely after reinitialising:
#   curl -fsSL https://raw.githubusercontent.com/SantinoVentre/agente/main/deploy/setup.sh -o setup.sh
#   sudo bash setup.sh
#
set -euo pipefail

REPO_URL="https://github.com/SantinoVentre/agente.git"
APP_DIR="/srv/agent/app"
BASE_DIR="/srv/agent"
DATA_DIRS=("$BASE_DIR/workspaces" "$BASE_DIR/logs" "$BASE_DIR/media")
AGENT_USER="agent"

echo "============================================"
echo "  Agent VPS Setup — Full Installation"
echo "============================================"
echo ""

# ── 1. System packages ──────────────────────────────────────────
echo "[1/9] Installing system packages..."
apt-get update -qq
apt-get install -y -qq \
    python3 python3-venv python3-pip \
    ffmpeg git curl ca-certificates gnupg lsb-release \
    libnss3 libnspr4 libdbus-1-3 libatk1.0-0t64 libatk-bridge2.0-0t64 \
    libcups2t64 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 \
    libxfixes3 libxrandr2 libgbm1 libpango-1.0-0 libcairo2 \
    libasound2t64 libatspi2.0-0t64 2>/dev/null || \
apt-get install -y -qq \
    python3 python3-venv python3-pip \
    ffmpeg git curl ca-certificates gnupg lsb-release \
    libnss3 libnspr4 libdbus-1-3 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 \
    libxfixes3 libxrandr2 libgbm1 libpango-1.0-0 libcairo2 \
    libasound2 libatspi2.0-0

# ── 2. Docker ────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    echo "[2/9] Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    systemctl enable --now docker
else
    echo "[2/9] Docker already installed ✓"
fi

# ── 3. Create agent user ────────────────────────────────────────
if ! id "$AGENT_USER" &>/dev/null; then
    echo "[3/9] Creating user '$AGENT_USER'..."
    useradd -r -m -s /bin/bash "$AGENT_USER"
else
    echo "[3/9] User '$AGENT_USER' exists ✓"
fi
usermod -aG docker "$AGENT_USER"

# ── 4. Directories ──────────────────────────────────────────────
echo "[4/9] Creating directories..."
mkdir -p "$BASE_DIR"
for d in "${DATA_DIRS[@]}"; do
    mkdir -p "$d"
done

# ── 5. Clone repository ─────────────────────────────────────────
if [ -d "$APP_DIR/.git" ]; then
    echo "[5/9] Repository exists, pulling latest..."
    cd "$APP_DIR"
    git pull --ff-only || true
else
    echo "[5/9] Cloning repository..."
    rm -rf "$APP_DIR"
    git clone "$REPO_URL" "$APP_DIR"
fi

# ── 5b. Fix permissions BEFORE venv creation ─────────────────────
echo "[5b] Setting permissions on $BASE_DIR..."
chown -R "$AGENT_USER:$AGENT_USER" "$BASE_DIR"

# ── 6. Python venv + dependencies (as agent) ────────────────────
echo "[6/9] Setting up Python venv and dependencies..."
cd "$APP_DIR"
sudo -u "$AGENT_USER" python3 -m venv .venv
sudo -u "$AGENT_USER" bash -c "source $APP_DIR/.venv/bin/activate && pip install --upgrade pip setuptools wheel -q && pip install -r $APP_DIR/requirements.txt -q"

# ── 7. Playwright browser (as agent) ────────────────────────────
echo "[7/9] Installing Playwright Chromium..."
sudo -u "$AGENT_USER" "$APP_DIR/.venv/bin/playwright" install chromium

# ── 8. Fix all permissions (again, after venv creation) ──────────
echo "[8/9] Final permission fix..."
chown -R "$AGENT_USER:$AGENT_USER" "$BASE_DIR"

# ── 9. Docker containers + Systemd ──────────────────────────────
echo "[9/9] Starting Docker containers and systemd service..."
cd "$APP_DIR"
docker compose up -d

cp "$APP_DIR/deploy/agent.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable agent

echo ""
echo "============================================"
echo "  ✅ Setup complete!"
echo "============================================"
echo ""
echo "NEXT STEPS:"
echo "  1. Edit the .env file with your API keys:"
echo "     sudo nano $APP_DIR/.env"
echo ""
echo "  2. Start the agent:"
echo "     sudo systemctl start agent"
echo ""
echo "  3. Check logs:"
echo "     sudo journalctl -u agent -f"
echo ""
echo "Required keys in .env:"
echo "  - TELEGRAM_BOT_TOKEN  (from @BotFather)"
echo "  - OPENROUTER_API_KEY  (from openrouter.ai/keys)"
echo "  - GITHUB_TOKEN        (from github.com/settings/tokens)"
echo "============================================"
