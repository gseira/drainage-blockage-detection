#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# start.sh  —  Drainage Blockage Detection server launcher
#
# Finds the right Python, activates the environment, and starts uvicorn.
# Called by the launchd agent (auto-start on login) or run manually.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "$(date '+%Y-%m-%d %H:%M:%S')  Starting Drainage Monitor…" >> "$LOG_DIR/server.log"
echo "  Project: $PROJECT_DIR" >> "$LOG_DIR/server.log"

cd "$PROJECT_DIR"

# ── Find Python ───────────────────────────────────────────────────────────────
# Priority: project venv → Anaconda base → other conda envs → system python3

PYTHON=""

# 1. Project-local venv
if [ -f "$PROJECT_DIR/venv/bin/python" ]; then
    source "$PROJECT_DIR/venv/bin/activate"
    PYTHON="$PROJECT_DIR/venv/bin/python"
    echo "  Env: project venv" >> "$LOG_DIR/server.log"

# 2. Anaconda base (detected on this machine at /opt/anaconda3)
elif [ -f "/opt/anaconda3/bin/python" ]; then
    source "/opt/anaconda3/etc/profile.d/conda.sh"
    conda activate base
    PYTHON="/opt/anaconda3/bin/python"
    echo "  Env: Anaconda base" >> "$LOG_DIR/server.log"

# 3. Other conda installations
elif command -v conda &>/dev/null; then
    for ENV_NAME in dissertation drainage cv ml base; do
        CONDA_PYTHON="$(conda run -n "$ENV_NAME" which python 2>/dev/null || true)"
        if [ -n "$CONDA_PYTHON" ]; then
            eval "$(conda shell.bash hook 2>/dev/null)"
            conda activate "$ENV_NAME"
            PYTHON="$CONDA_PYTHON"
            echo "  Env: conda/$ENV_NAME" >> "$LOG_DIR/server.log"
            break
        fi
    done
fi

# 4. System python3 fallback
if [ -z "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
    echo "  Env: system python3 ($PYTHON)" >> "$LOG_DIR/server.log"
fi

echo "  Python: $PYTHON" >> "$LOG_DIR/server.log"

# ── Start server ──────────────────────────────────────────────────────────────
exec "$PYTHON" -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info \
    2>&1 | tee -a "$LOG_DIR/server.log"
