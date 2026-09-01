#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────
# DrainWatch — start backend + frontend in one terminal
# Usage:  chmod +x start.sh && ./start.sh
# ─────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "▶  Starting FastAPI backend on http://127.0.0.1:8000 …"
uvicorn app.main:app --reload --port 8000 &
BACKEND_PID=$!

echo "▶  Waiting for backend to be ready …"
for i in $(seq 1 20); do
  if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "✔  Backend ready."
    break
  fi
  sleep 1
done

echo "▶  Starting Vite dev server on http://localhost:5173 …"
cd "Flood Watcher"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "  Backend  → http://127.0.0.1:8000"
echo "  Frontend → http://localhost:5173"
echo ""
echo "  Press Ctrl+C to stop both."

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" SIGINT SIGTERM
wait
