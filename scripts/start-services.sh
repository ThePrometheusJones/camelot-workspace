#!/usr/bin/env bash
# Start Docker support services (ChromaDB + SearXNG) before the main app.
# Called by systemd ExecStartPre. Idempotent — safe to run if already up.

set -euo pipefail

COMPOSE_FILE="/home/ken/camelot-workspace/docker-compose.camelot.yml"

# ponytail: one function, two calls
wait_for() { for _ in $(seq 1 "$2"); do curl -sf "$1" >/dev/null 2>&1 && echo "[camelot] $3 ready" && return 0; sleep 1; done; }

echo "[camelot] Starting Docker support services..."
docker compose -f "$COMPOSE_FILE" up -d

wait_for "http://localhost:8100/api/v2/heartbeat" 30 "ChromaDB"
wait_for "http://localhost:8889/" 15 "SearXNG"

echo "[camelot] Support services up"
