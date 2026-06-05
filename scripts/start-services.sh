#!/usr/bin/env bash
# Start Docker support services (ChromaDB + SearXNG) before the main app.
# Called by systemd ExecStartPre. Idempotent — safe to run if already up.

set -euo pipefail

COMPOSE_FILE="/home/ken/camelot-workspace/docker-compose.camelot.yml"

echo "[camelot] Starting Docker support services..."
docker compose -f "$COMPOSE_FILE" up -d

# Wait for ChromaDB to be healthy (max 30s)
echo "[camelot] Waiting for ChromaDB..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8100/api/v2/heartbeat >/dev/null 2>&1; then
        echo "[camelot] ChromaDB ready"
        break
    fi
    sleep 1
done

# Wait for SearXNG (max 15s)
echo "[camelot] Waiting for SearXNG..."
for i in $(seq 1 15); do
    if curl -sf http://localhost:8889/ >/dev/null 2>&1; then
        echo "[camelot] SearXNG ready"
        break
    fi
    sleep 1
done

echo "[camelot] Support services up"
