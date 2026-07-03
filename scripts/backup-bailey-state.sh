#!/usr/bin/env bash
# backup-bailey-state.sh — Guinevere's continuity, nightly, to the NAS.
#
# What it protects: app.db (26 tables of workspace state), vault_recall.db
# (Tier 2 long-term memory), the ChromaDB persistence dir (Tier 1 memory),
# and .env. These files collectively ARE her — everything else is
# reconstructable from git.
#
# SQLite is backed up with `sqlite3 .backup`, which takes a consistent
# snapshot of a live WAL database. Never raw-cp a live WAL db: you can
# capture the .db without its -wal and get a torn copy.
#
# Chroma: rsync of the persist dir on a live instance is *usually* fine
# (parquet/sqlite segments), but the belt-and-suspenders move is that Tier 1
# is rebuildable and Tier 2 (the irreplaceable vault knowledge) is sqlite
# and snapshotted properly. If you want Chroma bulletproof, add a
# `docker compose stop chromadb` / `start` bracket — costs ~5s of downtime
# at 03:00.
#
# Cron (as ken):
#   0 3 * * * /home/ken/camelot-workspace/scripts/backup-bailey-state.sh >> /home/ken/.local/state/bailey-backup.log 2>&1
#
# Restore test (do this once, not never):
#   sqlite3 restored.db "PRAGMA integrity_check;"
#
# ADJUST: paths below, NAS user, and dataset path on tank.

set -euo pipefail

# ------------------------- config -------------------------
REPO="/home/ken/camelot-workspace"
APP_DB="${REPO}/data/app.db"
VAULT_DB="/home/ken/.hermes/vault_recall.db"
CHROMA_DIR="${REPO}/data/chroma"             # host bind-mount of the volume
ENV_FILE="${REPO}/.env"

NAS="ken-optiplex-7050"                          # OptiPlex NAS over Tailscale
NAS_USER="ken"
NAS_PATH="/tank/backups/bailey"              # zfs dataset on the active pool

STAGE="/home/ken/.cache/bailey-backup"
KEEP_DAYS=14
STAMP="$(date +%Y%m%d-%H%M%S)"
# -----------------------------------------------------------

DEST="${STAGE}/${STAMP}"
mkdir -p "${DEST}"

echo "[$(date -Is)] backup start -> ${DEST}"

# 1. SQLite: consistent live snapshots
for db in "${APP_DB}" "${VAULT_DB}"; do
    if [[ -f "${db}" ]]; then
        name="$(basename "${db}")"
        sqlite3 "${db}" ".backup '${DEST}/${name}'"
        # cheap sanity check on the snapshot, not the live db
        if ! sqlite3 "${DEST}/${name}" "PRAGMA quick_check;" | grep -q '^ok$'; then
            echo "ERROR: quick_check failed on snapshot of ${name}" >&2
            exit 1
        fi
        echo "  snapshotted ${name} ($(du -h "${DEST}/${name}" | cut -f1))"
    else
        echo "  WARN: ${db} not found, skipping" >&2
    fi
done

# 2. Chroma persistence dir (see header note re: consistency)
if [[ -d "${CHROMA_DIR}" ]]; then
    tar -C "$(dirname "${CHROMA_DIR}")" -czf "${DEST}/chroma.tar.gz" \
        "$(basename "${CHROMA_DIR}")"
    echo "  archived chroma ($(du -h "${DEST}/chroma.tar.gz" | cut -f1))"
else
    echo "  WARN: ${CHROMA_DIR} not found, skipping" >&2
fi

# 3. .env (contains tokens — stays inside the tailnet, NAS perms apply)
[[ -f "${ENV_FILE}" ]] && install -m 600 "${ENV_FILE}" "${DEST}/env.snapshot"

# 4. Ship to NAS
rsync -a --mkpath "${DEST}/" "${NAS_USER}@${NAS}:${NAS_PATH}/${STAMP}/"
echo "  shipped to ${NAS}:${NAS_PATH}/${STAMP}"

# 5. Prune: local stage entirely, remote older than KEEP_DAYS
rm -rf "${DEST}"
ssh "${NAS_USER}@${NAS}" \
    "find '${NAS_PATH}' -maxdepth 1 -mindepth 1 -type d -mtime +${KEEP_DAYS} -exec rm -rf {} +"
echo "  pruned remote > ${KEEP_DAYS} days"

echo "[$(date -Is)] backup complete"

# Note: if /tank/backups is its own zfs dataset, zfs snapshots + the existing
# tank->coldtank replication give you versioning below this script for free.
