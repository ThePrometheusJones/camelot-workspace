#!/usr/bin/env bash
# wait-for-tailscale.sh — block until the Tailscale IPv4 address is actually
# bindable, not merely until tailscaled has started.
#
# Usage: wait-for-tailscale.sh [IP] [TIMEOUT_SECONDS]
# Called from systemd ExecStartPre. Exit 0 = IP is on an interface, safe to
# bind. Exit 1 = timed out; systemd will apply RestartSec and retry, but now
# the retry loop is in the *pre* step instead of uvicorn crash-looping.
#
# Why not `tailscale ip -4`? That reports the address from tailscaled's
# state before the kernel has it on tailscale0. uvicorn needs the kernel
# to have it. So we check the interface directly.

set -euo pipefail

WANT_IP="${1:-100.118.94.13}"
TIMEOUT="${2:-90}"
INTERVAL=2

elapsed=0
while (( elapsed < TIMEOUT )); do
    # Any interface is fine (tailscale0 normally, but don't assume the name)
    if ip -4 addr show 2>/dev/null | grep -qF "inet ${WANT_IP}/"; then
        echo "wait-for-tailscale: ${WANT_IP} is up after ${elapsed}s"
        exit 0
    fi
    sleep "${INTERVAL}"
    (( elapsed += INTERVAL ))
done

echo "wait-for-tailscale: timed out after ${TIMEOUT}s waiting for ${WANT_IP}" >&2
exit 1
