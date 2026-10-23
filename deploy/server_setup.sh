#!/usr/bin/env bash
# Prepares a fresh Ubuntu/Debian server for CardioLens (D153). Idempotent: the deploy
# workflow runs it before every update, and it does nothing when Docker is already there.
set -euo pipefail
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sh
fi
if ! docker compose version >/dev/null 2>&1; then
  apt-get update && apt-get install -y docker-compose-plugin
fi
systemctl enable --now docker >/dev/null 2>&1 || true
# The firewall, when ufw is on: SSH, HTTP (certificate check) and HTTPS.
if command -v ufw >/dev/null 2>&1 && ufw status | grep -q "Status: active"; then
  ufw allow OpenSSH >/dev/null; ufw allow 80/tcp >/dev/null; ufw allow 443/tcp >/dev/null; ufw allow 443/udp >/dev/null
fi
