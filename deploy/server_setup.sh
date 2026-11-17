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
# D279: the daily read-only security check for the security agent (07:50 Tashkent = 02:50 UTC), and one run now.
APP_DIR_SETUP=$(cd "$(dirname "$0")/.." && pwd)
if [ -d /etc/cron.d ]; then   # a server without cron must not stop the deploy (set -e)
  printf 'SHELL=/bin/bash\n50 2 * * * root bash %s/deploy/security_check.sh %s\n' "$APP_DIR_SETUP" "$APP_DIR_SETUP" \
    > /etc/cron.d/cardiovision-security && chmod 644 /etc/cron.d/cardiovision-security || true
fi
# D301: a daily encrypted backup of the site's data volume (03:20 Tashkent = 22:20 UTC), kept 14 days on this server.
if [ -d /etc/cron.d ]; then
  printf 'SHELL=/bin/bash\n20 22 * * * root bash %s/deploy/backup.sh\n' "$APP_DIR_SETUP" \
    > /etc/cron.d/cardiovision-backup && chmod 644 /etc/cron.d/cardiovision-backup || true
fi
bash "$APP_DIR_SETUP/deploy/security_check.sh" "$APP_DIR_SETUP" || true
