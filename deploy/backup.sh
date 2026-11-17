#!/usr/bin/env bash
# Daily backup of the site's data volume (D301): an encrypted archive on this server, 14 days kept.
# The archive holds what the volume holds (the quality log, saved analyses), so it is encrypted (AES-256, the key in
# /root/.cardiovision-backup.key, 0600, made on the first run) and readable by root only. It protects from a broken
# volume or a bad deploy; it does NOT protect from losing the server — an off-site copy is the owner's decision.
# Restore: openssl enc -d -aes-256-cbc -pbkdf2 -pass file:/root/.cardiovision-backup.key -in <file> | tar xz -C <dir>
set -euo pipefail
VOLUME=${BACKUP_VOLUME:-deploy_appdata}
OUT=${BACKUP_DIR:-/var/backups/cardiovision}
KEY=${BACKUP_KEY:-/root/.cardiovision-backup.key}
KEEP_DAYS=${BACKUP_KEEP_DAYS:-14}
umask 077
mkdir -p "$OUT"; chmod 700 "$OUT"
[ -s "$KEY" ] || { openssl rand -base64 48 > "$KEY"; chmod 600 "$KEY"; }
SRC=${BACKUP_SRC:-$(docker volume inspect -f '{{.Mountpoint}}' "$VOLUME")}
[ -d "$SRC" ] || { echo "backup: no data directory for volume $VOLUME" >&2; exit 1; }
name="$OUT/appdata-$(date -u +%Y%m%d-%H%M).tar.gz.enc"
tar -C "$SRC" -czf - . | openssl enc -aes-256-cbc -pbkdf2 -salt -pass "file:$KEY" -out "$name.tmp"
mv "$name.tmp" "$name"
find "$OUT" -name 'appdata-*.tar.gz.enc' -mtime "+$KEEP_DAYS" -delete
find "$OUT" -name '*.tmp' -mmin +120 -delete
echo "backup: $(basename "$name") $(du -h "$name" | cut -f1)"
