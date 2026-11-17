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
command -v flock >/dev/null || { echo "backup: flock is missing" >&2; exit 1; }
exec 9> "$OUT/.lock"; flock -n 9 || { echo "backup: another run is in progress" >&2; exit 0; }
# old archives go, but the newest one always stays (a row of failed nights must not leave nothing)
ls -t "$OUT"/appdata-*.tar.gz.enc 2>/dev/null | tail -n +2 | while read -r f; do
  [ -n "$(find "$f" -mtime "+$KEEP_DAYS")" ] && rm -f "$f"
done || true
find "$OUT" -name '*.tmp' -delete
need=$( (du -sk "$SRC" 2>/dev/null || true) | cut -f1); need=${need:-0}; free=$(df -Pk "$OUT" | awk 'NR==2 {print $4}')
[ "$free" -gt $((need + 1048576)) ] || { echo "backup: skipped, free ${free}K < data ${need}K + 1G" >&2; exit 1; }
name="$OUT/appdata-$(date -u +%Y%m%d-%H%M).tar.gz.enc"
trap 'rm -f "$name.tmp"' EXIT
set +e
tar -C "$SRC" -czf - . | openssl enc -aes-256-cbc -pbkdf2 -salt -pass "file:$KEY" -out "$name.tmp"
rc=("${PIPESTATUS[@]}")
set -e
# tar's 1 is "a file changed while it was read" (a live volume): the archive is still complete
{ [ "${rc[0]}" -le 1 ] && [ "${rc[1]}" -eq 0 ]; } || { echo "backup: failed (tar ${rc[0]}, openssl ${rc[1]})" >&2; exit 1; }
mv "$name.tmp" "$name"
echo "backup: $(basename "$name") $(du -h "$name" | cut -f1)"
