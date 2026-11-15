#!/usr/bin/env bash
# Daily read-only security check of the host (D279), for the security agent. Changes nothing: it only collects
# facts into /var/lib/cardiovision-security/report.txt, which the agents container reads (mounted read-only).
# No secret values are written: for deploy/.env only its permissions and the NAMES of its variables.
# Installed as /etc/cron.d/cardiovision-security by server_setup.sh.
set -uo pipefail
APP_DIR=${1:-/opt/cardiolens}
OUT=/var/lib/cardiovision-security
mkdir -p "$OUT"; chmod 755 "$OUT"
{
  echo "# security check $(date -u +%Y-%m-%dT%H:%MZ)"
  echo "## os"; (. /etc/os-release; echo "$PRETTY_NAME"); uname -r; uptime
  echo "## upgradable packages"; apt list --upgradable 2>/dev/null | tail -n +2 | wc -l
  apt list --upgradable 2>/dev/null | grep -i -- "-security" | head -30
  echo "## reboot required"; [ -f /var/run/reboot-required ] && echo yes || echo no
  echo "## listening ports"; ss -tulpn 2>/dev/null | awk 'NR==1 || /LISTEN|UNCONN/' | head -40
  echo "## firewall"; (command -v ufw >/dev/null && ufw status verbose) || echo "ufw not installed"
  echo "## ssh config"; sshd -T 2>/dev/null | grep -Ei '^(permitrootlogin|passwordauthentication|pubkeyauthentication|port|maxauthtries) '
  echo "## failed ssh logins, last 24 h"
  journalctl -u ssh -u sshd --since "24 hours ago" 2>/dev/null | grep -Ec "Failed password|Invalid user" || true
  journalctl -u ssh -u sshd --since "24 hours ago" 2>/dev/null | grep -Eo "from [0-9a-f.:]+" | sort | uniq -c | sort -rn | head -10
  echo "## successful ssh logins, last 24 h"
  journalctl -u ssh -u sshd --since "24 hours ago" 2>/dev/null | grep -E "Accepted (publickey|password)" | sed -E 's/.*Accepted ([a-z]+) for ([^ ]+) from ([^ ]+).*/\1 \2 \3/' | sort | uniq -c
  echo "## fail2ban"; (command -v fail2ban-client >/dev/null && fail2ban-client status 2>/dev/null) || echo "not installed"
  echo "## unattended-upgrades"; dpkg -l unattended-upgrades 2>/dev/null | grep -q '^ii' && echo installed || echo "not installed"
  echo "## deploy/.env"; stat -c '%a %U:%G' "$APP_DIR/deploy/.env" 2>/dev/null; sed -n 's/=.*//p' "$APP_DIR/deploy/.env" 2>/dev/null | paste -sd' '
  echo "## containers"; docker ps --format '{{.Names}} {{.Status}} {{.Image}}' 2>/dev/null
  echo "## disk"; df -h / | tail -1
  echo "## certificate"
  dom=$(sed -n 's/^DOMAIN=//p' "$APP_DIR/deploy/.env" 2>/dev/null)
  [ -n "$dom" ] && echo | timeout 15 openssl s_client -servername "$dom" -connect "$dom:443" 2>/dev/null | openssl x509 -noout -enddate 2>/dev/null
} > "$OUT/report.tmp" 2>&1
chmod 644 "$OUT/report.tmp"; mv "$OUT/report.tmp" "$OUT/report.txt"
