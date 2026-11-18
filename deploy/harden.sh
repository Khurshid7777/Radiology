#!/usr/bin/env bash
# Server hardening (D310), idempotent; run by server_setup.sh on every deploy. Never fails the deploy: each part
# runs on its own and a failure is only logged to /var/log/cardiovision-harden.log.
#   fail2ban (sshd + recidive) · ufw (SSH port first, then 80/443) · unattended security upgrades (no auto reboot)
#   · journald size cap · 2 GB swap when there is none · SSH: no root login; password login off ONLY with
#   HARDEN_SSH_PASSWORD_OFF=yes (deploy/target.env) — the owner's consent, since a password user would be locked out.
set -u
APP_DIR=${1:-/opt/cardiolens}
LOG=/var/log/cardiovision-harden.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
export DEBIAN_FRONTEND=noninteractive
have() { command -v "$1" >/dev/null 2>&1; }
pkg() { dpkg -s "$1" >/dev/null 2>&1 || { apt-get update -q >/dev/null 2>&1; apt-get install -y -q "$1" >/dev/null 2>&1; }; }
SSH_PORT=$(sshd -T 2>/dev/null | awk '$1=="port"{print $2; exit}'); SSH_PORT=${SSH_PORT:-22}

part_fail2ban() {
  pkg fail2ban || return 1
  cat > /etc/fail2ban/jail.d/cardiovision.local <<JAIL
[DEFAULT]
bantime = 1h
findtime = 10m
maxretry = 5
bantime.increment = true
bantime.maxtime = 1w

[sshd]
enabled = true
port = $SSH_PORT
backend = systemd

[recidive]
enabled = true
bantime = 1w
findtime = 1d
maxretry = 5
JAIL
  systemctl enable --now fail2ban >/dev/null 2>&1 && systemctl reload-or-restart fail2ban
}

part_ufw() {
  pkg ufw || return 1
  ufw allow "$SSH_PORT"/tcp >/dev/null || return 1        # SSH first: enabling must never cut the session
  ufw allow 80/tcp >/dev/null; ufw allow 443/tcp >/dev/null; ufw allow 443/udp >/dev/null
  ufw default deny incoming >/dev/null; ufw default allow outgoing >/dev/null
  ufw status | grep -q "Status: active" || ufw --force enable >/dev/null
}

part_upgrades() {
  pkg unattended-upgrades || return 1
  printf 'APT::Periodic::Update-Package-Lists "1";\nAPT::Periodic::Unattended-Upgrade "1";\n' \
    > /etc/apt/apt.conf.d/20auto-upgrades
  printf 'Unattended-Upgrade::Automatic-Reboot "false";\n' > /etc/apt/apt.conf.d/52cardiovision-no-reboot
}

part_journal() {
  mkdir -p /etc/systemd/journald.conf.d
  printf '[Journal]\nSystemMaxUse=500M\n' > /etc/systemd/journald.conf.d/cardiovision.conf
  systemctl restart systemd-journald >/dev/null 2>&1
}

part_swap() {
  [ -n "$(swapon --show --noheadings 2>/dev/null)" ] && return 0
  [ -f /swapfile ] || { fallocate -l 2G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=2048; } >/dev/null 2>&1
  chmod 600 /swapfile && mkswap /swapfile >/dev/null && swapon /swapfile || return 1
  grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
}

part_ssh() {
  local f=/etc/ssh/sshd_config.d/60-cardiovision.conf want
  want="PermitRootLogin no"
  if grep -qs '^HARDEN_SSH_PASSWORD_OFF=yes' "$APP_DIR/deploy/target.env"; then
    want="$want
PasswordAuthentication no
KbdInteractiveAuthentication no"
  fi
  [ -f "$f" ] && [ "$(cat "$f")" = "$want" ] && return 0
  local old; old=$(cat "$f" 2>/dev/null || true)
  printf '%s\n' "$want" > "$f"
  if sshd -t 2>/dev/null; then
    systemctl reload ssh >/dev/null 2>&1 || systemctl reload sshd >/dev/null 2>&1
  else                                                      # a broken config must never reach a restart
    if [ -n "$old" ]; then printf '%s\n' "$old" > "$f"; else rm -f "$f"; fi
    return 1
  fi
}

for p in fail2ban ufw upgrades journal swap ssh; do
  if "part_$p"; then log "ok $p"; else log "FAILED $p"; fi
done
exit 0
