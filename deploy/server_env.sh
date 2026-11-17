#!/usr/bin/env bash
# Keeps deploy/.env on the server (D153). The first run makes the clinic API key and the
# storage key there, so they never pass through GitHub; the share-link secret (D197) is made
# the same way on the first run that lacks it. Later runs keep them and refresh
# DOMAIN and whatever NAME=VALUE lines arrive on stdin (empty values are ignored):
#   ANTHROPIC_API_KEY     -> ANTHROPIC_API_KEY
#   TELEGRAM_BOT_TOKEN    -> ECGSCAN_BOT_TOKEN      (D161; with ECGSCAN_BOT_API_KEY = the clinic key)
#   TELEGRAM_BOT_ALLOWED  -> ECGSCAN_BOT_ALLOWED
#   AGENTS_TG_TOKEN, AGENTS_OWNER_ID, AGENTS_GH_TOKEN -> the same names (D279: the agents bot)
#   AGENTS_TG_COORD ... AGENTS_TG_NEWS                -> the same names (D298: a bot per agent)
#   AGENTS_API_KEY        -> the "agents:<key>" entry of ECGSCAN_API_KEYS (D274: a read-only key for the triage
#                            agent — the server lets it read GET /api/quality/summary only, without comments)
# Usage: printf 'NAME=VALUE\n...' | server_env.sh DOMAIN
set -euo pipefail
cd "$(dirname "$0")"
domain=$1
umask 077
if [ ! -f .env ]; then
  {
    echo "DOMAIN=$domain"
    echo "ECGSCAN_API_KEYS=clinic1:$(openssl rand -hex 24)"
    echo "ECGSCAN_STORAGE_KEY=$(openssl rand -base64 32 | tr '+/' '-_')"
    echo "ANTHROPIC_API_KEY="
    echo "CARDIOLENS_LLM_PROVIDER=auto"
  } > .env
  echo "deploy/.env created (keys: cat $(pwd)/.env)"
fi
set_var() {  # set_var NAME VALUE: replace the line or append it
  if grep -q "^$1=" .env; then sed -i "s|^$1=.*|$1=$2|" .env; else echo "$1=$2" >> .env; fi
}
set_var DOMAIN "$domain"
# D197: the share-link signing secret, made here once and never printed; kept across deploys
# so links outlive a restart.
if ! grep -q "^CARDIOLENS_SHARE_SECRET=..*" .env; then
  set_var CARDIOLENS_SHARE_SECRET "$(openssl rand -hex 32)"
  echo "CARDIOLENS_SHARE_SECRET made in deploy/.env"
fi
if [ ! -t 0 ]; then
  while IFS='=' read -r name value; do
    [ -n "${value:-}" ] || continue
    case "$name" in
      ANTHROPIC_API_KEY)    set_var ANTHROPIC_API_KEY "$value" ;;
      TELEGRAM_BOT_TOKEN)   set_var ECGSCAN_BOT_TOKEN "$value"
                            set_var ECGSCAN_BOT_API_KEY "$(sed -n 's/^ECGSCAN_API_KEYS=[^:]*:\([^,]*\).*/\1/p' .env)" ;;
      TELEGRAM_BOT_ALLOWED) set_var ECGSCAN_BOT_ALLOWED "$value" ;;
      AGENTS_API_KEY)       # letters, digits, - and _ only: the key goes through sed and the "name:key,name:key" list
                            if printf '%s' "$value" | grep -Eq '^[A-Za-z0-9_-]{24,}$'; then
                              rest=$(sed -n 's/^ECGSCAN_API_KEYS=//p' .env | tr ',' '\n' | grep -v '^agents:' | paste -sd, -)
                              set_var ECGSCAN_API_KEYS "$rest,agents:$value"
                            else echo "AGENTS_API_KEY ignored: 24+ characters of A-Z a-z 0-9 - _ expected" >&2; fi ;;
      # D279: the agents bot. Each value is checked for its shape, so nothing odd reaches sed or compose.
      AGENTS_TG_TOKEN)      if printf '%s' "$value" | grep -Eq '^[0-9]{6,12}:[A-Za-z0-9_-]{30,}$'; then
                              set_var AGENTS_TG_TOKEN "$value"; else echo "AGENTS_TG_TOKEN ignored: not a bot token" >&2; fi ;;
      AGENTS_TG_COORD|AGENTS_TG_DEV|AGENTS_TG_SCAN|AGENTS_TG_REVIEW|AGENTS_TG_SEC|AGENTS_TG_TRIAGE|AGENTS_TG_NEWS)
                            # D298: a bot per agent; only the bare token is accepted
                            if printf '%s' "$value" | grep -Eq '^[0-9]{6,12}:[A-Za-z0-9_-]{30,}$'; then
                              set_var "$name" "$value"; else echo "$name ignored: not a bot token (paste the token only)" >&2; fi ;;
      AGENTS_OWNER_ID)      if printf '%s' "$value" | grep -Eq '^[0-9]{4,15}$'; then
                              set_var AGENTS_OWNER_ID "$value"; else echo "AGENTS_OWNER_ID ignored: digits expected" >&2; fi ;;
      AGENTS_GH_TOKEN)      if printf '%s' "$value" | grep -Eq '^[A-Za-z0-9_]{30,}$'; then
                              set_var AGENTS_GH_TOKEN "$value"; else echo "AGENTS_GH_TOKEN ignored: not a GitHub token" >&2; fi ;;
    esac
  done
fi
# D279: the triage agent's read-only summary key. Made here once when no "agents:" key exists (so the owner does
# not have to make one), kept across deploys; the agents bot reads it as AGENTS_SUMMARY_KEY. Never printed.
if ! sed -n 's/^ECGSCAN_API_KEYS=//p' .env | tr ',' '\n' | grep -q '^agents:'; then
  set_var ECGSCAN_API_KEYS "$(sed -n 's/^ECGSCAN_API_KEYS=//p' .env),agents:$(openssl rand -hex 24)"
  echo "agents summary key made in deploy/.env"
fi
set_var AGENTS_SUMMARY_KEY "$(sed -n 's/^ECGSCAN_API_KEYS=//p' .env | tr ',' '\n' | sed -n 's/^agents://p' | head -1)"
