#!/usr/bin/env bash
# Keeps deploy/.env on the server (D153). The first run makes the clinic API key and the
# storage key there, so they never pass through GitHub; later runs keep them and refresh
# DOMAIN and whatever NAME=VALUE lines arrive on stdin (empty values are ignored):
#   ANTHROPIC_API_KEY     -> ANTHROPIC_API_KEY
#   TELEGRAM_BOT_TOKEN    -> ECGSCAN_BOT_TOKEN      (D161; with ECGSCAN_BOT_API_KEY = the clinic key)
#   TELEGRAM_BOT_ALLOWED  -> ECGSCAN_BOT_ALLOWED
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
if [ ! -t 0 ]; then
  while IFS='=' read -r name value; do
    [ -n "${value:-}" ] || continue
    case "$name" in
      ANTHROPIC_API_KEY)    set_var ANTHROPIC_API_KEY "$value" ;;
      TELEGRAM_BOT_TOKEN)   set_var ECGSCAN_BOT_TOKEN "$value"
                            set_var ECGSCAN_BOT_API_KEY "$(sed -n 's/^ECGSCAN_API_KEYS=[^:]*:\([^,]*\).*/\1/p' .env)" ;;
      TELEGRAM_BOT_ALLOWED) set_var ECGSCAN_BOT_ALLOWED "$value" ;;
    esac
  done
fi
