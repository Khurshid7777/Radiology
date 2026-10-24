#!/usr/bin/env bash
# Keeps deploy/.env on the server (D153). The first run makes the clinic API key and the
# storage key there, so they never pass through GitHub; later runs keep them and only
# refresh DOMAIN and, when given, ANTHROPIC_API_KEY.  Usage: server_env.sh DOMAIN [ANTHROPIC_KEY]
set -euo pipefail
cd "$(dirname "$0")"
domain=$1; anthropic=${2:-}
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
sed -i "s|^DOMAIN=.*|DOMAIN=$domain|" .env
if [ -n "$anthropic" ]; then sed -i "s|^ANTHROPIC_API_KEY=.*|ANTHROPIC_API_KEY=$anthropic|" .env; fi
