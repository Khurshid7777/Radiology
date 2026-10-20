#!/usr/bin/env bash
# CardioLens: local LLM (Ollama) + web UI in one command (TZ §12, §14, §17).
#
#   scripts/run_local.sh                 # venv -> ollama serve -> ollama pull -> uvicorn
#   scripts/run_local.sh --no-llm        # web UI only, deterministic narrative (stub)
#   CARDIOLENS_LLM_MODEL=qwen2.5:7b scripts/run_local.sh   # another model
#   PORT=8080 scripts/run_local.sh
#
# Nothing leaves the machine: Layers 1-5 are pure Python, Layer 6 talks to
# http://127.0.0.1:11434/v1 (Ollama's OpenAI-compatible endpoint). Without
# Ollama the service still starts and every report is written by the stub;
# the header badge and /health say which one is active.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${CARDIOLENS_LLM_MODEL:-llama3.1:8b}"
BASE_URL="${CARDIOLENS_LLM_BASE_URL:-http://127.0.0.1:11434/v1}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
WANT_LLM=1
for arg in "$@"; do
  case "$arg" in
    --no-llm) WANT_LLM=0 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

log() { printf '\033[1;34m[cardiolens]\033[0m %s\n' "$*"; }

# 1. Python environment -------------------------------------------------------
if [ ! -x .venv/bin/python ]; then
  log "creating .venv"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate
if ! python -c "import cardiolens, fastapi, uvicorn" 2>/dev/null; then
  log "installing cardiolens (pip install -e '.[dev]')"
  pip install -q --upgrade pip
  pip install -q -e ".[dev]"
fi

# 2. Local LLM server ---------------------------------------------------------
llm_up() { curl -fsS -m 2 "${BASE_URL%/}/models" >/dev/null 2>&1; }

if [ "$WANT_LLM" = 1 ]; then
  if llm_up; then
    log "LLM server already answers at $BASE_URL"
  elif command -v ollama >/dev/null 2>&1; then
    log "starting ollama serve (log: .ollama-serve.log)"
    nohup ollama serve >.ollama-serve.log 2>&1 &
    for _ in $(seq 1 30); do llm_up && break; sleep 1; done
    llm_up || { log "ollama did not come up in 30 s — see .ollama-serve.log"; }
  else
    cat >&2 <<EOF
[cardiolens] ollama is not installed. The service will start with the deterministic
[cardiolens] narrative (stub). To enable the local LLM, install Ollama and re-run:
[cardiolens]   Linux:   curl -fsSL https://ollama.com/install.sh | sh
[cardiolens]   macOS:   brew install ollama      (or https://ollama.com/download)
[cardiolens]   Windows: https://ollama.com/download  (then run this script in Git Bash / WSL)
[cardiolens] or point CARDIOLENS_LLM_BASE_URL at any OpenAI-compatible server
[cardiolens] (llama.cpp, vLLM, LM Studio).
EOF
  fi

  if llm_up && command -v ollama >/dev/null 2>&1; then
    if ! ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$MODEL"; then
      log "pulling model $MODEL (one-off download, several GB)"
      ollama pull "$MODEL"
    fi
    log "model $MODEL ready"
  fi
fi

# 3. Web service ---------------------------------------------------------------
export CARDIOLENS_LLM_MODEL="$MODEL"
export CARDIOLENS_LLM_BASE_URL="$BASE_URL"
export CARDIOLENS_LLM_PROVIDER="${CARDIOLENS_LLM_PROVIDER:-auto}"   # Claude key → local → stub
if [ "$WANT_LLM" = 0 ]; then export CARDIOLENS_LLM_PROVIDER=stub; fi

log "web UI:  http://$HOST:$PORT/        health: http://$HOST:$PORT/health"
# The API needs a key (D98). Without one configured, a key for this session is made up
# and printed: paste it into the "Ключ доступа API" field of the web page.
if [ -z "${ECGSCAN_API_KEYS:-}" ]; then
  KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(18))')"
  export ECGSCAN_API_KEYS="local:$KEY"
  echo "API key for this session (web page field «Ключ доступа API»): $KEY"
fi
exec uvicorn cardiolens.api.app:app --host "$HOST" --port "$PORT"
