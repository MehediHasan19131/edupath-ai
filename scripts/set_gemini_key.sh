#!/usr/bin/env bash
# Asks for your Gemini API key (hidden while you paste) and saves it into .env.
# Usage:  bash scripts/set_gemini_key.sh
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f .env ] || cp .env.example .env

read -rsp "Paste your Gemini API key, then press Enter: " KEY
echo
KEY="$(printf '%s' "$KEY" | tr -d '[:space:]"')"
if [ -z "$KEY" ]; then
  echo "Nothing pasted - .env not changed."
  exit 1
fi

if grep -q '^GEMINI_API_KEY=' .env; then
  sed -i '' "s|^GEMINI_API_KEY=.*|GEMINI_API_KEY=${KEY}|" .env
else
  echo "GEMINI_API_KEY=${KEY}" >> .env
fi
echo "Saved to $(pwd)/.env (${#KEY} characters). You can close this tab."
