#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR/frontend"

if [ -x "$ROOT_DIR/.local-node/bin/npm" ]; then
  export PATH="$ROOT_DIR/.local-node/bin:$PATH"
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm was not found. Install Node.js LTS or unpack portable Node into .local-node/."
  exit 1
fi

if [ ! -d "node_modules" ]; then
  npm ci
fi

echo "Starting frontend at http://127.0.0.1:5173"
npm run dev -- --host 127.0.0.1
