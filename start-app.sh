#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ELECTRON_RUN_AS_NODE=

cd "$ROOT_DIR/backend"
if [ ! -x ".venv/bin/python" ]; then
  python3.12 -m venv --without-pip .venv
fi
if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  wget -O /tmp/rtr-get-pip.py https://bootstrap.pypa.io/get-pip.py
  .venv/bin/python /tmp/rtr-get-pip.py
fi
if ! .venv/bin/python -c "import fastapi, uvicorn, cv2, onnxruntime, pydantic" >/dev/null 2>&1; then
  .venv/bin/python -m pip uninstall -y opencv-python opencv-python-headless >/dev/null 2>&1 || true
  .venv/bin/python -m pip install -r requirements.txt
fi

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

npm run build
./node_modules/.bin/electron .
