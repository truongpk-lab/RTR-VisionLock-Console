#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR/backend"

if [ ! -x ".venv/bin/python" ]; then
  python3.12 -m venv --without-pip .venv
fi

if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  wget -O /tmp/rtr-get-pip.py https://bootstrap.pypa.io/get-pip.py
  .venv/bin/python /tmp/rtr-get-pip.py
fi

.venv/bin/python -m pip uninstall -y opencv-python opencv-python-headless >/dev/null 2>&1 || true
.venv/bin/python -m pip install -r requirements.txt

echo "Starting backend at http://127.0.0.1:8000"
RTR_HOST=127.0.0.1 RTR_PORT=8000 RTR_RELOAD=0 .venv/bin/python -m app.main
