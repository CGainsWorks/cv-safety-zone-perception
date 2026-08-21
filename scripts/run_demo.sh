#!/usr/bin/env bash
# The demo, in one command, with the CAN interface picked for this machine.
#
#     scripts/run_demo.sh                 # simulated detector, geometry demo
#     scripts/run_demo.sh --detector onnx # the real model, for the timings
#
# Anything passed here goes through to scripts/run_demo.py.

set -euo pipefail
cd "$(dirname "$0")/.."

VENV_PY=".venv/bin/python"
[ -x "$VENV_PY" ] || VENV_PY=".venv/Scripts/python.exe"
[ -x "$VENV_PY" ] || VENV_PY="python3"

if [ "$(uname -s)" = "Linux" ] && ! ip link show vcan0 >/dev/null 2>&1; then
  echo "vcan0 is not up.  Run 'sudo scripts/setup_vcan.sh' for the socketcan" >&2
  echo "path or pass --can virtual to use the in-process bus." >&2
fi

mkdir -p results
exec "$VENV_PY" scripts/run_demo.py \
  --out results/demo.json \
  --timeline results/demo_timeline.csv \
  "$@"
