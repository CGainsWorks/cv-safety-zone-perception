#!/usr/bin/env bash
# One command from a fresh clone to something that runs.
#
#     scripts/setup.sh              # runtime only
#     scripts/setup.sh --export     # plus the AGPL export toolchain
#     scripts/setup.sh --data       # plus COCO val2017 (1 GB)
#
# Creates .venv, installs the pinned requirements, downloads and exports the
# detector weights and brings up vcan on Linux.  Everything it does is also a
# make target; this script exists so that a reader who has not read the
# Makefile still gets a working checkout.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
WANT_EXPORT=0
WANT_DATA=0
for arg in "$@"; do
  case "$arg" in
    --export) WANT_EXPORT=1 ;;
    --data)   WANT_DATA=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

if [ ! -d .venv ]; then
  echo "==> creating .venv"
  "$PYTHON" -m venv .venv
fi
VENV_PY=".venv/bin/python"
[ -x "$VENV_PY" ] || VENV_PY=".venv/Scripts/python.exe"

echo "==> installing requirements"
"$VENV_PY" -m pip install --upgrade pip >/dev/null
"$VENV_PY" -m pip install -r requirements.txt

if [ "$WANT_EXPORT" = "1" ]; then
  echo "==> installing the export toolchain (ultralytics is AGPL-3.0; see docs/licences.md)"
  "$VENV_PY" -m pip install -r requirements-export.txt
fi

if [ ! -f models/yolov8n.onnx ]; then
  if [ "$WANT_EXPORT" = "1" ]; then
    echo "==> exporting the detector"
    "$VENV_PY" scripts/fetch_model.py
  else
    echo "==> no models/yolov8n.onnx yet."
    echo "    Run 'scripts/setup.sh --export' to download and export it."
    echo "    Everything that does not need the detector works without it."
  fi
fi

if [ "$WANT_DATA" = "1" ]; then
  echo "==> fetching COCO val2017"
  "$VENV_PY" scripts/fetch_data.py
fi

if [ "$(uname -s)" = "Linux" ] && ! ip link show vcan0 >/dev/null 2>&1; then
  echo "==> vcan0 is not up.  Run: sudo scripts/setup_vcan.sh"
fi

echo
echo "done.  Next:"
echo "    make demo       # the whole pipeline on the synthetic cell"
echo "    make test       # the test suite"
