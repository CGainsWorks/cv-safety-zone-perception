#!/usr/bin/env bash
# Bring up a virtual CAN interface.  Linux only.
#
#     sudo scripts/setup_vcan.sh          # creates vcan0
#     sudo scripts/setup_vcan.sh vcan1
#
# vcan is a kernel module that gives you a CAN interface with no hardware
# behind it.  Frames go through the whole SocketCAN stack -- the same path a
# real interface takes, minus the transceiver -- which is why the latency
# measured over vcan is worth something and the latency measured over
# python-can's in-process "virtual" bus is not.
#
# On macOS and Windows there is no vcan.  The pipeline falls back to
# python-can's virtual bus automatically and every result file records which
# one it used.  See src/szp/safety/publisher.py.

set -euo pipefail

IFACE="${1:-vcan0}"

if [ "$(uname -s)" != "Linux" ]; then
  echo "vcan is a Linux kernel module; on $(uname -s) the pipeline uses" >&2
  echo "python-can's in-process virtual bus instead.  Nothing to do here." >&2
  exit 0
fi

if ip link show "$IFACE" >/dev/null 2>&1; then
  echo "$IFACE already exists"
else
  modprobe vcan
  ip link add dev "$IFACE" type vcan
  echo "created $IFACE"
fi

ip link set up "$IFACE"
ip -details -brief link show "$IFACE"

cat <<EOF

$IFACE is up.  Try it:

    python scripts/run_demo.py --can socketcan --channel $IFACE
    candump $IFACE          # from can-utils, if you have it

EOF
