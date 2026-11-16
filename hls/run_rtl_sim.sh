#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if ! command -v vitis_hls >/dev/null 2>&1; then
    echo "RTL verification requires Vitis HLS and its supported HDL simulator." >&2
    echo "No interface-specific standalone Verilator/Icarus driver is supplied." >&2
    exit 1
fi
vitis_hls -f vitis_hls.tcl cosim
