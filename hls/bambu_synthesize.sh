#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

TOP_FUNC="reduced_vgg_inference"
CLOCK_PERIOD=15
DEVICE="xc7z020-1clg400"

echo "=================================================="
echo "Starting Bambu C++ to Verilog Synthesis"
echo "Target Top Function: ${TOP_FUNC}"
echo "Clock Constraint:    ${CLOCK_PERIOD} ns (66.7 MHz)"
echo "Target Device:       ${DEVICE}"
echo "=================================================="

bambu --top-fname=$TOP_FUNC \
      --clock-period=$CLOCK_PERIOD \
      --device-name=$DEVICE \
      -I. \
      tiled_conv.cpp utils.cpp \
      -DNO_HLS \
      --compiler=I386_CLANG13 \
      --experimental-setup=BAMBU-BALANCED

echo "=================================================="
echo "Bambu HLS completed successfully!"
echo "Generated Verilog RTL: ${TOP_FUNC}.v"
echo "=================================================="
