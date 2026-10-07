#!/bin/bash
# Cycle-accurate RTL simulation with Verilator / Icarus Verilog
set -e

echo "=================================================="
echo "Starting Cycle-Accurate Verilog RTL Simulation"
echo "Top Module: reduced_vgg_inference"
echo "=================================================="

if command -v verilator &> /dev/null; then
    echo "Using Verilator for high-performance C++ co-simulation..."
    verilator --cc reduced_vgg_inference.v --exe tb_conv.cpp utils.cpp -DNO_HLS
    make -C obj_dir -f Vreduced_vgg_inference.mk
    ./obj_dir/Vreduced_vgg_inference
elif command -v iverilog &> /dev/null; then
    echo "Using Icarus Verilog for functional RTL verification..."
    iverilog -o sim_rtl tb_conv.v reduced_vgg_inference.v
    vvp sim_rtl
else
    echo "Neither Verilator nor Icarus Verilog found in PATH."
    exit 1
fi

echo "=================================================="
echo "Cycle-accurate RTL simulation completed successfully."
echo "=================================================="
