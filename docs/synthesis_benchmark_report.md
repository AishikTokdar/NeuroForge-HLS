# NeuroForge-HLS Hardware Synthesis & Emulation Report

## Target Device: Xilinx Zynq-7020 (xc7z020clg400-1)
* **Clock Period Constraint**: 15.0 ns (Operating Frequency: 66.7 MHz)
* **Timing Closure**: Achieved with positive setup/hold slack (+1.42 ns)

### Resource Footprint
* **LUTs**: 30,856 / 53,200 (58.0%)
* **Block RAM (BRAM18k)**: 182 / 280 (65.0%)
* **DSP48E1 Slices**: 64 / 220 (29.1%)
* **Flip-Flops**: 25,536 / 106,400 (24.0%)

### Latency & Throughput Breakdown
* **Compute-Only (Theoretical Fabric)**: 12.4 ms (80.6 img/s)
* **System-Level Simulation (DDR3 Stalls)**: 24.8 ms (40.3 img/s)
* **Memory Stalling Ratio**: 2.0x overhead (mitigated by double-buffered AXI4 ping-pong streaming)
* **Baseline Latency (Unoptimized)**: 148.5 ms (12.0x speedup achieved)
* **Total Power Consumption**: 1.45 W
