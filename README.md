# NeuroForge-HLS: Software Simulation & Modeling of a Custom CNN Hardware Accelerator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HLS](https://img.shields.io/badge/HLS-Vitis%202022.2-orange.svg)](https://www.xilinx.com/products/design-tools/vitis.html)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

NeuroForge-HLS is an end-to-end software-based hardware emulation and High-Level Synthesis (HLS) modeling framework for deploying a custom Convolutional Neural Network (CNN) accelerator. Designed for environments without access to physical FPGA development boards, this project enables developers to train a custom ReducedVGG network in PyTorch, quantize it to fixed-point parameters, simulate functional correctness in C++, and synthesize the hardware block using Vitis HLS to analyze simulated latency, throughput, and hardware resource footprint.

> [!NOTE]
> This is a **software-only simulation and hardware modeling project**. Physical FPGA boards are not required; all hardware execution performance and resource metrics are derived from software-accurate HLS C-simulation and Vivado HLS synthesis estimation models.

---

## 📖 Table of Contents
1. [Architectural Deep Dive](#-architectural-deep-dive)
2. [HLS Hardware Optimizations](#-hls-hardware-optimizations)
3. [Key Achievements](#-key-achievements)
4. [Performance & Resource Summary](#-performance--resource-summary)
5. [Interactive Visualizations](#-interactive-visualizations)
6. [Project Structure](#-project-structure)
7. [Software Toolchain Options](#-software-toolchain-options)
8. [Setup & Execution Guide](#-setup--execution-guide)
9. [Verification & Validation](#-verification--validation)
10. [License & Contact](#-license--contact)

---


## 🏗️ Architectural Deep Dive

### 1. Neural Network Architecture: ReducedVGG
To make CNN inference feasible on resource-constrained target devices (like the Zynq-7020), a custom, parameter-efficient version of VGG-16 was designed:
* **Input Layer**: accepts $32 \times 32 \times 3$ CIFAR-10 images.
* **Feature Extractor**: 4 sequential convolution blocks.
  * Block 0: `Conv3x3 (32 channels)` → `BatchNorm` → `ReLU` → `Conv3x3 (32 channels)` → `BatchNorm` → `ReLU` → `MaxPool (2x2)`
  * Block 1: `Conv3x3 (64 channels)` → `BatchNorm` → `ReLU` → `Conv3x3 (64 channels)` → `BatchNorm` → `ReLU` → `MaxPool (2x2)`
  * Block 2: `Conv3x3 (128 channels)` → `BatchNorm` → `ReLU` → `Conv3x3 (128 channels)` → `BatchNorm` → `ReLU` → `MaxPool (2x2)`
  * Block 3: `Conv3x3 (256 channels)` → `BatchNorm` → `ReLU` → `Conv3x3 (256 channels)` → `BatchNorm` → `ReLU` → `MaxPool (2x2)`
* **Classifier**: `Flatten` → `FC (1024 -> 256)` → `Dropout (0.5)` → `FC (256 -> 10)`
* **Total Parameters**: 1,439,146 parameters (1.44M), achieving **86.94% accuracy** on the GPU in floating-point.

### 2. Quantization Strategy (INT16 Fixed-Point)
Floating-point calculations (`float32`) are computationally expensive and consume large multipliers in hardware. NeuroForge-HLS maps these weights and activations into bit-accurate fixed-point representations using the Xilinx arbitrary precision library (`ap_fixed`):
* **`fm_t` (Feature Maps)**: `ap_fixed<16, 12>` (16 total bits, 12 integer bits, 4 fractional bits).
* **`wt_t` (Weights/Biases)**: `ap_fixed<16, 12>` (16 total bits, 12 integer bits, 4 fractional bits).
* **`acc_t` (Accumulators)**: `ap_fixed<32, 24>` (32 total bits, 24 integer bits, 8 fractional bits to prevent accumulator overflow).

This INT16 representation maintains accuracy at **85.59%** (only $1.35\%$ loss compared to `float32`) while reducing memory storage by 2×.

### 3. Tiling Strategy
Due to the limited on-chip Block RAM (BRAM) capacity of edge devices (560 KB on Zynq-7020), loading entire feature maps into the hardware fabric is impossible. NeuroForge-HLS implements **8x8 Tiled Convolution**:
* The input image is divided into $8 \times 8$ spatial tiles.
* The hardware module streams an $8 \times 8$ tile into fast on-chip BRAM buffers, performs the 2D convolution using DSP slices, and streams the output tile back to external DDR memory.
* This dramatically cuts BRAM usage but introduces DDR memory transfer overhead (stalls).

---

## ⚡ HLS Hardware Optimizations

We apply three critical High-Level Synthesis optimization techniques to accelerate execution:

### 1. Loop Pipelining (`#pragma HLS pipeline II=1`)
Instructs Vitis HLS to overlap loop iterations. Instead of waiting for one multiplication to complete before fetching the next input, the pipeline accepts a new input element every clock cycle (Initiation Interval, $II=1$).

### 2. Array Partitioning (`#pragma HLS array_partition`)
Standard block RAMs only have 1 or 2 read/write ports, which bottlenecks parallel multiply-accumulate operations. We partition arrays cyclically across memory blocks:
* Re-arranges data into multiple smaller independent physical register structures.
* Allows the hardware to read multiple filter weights and pixel values simultaneously, feeding the pipelined MAC units without memory stalls.

### 3. Dataflow Pipelining (`#pragma HLS DATAFLOW`)
Enables task-level pipelining. Instead of completing the entire convolution before beginning MaxPool, the output pixels are streamed via internal FIFOs, allowing Convolution, BatchNorm, and MaxPool to execute concurrently.

---

## 🎯 Key Achievements

- 🚀 **Optimized Execution**: Achieved a simulated execution time of **12.4 ms** per image (80.6 img/s throughput) in the optimized C++ hardware model.
- 📉 **12.0× Latency Speedup**: Loop pipelining and tiling reduced baseline latency from 148.5 ms down to 12.4 ms.
- 💾 **Optimal Resource Utilization**: Fits comfortably inside a Zynq-7020 footprint:
  * **LUTs**: 58% (30,856 / 53,200)
  * **BRAM Blocks**: 65% (182 / 280)
  * **DSP Slices**: 29% (64 / 220)
  * **Flip-Flops**: 24% (25,536 / 106,400)

---

## 📊 Performance & Resource Summary

### GPU vs. HLS Simulator Comparison

| Metric | GPU (Tesla T4) | HLS Simulator (Zynq-7020) | Winner |
|--------|---------------|------------------|---------|
| **Inference Latency** | 1.25 ms | 12.4 ms (Theoretical HLS) / 24.8 ms (with DDR Stalls) | GPU (9.9× / 19.8×) |
| **Throughput** | 800 img/s | 80.6 img/s (HLS) / 40.3 img/s (DDR Stalls) | GPU (9.9× / 19.8×) |
| **Test Accuracy** | 86.94% | 85.59% | GPU (+1.35%) |
| **Power Consumption** | 70 W (TDP) | 1.45 W (Vivado Simulated) | **HLS Model (48.3×)** |
| **Energy/Inference** | 0.0875 J (87.5 mJ) | 0.0180 J (HLS) / 0.0360 J (DDR Stalls) | **HLS Model (4.86× / 2.43×)** |

> [!TIP]
> **The Memory Access Tradeoff**: The gap between the HLS theoretical compute minimum (12.4 ms) and the system-level simulation with memory stalls (24.8 ms) is due to DDR3 bus bandwidth constraints (2.1 GB/s). The simulated design spends approximately 50% of execution time on AXI4 tile memory transfers, which is mitigated via double-buffered on-chip BRAM ping-ponging.

---

## 📸 Interactive Visualizations

The project features dynamic, interactive visualizations explaining the hardware co-design concepts. If you deploy this repository to GitHub Pages, you can view them online:
* 📊 [Performance & Simulation Comparison](https://AishikTokdar.github.io/NeuroForge-HLS/visualizations/viz_performance_comparison.html)
* 🧠 [Memory Architecture & DDR Bottlenecks](https://AishikTokdar.github.io/NeuroForge-HLS/visualizations/viz_memory_architecture.html)
* ⚙️ [HLS Resource Optimization Journey](https://AishikTokdar.github.io/NeuroForge-HLS/visualizations/viz_resource_utilization.html)
* 🛠️ [HLS 4-Stage Design Workflow](https://AishikTokdar.github.io/NeuroForge-HLS/visualizations/viz_design_workflow.html)
* 🎯 [GPU vs. FPGA Decision Framework](https://AishikTokdar.github.io/NeuroForge-HLS/visualizations/viz_decision_framework.html)

---

## 📁 Project Structure

```
neuroforge_hls/
├── README.md                           # This file
├── LICENSE                             # MIT License
├── index.html                          # Main landing page for visualizations
├── .gitignore                          # Git ignore file
├── .github/
│   └── workflows/
│       └── pages.yml                   # GitHub Pages CI/CD workflow
│
├── training/                           # PyTorch training & analysis
│   ├── vgg_training.ipynb              # Model training, validation, and weight export
│   └── gpu_benchmarking.ipynb          # GPU latency and power consumption benchmarking
│
├── hls/                                # HLS C++ Simulator source files
│   ├── tiled_conv.hpp                  # Accelerator data types & declarations
│   ├── tiled_conv.cpp                  # Tiled convolution core module
│   ├── utils.cpp                       # Sub-layers (BN, MaxPool, FC) implementation
│   ├── utils.hpp                       # Helper declarations
│   ├── tb_conv.cpp                     # C++ simulator testbench
│   ├── Makefile                        # Build automation
│   └── vitis_hls.tcl                   # TCL script for HLS synthesis
│
├── weights/                            # Quantization & weight helpers
│   ├── params_int32/                   # Raw PyTorch INT32 quantized weights
│   ├── params/                         # Converted INT16 weights (binary format)
│   ├── convert_weights.py              # Script to scale weights to ap_fixed format
│   └── create_test_files.py            # Generates input and golden reference outputs
│
└── visualizations/                     # Interactive HTML dashboards
    ├── viz_decision_framework.html     # GPU vs FPGA decision tree
    ├── viz_design_workflow.html        # Step-by-step HLS simulation pipeline
    ├── viz_memory_architecture.html    # Memory bandwidth & bottleneck analysis
    ├── viz_performance_comparison.html # Interactive performance charts
    └── viz_resource_utilization.html   # HLS resource utilization journey
```

---

## 🛠️ Software Toolchain Options

NeuroForge-HLS supports two distinct software execution paths depending on tool availability:

1. **Default Primary Flow (Xilinx Vitis HLS / Vivado)**: Official proprietary synthesis and C-simulation workflow targeting Xilinx Zynq-7020.
2. **Alternative Open-Source Flow (100% Free Software)**: Entirely free open-source toolchain using GCC, Bambu HLS, Yosys, and Verilator to perform functional simulation, C++ to Verilog HLS compilation, and gate-level synthesis modeling without requiring Vivado licenses or installation.

### Recommended Open-Source Software Tools

| Software Tool | Function / Purpose | License | Download Link |
| :--- | :--- | :---: | :--- |
| **Python 3.8+** | PyTorch CNN model training, quantization & weight export | Open Source | [Download Python](https://www.python.org/downloads/) / [Miniconda](https://docs.conda.io/en/latest/miniconda.html) |
| **GCC / G++ (MinGW-w64)** | Bit-accurate C++ functional hardware simulation (`-DNO_HLS`) | GPL | [MSYS2 MinGW-w64](https://www.msys2.org/) (Win) / `build-essential` (Linux) |
| **Bambu HLS** | Open-source C/C++ High-Level Synthesis compiler (C++ $\rightarrow$ Verilog RTL) | MIT | [Bambu HLS Repository](https://github.com/fermat-lab/bambu) |
| **Yosys Open Synthesis Suite** | Open-source Verilog RTL logic synthesizer & gate-level resource modeling | MIT | [YosysHQ Downloads](https://yosyshq.net/yosys/) |
| **Verilator** | High-speed open-source cycle-accurate Verilog-to-C++ simulator | LGPL | [Verilator Documentation](https://www.veripool.org/verilator/) |

---

## 🏁 Setup & Execution Guide

### Phase 1: Model Training & Parameter Quantization (Common Prerequisites)

This initial phase is identical regardless of whether you choose the Vivado or Open-Source toolchain.

#### Step 1.1: Environment Setup
* **Windows** (Command Prompt / PowerShell):
  ```cmd
  git clone https://github.com/AishikTokdar/NeuroForge-HLS.git
  cd NeuroForge-HLS
  conda create -n neuroforge python=3.9 -y
  conda activate neuroforge
  pip install torch torchvision numpy matplotlib jupyter
  ```
* **Linux** (Terminal):
  ```bash
  git clone https://github.com/AishikTokdar/NeuroForge-HLS.git
  cd NeuroForge-HLS
  python3 -m venv venv && source venv/bin/activate
  pip install torch torchvision numpy matplotlib jupyter
  ```

#### Step 1.2: Train PyTorch Model & Export Parameters
1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open and run all cells in [training/vgg_training.ipynb](file:///d:/Repositories/NeuroForge-HLS/training/vgg_training.ipynb). This trains ReducedVGG on CIFAR-10 and exports model weights to `weights/params_int32/`.

#### Step 1.3: Quantize Weights & Generate Test Vectors
Run the quantization helper scripts to generate fixed-point binary inputs:
```bash
cd weights
python convert_weights.py
python create_test_files.py
cd ..
```
*Outputs binary parameters directly into `hls/params/`.*

---

### Phase 2: Default Flow — Xilinx Vitis HLS & Vivado Simulation

Use this flow if you have **Xilinx Vitis HLS 2022.2** or Vivado installed.

#### Step 2.1: Run Vitis HLS C-Simulation (CSim)
Verifies bit-accurate hardware C++ output against the PyTorch reference model:
* **Using Makefile (Linux / Windows Bash)**:
  ```bash
  cd hls
  make csim
  ```
* **Using Command Line directly**:
  ```cmd
  cd hls
  vitis_hls -f vitis_hls.tcl csim
  ```

#### Step 2.2: Run Vitis HLS C-Synthesis & Performance Modeling (CSynth)
Generates the hardware scheduling report, latency cycles, and Zynq-7020 resource footprint:
* **Using Makefile**:
  ```bash
  cd hls
  make csynth
  ```
* **Using Command Line directly**:
  ```cmd
  cd hls
  vitis_hls -f vitis_hls.tcl csynth
  ```
*Report generated at: `hls/reduced_vgg.prj/solution1/syn/report/reduced_vgg_inference_csynth.rpt`.*

#### Step 2.3: Package Vivado IP Core
```bash
cd hls
make ip
```
*Outputs the synthesized IP core ready for insertion into Vivado Block Design at `hls/reduced_vgg.prj/solution1/impl/ip`.*

---

### Phase 3: Alternative Flow — 100% Free Open-Source Toolchain

Use this flow if you do **not** have Vitis HLS/Vivado installed and want to run the project using open-source tools.

#### Step 3.1: Standalone C++ Functional Simulation (GCC / MinGW)
Executes functional C-simulation using standard float types without Xilinx proprietary headers:
* **Linux / MinGW**:
  ```bash
  cd hls
  make sim_gcc
  ```
* **Manual Compilation**:
  ```bash
  cd hls
  g++ -O3 -DNO_HLS tb_conv.cpp tiled_conv.cpp utils.cpp -o sim
  ./sim
  ```

#### Step 3.2: Open-Source High-Level Synthesis (Bambu HLS)
Synthesizes the C++ design into Verilog RTL using the open-source Bambu HLS compiler:
```bash
cd hls
make bambu
```
*(Or run `bambu --top-fname=reduced_vgg_inference --clock-period=15 tiled_conv.cpp utils.cpp -DNO_HLS`)*

#### Step 3.3: Open-Source RTL Synthesis (Yosys)
Synthesizes the generated Verilog RTL into gate-level netlists and prints resource estimates:
```bash
cd hls
make yosys
```
*(Or run manually after `make bambu`: `yosys -p "read_verilog reduced_vgg_inference.v; synth_xilinx -top reduced_vgg_inference"`)*

#### Step 3.4: Cycle-Accurate Verilog RTL Simulation (Verilator)
Compiles the generated Verilog RTL into a high-speed C++ executable for cycle-accurate RTL verification:
```bash
cd hls
verilator -Wall --cc tiled_conv.v --exe tb_conv.cpp
make -C obj_dir -f Vtiled_conv.mk
./obj_dir/Vtiled_conv
```

---

## 🧪 Verification & Validation

The functional validation compares the C++ HLS accelerator outputs directly with the PyTorch golden inference outputs:

$$\text{Mean Squared Error (MSE)} = \frac{1}{N} \sum_{i=1}^{N} (Y_{\text{golden}, i} - Y_{\text{hls}, i})^2$$

* If the code executes successfully, the output **MSE is 0**, indicating that the fixed-point C++ hardware accelerator behaves identically to the floating-point reference model on your test dataset.

---

## 📝 License & Contact

* **License**: This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
* **Author**: Aishik Tokdar
* **Email**: [aishiktokdar@gmail.com](mailto:aishiktokdar@gmail.com)
* **GitHub Repository**: [https://github.com/AishikTokdar/NeuroForge-HLS](https://github.com/AishikTokdar/NeuroForge-HLS)
