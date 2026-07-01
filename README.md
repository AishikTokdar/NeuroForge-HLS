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
7. [Windows Setup & Execution Guide](#-windows-setup--execution-guide)
8. [Linux Setup & Execution Guide](#-linux-setup--execution-guide)
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
Floating-point calculations ($float32$) are computationally expensive and consume large multipliers in hardware. NeuroForge-HLS maps these weights and activations into bit-accurate fixed-point representations using the Xilinx arbitrary precision library (`ap_fixed`):
* **`fm_t` (Feature Maps)**: `ap_fixed<16, 12>` (16 total bits, 12 integer bits, 4 fractional bits).
* **`wt_t` (Weights/Biases)**: `ap_fixed<16, 12>` (16 total bits, 12 integer bits, 4 fractional bits).
* **`acc_t` (Accumulators)**: `ap_fixed<32, 24>` (32 total bits, 24 integer bits, 8 fractional bits to prevent accumulator overflow).

This INT16 representation maintains accuracy at **85.59%** (only $1.35\%$ loss compared to $float32$) while reducing memory storage by 2×.

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

- 🚀 **Optimized Execution**: Achieved a simulated execution time of **7.98 ms** per image (125.3 img/s throughput) in the optimized C++ design.
- 📉 **49.8× Latency Speedup**: Loop pipelining and tiling reduced baseline latency from 397 ms down to 7.98 ms.
- 💾 **Optimal Resource Utilization**: Fits comfortably inside a Zynq-7020 footprint:
  * **LUTs**: 62% (33,437 / 53,200)
  * **BRAM Blocks**: 70% (196 / 280)
  * **DSP Slices**: 14% (32 / 220)
  * **Flip-Flops**: 21% (23,210 / 106,400)

---

## 📊 Performance & Resource Summary

### GPU vs. HLS Simulator Comparison

| Metric | GPU (Tesla T4) | HLS Simulator (Zynq-7020) | Winner |
|--------|---------------|------------------|---------|
| **Inference Latency** | 1.296 ms | 7.98 ms (Theoretical HLS) / 466.53 ms (with DDR Stalls) | GPU (6.2× / 360×) |
| **Throughput** | 771.5 img/s | 125.3 img/s (HLS) / 2.14 img/s (DDR Stalls) | GPU (6.2× / 360×) |
| **Test Accuracy** | 86.94% | 85.59% | GPU (+1.35%) |
| **Power Consumption** | 70 W (TDP) | 1.451 W (Vivado Simulated) | **HLS Model (48×)** |
| **Energy/Inference** | 0.091 J | 0.0116 J (HLS) / 0.677 J (DDR Stalls) | **HLS Model (7.8×)** / GPU (7.4×) |

> [!TIP]
> **The Memory Bottleneck**: The gap between the HLS theoretical minimum (7.98 ms) and the system-level simulation including memory stalls (466.53 ms) is caused by DDR3 memory access bottlenecks. The simulated fabric spends 98.5% of its time waiting for DMA transfers to complete, underscoring the importance of memory-bandwidth-aware design in ML accelerators.

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

## 🏁 Windows Setup & Execution Guide

Follow these steps to run the training, weight conversion, C++ functional simulation, and Vitis HLS synthesis on Windows.

### Prerequisites on Windows
1. **Python Environment**: Install [Anaconda or Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. **C++ Compiler**: Install [MinGW-w64](https://www.mingw-w64.org/) (via MSYS2) or **Visual Studio Community Edition** (ensure "Desktop development with C++" workload is checked).
3. **Hardware Synthesis Tool (Optional)**: [Xilinx Vitis HLS 2022.2](https://www.xilinx.com/support/download.html).

---

### Step 1: Clone the Repository & Configure Python Environment
Open **Anaconda Prompt** or **Command Prompt** and run:
```cmd
# Clone the repository
git clone https://github.com/AishikTokdar/NeuroForge-HLS.git
cd NeuroForge-HLS

# Create and activate a conda environment
conda create -n neuroforge python=3.9 -y
conda activate neuroforge

# Install PyTorch (CPU version is sufficient for simulation)
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Install extra dependencies
pip install jupyter matplotlib numpy
```

---

### Step 2: Train and Export PyTorch Weights
1. Start Jupyter Notebook:
   ```cmd
   jupyter notebook
   ```
2. Navigate to and open `training/vgg_training.ipynb`.
3. Run all cells in the notebook. This will train the model on CIFAR-10, perform quantization checks, and export the raw model parameter weights into `weights/params_int32/`.

---

### Step 3: Quantize Weights & Generate Test Inputs
Now we scale the weights into the fixed-point formats and generate test image binary vectors:
```cmd
cd weights
python convert_weights.py
python create_test_files.py
cd ..
```
This converts the weights and generates the test image vectors directly inside the `hls/params/` directory.

---

### Step 4: Run C++ Functional Simulation (C-Sim)
This runs the hardware model in C++ to verify bit-level correctness against PyTorch's output.

#### Option A: Using MinGW (g++) on Windows (With Vitis HLS Installed)
If you have Vitis HLS installed and want to compile using the official Xilinx fixed-point libraries:
```cmd
cd hls
g++ -O3 -I"C:\Xilinx\Vitis_HLS\2022.2\include" tb_conv.cpp tiled_conv.cpp utils.cpp -o sim.exe
sim.exe
```

#### Option B: Using Visual Studio (MSVC) on Windows (With Vitis HLS Installed)
Open the **Developer Command Prompt for VS** and run:
```cmd
cd hls
cl /O2 /EHsc /I"C:\Xilinx\Vitis_HLS\2022.2\include" tb_conv.cpp tiled_conv.cpp utils.cpp /Fe:sim.exe
sim.exe
```

#### Option C: Standalone C++ Simulation (NO Vitis HLS/Vivado Installed)
If you do not have Vitis HLS/Vivado installed on your machine, compile using standard float types by defining the `NO_HLS` macro:
* Using MinGW (g++):
  ```cmd
  cd hls
  g++ -O3 -DNO_HLS tb_conv.cpp tiled_conv.cpp utils.cpp -o sim.exe
  sim.exe
  ```
* Using Visual Studio (MSVC CL):
  ```cmd
  cd hls
  cl /O2 /EHsc /DNO_HLS tb_conv.cpp tiled_conv.cpp utils.cpp /Fe:sim.exe
  sim.exe
  ```

*Expected Output:*
```
All model parameters and test vectors loaded successfully.
Beginning Reduced VGG Inference Simulation (4-Block VGG)...
========================================
Inference Simulation Complete
========================================
Simulation Time: 23 ms

========================================
Performance Metrics
========================================
MSE:              0
Error count:      0/10
Expected Accuracy: 85.59%
Model:            Reduced VGG (INT32 quantized)
Parameters:       1,439,146

========================================
TEST PASSED - Simulation SUCCESSFUL!
SCORE = Accuracy / Latency(ms) = 3.7213 acc/ms
========================================
```

---

### Step 5: Run HLS Synthesis (Hardware Modeling)
To estimate the latency, cycles, and resource footprint, launch the **Vivado HLS / Vitis HLS Command Prompt** and run:
```cmd
cd hls
vitis_hls -f vitis_hls.tcl
```
This script will parse your C++ code, execute scheduling, apply optimizations, and write the report files to `hls/reduced_vgg.prj/solution1/syn/report/reduced_vgg_inference_csynth.rpt`.

---

## 🐧 Linux Setup & Execution Guide

Follow these steps to configure and run the project under a Linux environment (Ubuntu/CentOS).

### Prerequisites on Linux
1. **Compiler Tools**: Install `gcc`, `g++`, and `make` (typically via `build-essential`).
2. **Python Environment**: Python 3.8+ with virtualenv support.
3. **Hardware Synthesis Tool (Optional)**: Vitis HLS 2022.2.

---

### Step 1: Install System Dependencies & Python Setup
Open your terminal and run:
```bash
# Update packages and install compiler tools
sudo apt-get update
sudo apt-get install build-essential gitpython -y

# Clone the repository
git clone https://github.com/AishikTokdar/NeuroForge-HLS.git
cd NeuroForge-HLS

# Set up python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch and dependencies
pip install --upgrade pip
pip install torch torchvision numpy matplotlib jupyter
```

---

### Step 2: Run Training & Weight Conversion
1. Launch Jupyter and run `training/vgg_training.ipynb`:
   ```bash
   jupyter notebook --no-browser
   ```
2. Once the notebook finishes exporting the raw parameters, run the quantization scripts:
   ```bash
   cd weights
   python3 convert_weights.py
   python3 create_test_files.py
   cd ..
   ```
This converts the weights and generates the test image vectors directly inside the `hls/params/` directory.

---

### Step 3: Compile and Run C++ Functional Simulation
On Linux, we can use the pre-configured `Makefile`.

#### Option A: Running with Vitis HLS environment sourced
If Vitis HLS is installed on your Linux machine:
```bash
# Source Xilinx tools (path may vary depending on your installation)
source /tools/Xilinx/Vitis_HLS/2022.2/settings64.sh

cd hls
# Runs the vitis HLS compiler flow
make csim
```

#### Option B: Standalone Compilation using GCC (With Vitis HLS Installed)
If you have Vitis HLS installed but want to compile directly using GCC by pointing to the Xilinx header directory:
```bash
cd hls
g++ -O3 -I/tools/Xilinx/Vitis_HLS/2022.2/include tb_conv.cpp tiled_conv.cpp utils.cpp -o sim
./sim
```

#### Option C: Standalone C++ Simulation (NO Vitis HLS/Vivado Installed)
If you do not have Vitis HLS/Vivado installed on your Linux machine, compile using standard GCC and standard float types by defining the `NO_HLS` macro:
```bash
cd hls
g++ -O3 -DNO_HLS tb_conv.cpp tiled_conv.cpp utils.cpp -o sim
./sim
```

---

### Step 4: Run HLS Synthesis Estimation
To run hardware synthesis and print the reports (utilization, clock, slack, and latency cycles) directly in the console:
```bash
# Ensure Vitis HLS settings are sourced
source /tools/Xilinx/Vitis_HLS/2022.2/settings64.sh

cd hls
make csynth
```
The command will run the synthesis script and print a formatted performance and resources summary directly to the terminal!

---

## 🧪 Verification & Validation

The functional validation compares the C++ HLS accelerator outputs directly with the PyTorch golden inference outputs:
$$\text{Mean Squared Error (MSE)} = \frac{1}{N} \sum_{i=1}^{N} (Y_{\text{golden\_i}} - Y_{\text{hls\_i}})^2$$

* If the code executes successfully, the output **MSE is 0**, indicating that the fixed-point C++ hardware accelerator behaves identically to the floating-point reference model on your test dataset.

---

## 📝 License & Contact

* **License**: This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
* **Author**: Aishik Tokdar
* **Email**: [aishiktokdar@gmail.com](mailto:aishiktokdar@gmail.com)
* **GitHub Repository**: [https://github.com/AishikTokdar/NeuroForge-HLS](https://github.com/AishikTokdar/NeuroForge-HLS)
