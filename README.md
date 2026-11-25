# NeuroForge-HLS

ReducedVGG inference in C++ for High-Level Synthesis (HLS), with PyTorch training, comprehensive classification analysis, and verified software simulation on CIFAR-10.

---

## Table of Contents

- [Project Status](#project-status)
- [Software Verification and Findings](#software-verification-and-findings)
- [Architecture and Arithmetic](#architecture-and-arithmetic)
- [Weight Recovery and Retraining](#weight-recovery-and-retraining)
- [Classification Metrics and Diagnostics](#classification-metrics-and-diagnostics)
- [Numeric-Range Audit](#numeric-range-audit)
- [HLS and RTL Synthesis Flows](#hls-and-rtl-synthesis-flows)
- [Reproduction and Verification Guide](#reproduction-and-verification-guide)
- [Project Website and GitHub Pages](#project-website-and-github-pages)
- [License and Author](#license-and-author)

---

## Project Status

**Ready for software simulation and demonstration:**
- **72 automated tests pass** across unit testing, operator parity, full inference, packaging, and validation contracts.
- **Standalone C++ testbench passes** with MSE **2.09894e-12** and **0/10 errors** against genuine nonzero golden logits.
- **All 10,000 CIFAR-10 test predictions match PyTorch** with 100% classification agreement and a maximum absolute logit difference of **0.0000190735**.
- **FPGA deployment status:** Vendor fixed-point simulation, Vitis HLS synthesis, RTL co-simulation, routed timing closure, and physical board/system integration remain pending vendor toolchain availability.

**Zero-Weight Issue Resolved:**
The initial repository exports contained zeroed weight arrays resulting from an integer-cast export flaw. A replacement ReducedVGG model was trained using the `tf-gpu` environment and an NVIDIA RTX 3050 laptop GPU. Both floating-point and quantization-aware checkpoints, validated little-endian Q16.16 parameters, converted Q12.4 parameters, a real-image golden vector, and verified test predictions are bundled directly in the repository.

---

## Software Verification and Findings

### Verification Scope and Verdict

- **Software inference and replacement trained weights are verified.** The C++ host implementation executes the ReducedVGG network and matches PyTorch numerical behavior to floating-point precision.
- **Hardware synthesis and physical operation remain unverified.** The GCC `NO_HLS` build uses floating-point arithmetic to verify control logic, data movement, tiling, and layer sequencing; it does not simulate AMD fixed-point truncation, saturation, AXI bus handshakes, or hardware timing closure.

### Key Issues Identified and Corrected

| Severity | Component | Finding | Correction and Verification |
| :--- | :--- | :--- | :--- |
| **Critical** | Weight Exports | The original eight convolution and two fully connected weight arrays contained only zeros; the golden test vector had zero logits. | Retrained replacement ReducedVGG on CIFAR-10. Generated genuine FP32, Q16.16, and Q12.4 binaries. Golden vector updated to a real test image with nonzero logits. |
| **High** | Network Topology | The final 2x2 max pool was missing before FC1. FC1 read only the first 1,024 elements of an unpooled 4,096-element buffer. | Inserted `pool4` and corrected buffer addressing to read the proper 256x2x2 feature map. Tests verify layer-by-layer dimension alignment. |
| **High** | Parameter Export | Notebook exported raw FP32 bytes into `params_int32/`, while loaders interpreted them as scaled integers. | Enforced canonical little-endian FP32 export to `params_fp32/` and true integer scaling (factor 65,536) for `params_int32/`. Export manifest records SHA-256 hashes. |
| **High** | BatchNorm Precision | BatchNorm $\epsilon = 10^{-5}$ vanished in the accumulator due to insufficient fractional bits, risking division by zero. | Defined intermediate BatchNorm type with 24 fractional bits (`ap_fixed<48,24>`), preventing epsilon collapse. Host tests exercise zero and small variance. |
| **High** | C++ Testbench | Testbench returned status `0` even when printing `TEST FAILED`, accepted nonfinite values, and checked only 2 of 10 weight matrices. | Replaced with strict validation: nonzero exit codes on error, finite check on all logits, file size validation, and all 10 weight matrices checked for corruption. |
| **High** | Fixed-Point Loader | Loader aliased raw `int16_t` buffers into `ap_fixed` objects without proper arithmetic decoding. | Replaced object aliasing with exact numeric Q12.4 fixed-point decoding. |
| **High** | Verification Scripts | `parse_yosys_report.py` and `verify_quantization.py` printed hardcoded dummy metrics without reading input files. | Rewrote both utilities to parse real JSON/binary data, compute actual SQNR/MSE, and fail gracefully on corrupt or missing inputs. |
| **High** | Tiling & Boundary | Convolution loop over-read memory on partial channel groups; uninitialized input halos occurred with zero padding. | Added explicit masking on channel tails, complete 3x3 halo loading, and runtime dimension assertions. |
| **Medium** | Path Resolution | Build and test scripts assumed the current working directory was the repository root. | Updated scripts to resolve paths relative to their own source location. |
| **Medium** | Benchmark Timing | Simulation profiler included compiler startup and process invocation in latency measurements. | Refactored profiler to measure pure execution latency over multiple warm iterations. |

---

## Architecture and Arithmetic

### ReducedVGG Layer Specification

The network processes CIFAR-10 images ($3 \times 32 \times 32$, RGB) using standard fixed CIFAR normalization (mean: `[0.4914, 0.4822, 0.4465]`, std: `[0.2470, 0.2435, 0.2616]`):

1. **Block 1 (Channels: 32):**
   - Conv1_1 ($3 \times 3$, pad 1, stride 1) $\rightarrow$ BN $\rightarrow$ ReLU
   - Conv1_2 ($3 \times 3$, pad 1, stride 1) $\rightarrow$ BN $\rightarrow$ ReLU
   - MaxPool1 ($2 \times 2$, stride 2) $\rightarrow$ Feature map: $32 \times 16 \times 16$
2. **Block 2 (Channels: 64):**
   - Conv2_1 ($3 \times 3$, pad 1, stride 1) $\rightarrow$ BN $\rightarrow$ ReLU
   - Conv2_2 ($3 \times 3$, pad 1, stride 1) $\rightarrow$ BN $\rightarrow$ ReLU
   - MaxPool2 ($2 \times 2$, stride 2) $\rightarrow$ Feature map: $64 \times 8 \times 8$
3. **Block 3 (Channels: 128):**
   - Conv3_1 ($3 \times 3$, pad 1, stride 1) $\rightarrow$ BN $\rightarrow$ ReLU
   - Conv3_2 ($3 \times 3$, pad 1, stride 1) $\rightarrow$ BN $\rightarrow$ ReLU
   - MaxPool3 ($2 \times 2$, stride 2) $\rightarrow$ Feature map: $128 \times 4 \times 4$
4. **Block 4 (Channels: 256):**
   - Conv4_1 ($3 \times 3$, pad 1, stride 1) $\rightarrow$ BN $\rightarrow$ ReLU
   - Conv4_2 ($3 \times 3$, pad 1, stride 1) $\rightarrow$ BN $\rightarrow$ ReLU
   - MaxPool4 ($2 \times 2$, stride 2) $\rightarrow$ Feature map: $256 \times 2 \times 2$ (1,024 activations)
5. **Classifier:**
   - FC1 ($1024 \rightarrow 256$) $\rightarrow$ ReLU
   - FC2 ($256 \rightarrow 10$) $\rightarrow$ 10 output class logits (inference disables dropout)

### Tiling and Memory Strategy

- **Ping-Pong Buffers:** Two full-size feature-map buffers alternate across sequential layer calls (max storage: $2 \times 32,768$ 16-bit elements = 128 KiB).
- **Spatial Tiling:** Tiled convolution processes up to $8 \times 8$ output spatial tiles with a $10 \times 10$ input halo buffer (50 KiB) to support $3 \times 3$ kernels.
- **Channel Chunking:** Output channels are computed in groups of 8 (weight tile: $8 \times 256 \times 3 \times 3$ elements = 36 KiB).
- **Hardware Types:**
  - Feature maps & weights: `ap_fixed<16, 12>` (Q12.4 signed: 12 integer bits including sign, 4 fractional bits).
  - Accumulators: `ap_fixed<32, 24>` (24 integer bits, 8 fractional bits; max magnitude: 8,388,608).
  - BatchNorm intermediates: `ap_fixed<48, 24>` (preserves epsilon resolution).

---

## Weight Recovery and Retraining

### Training Procedure

- **Dataset:** Official CIFAR-10 (45,000 training, 5,000 validation, 10,000 test images).
- **Optimization:** AdamW optimizer with cosine learning-rate decay; training data augmented with random horizontal flips and $32 \times 32$ crops (padding 4).
- **Schedule:** 20 floating-point epochs followed by 5 quantization-aware fine-tuning (QAT) epochs (batch size 128).
- **QAT Modeling:** Simulates nearest-even rounding on Q12.4 parameters and floor rounding on Q12.4 activations, freezing batch normalization statistics during fine-tuning.
- **Model Checkpoints:** Selected strictly on validation accuracy; the official 10,000-image test split remained completely untouched during training.

### Accuracy Progression Across Representations

| Representation / Execution Mode | Evaluated Split | Accuracy | Notes |
| :--- | :--- | :---: | :--- |
| **PyTorch FP32 Baseline** | Official 10k Test | **88.50%** | Independent floating-point master checkpoint (`weights/reduced_vgg_fp32.pth`) |
| **QAT Checkpoint (Approximate Q12.4)** | Official 10k Test | **87.83%** | Software Q12.4 activation and weight simulation model |
| **Exported Q12.4 in PyTorch** | Official 10k Test | **87.26%** | Loaded from exported binary parameters; evaluates exact Q12.4 weights |
| **Exported Q12.4 in C++ (NO_HLS)** | Official 10k Test | **87.26%** | **Actual host C++ implementation**; 100% agreement with PyTorch across all 10,000 images |

### Parameter Artifacts in the Repository

- `weights/reduced_vgg_retrained.pth`: Primary quantization-aware trained PyTorch checkpoint.
- `weights/reduced_vgg_fp32.pth`: Baseline floating-point PyTorch checkpoint (88.50%).
- `weights/params_fp32/`: Full-precision IEEE-754 binary dumps of all learned parameters.
- `weights/params_int32/`: Genuine Q16.16 scaled integer binaries (scale factor 65,536).
- `hls/params/` and `weights/params/`: Converted Q12.4 INT16 binaries used directly by the C++ testbench and HLS inference engine.
- `weights/export_manifest.json`: Cryptographic SHA-256 hashes verifying checkpoint and parameter file integrity.

---

## Classification Metrics and Diagnostics

The classification performance was evaluated on all 10,000 official CIFAR-10 test images using the C++ host classifier:

### Summary Performance

| Metric | Measured Value | Notes |
| :--- | :---: | :--- |
| **Top-1 Accuracy** | **87.26%** | 8,726 correct, 1,274 incorrect (95% Wilson Interval: 86.59%–87.90%) |
| **Macro Precision** | **87.18%** | Unweighted average over all 10 classes |
| **Macro Recall** | **87.26%** | Equal to accuracy due to balanced 1,000 images/class |
| **Macro F1 Score** | **87.18%** | Harmonic mean of precision and recall |
| **Balanced Accuracy** | **87.26%** | Balanced across support |
| **Matthews Corr. Coeff. (MCC)** | **0.858532** | High correlation between predictions and ground truth |
| **Cohen's Kappa** | **0.858444** | Inter-rater agreement over chance |
| **Top-3 Accuracy** | **98.00%** | True class ranked in top 3 predictions |
| **Top-5 Accuracy** | **99.48%** | True class ranked in top 5 predictions |
| **Macro ROC-AUC (OvR)** | **0.990274** | One-vs-rest area under receiver operating characteristic |
| **Macro Average Precision (OvR)** | **0.941763** | Area under the precision-recall curve |
| **Mean Reciprocal Rank (MRR)** | **0.926421** | Average multiplicative inverse of true rank |

### Per-Class Breakdown

| Class | Support | Precision | Recall | F1 Score | Primary Confusions |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **airplane** | 1,000 | 85.51% | 90.30% | 87.84% | Confused with ship (32) and bird (27) |
| **automobile**| 1,000 | 90.20% | **96.60%** | 93.29% | Highest class recall; confused with truck (24) |
| **bird** | 1,000 | 85.33% | 79.70% | 82.42% | Confused with airplane (46) and frog (43) |
| **cat** | 1,000 | 76.38% | **72.10%** | 74.18% | Lowest class recall; confused with dog (107) |
| **deer** | 1,000 | 88.13% | 86.10% | 87.10% | Confused with horse (41) and bird (39) |
| **dog** | 1,000 | 81.59% | 81.10% | 81.34% | Confused with cat (103) |
| **frog** | 1,000 | 88.91% | 92.20% | 90.53% | Confused with cat (24) and bird (22) |
| **horse** | 1,000 | 90.53% | 89.90% | 90.22% | Confused with deer (39) and dog (26) |
| **ship** | 1,000 | 94.07% | 92.00% | 93.02% | Confused with airplane (44) |
| **truck** | 1,000 | 91.14% | 92.60% | 91.87% | Confused with automobile (50) |

### Confidence Threshold Analysis (Selective Classification)

Accepting predictions only when the maximum softmax probability exceeds a confidence threshold produces a tunable precision/coverage tradeoff:

| Confidence Gate | Test Coverage | Accepted Accuracy | Macro Recall (Rejected = Error) | Images Rejected |
| :---: | :---: | :---: | :---: | :---: |
| **$\ge$ 0%** | 100.00% | 87.26% | 87.26% | 0 |
| **$\ge$ 50%** | 95.61% | 89.57% | 85.64% | 439 |
| **$\ge$ 70%** | 87.16% | 92.84% | 80.92% | 1,284 |
| **$\ge$ 80%** | 82.36% | 94.49% | 77.82% | 1,764 |
| **$\ge$ 90%** | 76.02% | **96.25%** | 73.17% | 2,398 |
| **$\ge$ 95%** | 70.60% | **97.35%** | 68.73% | 2,940 |
| **$\ge$ 99%** | 59.93% | **98.67%** | 59.13% | 4,007 |

### Calibration Diagnostics

- **Negative Log-Likelihood (NLL):** 0.407236
- **Multiclass Brier Score:** 0.188036
- **Expected Calibration Error (ECE, 15 bins):** 3.98%
- **Mean Prediction Confidence:** 91.20%

---

## Numeric-Range Audit

To confirm that Q12.4 activations and partial sums do not suffer catastrophic overflow or truncation in fixed-point registers, an extensive range audit was conducted on the full test set:

- **Total Checked Values:** **2,490,980,000** values examined across input layers, convolutions, batch normalization, activations, and fully connected operations.
- **Observed Q12.4 Cast Overflows:** **0 overflows**.
- **Maximum Observed Partial-Sum Bound:** **99.26171875**.
- **Hardware Accumulator Capacity:** **8,388,608** (`ap_fixed<32,24>`), providing a safety margin of over $84,500\times$.

Per-layer bounds and statistics are archived in `docs/numeric_range_audit.json`.

---

## HLS and RTL Synthesis Flows

### Vendor Flow (AMD Vitis / Vivado)

From the `hls/` directory:

```bash
# C Simulation (compiles and runs testbench with Vitis ap_fixed types)
make csim

# C Synthesis (generates RTL from C++ source using target device constraints)
make csynth

# C/RTL Co-simulation (verifies generated RTL against C testbench)
make cosim
```

*Note: Requires Vitis HLS and Vivado installed on PATH. The target part is Xilinx Zynq-7000 `xc7z020clg400-1` with a 15 ns clock constraint.*

### Experimental Open-Source Synthesis Flow

For environments without commercial EDA tools, the repository supports an open-source synthesis flow using Bambu HLS and Yosys:

```bash
# Generate Verilog RTL using Bambu High-Level Synthesis
make bambu

# Run technology mapping and cell synthesis using Yosys
make yosys

# Parse synthesized resource metrics (LUTs, registers, DSPs)
python parse_yosys_report.py yosys_stats.json
```

*Note: The open-source Bambu flow operates on floating-point C++ and is intended for structural analysis and toolchain exploration.*

---

## Reproduction and Verification Guide

### 1. Run the Full Test Suite

Install testing dependencies and run the complete test suite:

```powershell
python -m pip install -r requirements-test.txt
python -m pytest tests -q
```

### 2. Standalone C++ Simulation

Compile and execute the standalone C++ simulation testbench:

```powershell
python hls/tb_standalone_runner.py
```

*Compiles `hls/tb_conv.cpp`, `hls/tiled_conv.cpp`, and `hls/utils.cpp` with GCC `-O3 -DNO_HLS`, loads converted weights and input test vectors, and verifies logits against golden outputs.*

### 3. Verify Quantization Fidelity

Validate parameter files and compute SQNR across all 52 weight and bias tensors:

```powershell
python weights/verify_quantization.py
```

### 4. Regenerate Classification Reports

Recompute classification metrics, per-class tables, and diagnostic plots from the saved 10,000 test predictions:

```powershell
python scripts/classification_metrics.py
```

*Outputs JSON (`docs/classification_metrics.json`), CSV (`docs/classification_metrics.csv`), and publication-ready plots (`docs/classification_metrics.png`, `docs/classification_metrics.svg`).*

### 5. Full GPU Re-Verification (Optional)

If PyTorch with CUDA is available, verify all 10,000 predictions against PyTorch in parallel:

```powershell
conda run -n tf-gpu python scripts/validate_export.py --device cuda
```

### 6. Model Retraining and Export

To retrain the ReducedVGG network from scratch:

```powershell
conda run -n tf-gpu python training/train_replacement.py --device cuda --epochs 20 --qat-epochs 5
```

To re-export and convert parameters from a PyTorch checkpoint:

```powershell
python weights/export_checkpoint.py weights/reduced_vgg_retrained.pth
python weights/convert_weights.py
python weights/convert_weights.py --output weights/params
python weights/verify_quantization.py
```

---

## Project Website and GitHub Pages

The repository includes a modern, responsive project website and interactive visualization suite deployed via GitHub Pages:

- **Live Routes:**
  - `index.html`: Project landing page, architecture overview, interactive tiling workbench, and verification summary.
  - `visualizations/viz_performance_comparison.html`: Interactive classification metrics, sortable class recall, and confusion matrix.
  - `visualizations/viz_memory_architecture.html`: Interactive tile/halo visualizer and ping-pong memory allocation explorer.
  - `visualizations/viz_decision_framework.html`: Verification scope advisor and toolchain decision guide.
  - `visualizations/viz_resource_utilization.html`: Numeric-range audit inspector and partial-sum distribution explorer.
  - `visualizations/viz_design_workflow.html`: End-to-end design flow from PyTorch training to HLS synthesis.

### Zero-Dependency Architecture

- **No Runtime CDN Dependencies:** Uses local typography (`Georgia` headings, system sans-serif, and monospace accents), custom SVG renderers, and zero external font or script CDNs.
- **Theme Support:** Native light/dark theme toggle with system preference detection and local storage persistence (`assets/theme.js`).
- **Verified Build Pipeline:** `scripts/build_pages.py` stages public-only assets into a self-contained directory, validating all local references, JSON data schemas, and image paths before deployment.

### Validate Website Locally

```powershell
python scripts/build_pages.py --output .pages-staging
node tests/audit_pages.cjs .pages-staging
python -m unittest discover -s tests -p test_pages.py -v
```

---

## License and Author

- **Author:** Aishik Tokdar ([GitHub Profile](https://github.com/AishikTokdar))
- **License:** Licensed under the [MIT License](LICENSE).
