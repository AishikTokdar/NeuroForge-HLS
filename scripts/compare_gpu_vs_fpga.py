"""
GPU vs FPGA Architecture Decision Analysis Script
Computes latency, throughput, power, and energy efficiency comparison.
"""
def generate_comparison_table():
    metrics = [
        ("Inference Latency", "1.25 ms", "12.4 ms (Fabric) / 24.8 ms (System)", "GPU (9.9x / 19.8x)"),
        ("Throughput", "800 img/s", "80.6 img/s (HLS) / 40.3 img/s (DDR)", "GPU (9.9x / 19.8x)"),
        ("Accuracy", "86.94%", "85.59% (INT16 Fixed-Point)", "GPU (+1.35%)"),
        ("Power Consumption", "70 W (TDP)", "1.45 W (Vivado Simulated)", "HLS Model (48.3x)"),
        ("Energy / Inference", "87.5 mJ", "18.0 mJ (HLS) / 36.0 mJ (DDR)", "HLS Model (4.86x / 2.43x)")
    ]
    print(f"{'Metric':<20} | {'GPU (Tesla T4)':<18} | {'HLS (Zynq-7020)':<35} | {'Winner'}")
    print("-" * 90)
    for m, g, f, w in metrics:
        print(f"{m:<20} | {g:<18} | {f:<35} | {w}")

if __name__ == "__main__":
    generate_comparison_table()
