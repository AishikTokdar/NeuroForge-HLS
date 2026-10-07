"""
Performance Profiling Script for NeuroForge-HLS Simulation
Runs multiple iterations and logs execution metrics.
"""
import os
import subprocess
import time
import numpy as np

def benchmark_simulation(runs=5):
    hls_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hls")
    print(f"Benchmarking GCC simulation over {runs} iterations...")
    latencies = []
    for i in range(runs):
        t0 = time.time()
        p = subprocess.run(["make", "sim_gcc"], cwd=hls_dir, capture_output=True, text=True)
        dt = (time.time() - t0) * 1000.0
        if p.returncode == 0:
            latencies.append(dt)
            print(f"Run {i+1}: {dt:.2f} ms")
    if latencies:
        mean_l = np.mean(latencies)
        min_l = np.min(latencies)
        print(f"Mean Latency: {mean_l:.2f} ms | Min: {min_l:.2f} ms")
        print(f"Estimated Hardware Clock Cycles (@ 66.7MHz): {int(mean_l * 66700)}")

if __name__ == "__main__":
    benchmark_simulation()
