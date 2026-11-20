"""Benchmark host C++ inference; CPU timing cannot estimate FPGA clock cycles."""
import argparse
import re
import subprocess
import sys
from pathlib import Path
import numpy as np


def benchmark_simulation(runs=5):
    if runs < 1:
        raise ValueError("runs must be positive")
    hls_dir = Path(__file__).resolve().parents[1] / "hls"
    subprocess.run([sys.executable, str(hls_dir / "tb_standalone_runner.py")], check=True)
    executable = hls_dir / ("sim.exe" if sys.platform == "win32" else "sim")
    latencies = []
    for i in range(runs):
        result = subprocess.run([str(executable)], cwd=hls_dir, capture_output=True,
                                text=True, check=True, timeout=300)
        if "TEST PASSED" not in result.stdout:
            raise RuntimeError("Output validation did not pass")
        match = re.search(r"Simulation Time:\s+([0-9.]+) ms", result.stdout)
        if match is None:
            raise ValueError("Missing inference timing")
        latencies.append(float(match.group(1)))
        print(f"Host inference {i + 1}: {latencies[-1]:.3f} ms")
    print(f"Host mean: {np.mean(latencies):.3f} ms; minimum: {min(latencies):.3f} ms")
    return latencies


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    benchmark_simulation(parser.parse_args().runs)
