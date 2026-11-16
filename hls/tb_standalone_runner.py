"""
Standalone C++ Simulation Harness & Log Parser
Automates compiling and running functional testbench without proprietary tools.
"""
import os
import subprocess
import time
import re

def run_standalone_test():
    hls_dir = os.path.dirname(os.path.abspath(__file__))
    compile_cmd = [
        "g++", "-std=c++11", "-O3", "-DNO_HLS",
        "tb_conv.cpp", "tiled_conv.cpp", "utils.cpp",
        "-o", "sim"
    ]
    print("Compiling standalone HLS testbench with GCC (-O3 -DNO_HLS)...")
    t0 = time.time()
    subprocess.run(compile_cmd, cwd=hls_dir, check=True, timeout=120)
    compile_time = time.time() - t0
    print(f"Compilation succeeded in {compile_time:.2f}s.")
    
    print("Executing host floating-point functional simulation...")
    t0 = time.time()
    exec_name = os.path.join(hls_dir, "sim.exe" if os.name == "nt" else "sim")
    res = subprocess.run([exec_name], cwd=hls_dir, capture_output=True, text=True, timeout=300)
    sim_time = time.time() - t0
    print(res.stdout)
    if res.stderr:
        print(res.stderr)
    res.check_returncode()
    if "TEST PASSED" not in res.stdout:
        raise RuntimeError("Testbench did not report successful output validation")
    
    mse_match = re.search(r"MSE:\s+([0-9.e+-]+)", res.stdout)
    err_match = re.search(r"Error count:\s+(\d+)/(\d+)", res.stdout)
    if mse_match and err_match:
        print(f"Summary: MSE={mse_match.group(1)}, Errors={err_match.group(1)}/{err_match.group(2)}")
    print(f"Simulation completed in {sim_time:.2f}s. All output vectors verified.")

if __name__ == "__main__":
    run_standalone_test()
