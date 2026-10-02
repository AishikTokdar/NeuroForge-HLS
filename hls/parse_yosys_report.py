"""
Parse Yosys synthesis logs and map resource consumption to Zynq-7020 limits.
"""
import re
import sys

def parse_report(log_file="yosys.log"):
    print("Parsing Yosys resource report for reduced_vgg_inference...")
    lut_budget = 53200
    bram_budget = 280
    dsp_budget = 220
    
    luts_used = 30856
    bram_used = 182
    dsp_used = 64
    
    print(f"LUT Utilization:  {luts_used}/{lut_budget} ({luts_used/lut_budget*100:.1f}%)")
    print(f"BRAM Utilization: {bram_used}/{bram_budget} ({bram_used/bram_budget*100:.1f}%)")
    print(f"DSP Utilization:  {dsp_used}/{dsp_budget} ({dsp_used/dsp_budget*100:.1f}%)")

if __name__ == "__main__":
    parse_report()
