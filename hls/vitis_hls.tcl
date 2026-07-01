# Optimized Vitis HLS TCL Script for VGG Accelerator
# Focus: Maximum throughput with aggressive optimizations

# Determine which stages to run (default to both if no args are passed)
set run_csim 1
set run_csynth 1

if { [llength $argv] > 0 } {
    set mode [lindex $argv 0]
    if { $mode == "csim" } {
        set run_csim 1
        set run_csynth 0
    } elseif { $mode == "csynth" } {
        set run_csim 0
        set run_csynth 1
    }
}

# Set project name
set hls_prj reduced_vgg.prj

# Open/reset the project
open_project ${hls_prj} -reset

# Set top module
set_top reduced_vgg_inference

# Add design and testbench files
add_files utils.cpp
add_files tiled_conv.cpp
add_files -tb tb_conv.cpp

# Open solution
open_solution "solution1"

# Set target device (Zynq-7020)
set_part {xc7z020clg400-1}

# Clock period: 15ns (66.7 MHz) - More relaxed for better optimization
create_clock -period 15 -name default

# ============================================
# CRITICAL OPTIMIZATION SETTINGS
# ============================================

# Reduce AXI latency (was 64, now 16)
config_interface -m_axi_latency 16

# Enable aggressive optimization
config_compile -pipeline_loops 64

# Enable more aggressive scheduling
config_schedule -effort high

# Enable automatic dataflow detection
config_dataflow -default_channel fifo -fifo_depth 2

# ============================================
# Run C simulation if requested
# ============================================
if { ${run_csim} } {
    puts "INFO: Running C Simulation..."
    csim_design
}

# ============================================
# Run C synthesis if requested
# ============================================
if { ${run_csynth} } {
    puts "INFO: Running C Synthesis..."
    csynth_design

    # Export IP for Vivado
    export_design -format ip_catalog \
        -description "VGG CIFAR-10 Accelerator - Optimized" \
        -vendor "aishiktokdar.github.io" \
        -library "vgg" \
        -version "1.0" \
        -display_name "VGG_REDUCED_OPT"
}

exit
