cd [file dirname [file normalize [info script]]]

set run_csim 1
set run_csynth 1
set run_cosim 0

if { [llength $argv] > 0 } {
    set mode [lindex $argv 0]
    if { $mode == "csim" } {
        set run_csim 1
        set run_csynth 0
    } elseif { $mode == "csynth" } {
        set run_csim 0
        set run_csynth 1
    } elseif { $mode == "cosim" } {
        set run_cosim 1
    } else {
        error "Unknown mode: $mode (expected csim, csynth, or cosim)"
    }
}

set hls_prj reduced_vgg.prj

open_project ${hls_prj} -reset

set_top reduced_vgg_inference

add_files utils.cpp
add_files tiled_conv.cpp
add_files -tb tb_conv.cpp
add_files -tb params

open_solution "solution1"

set_part {xc7z020clg400-1}

create_clock -period 15 -name default

config_interface -m_axi_latency 16
config_compile -pipeline_loops 64
config_schedule -effort high
config_dataflow -default_channel fifo -fifo_depth 2

if { ${run_csim} } {
    puts "INFO: Running C Simulation..."
    csim_design
}

if { ${run_csynth} } {
    puts "INFO: Running C Synthesis..."
    csynth_design

    export_design -format ip_catalog \
        -description "VGG CIFAR-10 Accelerator - Optimized" \
        -vendor "aishiktokdar.github.io" \
        -library "vgg" \
        -version "1.0" \
        -display_name "VGG_REDUCED_OPT"
}

if { ${run_cosim} } {
    cosim_design -rtl verilog
}

exit
