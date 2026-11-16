#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "tiled_conv.hpp"
#include "utils.hpp"

using namespace std;
using namespace std::chrono;

#define W1_SIZE (32 * 3 * 3 * 3)
#define B1_SIZE 32
#define BN1_SIZE 32
#define W2_SIZE (32 * 32 * 3 * 3)
#define B2_SIZE 32
#define BN2_SIZE 32

#define W3_SIZE (64 * 32 * 3 * 3)
#define B3_SIZE 64
#define BN3_SIZE 64
#define W4_SIZE (64 * 64 * 3 * 3)
#define B4_SIZE 64
#define BN4_SIZE 64

#define W5_SIZE (128 * 64 * 3 * 3)
#define B5_SIZE 128
#define BN5_SIZE 128
#define W6_SIZE (128 * 128 * 3 * 3)
#define B6_SIZE 128
#define BN6_SIZE 128

#define W7_SIZE (256 * 128 * 3 * 3)
#define B7_SIZE 256
#define BN7_SIZE 256
#define W8_SIZE (256 * 256 * 3 * 3)
#define B8_SIZE 256
#define BN8_SIZE 256

#define W_FC1_SIZE (256 * 1024)
#define B_FC1_SIZE 256
#define W_FC2_SIZE (10 * 256)
#define B_FC2_SIZE 10

#define INPUT_SIZE (3 * 32 * 32)
#define OUTPUT_SIZE 10

fm_t input_fm_mem[INPUT_SIZE];
fm_t output_fm_mem[OUTPUT_SIZE];
float output_golden_mem[OUTPUT_SIZE];

wt_t W_1_mem[W1_SIZE]; wt_t B_1_mem[B1_SIZE];
wt_t W_2_mem[W2_SIZE]; wt_t B_2_mem[B2_SIZE];
wt_t W_3_mem[W3_SIZE]; wt_t B_3_mem[B3_SIZE];
wt_t W_4_mem[W4_SIZE]; wt_t B_4_mem[B4_SIZE];
wt_t W_5_mem[W5_SIZE]; wt_t B_5_mem[B5_SIZE];
wt_t W_6_mem[W6_SIZE]; wt_t B_6_mem[B6_SIZE];
wt_t W_7_mem[W7_SIZE]; wt_t B_7_mem[B7_SIZE];
wt_t W_8_mem[W8_SIZE]; wt_t B_8_mem[B8_SIZE];

wt_t G_1_mem[BN1_SIZE], T_1_mem[BN1_SIZE], M_1_mem[BN1_SIZE], V_1_mem[BN1_SIZE];
wt_t G_2_mem[BN2_SIZE], T_2_mem[BN2_SIZE], M_2_mem[BN2_SIZE], V_2_mem[BN2_SIZE];
wt_t G_3_mem[BN3_SIZE], T_3_mem[BN3_SIZE], M_3_mem[BN3_SIZE], V_3_mem[BN3_SIZE];
wt_t G_4_mem[BN4_SIZE], T_4_mem[BN4_SIZE], M_4_mem[BN4_SIZE], V_4_mem[BN4_SIZE];
wt_t G_5_mem[BN5_SIZE], T_5_mem[BN5_SIZE], M_5_mem[BN5_SIZE], V_5_mem[BN5_SIZE];
wt_t G_6_mem[BN6_SIZE], T_6_mem[BN6_SIZE], M_6_mem[BN6_SIZE], V_6_mem[BN6_SIZE];
wt_t G_7_mem[BN7_SIZE], T_7_mem[BN7_SIZE], M_7_mem[BN7_SIZE], V_7_mem[BN7_SIZE];
wt_t G_8_mem[BN8_SIZE], T_8_mem[BN8_SIZE], M_8_mem[BN8_SIZE], V_8_mem[BN8_SIZE];

wt_t W_FC1_mem[W_FC1_SIZE]; wt_t B_FC1_mem[B_FC1_SIZE];
wt_t W_FC2_mem[W_FC2_SIZE]; wt_t B_FC2_mem[B_FC2_SIZE];

template <typename T, size_t N>
void read_file(const char *path, T (&buffer)[N]) {
    ifstream file(path, ios::in | ios::binary);

    if (!file.is_open()) {
        std::string alt = std::string("../../../../") + path;
        file.open(alt.c_str(), ios::in | ios::binary);
    }

    if (!file.is_open()) {
        cerr << "ERROR: Could not open file: " << path << endl;
        exit(1);
    }

    std::string filepath(path);
    const size_t element_bytes = filepath.find("fp32") != std::string::npos ? 4 : 2;
    file.seekg(0, ios::end);
    if (file.tellg() != static_cast<std::streamoff>(N * element_bytes)) {
        cerr << "ERROR: Incorrect binary file size: " << path << endl;
        exit(1);
    }
    file.seekg(0, ios::beg);
    if (filepath.find("fp32") != std::string::npos) {
        std::vector<float> temp(N);
        file.read(reinterpret_cast<char *>(temp.data()), N * sizeof(float));
        for (size_t i = 0; i < N; i++) {
            if (!std::isfinite(temp[i])) {
                cerr << "ERROR: Nonfinite FP32 test data: " << path << endl;
                exit(1);
            }
            buffer[i] = (T)temp[i];
        }
    } else {
        std::vector<int16_t> temp(N);
        file.read(reinterpret_cast<char *>(temp.data()), N * sizeof(int16_t));
        for (size_t i = 0; i < N; i++) {
            buffer[i] = (T)(temp[i] / 16.0f);
        }
    }
    if (!file) {
        cerr << "ERROR: Could not read complete file: " << path << endl;
        exit(1);
    }
    file.close();
}

void read_bin_files()
{
    read_file("params/input_fp32.bin", input_fm_mem);
    read_file("params/output_golden_fp32.bin", output_golden_mem);

    read_file("params/features_0_block_0_weight.bin", W_1_mem); read_file("params/features_0_block_0_bias.bin", B_1_mem);
    read_file("params/features_0_block_1_weight.bin", G_1_mem); read_file("params/features_0_block_1_bias.bin", T_1_mem);
    read_file("params/features_0_block_1_running_mean.bin", M_1_mem); read_file("params/features_0_block_1_running_var.bin", V_1_mem);

    read_file("params/features_0_block_3_weight.bin", W_2_mem); read_file("params/features_0_block_3_bias.bin", B_2_mem);
    read_file("params/features_0_block_4_weight.bin", G_2_mem); read_file("params/features_0_block_4_bias.bin", T_2_mem);
    read_file("params/features_0_block_4_running_mean.bin", M_2_mem); read_file("params/features_0_block_4_running_var.bin", V_2_mem);

    read_file("params/features_1_block_0_weight.bin", W_3_mem); read_file("params/features_1_block_0_bias.bin", B_3_mem);
    read_file("params/features_1_block_1_weight.bin", G_3_mem); read_file("params/features_1_block_1_bias.bin", T_3_mem);
    read_file("params/features_1_block_1_running_mean.bin", M_3_mem); read_file("params/features_1_block_1_running_var.bin", V_3_mem);

    read_file("params/features_1_block_3_weight.bin", W_4_mem); read_file("params/features_1_block_3_bias.bin", B_4_mem);
    read_file("params/features_1_block_4_weight.bin", G_4_mem); read_file("params/features_1_block_4_bias.bin", T_4_mem);
    read_file("params/features_1_block_4_running_mean.bin", M_4_mem); read_file("params/features_1_block_4_running_var.bin", V_4_mem);

    read_file("params/features_2_block_0_weight.bin", W_5_mem); read_file("params/features_2_block_0_bias.bin", B_5_mem);
    read_file("params/features_2_block_1_weight.bin", G_5_mem); read_file("params/features_2_block_1_bias.bin", T_5_mem);
    read_file("params/features_2_block_1_running_mean.bin", M_5_mem); read_file("params/features_2_block_1_running_var.bin", V_5_mem);

    read_file("params/features_2_block_3_weight.bin", W_6_mem); read_file("params/features_2_block_3_bias.bin", B_6_mem);
    read_file("params/features_2_block_4_weight.bin", G_6_mem); read_file("params/features_2_block_4_bias.bin", T_6_mem);
    read_file("params/features_2_block_4_running_mean.bin", M_6_mem); read_file("params/features_2_block_4_running_var.bin", V_6_mem);

    read_file("params/features_3_block_0_weight.bin", W_7_mem); read_file("params/features_3_block_0_bias.bin", B_7_mem);
    read_file("params/features_3_block_1_weight.bin", G_7_mem); read_file("params/features_3_block_1_bias.bin", T_7_mem);
    read_file("params/features_3_block_1_running_mean.bin", M_7_mem); read_file("params/features_3_block_1_running_var.bin", V_7_mem);

    read_file("params/features_3_block_3_weight.bin", W_8_mem); read_file("params/features_3_block_3_bias.bin", B_8_mem);
    read_file("params/features_3_block_4_weight.bin", G_8_mem); read_file("params/features_3_block_4_bias.bin", T_8_mem);
    read_file("params/features_3_block_4_running_mean.bin", M_8_mem); read_file("params/features_3_block_4_running_var.bin", V_8_mem);

    read_file("params/classifier_1_weight.bin", W_FC1_mem);
    read_file("params/classifier_1_bias.bin", B_FC1_mem);
    read_file("params/classifier_4_weight.bin", W_FC2_mem);
    read_file("params/classifier_4_bias.bin", B_FC2_mem);

    std::cout << "All model parameters and test vectors loaded successfully." << std::endl;
}

int main()
{
    read_bin_files();

    if (std::all_of(W_1_mem, W_1_mem + W1_SIZE, [](wt_t x) { return x == 0; }) ||
        std::all_of(W_2_mem, W_2_mem + W2_SIZE, [](wt_t x) { return x == 0; }) ||
        std::all_of(W_3_mem, W_3_mem + W3_SIZE, [](wt_t x) { return x == 0; }) ||
        std::all_of(W_4_mem, W_4_mem + W4_SIZE, [](wt_t x) { return x == 0; }) ||
        std::all_of(W_5_mem, W_5_mem + W5_SIZE, [](wt_t x) { return x == 0; }) ||
        std::all_of(W_6_mem, W_6_mem + W6_SIZE, [](wt_t x) { return x == 0; }) ||
        std::all_of(W_7_mem, W_7_mem + W7_SIZE, [](wt_t x) { return x == 0; }) ||
        std::all_of(W_8_mem, W_8_mem + W8_SIZE, [](wt_t x) { return x == 0; }) ||
        std::all_of(W_FC1_mem, W_FC1_mem + W_FC1_SIZE, [](wt_t x) { return x == 0; }) ||
        std::all_of(W_FC2_mem, W_FC2_mem + W_FC2_SIZE, [](wt_t x) { return x == 0; })) {
        cerr << "ERROR: Degenerate all-zero model weights; inference validation is invalid." << endl;
        return 1;
    }

    std::cout << "Beginning Reduced VGG Inference Simulation (4-Block VGG)..." << std::endl;
    auto start = high_resolution_clock::now();

    reduced_vgg_inference(
        input_fm_mem,
        W_1_mem, B_1_mem, G_1_mem, T_1_mem, M_1_mem, V_1_mem,
        W_2_mem, B_2_mem, G_2_mem, T_2_mem, M_2_mem, V_2_mem,
        W_3_mem, B_3_mem, G_3_mem, T_3_mem, M_3_mem, V_3_mem,
        W_4_mem, B_4_mem, G_4_mem, T_4_mem, M_4_mem, V_4_mem,
        W_5_mem, B_5_mem, G_5_mem, T_5_mem, M_5_mem, V_5_mem,
        W_6_mem, B_6_mem, G_6_mem, T_6_mem, M_6_mem, V_6_mem,
        W_7_mem, B_7_mem, G_7_mem, T_7_mem, M_7_mem, V_7_mem,
        W_8_mem, B_8_mem, G_8_mem, T_8_mem, M_8_mem, V_8_mem,
        W_FC1_mem, B_FC1_mem, W_FC2_mem, B_FC2_mem,
        output_fm_mem);

    auto end = high_resolution_clock::now();
    double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "========================================" << std::endl;
    std::cout << "Inference Simulation Complete" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Simulation Time: " << duration_ms << " ms" << std::endl;

    long double mse = 0.0;
    int error_count = 0;

    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        float expected = output_golden_mem[i];
        float actual = (float)output_fm_mem[i];

        mse += std::pow((long double)(expected - actual), 2.0);

        if (!std::isfinite(actual) || !std::isfinite(expected) ||
            std::abs(expected - actual) > 0.1f) {
            error_count++;
        }
    }

    mse = mse / OUTPUT_SIZE;

    std::cout << "\n========================================" << std::endl;
    std::cout << "Performance Metrics" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "MSE:              " << mse << std::endl;
    std::cout << "Error count:      " << error_count << "/" << OUTPUT_SIZE << std::endl;
    std::cout << "Dataset accuracy: not measured by this single-vector test" << std::endl;
#ifdef NO_HLS
    std::cout << "Arithmetic:       host floating point (not bit-accurate RTL)" << std::endl;
#else
    std::cout << "Arithmetic:       ap_fixed C simulation" << std::endl;
#endif
    std::cout << "Parameters:       1,439,146" << std::endl;

    std::cout << "\n========================================" << std::endl;
    bool passed = std::isfinite((double)mse) && error_count == 0;
    if (passed)
    {
        std::cout << "TEST PASSED - Simulation SUCCESSFUL!" << std::endl;
    }
    else
    {
        std::cout << "TEST FAILED" << std::endl;
    }
    std::cout << "========================================" << std::endl;

    return passed ? 0 : 1;
}
