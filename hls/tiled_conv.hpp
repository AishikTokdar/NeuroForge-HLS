#ifndef __TILED_CONV_HPP__
#define __TILED_CONV_HPP__

#ifdef NO_HLS
#include <cmath>
typedef float fm_t, wt_t;
typedef float acc_t;
typedef double bn_t;
namespace hls {
    template <typename T>
    T sqrt(T val) {
        return std::sqrt(val);
    }
}
#else
#include <ap_fixed.h>
#include <hls_math.h>
typedef ap_fixed<16, 12> fm_t, wt_t;
typedef ap_fixed<32, 24> acc_t;
typedef ap_fixed<48, 24> bn_t;
#endif

#define K 3
#define FC_IN_SIZE 1024
#define FC1_OUT_SIZE 256
#define FC_OUT_SIZE 10

#define MAX_TILE_HW 8
#define MAX_MARGIN 1
#define MAX_IN_BUF_HW (MAX_TILE_HW + 2 * MAX_MARGIN)
#define MAX_OUT_BUF_DEPTH 8
#define MAX_M 256

#define BN_EPSILON 1e-5f
#define FM_BUF_SIZE 32768

#define GET_3D_ADDR(m, r, c, width) ((m) * (width) * (width) + (r) * (width) + (c))

void reduced_vgg_inference(
	fm_t *input_fm,
	wt_t *W_1, wt_t *B_1, wt_t *G_1, wt_t *T_1, wt_t *M_1, wt_t *V_1,
	wt_t *W_2, wt_t *B_2, wt_t *G_2, wt_t *T_2, wt_t *M_2, wt_t *V_2,
	wt_t *W_3, wt_t *B_3, wt_t *G_3, wt_t *T_3, wt_t *M_3, wt_t *V_3,
	wt_t *W_4, wt_t *B_4, wt_t *G_4, wt_t *T_4, wt_t *M_4, wt_t *V_4,
	wt_t *W_5, wt_t *B_5, wt_t *G_5, wt_t *T_5, wt_t *M_5, wt_t *V_5,
	wt_t *W_6, wt_t *B_6, wt_t *G_6, wt_t *T_6, wt_t *M_6, wt_t *V_6,
	wt_t *W_7, wt_t *B_7, wt_t *G_7, wt_t *T_7, wt_t *M_7, wt_t *V_7,
	wt_t *W_8, wt_t *B_8, wt_t *G_8, wt_t *T_8, wt_t *M_8, wt_t *V_8,
	wt_t *W_FC1, wt_t *B_FC1, wt_t *W_FC2, wt_t *B_FC2,
	fm_t *output_fm);

void tiled_conv_layer(
	fm_t *input_fm,
	wt_t *weights,
	wt_t *biases,
	fm_t *output_fm,
	int M, int I, int N, int O, int S, int P,
	int TILE_HW, int OUT_BUF_DEPTH);

void batch_norm_relu_layer(
	fm_t *input_fm,
	wt_t *gamma, wt_t *beta, wt_t *mean, wt_t *variance,
	fm_t *output_fm,
	int M, int I);

#endif
