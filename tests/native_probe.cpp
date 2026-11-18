#include "tiled_conv.hpp"
#include "utils.hpp"
extern "C" {
void probe_conv(float* x, float* w, float* b, float* y, int m, int i, int n, int o, int p, int tile, int depth) {
    tiled_conv_layer(x,w,b,y,m,i,n,o,1,p,tile,depth);
}
void probe_bn(float* x,float* g,float* b,float* mean,float* var,float* y,int m,int i) {
    batch_norm_relu_layer(x,g,b,mean,var,y,m,i);
}
void probe_pool(float* x,float* y,int m,int i,int o) { max_pool_2x2(x,y,m,i,o); }
void probe_fc(float* x,float* w,float* b,float* y,int in,int out) { fully_connected_layer(x,w,b,y,in,out); }
void probe_infer(float** p) {
    reduced_vgg_inference(p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12],p[13],p[14],p[15],p[16],p[17],p[18],p[19],p[20],p[21],p[22],p[23],p[24],p[25],p[26],p[27],p[28],p[29],p[30],p[31],p[32],p[33],p[34],p[35],p[36],p[37],p[38],p[39],p[40],p[41],p[42],p[43],p[44],p[45],p[46],p[47],p[48],p[49],p[50],p[51],p[52],p[53]);
}
}
