#include "ps_dsamnet.h"
#include "ps_conv_fast.h"
#include <cstdio>

// Same folded, strength-reduced pattern as the validated HLS code --
// no hardware pragmas needed here (this runs as plain software), but
// the address math stays identical on purpose so the two are easy to
// cross-check line by line.

template<
    int IC, int OC, int IH, int IW, int OH, int OW,
    int K, int STRIDE, int DIL, int PAD, bool RELU
>
static void conv2d_ref(
    const data_t *in, const data_t *weights, const data_t *biases,
    int w_off, int b_off, data_t *out)
{
    for (int oc = 0; oc < OC; oc++) {
        int w_oc_base = w_off + oc * IC * K * K;
        for (int oy = 0; oy < OH; oy++) {
            for (int ox = 0; ox < OW; ox++) {
                float acc = 0.0f;
                int w_ic_base = w_oc_base;
                int in_ic_base = 0;
                for (int ic = 0; ic < IC; ic++) {
                    for (int ky = 0; ky < K; ky++) {
                        int iy = oy*STRIDE - PAD + ky*DIL;
                        if (iy < 0 || iy >= IH) { continue; }
                        for (int kx = 0; kx < K; kx++) {
                            int ix = ox*STRIDE - PAD + kx*DIL;
                            if (ix < 0 || ix >= IW) continue;
                            int in_idx = in_ic_base + iy*IW + ix;
                            int w_idx  = w_ic_base + ky*K + kx;
                            acc += in[in_idx] * weights[w_idx];
                        }
                    }
                    w_ic_base += K*K;
                    in_ic_base += IH*IW;
                }
                acc += biases[b_off + oc];
                int out_idx = oc*OH*OW + oy*OW + ox;
                out[out_idx] = RELU ? (acc > 0.0f ? acc : 0.0f) : acc;
            }
        }
    }
}

template<int IC, int OC, int IH, int IW, int OH, int OW, int STRIDE>
static void conv1x1_ref(
    const data_t *in, const data_t *weights, const data_t *biases,
    int w_off, int b_off, data_t *out)
{
    for (int oc = 0; oc < OC; oc++) {
        int w_oc_base = w_off + oc * IC;
        for (int oy = 0; oy < OH; oy++) {
            for (int ox = 0; ox < OW; ox++) {
                float acc = 0.0f;
                int iy = oy * STRIDE, ix = ox * STRIDE;
                int in_ic_base = 0;
                for (int ic = 0; ic < IC; ic++) {
                    int in_idx = in_ic_base + iy*IW + ix;
                    int w_idx  = w_oc_base + ic;
                    acc += in[in_idx] * weights[w_idx];
                    in_ic_base += IH*IW;
                }
                acc += biases[b_off + oc];
                out[oc*OH*OW + oy*OW + ox] = acc;   // no ReLU -- residual add applies it
            }
        }
    }
}

// ---- front ends used by the network: fast kernels, or the original loops (USE_FAST_CONV = 0) ----
template<
    int IC, int OC, int IH, int IW, int OH, int OW,
    int K, int STRIDE, int DIL, int PAD, bool RELU
>
static inline void conv2d(
    const data_t *in, const data_t *weights, const data_t *biases,
    int w_off, int b_off, data_t *out)
{
#if USE_FAST_CONV
    conv_fast<IC,OC,IH,IW,OH,OW,K,STRIDE,DIL,PAD,RELU>(in, weights, biases, w_off, b_off, out);
#else
    conv2d_ref<IC,OC,IH,IW,OH,OW,K,STRIDE,DIL,PAD,RELU>(in, weights, biases, w_off, b_off, out);
#endif
}

template<int IC, int OC, int IH, int IW, int OH, int OW, int STRIDE>
static inline void conv1x1(
    const data_t *in, const data_t *weights, const data_t *biases,
    int w_off, int b_off, data_t *out)
{
#if USE_FAST_CONV
    conv_fast<IC,OC,IH,IW,OH,OW,1,STRIDE,1,0,false>(in, weights, biases, w_off, b_off, out);
#else
    conv1x1_ref<IC,OC,IH,IW,OH,OW,STRIDE>(in, weights, biases, w_off, b_off, out);
#endif
}

template<int C, int H, int W>
static void residual_add_relu(const data_t *a, const data_t *b, data_t *out)
{
    const int N = C * H * W;
    for (int i = 0; i < N; i++) {
        float v = a[i] + b[i];
        out[i] = (v > 0.0f) ? v : 0.0f;
    }
}

#if PIPELINE_MODE != MODE_LL_NOPOOL
static void maxpool_3x3_s2_p1(const data_t *in, data_t *out)
{
    for (int c = 0; c < L0_C; c++) {
        for (int oy = 0; oy < POOL_H; oy++) {
            for (int ox = 0; ox < POOL_W; ox++) {
                bool found = false;
                float maxval = 0.0f;
                for (int ky = 0; ky < 3; ky++) {
                    int iy = oy*2 - 1 + ky;
                    if (iy < 0 || iy >= IMG_H) continue;
                    for (int kx = 0; kx < 3; kx++) {
                        int ix = ox*2 - 1 + kx;
                        if (ix < 0 || ix >= IMG_W) continue;
                        float v = in[c*IMG_H*IMG_W + iy*IMG_W + ix];
                        if (!found || v > maxval) { maxval = v; found = true; }
                    }
                }
                out[c*POOL_H*POOL_W + oy*POOL_W + ox] = maxval;
            }
        }
    }
}
#endif

void ps_backbone(
    const data_t *image, const data_t *weights, const data_t *biases,
    data_t *f2, data_t *f3, data_t *f4, data_t *x,
    data_t *work0, data_t *work1, data_t *work2, data_t *work3, data_t *work4)
{
#if PIPELINE_MODE == MODE_LL_NOPOOL
    // Input is the 128x128 LL image. The stem output (64 x 128 x 128) already has the size the
    // max-pool would have produced, so the pool is skipped and the result goes straight to work1.
    conv2d<IN_C,L0_C,LL_H,LL_W,LL_H,LL_W,7,1,1,3,true>(image, weights, biases, W0, B0, work1);
#else
    conv2d<IN_C,L0_C,IMG_H,IMG_W,IMG_H,IMG_W,7,1,1,3,true>(image, weights, biases, W0, B0, work0);
    maxpool_3x3_s2_p1(work0, work1);
#endif

    // Layer1
    conv2d<L1_C,L1_C,L1_H,L1_W,L1_H,L1_W,3,1,1,1,true>(work1, weights, biases, W1, B1, work2);
    conv2d<L1_C,L1_C,L1_H,L1_W,L1_H,L1_W,3,1,1,1,false>(work2, weights, biases, W2, B2, work3);
    residual_add_relu<L1_C,L1_H,L1_W>(work3, work1, work2);
    conv2d<L1_C,L1_C,L1_H,L1_W,L1_H,L1_W,3,1,1,1,true>(work2, weights, biases, W3, B3, work3);
    conv2d<L1_C,L1_C,L1_H,L1_W,L1_H,L1_W,3,1,1,1,false>(work3, weights, biases, W4, B4, f2);
    residual_add_relu<L1_C,L1_H,L1_W>(f2, work2, f2);
    printf("  [PS backbone] layer1 done\n"); fflush(stdout);

    // Layer2
    conv2d<L1_C,L2_C,L1_H,L1_W,L2_H,L2_W,3,2,1,1,true>(f2, weights, biases, W5, B5, work0);
    conv2d<L2_C,L2_C,L2_H,L2_W,L2_H,L2_W,3,1,1,1,false>(work0, weights, biases, W6, B6, work1);
    conv1x1<L1_C,L2_C,L1_H,L1_W,L2_H,L2_W,2>(f2, weights, biases, W7, B7, work2);
    residual_add_relu<L2_C,L2_H,L2_W>(work1, work2, work3);
    conv2d<L2_C,L2_C,L2_H,L2_W,L2_H,L2_W,3,1,1,1,true>(work3, weights, biases, W8, B8, work1);
    conv2d<L2_C,L2_C,L2_H,L2_W,L2_H,L2_W,3,1,1,1,false>(work1, weights, biases, W9, B9, f3);
    residual_add_relu<L2_C,L2_H,L2_W>(f3, work3, f3);
    printf("  [PS backbone] layer2 done\n"); fflush(stdout);

    // Layer3
    conv2d<L2_C,L3_C,L2_H,L2_W,L3_H,L3_W,3,2,1,1,true>(f3, weights, biases, W10, B10, work0);
    conv2d<L3_C,L3_C,L3_H,L3_W,L3_H,L3_W,3,1,1,1,false>(work0, weights, biases, W11, B11, work1);
    conv1x1<L2_C,L3_C,L2_H,L2_W,L3_H,L3_W,2>(f3, weights, biases, W12, B12, work2);
    residual_add_relu<L3_C,L3_H,L3_W>(work1, work2, work3);
    conv2d<L3_C,L3_C,L3_H,L3_W,L3_H,L3_W,3,1,1,1,true>(work3, weights, biases, W13, B13, work1);
    conv2d<L3_C,L3_C,L3_H,L3_W,L3_H,L3_W,3,1,1,1,false>(work1, weights, biases, W14, B14, f4);
    residual_add_relu<L3_C,L3_H,L3_W>(f4, work3, f4);
    printf("  [PS backbone] layer3 done\n"); fflush(stdout);

    // Layer4 (multi-grid dilation 2,4,8)
    conv2d<L3_C,L4_C,L3_H,L3_W,L4_H,L4_W,3,1,2,2,true>(f4, weights, biases, W15, B15, work0);
    conv2d<L4_C,L4_C,L4_H,L4_W,L4_H,L4_W,3,1,1,1,false>(work0, weights, biases, W16, B16, work1);
    conv1x1<L3_C,L4_C,L3_H,L3_W,L4_H,L4_W,1>(f4, weights, biases, W17, B17, work2);
    residual_add_relu<L4_C,L4_H,L4_W>(work1, work2, work3);

    conv2d<L4_C,L4_C,L4_H,L4_W,L4_H,L4_W,3,1,4,4,true>(work3, weights, biases, W18, B18, work1);
    conv2d<L4_C,L4_C,L4_H,L4_W,L4_H,L4_W,3,1,1,1,false>(work1, weights, biases, W19, B19, work2);
    residual_add_relu<L4_C,L4_H,L4_W>(work2, work3, work4);

    conv2d<L4_C,L4_C,L4_H,L4_W,L4_H,L4_W,3,1,8,8,true>(work4, weights, biases, W20, B20, work1);
    conv2d<L4_C,L4_C,L4_H,L4_W,L4_H,L4_W,3,1,1,1,false>(work1, weights, biases, W21, B21, work2);
    residual_add_relu<L4_C,L4_H,L4_W>(work2, work4, x);
    printf("  [PS backbone] layer4 done\n"); fflush(stdout);
}
