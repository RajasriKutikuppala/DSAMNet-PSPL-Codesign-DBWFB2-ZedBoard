#include "ps_dsamnet.h"
#include <cstdio>
#include <cmath>

template<int IC, int OC, int H, int W, bool RELU>
static void conv1x1_relu(
    const data_t *in, const data_t *weights, const data_t *biases,
    int w_off, int b_off, data_t *out)
{
    for (int oc = 0; oc < OC; oc++) {
        int w_oc_base = w_off + oc * IC;
        for (int y = 0; y < H; y++) {
            for (int xx = 0; xx < W; xx++) {
                float acc = 0.0f;
                int in_ic_base = 0;
                for (int ic = 0; ic < IC; ic++) {
                    acc += in[in_ic_base + y*W + xx] * weights[w_oc_base + ic];
                    in_ic_base += H*W;
                }
                acc += biases[b_off + oc];
                float r = RELU ? (acc > 0.0f ? acc : 0.0f) : acc;
                out[oc*H*W + y*W + xx] = r;
            }
        }
    }
}

template<int IC, int OC, int H, int W, bool RELU>
static void conv3x3_s1p1_relu(
    const data_t *in, const data_t *weights, const data_t *biases,
    int w_off, int b_off, data_t *out)
{
    for (int oc = 0; oc < OC; oc++) {
        int w_oc_base = w_off + oc * IC * 9;
        for (int oy = 0; oy < H; oy++) {
            for (int ox = 0; ox < W; ox++) {
                float acc = 0.0f;
                int w_ic_base = w_oc_base;
                int in_ic_base = 0;
                for (int ic = 0; ic < IC; ic++) {
                    for (int ky = 0; ky < 3; ky++) {
                        int iy = oy - 1 + ky;
                        if (iy < 0 || iy >= H) { continue; }
                        for (int kx = 0; kx < 3; kx++) {
                            int ix = ox - 1 + kx;
                            if (ix < 0 || ix >= W) continue;
                            acc += in[in_ic_base + iy*W + ix] * weights[w_ic_base + ky*3 + kx];
                        }
                    }
                    w_ic_base += 9;
                    in_ic_base += H*W;
                }
                acc += biases[b_off + oc];
                float r = RELU ? (acc > 0.0f ? acc : 0.0f) : acc;
                out[oc*H*W + oy*W + ox] = r;
            }
        }
    }
}

template<int C, int IH, int IW, int OH, int OW>
static void bilinear_resize_align_corners(const data_t *in, data_t *out)
{
    float scale_h = (OH > 1) ? float(IH - 1) / float(OH - 1) : 0.0f;
    float scale_w = (OW > 1) ? float(IW - 1) / float(OW - 1) : 0.0f;

    for (int c = 0; c < C; c++) {
        for (int oy = 0; oy < OH; oy++) {
            float iy_f = oy * scale_h;
            int y0 = (int)iy_f;
            int y1 = (y0 + 1 < IH) ? y0 + 1 : y0;
            float wy = iy_f - y0;

            for (int ox = 0; ox < OW; ox++) {
                float ix_f = ox * scale_w;
                int x0 = (int)ix_f;
                int x1 = (x0 + 1 < IW) ? x0 + 1 : x0;
                float wx = ix_f - x0;

                int base = c*IH*IW;
                float v00 = in[base + y0*IW + x0];
                float v01 = in[base + y0*IW + x1];
                float v10 = in[base + y1*IW + x0];
                float v11 = in[base + y1*IW + x1];

                float top = v00*(1.0f-wx) + v01*wx;
                float bot = v10*(1.0f-wx) + v11*wx;
                out[c*OH*OW + oy*OW + ox] = top*(1.0f-wy) + bot*wy;
            }
        }
    }
}

template<int C, int H, int W>
static void concat4(const data_t *a, const data_t *b, const data_t *c_, const data_t *d, data_t *out)
{
    const int HW = H * W;
    for (int c = 0; c < C; c++) for (int p = 0; p < HW; p++) out[c*HW + p] = a[c*HW + p];
    for (int c = 0; c < C; c++) for (int p = 0; p < HW; p++) out[(C+c)*HW + p] = b[c*HW + p];
    for (int c = 0; c < C; c++) for (int p = 0; p < HW; p++) out[(2*C+c)*HW + p] = c_[c*HW + p];
    for (int c = 0; c < C; c++) for (int p = 0; p < HW; p++) out[(3*C+c)*HW + p] = d[c*HW + p];
}

void ps_decoder(
    const data_t *f2, const data_t *f3, const data_t *f4, const data_t *x,
    const data_t *weights, const data_t *biases,
    data_t *dr_x2, data_t *dr_x3, data_t *dr_x4, data_t *dr_x5,
    data_t *dr_raw, data_t *dr_cat, data_t *dr_lc0,
    data_t *out)
{
    // DR2: f2 already at DEC_H/DEC_W resolution, no resize needed
    conv1x1_relu<L1_C,DR_C,L1_H,L1_W,true>(f2, weights, biases, DR2_W, DR2_B, dr_x2);

    conv1x1_relu<L2_C,DR_C,L2_H,L2_W,true>(f3, weights, biases, DR3_W, DR3_B, dr_raw);
    bilinear_resize_align_corners<DR_C,L2_H,L2_W,DEC_H,DEC_W>(dr_raw, dr_x3);

    conv1x1_relu<L3_C,DR_C,L3_H,L3_W,true>(f4, weights, biases, DR4_W, DR4_B, dr_raw);
    bilinear_resize_align_corners<DR_C,L3_H,L3_W,DEC_H,DEC_W>(dr_raw, dr_x4);

    conv1x1_relu<L4_C,DR_C,L4_H,L4_W,true>(x, weights, biases, DR5_W, DR5_B, dr_raw);
    bilinear_resize_align_corners<DR_C,L4_H,L4_W,DEC_H,DEC_W>(dr_raw, dr_x5);

    // Concat order per PyTorch: (DR5-resized, DR2, DR3-resized, DR4-resized)
    concat4<DR_C,DEC_H,DEC_W>(dr_x5, dr_x2, dr_x3, dr_x4, dr_cat);

    conv3x3_s1p1_relu<4*DR_C,256,DEC_H,DEC_W,true>(dr_cat, weights, biases, LC0_W, LC0_B, dr_lc0);
    conv1x1_relu<256,DEC_OC,DEC_H,DEC_W,true>(dr_lc0, weights, biases, LC4_W, LC4_B, out);

    printf("  [PS decoder] done\n"); fflush(stdout);
}
