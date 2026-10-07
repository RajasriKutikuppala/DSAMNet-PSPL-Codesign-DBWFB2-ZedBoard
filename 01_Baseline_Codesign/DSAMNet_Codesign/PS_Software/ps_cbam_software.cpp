#include "ps_dsamnet.h"
#include <cstdio>
#include <cmath>

// Software fallback for CBAM -- used to validate the FULL pipeline
// end-to-end today, before real hardware/XSA exist. No overflow
// concerns at all here (that was purely an ap_fixed<16,4> issue in
// the HLS version) -- plain float has enormous range.
//
// Swap the call in ps_main.cpp to ps_cbam_hw(...) once the bitstream
// and XSA are ready; this function's math is otherwise identical to
// the validated cbam_accel.cpp (Corr=0.999997 against real PyTorch
// output), so results should match closely either way.

static void channel_attention(const data_t *in, const data_t *weights, data_t *ca_map)
{
    data_t avg_pool[DEC_OC], max_pool[DEC_OC];
    for (int c = 0; c < DEC_OC; c++) {
        float sum = 0.0f;
        float maxv = in[c*DEC_H*DEC_W];
        for (int y = 0; y < DEC_H; y++)
            for (int x = 0; x < DEC_W; x++) {
                float v = in[c*DEC_H*DEC_W + y*DEC_W + x];
                sum += v;
                if (v > maxv) maxv = v;
            }
        avg_pool[c] = sum / (DEC_H*DEC_W);
        max_pool[c] = maxv;
    }

    data_t avg_fc1[CA_MID], max_fc1[CA_MID];
    for (int m = 0; m < CA_MID; m++) {
        float a_acc = 0.0f, m_acc = 0.0f;
        for (int c = 0; c < DEC_OC; c++) {
            float w = weights[CA_FC1_W + m*DEC_OC + c];
            a_acc += avg_pool[c] * w;
            m_acc += max_pool[c] * w;
        }
        avg_fc1[m] = (a_acc > 0.0f) ? a_acc : 0.0f;
        max_fc1[m] = (m_acc > 0.0f) ? m_acc : 0.0f;
    }

    for (int c = 0; c < DEC_OC; c++) {
        float a_acc = 0.0f, m_acc = 0.0f;
        for (int m = 0; m < CA_MID; m++) {
            float w = weights[CA_FC2_W + c*CA_MID + m];
            a_acc += avg_fc1[m] * w;
            m_acc += max_fc1[m] * w;
        }
        float sum = a_acc + m_acc;
        ca_map[c] = 1.0f / (1.0f + expf(-sum));
    }
}

static void spatial_attention(const data_t *in, const data_t *weights, data_t *out)
{
    static data_t avg_ch[DEC_H*DEC_W], max_ch[DEC_H*DEC_W], sa_map[DEC_H*DEC_W];

    for (int y = 0; y < DEC_H; y++) {
        for (int x = 0; x < DEC_W; x++) {
            float sum = 0.0f;
            float maxv = in[y*DEC_W + x];
            for (int c = 0; c < DEC_OC; c++) {
                float v = in[c*DEC_H*DEC_W + y*DEC_W + x];
                sum += v;
                if (v > maxv) maxv = v;
            }
            avg_ch[y*DEC_W + x] = sum / DEC_OC;
            max_ch[y*DEC_W + x] = maxv;
        }
    }

    const int PAD = SA_K / 2;
    for (int oy = 0; oy < DEC_H; oy++) {
        for (int ox = 0; ox < DEC_W; ox++) {
            float acc = 0.0f;
            for (int ky = 0; ky < SA_K; ky++) {
                int iy = oy - PAD + ky;
                if (iy < 0 || iy >= DEC_H) continue;
                for (int kx = 0; kx < SA_K; kx++) {
                    int ix = ox - PAD + kx;
                    if (ix < 0 || ix >= DEC_W) continue;
                    int p = iy*DEC_W + ix;
                    acc += avg_ch[p] * weights[SA_CONV1_W + 0*SA_K*SA_K + ky*SA_K + kx];
                    acc += max_ch[p] * weights[SA_CONV1_W + 1*SA_K*SA_K + ky*SA_K + kx];
                }
            }
            sa_map[oy*DEC_W + ox] = 1.0f / (1.0f + expf(-acc));
        }
    }

    for (int c = 0; c < DEC_OC; c++)
        for (int y = 0; y < DEC_H; y++)
            for (int x = 0; x < DEC_W; x++) {
                int idx = c*DEC_H*DEC_W + y*DEC_W + x;
                out[idx] = in[idx] * sa_map[y*DEC_W + x];
            }
}

void ps_cbam_software(const data_t *in, const data_t *weights, data_t *after_ca, data_t *out)
{
    data_t ca_map[DEC_OC];
    channel_attention(in, weights, ca_map);

    for (int c = 0; c < DEC_OC; c++)
        for (int y = 0; y < DEC_H; y++)
            for (int x = 0; x < DEC_W; x++) {
                int idx = c*DEC_H*DEC_W + y*DEC_W + x;
                after_ca[idx] = in[idx] * ca_map[c];
            }

    spatial_attention(after_ca, weights, out);
    printf("  [PS CBAM-software] done\n"); fflush(stdout);
}

void ps_channelwise_distance(const data_t *a, const data_t *b, data_t *dist_lowres)
{
    const float EPS = 1e-6f;
    for (int y = 0; y < DEC_H; y++) {
        for (int x = 0; x < DEC_W; x++) {
            float sum_sq = 0.0f;
            for (int c = 0; c < DEC_OC; c++) {
                float d = a[c*DEC_H*DEC_W + y*DEC_W + x] - b[c*DEC_H*DEC_W + y*DEC_W + x] + EPS;
                sum_sq += d*d;
            }
            dist_lowres[y*DEC_W + x] = sqrtf(sum_sq);
        }
    }
}

void ps_bilinear_resize_1ch(const data_t *in, data_t *out)
{
    float scale_h = float(DEC_H - 1) / float(OUT_H - 1);
    float scale_w = float(DEC_W - 1) / float(OUT_W - 1);

    for (int oy = 0; oy < OUT_H; oy++) {
        float iy_f = oy * scale_h;
        int y0 = (int)iy_f;
        int y1 = (y0 + 1 < DEC_H) ? y0 + 1 : y0;
        float wy = iy_f - y0;

        for (int ox = 0; ox < OUT_W; ox++) {
            float ix_f = ox * scale_w;
            int x0 = (int)ix_f;
            int x1 = (x0 + 1 < DEC_W) ? x0 + 1 : x0;
            float wx = ix_f - x0;

            float v00 = in[y0*DEC_W + x0];
            float v01 = in[y0*DEC_W + x1];
            float v10 = in[y1*DEC_W + x0];
            float v11 = in[y1*DEC_W + x1];

            float top = v00*(1.0f-wx) + v01*wx;
            float bot = v10*(1.0f-wx) + v11*wx;
            out[oy*OUT_W + ox] = top*(1.0f-wy) + bot*wy;
        }
    }
}

// Per the DSAMNet paper (Section IV): threshold the distance map to
// get the final binary change map. Paper's implementation uses
// threshold = 1.0.
void ps_threshold_change_map(const data_t *dist, unsigned char *change_map, int n, float threshold)
{
    for (int i = 0; i < n; i++)
        change_map[i] = (dist[i] > threshold) ? 1 : 0;
}
