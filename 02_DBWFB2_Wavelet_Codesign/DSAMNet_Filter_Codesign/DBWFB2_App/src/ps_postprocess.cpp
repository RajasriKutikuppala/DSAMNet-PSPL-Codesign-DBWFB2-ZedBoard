#include "ps_dsamnet.h"
#include <cmath>

// Post-processing functions used by ps_main.cpp regardless of whether
// CBAM runs on hardware or software -- distance calculation, upsampling,
// and thresholding are always PS-side software in this architecture.
// (Split out from the earlier combined file so the build doesn't need
// to include the unused CBAM software fallback at all.)

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
