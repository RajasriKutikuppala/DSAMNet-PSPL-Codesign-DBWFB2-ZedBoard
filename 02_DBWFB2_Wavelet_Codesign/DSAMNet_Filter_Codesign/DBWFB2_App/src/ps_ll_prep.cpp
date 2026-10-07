#include "ps_dsamnet.h"

// Converts the int8 LL image produced by the PL filter into the float network input.
// LL pixel (i,m) is centred on input pixel (2i,2m), so for the upsampling mode every
// full-resolution pixel (y,x) is read from LL position (y/2, x/2) with linear interpolation
// and the edge sample repeated beyond the last LL pixel.

#if PIPELINE_MODE == MODE_LL_UPSAMPLE

void ps_ll_prepare_input(const int8_t *ll, data_t *image)
{
    for (int c = 0; c < IN_C; c++) {
        const int8_t *src = ll + c * LL_H * LL_W;
        data_t *dst = image + c * IMG_H * IMG_W;
        for (int y = 0; y < IMG_H; y++) {
            const float v  = (float)y * 0.5f;
            const int   y0 = (int)v;
            const int   y1 = (y0 + 1 < LL_H) ? (y0 + 1) : (LL_H - 1);
            const float wy = v - (float)y0;
            for (int x = 0; x < IMG_W; x++) {
                const float u  = (float)x * 0.5f;
                const int   x0 = (int)u;
                const int   x1 = (x0 + 1 < LL_W) ? (x0 + 1) : (LL_W - 1);
                const float wx = u - (float)x0;
                const float a = (float)src[y0 * LL_W + x0];
                const float b = (float)src[y0 * LL_W + x1];
                const float p = (float)src[y1 * LL_W + x0];
                const float q = (float)src[y1 * LL_W + x1];
                const float top = a + (b - a) * wx;
                const float bot = p + (q - p) * wx;
                dst[y * IMG_W + x] = (top + (bot - top) * wy) / 127.0f;
            }
        }
    }
}

#elif PIPELINE_MODE == MODE_LL_NOPOOL

void ps_ll_prepare_input(const int8_t *ll, data_t *image)
{
    for (int i = 0; i < IN_C * LL_H * LL_W; i++) image[i] = (float)ll[i] / 127.0f;
}

#else   // MODE_BASELINE: the filter is not used

void ps_ll_prepare_input(const int8_t *, data_t *) {}

#endif
