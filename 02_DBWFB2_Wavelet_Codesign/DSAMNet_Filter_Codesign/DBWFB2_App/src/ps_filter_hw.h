#ifndef PS_FILTER_HW_H
#define PS_FILTER_HW_H
#include <stdint.h>

// Geometry of the hardware engine (must match dbwfb2_ll_engine W / H / NP in Vivado)
#define FILTER_PLANES 3
#define FILTER_H      256
#define FILTER_W      256
#define FILTER_IN_BYTES   (FILTER_PLANES * FILTER_H * FILTER_W)              // 196608
#define FILTER_OUT_BYTES  (FILTER_PLANES * (FILTER_H / 2) * (FILTER_W / 2))  //  49152

struct FilterTiming {
    double prep_s;    // copy into the DMA buffer + cache flush
    double dma_s;     // DMA start -> both channels idle (this is the filter time)
    double post_s;    // cache invalidate + copy out
};

// One-time DMA setup. Returns false (and prints why) on failure.
bool ps_filter_init();

// in : FILTER_IN_BYTES  int8, planar CHW 3x256x256
// out: FILTER_OUT_BYTES int8, planar CHW 3x128x128 (LL subband)
// Returns false on DMA error or timeout.
bool ps_filter_run(const int8_t *in, int8_t *out, FilterTiming *t);

#endif
