#ifndef PS_BENCH_H
#define PS_BENCH_H
// ============================================================================
// Quick convolution benchmark (header only, called from main() when RUN_BENCH = 1).
// For five real layer shapes of the network it runs the ORIGINAL loops and the NEW kernel on the
// same random data and prints: time, millions of multiply-accumulates per second, speed-up, and
// whether the two results are identical. Output channels are reduced to 8 so the whole test
// takes about a minute even with the slow original loops.
// ============================================================================
#include "ps_dsamnet.h"
#include "ps_conv_fast.h"
#include <cstdio>
#include <cmath>

namespace psb {

static unsigned seed = 20261006u;
static inline float rnd() { seed = seed * 1664525u + 1013904223u; return ((seed >> 8) & 0xFFFF) / 65535.0f - 0.5f; }

static float b_in  [384 * 128 * 128];     // largest input used below (25 MB)
static float b_w   [8 * 384 * 9 + 64];
static float b_bias[16];
static float b_ref [8 * 256 * 256];       // largest output used below
static float b_fast[8 * 256 * 256];

struct Result { double ref_s, fast_s, macs; long diff; double maxd; };

template<int IC, int OC, int IH, int IW, int OH, int OW, int K, int S, int D, int P, bool RELU>
static Result one(const char *name)
{
    const int n_in = IC * IH * IW, n_w = OC * IC * K * K, n_out = OC * OH * OW;
    for (int i = 0; i < n_in; i++) b_in[i] = rnd() * 4.0f;
    for (int i = 0; i < n_w;  i++) b_w[i]  = rnd() * 0.3f;
    for (int i = 0; i < OC;   i++) b_bias[i] = rnd();

    double t0 = ps_timer_now_s();
    conv_ref <IC,OC,IH,IW,OH,OW,K,S,D,P,RELU>(b_in, b_w, b_bias, 0, 0, b_ref);
    double t1 = ps_timer_now_s();
    conv_fast<IC,OC,IH,IW,OH,OW,K,S,D,P,RELU>(b_in, b_w, b_bias, 0, 0, b_fast);
    double t2 = ps_timer_now_s();

    long diff = 0; double maxd = 0.0;
    for (int i = 0; i < n_out; i++) {
        double d = fabs((double)b_ref[i] - (double)b_fast[i]);
        if (b_ref[i] != b_fast[i]) diff++;
        if (d > maxd) maxd = d;
    }
    Result r; r.ref_s = t1 - t0; r.fast_s = t2 - t1; r.macs = (double)n_out * IC * K * K; r.diff = diff; r.maxd = maxd;
    printf("  %-26s old %6.2f s (%6.1f M/s) | new %6.2f s (%7.1f M/s) | x%-5.1f | %s (max diff %g)\n",
           name, r.ref_s, r.macs / r.ref_s / 1e6, r.fast_s, r.macs / r.fast_s / 1e6,
           r.ref_s / r.fast_s, diff == 0 ? "IDENTICAL" : "DIFFERENT", maxd);
    return r;
}

} // namespace psb

static void ps_bench()
{
    printf("\n============================================================\n");
    printf(" CONVOLUTION BENCHMARK  (old loops vs new kernel, same data)\n");
    printf(" NEON compiled in: %s\n", PS_CONV_USES_NEON ? "YES" : "NO (plain C path)");
    printf("============================================================\n");
    using namespace psb;
    Result r[5];
    r[0] = one<64,  8, 128,128,128,128, 3,1,1,1, true >("layer1 3x3, 64 ch @128");
    r[1] = one<384, 8, 128,128,128,128, 3,1,1,1, true >("decoder 3x3, 384 ch @128");
    r[2] = one<512, 8,  32, 32, 32, 32, 3,1,4,4, true >("layer4 3x3 dil 4, 512 ch");
    r[3] = one<64,  8, 128,128, 64, 64, 3,2,1,1, true >("layer2 3x3 stride 2");
    r[4] = one<3,   8, 256,256,256,256, 7,1,1,3, true >("stem 7x7, 3 ch @256");

    double mac = 0, tr = 0, tf = 0; long bad = 0;
    for (int i = 0; i < 5; i++) { mac += r[i].macs; tr += r[i].ref_s; tf += r[i].fast_s; bad += r[i].diff; }
    const double full_macs = 71.5e9;            // one image pair at 256x256 (measured by counting)
    printf("\n  overall: old %.1f M/s, new %.1f M/s  (x%.1f)   results %s\n",
           mac / tr / 1e6, mac / tf / 1e6, tr / tf, bad == 0 ? "IDENTICAL" : "DIFFERENT - DO NOT USE");
    printf("  ESTIMATE for one image pair (71.5 G multiply-accumulates): old %.0f s | new %.0f s\n",
           full_macs / (mac / tr), full_macs / (mac / tf));
    printf("  (estimate only: the real run also includes the small non-convolution steps)\n");
    printf("============================================================\n");
}
#endif
