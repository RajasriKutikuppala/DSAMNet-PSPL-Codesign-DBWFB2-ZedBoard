#include "ps_cbam_hw.h"
#include "xcbam_accel.h"
#include "xil_cache.h"
#include "xil_types.h"
#include "xstatus.h"
#include <cstdio>
#include <cstdint>
#include <cstring>

#define CBAM_BASEADDR  0x40000000ULL
#define CBAM_ELEMS     (DEC_OC * DEC_H * DEC_W)
#define FX_SCALE       4096.0f            // ap_fixed<16,4>: 12 fractional bits
#define POLL_TIMEOUT   200000000UL

// DDR staging buffers in the exact format the IP uses: int16 (Q4.12).
// 4 KB aligned so AXI bursts never straddle a 4 KB boundary unexpectedly.
alignas(4096) static int16_t g_in_fx[CBAM_ELEMS];
alignas(4096) static int16_t g_after_ca_fx[CBAM_ELEMS];   // IP-internal scratch
alignas(4096) static int16_t g_out_fx[CBAM_ELEMS];
alignas(64)   static int16_t g_w_fx[CBAM_WEIGHT_COUNT];

static XCbam_accel g_cbam;
static bool g_init = false;

// float -> Q4.12, round to nearest, saturate (never wrap).
static inline int16_t to_fx(float v, unsigned long &sat)
{
    if (v != v) return 0;                       // NaN guard
    float s = v * FX_SCALE;
    if (s >=  32767.0f) { sat++; return  32767; }
    if (s <= -32768.0f) { sat++; return -32768; }
    return (int16_t)(s >= 0.0f ? s + 0.5f : s - 0.5f);
}

bool ps_cbam_hw(const data_t *in, const data_t *weights, data_t *out)
{
    if (!g_init) {
        if (XCbam_accel_Initialize(&g_cbam, CBAM_BASEADDR) != XST_SUCCESS) {
            printf("  [CBAM-PL] ERROR: driver initialize failed\n");
            return false;
        }
        g_init = true;
    }

    // ---- 1. float -> Q4.12 ------------------------------------------------
    unsigned long sat_in = 0, sat_w = 0;
    float in_min = 1e30f, in_max = -1e30f;
    for (int i = 0; i < CBAM_ELEMS; i++) {
        float v = in[i];
        if (v < in_min) in_min = v;
        if (v > in_max) in_max = v;
        g_in_fx[i] = to_fx(v, sat_in);
    }
    for (int i = 0; i < CBAM_WEIGHT_COUNT; i++)
        g_w_fx[i] = to_fx(weights[i], sat_w);

    printf("  [CBAM-PL] input range [%.4f, %.4f] (Q4.12 limit +-8.0), saturated: in=%lu weights=%lu\n",
           in_min, in_max, sat_in, sat_w);

    // ---- 2. mark outputs with a sentinel (CBAM output is never negative) ---
    memset(g_after_ca_fx, 0x80, sizeof(g_after_ca_fx));   // 0x8080 = -7.97
    memset(g_out_fx,      0x80, sizeof(g_out_fx));

    // ---- 3. make DDR match the CPU cache before the PL looks at it --------
    Xil_DCacheFlushRange((UINTPTR)g_in_fx,       sizeof(g_in_fx));
    Xil_DCacheFlushRange((UINTPTR)g_w_fx,        sizeof(g_w_fx));
    Xil_DCacheFlushRange((UINTPTR)g_after_ca_fx, sizeof(g_after_ca_fx));
    Xil_DCacheFlushRange((UINTPTR)g_out_fx,      sizeof(g_out_fx));

    // ---- 4. program buffer addresses, start, wait --------------------------
    XCbam_accel_Set_in_r    (&g_cbam, (u64)(uintptr_t)g_in_fx);
    XCbam_accel_Set_weights (&g_cbam, (u64)(uintptr_t)g_w_fx);
    XCbam_accel_Set_after_ca(&g_cbam, (u64)(uintptr_t)g_after_ca_fx);
    XCbam_accel_Set_out_r   (&g_cbam, (u64)(uintptr_t)g_out_fx);

    double t0 = ps_timer_now_s();
    XCbam_accel_Start(&g_cbam);

    unsigned long polls = 0;
    while (!XCbam_accel_IsDone(&g_cbam)) {
        if (++polls >= POLL_TIMEOUT) {
            printf("  [CBAM-PL] ERROR: timeout, accelerator never asserted DONE\n");
            return false;
        }
    }
    double hw_s = ps_timer_now_s() - t0;

    // ---- 5. drop stale cache lines, then read what the PL wrote ------------
    Xil_DCacheInvalidateRange((UINTPTR)g_after_ca_fx, sizeof(g_after_ca_fx));
    Xil_DCacheInvalidateRange((UINTPTR)g_out_fx,      sizeof(g_out_fx));

    unsigned long unwritten = 0;
    double sum = 0.0;
    float o_min = 1e30f, o_max = -1e30f;
    for (int i = 0; i < CBAM_ELEMS; i++) {
        int16_t q = g_out_fx[i];
        if (q == (int16_t)0x8080) unwritten++;
        float f = (float)q / FX_SCALE;
        out[i] = f;
        sum += f;
        if (f < o_min) o_min = f;
        if (f > o_max) o_max = f;
    }

    printf("  [CBAM-PL] done in %.4f s (%lu polls) | out min=%.4f max=%.4f mean=%.6f | unwritten=%lu\n",
           hw_s, polls, o_min, o_max, sum / CBAM_ELEMS, unwritten);

    if (unwritten != 0) {
        printf("  [CBAM-PL] ERROR: hardware did not write the whole output buffer\n");
        return false;
    }
    return true;
}