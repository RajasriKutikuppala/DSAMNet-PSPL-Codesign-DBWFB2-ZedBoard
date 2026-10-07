#ifndef PS_CONV_FAST_H
#define PS_CONV_FAST_H
// ============================================================================
// Fast float32 convolution for the PS (Cortex-A9).
//
// Why the old loops were slow (measured: about 21 million multiply-accumulates per second):
//   for every output pixel the code walked across all input-channel planes. The planes are
//   64 KB apart, so they all land on the same cache sets and evict each other.
//
// What this does instead:
//   * works on one output ROW at a time, accumulating all taps of all input channels into it
//     (the 4 output rows stay in the L1 cache),
//   * four output channels per pass, so every input value that is loaded is used four times,
//   * the innermost loop is a plain stride-1 loop over x (NEON, 4 floats at a time, when the
//     compiler flag -mfpu=neon is given; plain C otherwise).
//
// Results are identical to the old loops: every output element receives its products in the
// same order (input channel, ky, kx), out-of-range taps are skipped the same way, the bias is
// added last and ReLU applied last. The only possible difference is NEON flushing denormal
// numbers (below 1e-38) to zero.
// ============================================================================
#if defined(__ARM_NEON) && !defined(PS_NO_NEON)
#include <arm_neon.h>
#define PS_CONV_USES_NEON 1
#else
#define PS_CONV_USES_NEON 0
#endif

// ---- the original loops, kept verbatim as the reference (used by the benchmark and tests) ----
template<int IC, int OC, int IH, int IW, int OH, int OW, int K, int S, int D, int P, bool RELU>
static void conv_ref(const float *in, const float *weights, const float *biases,
                     int w_off, int b_off, float *out)
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
                        int iy = oy * S - P + ky * D;
                        if (iy < 0 || iy >= IH) continue;
                        for (int kx = 0; kx < K; kx++) {
                            int ix = ox * S - P + kx * D;
                            if (ix < 0 || ix >= IW) continue;
                            acc += in[in_ic_base + iy * IW + ix] * weights[w_ic_base + ky * K + kx];
                        }
                    }
                    w_ic_base += K * K;
                    in_ic_base += IH * IW;
                }
                acc += biases[b_off + oc];
                out[oc * OH * OW + oy * OW + ox] = RELU ? (acc > 0.0f ? acc : 0.0f) : acc;
            }
        }
    }
}

// ---- fast version: same arguments, same weight layout [oc][ic][ky][kx] ----
template<int IC, int OC, int IH, int IW, int OH, int OW, int K, int S, int D, int P, bool RELU>
static void conv_fast(const float *__restrict in, const float *__restrict weights,
                      const float *__restrict biases, int w_off, int b_off, float *__restrict out)
{
    static_assert(OC % 4 == 0, "conv_fast needs OC to be a multiple of 4");
    for (int oc0 = 0; oc0 < OC; oc0 += 4) {
        const float *wb0 = weights + w_off + (oc0 + 0) * IC * K * K;
        const float *wb1 = weights + w_off + (oc0 + 1) * IC * K * K;
        const float *wb2 = weights + w_off + (oc0 + 2) * IC * K * K;
        const float *wb3 = weights + w_off + (oc0 + 3) * IC * K * K;
        for (int oy = 0; oy < OH; oy++) {
            float *__restrict o0 = out + (oc0 + 0) * OH * OW + oy * OW;
            float *__restrict o1 = out + (oc0 + 1) * OH * OW + oy * OW;
            float *__restrict o2 = out + (oc0 + 2) * OH * OW + oy * OW;
            float *__restrict o3 = out + (oc0 + 3) * OH * OW + oy * OW;
            for (int ox = 0; ox < OW; ox++) { o0[ox] = 0.0f; o1[ox] = 0.0f; o2[ox] = 0.0f; o3[ox] = 0.0f; }

            for (int ic = 0; ic < IC; ic++) {
                const float *plane = in + ic * IH * IW;
                const float *wk0 = wb0 + ic * K * K;
                const float *wk1 = wb1 + ic * K * K;
                const float *wk2 = wb2 + ic * K * K;
                const float *wk3 = wb3 + ic * K * K;
                for (int ky = 0; ky < K; ky++) {
                    const int iy = oy * S - P + ky * D;
                    if (iy < 0 || iy >= IH) continue;
                    const float *irow = plane + iy * IW;
                    for (int kx = 0; kx < K; kx++) {
                        const int off = kx * D - P;                    // input column = ox*S + off
                        int lo = 0;
                        if (off < 0) lo = (-off + S - 1) / S;          // first ox with column >= 0
                        const int lim = IW - 1 - off;
                        if (lim < 0) continue;
                        int hi = lim / S;                              // last ox with column <= IW-1
                        if (hi > OW - 1) hi = OW - 1;
                        if (lo > hi) continue;
                        const int n = hi - lo + 1;
                        const float *src = irow + lo * S + off;
                        float *__restrict p0 = o0 + lo;
                        float *__restrict p1 = o1 + lo;
                        float *__restrict p2 = o2 + lo;
                        float *__restrict p3 = o3 + lo;
                        const float a0 = wk0[ky * K + kx];
                        const float a1 = wk1[ky * K + kx];
                        const float a2 = wk2[ky * K + kx];
                        const float a3 = wk3[ky * K + kx];
                        int i = 0;
#if PS_CONV_USES_NEON
                        if (S == 1) {
                            const float32x4_t va0 = vdupq_n_f32(a0);
                            const float32x4_t va1 = vdupq_n_f32(a1);
                            const float32x4_t va2 = vdupq_n_f32(a2);
                            const float32x4_t va3 = vdupq_n_f32(a3);
                            for (; i + 4 <= n; i += 4) {
                                const float32x4_t s = vld1q_f32(src + i);
                                vst1q_f32(p0 + i, vmlaq_f32(vld1q_f32(p0 + i), va0, s));
                                vst1q_f32(p1 + i, vmlaq_f32(vld1q_f32(p1 + i), va1, s));
                                vst1q_f32(p2 + i, vmlaq_f32(vld1q_f32(p2 + i), va2, s));
                                vst1q_f32(p3 + i, vmlaq_f32(vld1q_f32(p3 + i), va3, s));
                            }
                        }
#endif
                        for (; i < n; i++) {
                            const float s = src[i * S];
                            p0[i] += a0 * s;
                            p1[i] += a1 * s;
                            p2[i] += a2 * s;
                            p3[i] += a3 * s;
                        }
                    }
                }
            }

            const float b0 = biases[b_off + oc0 + 0], b1 = biases[b_off + oc0 + 1];
            const float b2 = biases[b_off + oc0 + 2], b3 = biases[b_off + oc0 + 3];
            for (int ox = 0; ox < OW; ox++) {
                float v0 = o0[ox] + b0, v1 = o1[ox] + b1, v2 = o2[ox] + b2, v3 = o3[ox] + b3;
                o0[ox] = RELU ? (v0 > 0.0f ? v0 : 0.0f) : v0;
                o1[ox] = RELU ? (v1 > 0.0f ? v1 : 0.0f) : v1;
                o2[ox] = RELU ? (v2 > 0.0f ? v2 : 0.0f) : v2;
                o3[ox] = RELU ? (v3 > 0.0f ? v3 : 0.0f) : v3;
            }
        }
    }
}
#endif
