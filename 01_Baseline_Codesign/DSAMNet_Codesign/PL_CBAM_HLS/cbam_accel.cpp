#include "cbam_accel.h"
#include <cstdio>
#include <cmath>

// Identical math to the validated dsamnet_cbam.cpp from the full-network
// project -- same channel/spatial attention logic, same acc_t division
// fix, same BRAM binding for the internal scratch buffers.

static void channel_attention(const data_t * __restrict in, const data_t * __restrict weights, data_t * __restrict ca_map)
{
#pragma HLS INLINE off
    data_t avg_pool[DEC_OC];
    data_t max_pool[DEC_OC];

    for (int c = 0; c < DEC_OC; c++) {
        acc_t sum = 0;
        data_t maxv = in[(int)c*DEC_H*DEC_W];
        for (int y = 0; y < DEC_H; y++) {
            for (int x = 0; x < DEC_W; x++) {
                #pragma HLS PIPELINE II=1
                int idx = (int)c*DEC_H*DEC_W + (int)y*DEC_W + x;
                data_t v = in[idx];
                sum += v;
                if (v > maxv) maxv = v;
            }
        }
        avg_pool[c] = data_t(sum / acc_t(DEC_H * DEC_W));
        max_pool[c] = maxv;
    }

    data_t avg_fc1[CA_MID], max_fc1[CA_MID];
    for (int m = 0; m < CA_MID; m++) {
        acc_t a_acc = 0, m_acc = 0;
        for (int c = 0; c < DEC_OC; c++) {
            #pragma HLS PIPELINE II=1
            int widx = CA_FC1_W + (int)m*DEC_OC + c;
            a_acc += avg_pool[c] * weights[widx];
            m_acc += max_pool[c] * weights[widx];
        }
        avg_fc1[m] = (a_acc > 0) ? data_t(a_acc) : data_t(0);
        max_fc1[m] = (m_acc > 0) ? data_t(m_acc) : data_t(0);
    }

    for (int c = 0; c < DEC_OC; c++) {
        acc_t a_acc = 0, m_acc = 0;
        for (int m = 0; m < CA_MID; m++) {
            #pragma HLS PIPELINE II=1
            int widx = CA_FC2_W + (int)c*CA_MID + m;
            a_acc += avg_fc1[m] * weights[widx];
            m_acc += max_fc1[m] * weights[widx];
        }
        acc_t sum = a_acc + m_acc;
        double s = 1.0 / (1.0 + exp(-(double)sum));
        ca_map[c] = data_t(s);
    }
}

static void spatial_attention(const data_t * __restrict in, const data_t * __restrict weights, data_t * __restrict out)
{
#pragma HLS INLINE off
    static data_t s_avg_ch[DEC_H * DEC_W];
    static data_t s_max_ch[DEC_H * DEC_W];
    static data_t s_sa_map[DEC_H * DEC_W];
#pragma HLS bind_storage variable=s_avg_ch type=RAM_2P impl=BRAM
#pragma HLS bind_storage variable=s_max_ch type=RAM_2P impl=BRAM
#pragma HLS bind_storage variable=s_sa_map type=RAM_2P impl=BRAM

    // RESTRUCTURED: the original version read in[c*H*W + y*W + x] for
    // c=0..63 per pixel -- a stride-16384 scatter, 64 individually
    // non-burstable AXI reads per pixel, over a million total fragmented
    // transactions. This is a categorically different (and much worse)
    // access pattern than anything addressed by burst-length/outstanding
    // pragma tuning, and is the most likely real cause of the hardware
    // hang: idealized C-Sim and even RTL cosim never model how badly
    // real DDR/SmartConnect arbitration handles this many scattered
    // small transactions.
    //
    // Fix: iterate channel-major instead of pixel-major. Each channel
    // plane (16384 contiguous elements) is read as ONE large burst, and
    // its contribution to every pixel's running avg/max is accumulated
    // into on-chip BRAM buffers. Same math, same result -- 64 big bursts
    // instead of a million tiny scattered reads.
    for (int c = 0; c < DEC_OC; c++) {
        for (int p = 0; p < DEC_H * DEC_W; p++) {
            #pragma HLS PIPELINE II=1
            data_t v = in[(int)c*DEC_H*DEC_W + p];
            if (c == 0) {
                s_avg_ch[p] = data_t(acc_t(v) / acc_t(DEC_OC));  // seed with first channel's contribution
                s_max_ch[p] = v;
            } else {
                s_avg_ch[p] = data_t(acc_t(s_avg_ch[p]) + acc_t(v) / acc_t(DEC_OC));
                if (v > s_max_ch[p]) s_max_ch[p] = v;
            }
        }
    }

    const int PAD = SA_K / 2;
    for (int oy = 0; oy < DEC_H; oy++) {
        for (int ox = 0; ox < DEC_W; ox++) {
            acc_t acc = 0;
            for (int ky = 0; ky < SA_K; ky++) {
                int iy = oy - PAD + ky;
                if (iy < 0 || iy >= DEC_H) continue;
                for (int kx = 0; kx < SA_K; kx++) {
                    #pragma HLS PIPELINE II=1
                    int ix = ox - PAD + kx;
                    if (ix < 0 || ix >= DEC_W) continue;
                    int p = (int)iy*DEC_W + ix;
                    int w0 = SA_CONV1_W + 0*SA_K*SA_K + ky*SA_K + kx;
                    int w1 = SA_CONV1_W + 1*SA_K*SA_K + ky*SA_K + kx;
                    acc += s_avg_ch[p] * weights[w0];
                    acc += s_max_ch[p] * weights[w1];
                }
            }
            double s = 1.0 / (1.0 + exp(-(double)acc));
            s_sa_map[(int)oy*DEC_W + ox] = data_t(s);
        }
    }

    for (int c = 0; c < DEC_OC; c++) {
        for (int y = 0; y < DEC_H; y++) {
            for (int x = 0; x < DEC_W; x++) {
                #pragma HLS PIPELINE II=1
                int idx = (int)c*DEC_H*DEC_W + (int)y*DEC_W + x;
                out[idx] = in[idx] * s_sa_map[(int)y*DEC_W + x];
            }
        }
    }
}

void cbam_accel(
    const data_t *in,
    const data_t *weights,
    data_t *after_ca,
    data_t *out)
{
#pragma HLS INTERFACE s_axilite port=return  bundle=CTRL
// depth= added for C/RTL co-simulation only -- tells the RTL simulator
// how large a memory model to build for testing. Real hardware/DDR is
// unaffected (depth is simulation-only metadata, not a hardware limit).
// Values match the actual buffer sizes: in/after_ca/out are
// DEC_OC*DEC_H*DEC_W = 1,048,576 elements; weights is CBAM_WEIGHT_COUNT.
// out moved to its own bundle (gmem2) -- it was sharing gmem1 with
// after_ca, and spatial_attention's final loop simultaneously reads
// after_ca and writes out on that same physical port. Synthesis
// confirmed both a large burst READ and a large burst WRITE inferred
// on gmem1 at the exact same line (107) -- a real read/write
// contention hazard on a shared AXI master, invisible to C-Simulation
// (which has no concept of AXI ports at all) but fatal in real RTL:
// this is very likely why cosim (and the board) hang at 0% progress
// while C-sim passes cleanly.
// max_*_burst_length / num_*_outstanding added to tune the HLS-generated
// AXI master adapter's in-flight transaction limits -- default outstanding
// limits can stall on the kind of large, repeated burst pattern this
// kernel issues, in a way that never shows up with small/simple test data.
#pragma HLS INTERFACE m_axi     port=in         offset=slave bundle=gmem0 depth=1048576 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi     port=weights    offset=slave bundle=gmem0 depth=1122    max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi     port=after_ca   offset=slave bundle=gmem1 depth=1048576 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=8 num_write_outstanding=8
#pragma HLS INTERFACE m_axi     port=out        offset=slave bundle=gmem2 depth=1048576 max_write_burst_length=64 num_write_outstanding=8

#ifndef __SYNTHESIS__
    printf("[CBAM_ACCEL] start\n"); fflush(stdout);
#endif

    data_t ca_map[DEC_OC];
    channel_attention(in, weights, ca_map);

    for (int c = 0; c < DEC_OC; c++) {
        for (int y = 0; y < DEC_H; y++) {
            for (int x = 0; x < DEC_W; x++) {
                #pragma HLS PIPELINE II=1
                int idx = (int)c*DEC_H*DEC_W + (int)y*DEC_W + x;
                after_ca[idx] = in[idx] * ca_map[c];
            }
        }
    }

#ifndef __SYNTHESIS__
    printf("[CBAM_ACCEL] channel attention done\n"); fflush(stdout);
#endif

    spatial_attention(after_ca, weights, out);

#ifndef __SYNTHESIS__
    printf("[CBAM_ACCEL] complete\n"); fflush(stdout);
#endif
}