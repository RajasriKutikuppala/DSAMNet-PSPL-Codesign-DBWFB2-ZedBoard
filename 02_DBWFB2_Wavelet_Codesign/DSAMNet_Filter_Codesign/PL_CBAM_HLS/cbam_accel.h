#ifndef CBAM_ACCEL_H
#define CBAM_ACCEL_H

#include "ap_fixed.h"

// Standalone CBAM accelerator -- self-contained, no dependency on the
// full-network headers. This is the ONLY piece of DSAMNet running in
// PL; everything else (backbone, decoder, distance, resize) runs as
// plain C++ software on the ARM PS.

typedef ap_fixed<16,4> data_t;
typedef ap_fixed<28,16> acc_t;

#define DEC_OC 64
#define DEC_H  128
#define DEC_W  128

// ---- channel-sum accumulator (FIX) ------------------------------------------------------
typedef ap_fixed<32,20> sum_t;
#define DEC_HW_SHIFT 14          // log2(DEC_H * DEC_W)
static_assert((1 << DEC_HW_SHIFT) == DEC_H * DEC_W, "DEC_HW_SHIFT must be log2(DEC_H*DEC_W)");

#define CA_RATIO 8
#define CA_MID (DEC_OC / CA_RATIO)   // 8
#define SA_K 7

// Flat weight offsets -- identical layout to the full-network version,
// so the SAME weights_cbam0_flat.bin / weights_cbam1_flat.bin files
// already built earlier in this project work here unchanged.
#define CA_FC1_W   0
#define CA_FC2_W   512
#define SA_CONV1_W 1024
#define CBAM_WEIGHT_COUNT 1122

// in, after_ca, out are all DEC_OC*DEC_H*DEC_W flat (external memory).
// PS writes `in` (the decoder output for whichever branch) and the
// right weight set (cbam0 or cbam1), triggers this IP, and reads `out`
// back once done.
void cbam_accel(
    const data_t *in,
    const data_t *weights,
    data_t *after_ca,
    data_t *out
);

#endif