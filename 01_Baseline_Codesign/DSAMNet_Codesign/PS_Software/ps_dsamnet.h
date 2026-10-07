#ifndef PS_DSAMNET_H
#define PS_DSAMNET_H

// ============================================================
// PS-side (ARM software) DSAMNet -- plain float throughout.
//
// No ap_fixed here: this code never gets synthesized into hardware,
// it runs as a normal program on the Zynq's ARM cores, which have a
// real FPU. float is fast AND more numerically accurate here than
// our fixed-point HLS path (matches PyTorch's own float32 math).
//
// Same dimension names and same flat weight-offset scheme as the
// validated HLS code, on purpose -- easy to cross-check the two.
// ============================================================

typedef float data_t;

#define IMG_H 256
#define IMG_W 256
#define IN_C  3

#define L0_C 64
#define POOL_H 128
#define POOL_W 128

#define L1_C 64
#define L1_H 128
#define L1_W 128

#define L2_C 128
#define L2_H 64
#define L2_W 64

#define L3_C 256
#define L3_H 32
#define L3_W 32

#define L4_C 512
#define L4_H 32
#define L4_W 32

#define DR_C   96
#define DEC_OC 64
#define DEC_H  128
#define DEC_W  128

#define CA_RATIO 8
#define CA_MID (DEC_OC / CA_RATIO)
#define SA_K 7

#define OUT_H 256
#define OUT_W 256

// ---- Flat tensor sizes ----
#define IMAGE_SIZE (IN_C * IMG_H * IMG_W)
#define F2_SIZE (L1_C * L1_H * L1_W)
#define F3_SIZE (L2_C * L2_H * L2_W)
#define F4_SIZE (L3_C * L3_H * L3_W)
#define X_SIZE  (L4_C * L4_H * L4_W)
#define T1_SIZE (L0_C * IMG_H * IMG_W)
#define DEC_OUT_SIZE (DEC_OC * DEC_H * DEC_W)

// ---- Backbone flat weight offsets (identical to the validated HLS scheme) ----
#define W0  0
#define W1  9408
#define W2  46272
#define W3  83136
#define W4  120000
#define W5  156864
#define W6  230592
#define W7  378048
#define W8  386240
#define W9  533696
#define W10 681152
#define W11 976064
#define W12 1565888
#define W13 1598656
#define W14 2188480
#define W15 2778304
#define W16 3957952
#define W17 6317248
#define W18 6448320
#define W19 8807616
#define W20 11166912
#define W21 13526208
#define BACKBONE_WEIGHT_COUNT 15885504

#define B0  0
#define B1  64
#define B2  128
#define B3  192
#define B4  256
#define B5  320
#define B6  448
#define B7  576
#define B8  704
#define B9  832
#define B10 960
#define B11 1216
#define B12 1472
#define B13 1728
#define B14 1984
#define B15 2240
#define B16 2752
#define B17 3264
#define B18 3776
#define B19 4288
#define B20 4800
#define B21 5312
#define BACKBONE_BIAS_COUNT 5824

// ---- Decoder flat weight offsets ----
#define DR2_W 0
#define DR3_W 6144
#define DR4_W 18432
#define DR5_W 43008
#define LC0_W 92160
#define LC4_W 976896
#define DECODER_WEIGHT_COUNT 993280

#define DR2_B 0
#define DR3_B 96
#define DR4_B 192
#define DR5_B 288
#define LC0_B 384
#define LC4_B 640
#define DECODER_BIAS_COUNT 704

// ---- CBAM flat weight offsets (no bias) ----
#define CA_FC1_W   0
#define CA_FC2_W   512
#define SA_CONV1_W 1024
#define CBAM_WEIGHT_COUNT 1122

// ============================================================
// Function declarations
// ============================================================

// Backbone: image -> f2,f3,f4,x. Caller supplies work0..4 scratch
// (each sized T1_SIZE) -- plain heap/stack allocated, no external-
// memory concerns here, this is just software.
void ps_backbone(
    const data_t *image, const data_t *weights, const data_t *biases,
    data_t *f2, data_t *f3, data_t *f4, data_t *x,
    data_t *work0, data_t *work1, data_t *work2, data_t *work3, data_t *work4
);

// Decoder: f2,f3,f4,x -> DEC_OC x DEC_H x DEC_W feature map.
// Scratch buffers sized generously by the caller (see ps_main.cpp).
void ps_decoder(
    const data_t *f2, const data_t *f3, const data_t *f4, const data_t *x,
    const data_t *weights, const data_t *biases,
    data_t *dr_x2, data_t *dr_x3, data_t *dr_x4, data_t *dr_x5,
    data_t *dr_raw, data_t *dr_cat, data_t *dr_lc0,
    data_t *out
);

// CBAM -- SOFTWARE version, used for testing today without hardware.
// Swap the call site in ps_main.cpp to ps_cbam_hw(...) once the real
// bitstream/XSA are ready (see ps_cbam_hw_driver.h, built later).
void ps_cbam_software(
    const data_t *in, const data_t *weights,
    data_t *after_ca, data_t *out
);

// Channel-wise Euclidean distance between two CBAM-refined feature maps.
void ps_channelwise_distance(const data_t *a, const data_t *b, data_t *dist_lowres);

// Bilinear upsample DEC_H x DEC_W (1 channel) -> OUT_H x OUT_W.
void ps_bilinear_resize_1ch(const data_t *in, data_t *out);

// Per the DSAMNet paper: threshold the distance map at `threshold`
// (paper uses 1.0) to produce the final binary change map: 1 = changed,
// 0 = unchanged.
void ps_threshold_change_map(const data_t *dist, unsigned char *change_map, int n, float threshold);

#endif
