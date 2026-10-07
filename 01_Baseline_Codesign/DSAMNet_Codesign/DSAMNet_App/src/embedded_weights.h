#ifndef EMBEDDED_WEIGHTS_H
#define EMBEDDED_WEIGHTS_H

// These symbols are defined in embedded_weights.S (the linker resolves
// them). "_start"/"_end" bracket the raw bytes of each embedded file --
// (end - start) gives the exact byte count, useful as a build sanity check.

#ifdef __cplusplus
extern "C" {
#endif

extern const unsigned char g_image1_bin_start[];
extern const unsigned char g_image1_bin_end[];
extern const unsigned char g_image2_bin_start[];
extern const unsigned char g_image2_bin_end[];

extern const unsigned char g_backbone_weights_start[];
extern const unsigned char g_backbone_weights_end[];
extern const unsigned char g_backbone_biases_start[];
extern const unsigned char g_backbone_biases_end[];

extern const unsigned char g_decoder_weights_start[];
extern const unsigned char g_decoder_weights_end[];
extern const unsigned char g_decoder_biases_start[];
extern const unsigned char g_decoder_biases_end[];

extern const unsigned char g_cbam0_weights_start[];
extern const unsigned char g_cbam0_weights_end[];
extern const unsigned char g_cbam1_weights_start[];
extern const unsigned char g_cbam1_weights_end[];

extern const unsigned char g_golden_decoder_start[];
extern const unsigned char g_golden_decoder_end[];

extern const unsigned char g_golden_final_dist_start[];
extern const unsigned char g_golden_final_dist_end[];

#ifdef __cplusplus
}
#endif

#endif