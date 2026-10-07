// ============================================================
// Directly embeds the real weight/bias/image .bin files (already
// on your PC from earlier in this project) into the compiled
// program, using the GNU assembler's .incbin directive -- copies
// raw bytes at build time, no text serialization, so file size is
// not a compile-time problem the way it was for the earlier
// CBAM-only test's C-array approach.
//
// Same underlying idea as your wavelet project's lena_data.h
// (data embedded directly in the program), just a mechanism that
// scales to ~68MB instead of a few KB.
// ============================================================

    .section .rodata

    .align 4
    .global g_image1_bin_start
g_image1_bin_start:
    .incbin "C:/Users/HP/Desktop/BTP_RemoteSensing/DSAMNet/hls_golden_inputs/image1_int8.bin"
    .global g_image1_bin_end
g_image1_bin_end:

    .align 4
    .global g_image2_bin_start
g_image2_bin_start:
    .incbin "C:/Users/HP/Desktop/BTP_RemoteSensing/DSAMNet/hls_golden_inputs/image2_int8.bin"
    .global g_image2_bin_end
g_image2_bin_end:

    .align 4
    .global g_backbone_weights_start
g_backbone_weights_start:
    .incbin "C:/Users/HP/Desktop/BTP_RemoteSensing/DSAMNet/hls_weights_flat/weights_backbone_flat.bin"
    .global g_backbone_weights_end
g_backbone_weights_end:

    .align 4
    .global g_backbone_biases_start
g_backbone_biases_start:
    .incbin "C:/Users/HP/Desktop/BTP_RemoteSensing/DSAMNet/hls_weights_flat/biases_backbone_flat.bin"
    .global g_backbone_biases_end
g_backbone_biases_end:

    .align 4
    .global g_decoder_weights_start
g_decoder_weights_start:
    .incbin "C:/Users/HP/Desktop/BTP_RemoteSensing/DSAMNet/hls_weights_flat/weights_decoder_flat.bin"
    .global g_decoder_weights_end
g_decoder_weights_end:

    .align 4
    .global g_decoder_biases_start
g_decoder_biases_start:
    .incbin "C:/Users/HP/Desktop/BTP_RemoteSensing/DSAMNet/hls_weights_flat/biases_decoder_flat.bin"
    .global g_decoder_biases_end
g_decoder_biases_end:

    .align 4
    .global g_cbam0_weights_start
g_cbam0_weights_start:
    .incbin "C:/Users/HP/Desktop/BTP_RemoteSensing/DSAMNet/hls_weights_flat/weights_cbam0_flat.bin"
    .global g_cbam0_weights_end
g_cbam0_weights_end:

    .align 4
    .global g_cbam1_weights_start
g_cbam1_weights_start:
    .incbin "C:/Users/HP/Desktop/BTP_RemoteSensing/DSAMNet/hls_weights_flat/weights_cbam1_flat.bin"
    .global g_cbam1_weights_end
g_cbam1_weights_end:

    .align 4
    .global g_golden_decoder_start
g_golden_decoder_start:
    .incbin "C:/Users/HP/Desktop/BTP_RemoteSensing/DSAMNet/golden_reference/golden_decoder_branch1.bin"
    .global g_golden_decoder_end
g_golden_decoder_end:

    .align 4
    .global g_golden_final_dist_start
g_golden_final_dist_start:
    .incbin "C:/Users/HP/Desktop/BTP_RemoteSensing/DSAMNet/golden_reference/final_dist_256x256.bin"
    .global g_golden_final_dist_end
g_golden_final_dist_end: