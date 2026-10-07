#include "ps_dsamnet.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

// ============================================================
// PS-side DSAMNet application.
//
// Run 1 (today, on your PC): validates the whole pipeline against
// the real PyTorch golden output, using the software CBAM fallback.
// No board, no Vitis, no XSA needed for this -- just a normal
// compile+run, exactly like the g++ verification tests used earlier
// in this project.
//
// Run 2 (later, on the ZedBoard): identical code, except the CBAM
// call is swapped to the real hardware IP via ps_cbam_hw(...), once
// the bitstream and XSA exist. printf() already goes to UART
// automatically in a Xilinx bare-metal application -- no separate
// UART code needed.
// ============================================================

static bool load_binary(const char *filename, void *buffer, size_t bytes)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) { printf("ERROR: cannot open %s\n", filename); return false; }
    size_t n = fread(buffer, 1, bytes, fp);
    fclose(fp);
    if (n != bytes) { printf("ERROR: %s expected %zu bytes, got %zu\n", filename, bytes, n); return false; }
    return true;
}

static double elapsed_seconds(clock_t t0) { return double(clock() - t0) / CLOCKS_PER_SEC; }

int main()
{
    printf("============================================\n");
    printf(" DSAMNet PS Software (float, no ap_fixed)\n");
    printf(" Change detection: was this area changed between T1 and T2?\n");
    printf("============================================\n");

    const char *BASE = "C:/Users/HP/Desktop/BTP_RemoteSensing/DSAMNet";
    char path[512];
    clock_t t0 = clock();

    // ---- Images ----
    static data_t image1[IMAGE_SIZE], image2[IMAGE_SIZE];
    {
        signed char tmp[IMAGE_SIZE];
        snprintf(path, sizeof(path), "%s/hls_golden_inputs/image1_int8.bin", BASE);
        if (!load_binary(path, tmp, IMAGE_SIZE)) return 1;
        for (int i = 0; i < IMAGE_SIZE; i++) image1[i] = ((float)tmp[i]) / 127.0f;

        snprintf(path, sizeof(path), "%s/hls_golden_inputs/image2_int8.bin", BASE);
        if (!load_binary(path, tmp, IMAGE_SIZE)) return 1;
        for (int i = 0; i < IMAGE_SIZE; i++) image2[i] = ((float)tmp[i]) / 127.0f;
    }
    printf("[MAIN] Images loaded (t=%.1fs)\n", elapsed_seconds(t0));

    // ---- Weights -- already FP32, load directly, no cast needed ----
    static data_t backbone_weights[BACKBONE_WEIGHT_COUNT];
    static data_t backbone_biases[BACKBONE_BIAS_COUNT];
    static data_t decoder_weights[DECODER_WEIGHT_COUNT];
    static data_t decoder_biases[DECODER_BIAS_COUNT];
    static data_t cbam0_weights[CBAM_WEIGHT_COUNT];
    static data_t cbam1_weights[CBAM_WEIGHT_COUNT];

    #define LOADW(varname, filename, count) \
        snprintf(path, sizeof(path), "%s/hls_weights_flat/" filename, BASE); \
        if (!load_binary(path, varname, (size_t)(count) * sizeof(float))) return 1;

    LOADW(backbone_weights, "weights_backbone_flat.bin", BACKBONE_WEIGHT_COUNT);
    LOADW(backbone_biases,  "biases_backbone_flat.bin",  BACKBONE_BIAS_COUNT);
    LOADW(decoder_weights,  "weights_decoder_flat.bin",  DECODER_WEIGHT_COUNT);
    LOADW(decoder_biases,   "biases_decoder_flat.bin",   DECODER_BIAS_COUNT);
    LOADW(cbam0_weights,    "weights_cbam0_flat.bin",    CBAM_WEIGHT_COUNT);
    LOADW(cbam1_weights,    "weights_cbam1_flat.bin",    CBAM_WEIGHT_COUNT);
    printf("[MAIN] Weights loaded (t=%.1fs)\n", elapsed_seconds(t0));

    // ---- Scratch buffers (plain software memory, no external-memory
    // concerns here -- that was an HLS/BRAM issue, not a software one) ----
    static data_t f2[F2_SIZE], f3[F3_SIZE], f4[F4_SIZE], x_t[X_SIZE];
    static data_t work0[T1_SIZE], work1[T1_SIZE], work2[T1_SIZE], work3[T1_SIZE], work4[T1_SIZE];
    static data_t dr_x2[DR_C*DEC_H*DEC_W], dr_x3[DR_C*DEC_H*DEC_W], dr_x4[DR_C*DEC_H*DEC_W], dr_x5[DR_C*DEC_H*DEC_W];
    static data_t dr_raw[DR_C*L2_H*L2_W];
    static data_t dr_cat[4*DR_C*DEC_H*DEC_W];
    static data_t dr_lc0[256*DEC_H*DEC_W];
    static data_t dec_tmp[DEC_OUT_SIZE];
    static data_t cbam_after_ca[DEC_OUT_SIZE];
    static data_t cbam_out1[DEC_OUT_SIZE], cbam_out2[DEC_OUT_SIZE];
    static data_t dist_lowres[DEC_H*DEC_W];
    static data_t final_dist[OUT_H*OUT_W];
    static unsigned char change_map[OUT_H*OUT_W];

    // ---- Branch 1 ----
    printf("\n[MAIN] Branch1 (image1) backbone starting...\n");
    ps_backbone(image1, backbone_weights, backbone_biases, f2, f3, f4, x_t, work0, work1, work2, work3, work4);
    printf("[MAIN] Branch1 decoder starting (t=%.1fs)...\n", elapsed_seconds(t0));
    ps_decoder(f2, f3, f4, x_t, decoder_weights, decoder_biases, dr_x2, dr_x3, dr_x4, dr_x5, dr_raw, dr_cat, dr_lc0, dec_tmp);
    printf("[MAIN] Branch1 CBAM starting (t=%.1fs)...\n", elapsed_seconds(t0));
    // ---- SWAP POINT: replace with ps_cbam_hw(...) once XSA is ready ----
    ps_cbam_software(dec_tmp, cbam0_weights, cbam_after_ca, cbam_out1);
    printf("[MAIN] Branch1 complete (t=%.1fs)\n", elapsed_seconds(t0));

    // ---- Branch 2 (reuses the same scratch buffers) ----
    printf("\n[MAIN] Branch2 (image2) backbone starting...\n");
    ps_backbone(image2, backbone_weights, backbone_biases, f2, f3, f4, x_t, work0, work1, work2, work3, work4);
    printf("[MAIN] Branch2 decoder starting (t=%.1fs)...\n", elapsed_seconds(t0));
    ps_decoder(f2, f3, f4, x_t, decoder_weights, decoder_biases, dr_x2, dr_x3, dr_x4, dr_x5, dr_raw, dr_cat, dr_lc0, dec_tmp);
    printf("[MAIN] Branch2 CBAM starting (t=%.1fs)...\n", elapsed_seconds(t0));
    ps_cbam_software(dec_tmp, cbam1_weights, cbam_after_ca, cbam_out2);
    printf("[MAIN] Branch2 complete (t=%.1fs)\n", elapsed_seconds(t0));

    // ---- Distance + upsample + threshold ----
    ps_channelwise_distance(cbam_out1, cbam_out2, dist_lowres);
    ps_bilinear_resize_1ch(dist_lowres, final_dist);
    ps_threshold_change_map(final_dist, change_map, OUT_H*OUT_W, 1.0f);  // paper's threshold
    printf("[MAIN] Distance + threshold complete (t=%.1fs)\n", elapsed_seconds(t0));

    // ---- Report the actual product: how much of the scene changed ----
    long changed_pixels = 0;
    for (int i = 0; i < OUT_H*OUT_W; i++) if (change_map[i]) changed_pixels++;
    double change_pct = 100.0 * changed_pixels / (OUT_H*OUT_W);

    printf("\n============================================\n");
    printf(" RESULT\n");
    printf("============================================\n");
    printf("Output size: %d x %d\n", OUT_H, OUT_W);
    printf("Changed pixels: %ld / %d (%.2f%%)\n", changed_pixels, OUT_H*OUT_W, change_pct);
    printf("Verdict: %s\n", change_pct > 1.0 ? "CHANGE DETECTED between T1 and T2" : "No significant change detected");

    // ---- Validation against the real PyTorch golden output ----
    static float golden[OUT_H*OUT_W];
    snprintf(path, sizeof(path), "%s/golden_reference/final_dist_256x256.bin", BASE);
    if (load_binary(path, golden, OUT_H*OUT_W*sizeof(float))) {
        double mae=0, mse=0, maxerr=0;
        double st=0, sg=0, stg=0, st2=0, sg2=0;
        int n = OUT_H*OUT_W;
        for (int i = 0; i < n; i++) {
            double a = final_dist[i], b = golden[i];
            double e = fabs(a-b);
            mae += e; mse += e*e;
            if (e > maxerr) maxerr = e;
            st+=a; sg+=b; stg+=a*b; st2+=a*a; sg2+=b*b;
        }
        mae /= n; mse = sqrt(mse/n);
        double num = n*stg - st*sg;
        double den = sqrt((n*st2-st*st)*(n*sg2-sg*sg));
        double corr = (den!=0) ? num/den : 0;
        printf("\n[VALIDATION vs real PyTorch DSAMNet output]\n");
        printf("MAE=%.6f RMSE=%.6f MaxErr=%.6f Corr=%.6f\n", mae, mse, maxerr, corr);
    } else {
        printf("\n[VALIDATION] golden file not found -- skipping comparison.\n");
    }

    printf("\n[MAIN] Total time: %.1fs\n", elapsed_seconds(t0));
    return 0;
}
