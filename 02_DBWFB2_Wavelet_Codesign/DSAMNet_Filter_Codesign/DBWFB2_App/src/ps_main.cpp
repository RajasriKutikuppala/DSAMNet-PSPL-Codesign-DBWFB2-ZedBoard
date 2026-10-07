#include "ps_dsamnet.h"
#include "embedded_weights.h"
#include "ps_cbam_hw.h"
#include "ps_filter_hw.h"
#include "ps_bench.h"

#include <cstdio>
#include <cmath>
#include <cstdint>

// ============================================================================
// DSAMNet change detection on the ZedBoard (Zynq-7000), bare-metal.
//
//   PL : DBWFB2 LL filter (AXI DMA -> filter -> AXI DMA)   [modes 1 and 2]
//        CBAM attention (HLS IP)
//   PS : backbone, decoder, distance, resize, threshold
//
//   The pipeline mode is chosen by PIPELINE_MODE in ps_dsamnet.h.
// ============================================================================

static const char *mode_name()
{
#if PIPELINE_MODE == MODE_BASELINE
    return "0 BASELINE   : 3x256x256 -> DSAMNet (no filter)";
#elif PIPELINE_MODE == MODE_LL_UPSAMPLE
    return "1 LL-UPSAMPLE: PL filter -> LL 3x128x128 -> PS upsample to 256 -> DSAMNet";
#else
    return "2 LL-NOPOOL  : PL filter -> LL 3x128x128 -> stem on 128x128, no max-pool";
#endif
}

static bool check_size(const char *name, const unsigned char *start,
                       const unsigned char *end, size_t expected)
{
    size_t actual = (size_t)(end - start);
    if (actual != expected) {
        printf("[ERROR] %-18s embedded %zu bytes, expected %zu\n", name, actual, expected);
        return false;
    }
    printf("[CHECK] %-18s OK (%zu bytes)\n", name, actual);
    return true;
}

// ---- inference buffers (file scope: zero-initialised, shared by both branches) ----
static data_t image1[IMAGE_SIZE], image2[IMAGE_SIZE];
static data_t f2[F2_SIZE], f3[F3_SIZE], f4[F4_SIZE], x_t[X_SIZE];
static data_t work0[T1_SIZE], work1[T1_SIZE], work2[T1_SIZE], work3[T1_SIZE], work4[T1_SIZE];
static data_t dr_x2[DR_C * DEC_H * DEC_W], dr_x3[DR_C * DEC_H * DEC_W];
static data_t dr_x4[DR_C * DEC_H * DEC_W], dr_x5[DR_C * DEC_H * DEC_W];
static data_t dr_raw[DR_C * L2_H * L2_W];
static data_t dr_cat[4 * DR_C * DEC_H * DEC_W];
static data_t dr_lc0[256 * DEC_H * DEC_W];
static data_t dec_tmp[DEC_OUT_SIZE];
static data_t cbam_out1[DEC_OUT_SIZE], cbam_out2[DEC_OUT_SIZE];
static data_t dist_lowres[DEC_H * DEC_W];
static data_t final_dist[OUT_H * OUT_W];
static unsigned char change_map[OUT_H * OUT_W];
static int8_t ll_i8[FILTER_OUT_BYTES];            // LL image returned by the PL filter

struct Weights {
    const data_t *bb_w, *bb_b;     // backbone
    const data_t *dec_w, *dec_b;   // decoder
};

struct BranchTimes { double filter, backbone, decoder, cbam; };

// One Siamese branch: [PL filter] -> backbone (PS) -> decoder (PS) -> CBAM (PL).
static bool run_branch(const char *tag, const unsigned char *raw, const unsigned char *ll_ref,
                       data_t *image, const Weights &w, const data_t *cbam_w, data_t *cbam_out,
                       BranchTimes *bt)
{
    double t = ps_timer_now_s();

#if PIPELINE_MODE == MODE_BASELINE
    (void)ll_ref;
    const signed char *r = (const signed char *)raw;
    for (int i = 0; i < IMAGE_SIZE; i++) image[i] = (float)r[i] / 127.0f;
#else
    printf("\n[%s] PL filter (DMA -> DBWFB2 LL engine -> DMA) ...\n", tag);
    FilterTiming ft;
    if (!ps_filter_run((const int8_t *)raw, ll_i8, &ft)) {
        printf("[%s] FATAL: PL filter failed\n", tag);
        return false;
    }
    const int8_t *ref = (const int8_t *)ll_ref;
    long bad = 0;
    for (int i = 0; i < FILTER_OUT_BYTES; i++) if (ll_i8[i] != ref[i]) bad++;
    printf("[%s] filter %.3f ms (DMA) | LL vs Python reference: %ld mismatches of %d %s\n",
           tag, ft.dma_s * 1e3, bad, FILTER_OUT_BYTES, bad == 0 ? "(bit-exact)" : "<-- CHECK");
    ps_ll_prepare_input(ll_i8, image);
#endif
    bt->filter = ps_timer_now_s() - t;  t = ps_timer_now_s();

    printf("[%s] backbone (PS) ...\n", tag);
    ps_backbone(image, w.bb_w, w.bb_b, f2, f3, f4, x_t, work0, work1, work2, work3, work4);
    bt->backbone = ps_timer_now_s() - t;  t = ps_timer_now_s();

    printf("[%s] decoder (PS) ...\n", tag);
    ps_decoder(f2, f3, f4, x_t, w.dec_w, w.dec_b,
               dr_x2, dr_x3, dr_x4, dr_x5, dr_raw, dr_cat, dr_lc0, dec_tmp);
    bt->decoder = ps_timer_now_s() - t;  t = ps_timer_now_s();

#if PIPELINE_MODE == MODE_BASELINE
    // Diagnostic only valid for the unfiltered branch-1 input: decoder vs PyTorch decoder output
    if (tag[0] == 'T' && tag[1] == '1') {
        size_t exp_bytes = (size_t)DEC_OUT_SIZE * 4;
        size_t got_bytes = (size_t)(g_golden_decoder_end - g_golden_decoder_start);
        if (got_bytes == exp_bytes) {
            const float *gd = (const float *)g_golden_decoder_start;
            double sum_ours = 0, sum_gold = 0, mae = 0;
            for (int i = 0; i < DEC_OUT_SIZE; i++) {
                sum_ours += dec_tmp[i];
                sum_gold += gd[i];
                mae += fabs((double)dec_tmp[i] - (double)gd[i]);
            }
            printf("  [DIAG] decoder mean: ours=%.6f golden=%.6f  MAE=%.6f\n",
                   sum_ours / DEC_OUT_SIZE, sum_gold / DEC_OUT_SIZE, mae / DEC_OUT_SIZE);
        }
        t = ps_timer_now_s();
    }
#endif

    printf("[%s] CBAM (PL) ...\n", tag);
    bool ok = ps_cbam_hw(dec_tmp, cbam_w, cbam_out);
    bt->cbam = ps_timer_now_s() - t;

    printf("[%s] times: filter %.3f s | backbone %.2f s | decoder %.2f s | CBAM %.3f s\n",
           tag, bt->filter, bt->backbone, bt->decoder, bt->cbam);
    return ok;
}

int main()
{
    ps_timer_init();
    double t_start = ps_timer_now_s();

#if RUN_BENCH
    ps_bench();           // quick speed test only; set RUN_BENCH to 0 in ps_dsamnet.h for the full run
    return 0;
#endif

    printf("\n============================================================\n");
    printf(" DSAMNet + DBWFB2 LL front end  (ZedBoard, Zynq-7000)\n");
    printf(" Mode %s\n", mode_name());
    printf("============================================================\n");

    // ---- 1. embedded data sanity ----
    bool ok = true;
    ok &= check_size("image1",           g_image1_bin_start,       g_image1_bin_end,       IMAGE_SIZE);
    ok &= check_size("image2",           g_image2_bin_start,       g_image2_bin_end,       IMAGE_SIZE);
    ok &= check_size("backbone_weights", g_backbone_weights_start, g_backbone_weights_end, (size_t)BACKBONE_WEIGHT_COUNT * 4);
    ok &= check_size("backbone_biases",  g_backbone_biases_start,  g_backbone_biases_end,  (size_t)BACKBONE_BIAS_COUNT * 4);
    ok &= check_size("decoder_weights",  g_decoder_weights_start,  g_decoder_weights_end,  (size_t)DECODER_WEIGHT_COUNT * 4);
    ok &= check_size("decoder_biases",   g_decoder_biases_start,   g_decoder_biases_end,   (size_t)DECODER_BIAS_COUNT * 4);
    ok &= check_size("cbam0_weights",    g_cbam0_weights_start,    g_cbam0_weights_end,    (size_t)CBAM_WEIGHT_COUNT * 4);
    ok &= check_size("cbam1_weights",    g_cbam1_weights_start,    g_cbam1_weights_end,    (size_t)CBAM_WEIGHT_COUNT * 4);
    ok &= check_size("LL reference 1",   g_ll1_ref_start,          g_ll1_ref_end,          FILTER_OUT_BYTES);
    ok &= check_size("LL reference 2",   g_ll2_ref_start,          g_ll2_ref_end,          FILTER_OUT_BYTES);
    if (!ok) { printf("[FATAL] embedded data mismatch - regenerate the reference files / rebuild\n"); return 1; }

#if PIPELINE_MODE != MODE_BASELINE
    if (!ps_filter_init()) return 1;
#endif

    Weights w;
    w.bb_w  = (const data_t *)g_backbone_weights_start;
    w.bb_b  = (const data_t *)g_backbone_biases_start;
    w.dec_w = (const data_t *)g_decoder_weights_start;
    w.dec_b = (const data_t *)g_decoder_biases_start;

    // ---- 2. both Siamese branches (CBAM0 for T1, CBAM1 for T2) ----
    BranchTimes b1 = {0, 0, 0, 0}, b2 = {0, 0, 0, 0};
    if (!run_branch("T1", g_image1_bin_start, g_ll1_ref_start, image1, w,
                    (const data_t *)g_cbam0_weights_start, cbam_out1, &b1)) {
        printf("[FATAL] T1 branch failed\n");
        return 1;
    }
    if (!run_branch("T2", g_image2_bin_start, g_ll2_ref_start, image2, w,
                    (const data_t *)g_cbam1_weights_start, cbam_out2, &b2)) {
        printf("[FATAL] T2 branch failed\n");
        return 1;
    }

    // ---- 3. distance -> upsample -> threshold (PS) ----
    double t = ps_timer_now_s();
    ps_channelwise_distance(cbam_out1, cbam_out2, dist_lowres);
    ps_bilinear_resize_1ch(dist_lowres, final_dist);
    const float THRESHOLD = 1.0f;                       // value used by the paper
    ps_threshold_change_map(final_dist, change_map, OUT_H * OUT_W, THRESHOLD);
    double t_post = ps_timer_now_s() - t;

    long changed = 0;
    double dsum = 0.0, dmin = 1e30, dmax = -1e30;
    for (int i = 0; i < OUT_H * OUT_W; i++) {
        if (change_map[i]) changed++;
        double d = final_dist[i];
        dsum += d;
        if (d < dmin) dmin = d;
        if (d > dmax) dmax = d;
    }
    const double N = (double)OUT_H * OUT_W;

    printf("\n============================================================\n");
    printf(" RESULT   (mode %d)\n", PIPELINE_MODE);
    printf("============================================================\n");
    printf("Changed pixels : %ld / %d (%.2f %%)\n", changed, OUT_H * OUT_W, 100.0 * changed / N);
    printf("Distance       : mean %.4f  min %.4f  max %.4f\n", dsum / N, dmin, dmax);
    printf("Verdict        : %s\n", (100.0 * changed / N) > 1.0 ? "CHANGE DETECTED between T1 and T2" : "No significant change detected");

    // ---- 4. comparison with the PyTorch output of the UNFILTERED network ----
    if (check_size("golden_final_dist", g_golden_final_dist_start, g_golden_final_dist_end, (size_t)OUT_H * OUT_W * 4)) {
        const float *g = (const float *)g_golden_final_dist_start;
        const int n = OUT_H * OUT_W;
        double mae = 0, mse = 0, maxe = 0, st = 0, sg = 0, stg = 0, st2 = 0, sg2 = 0;
        for (int i = 0; i < n; i++) {
            double a = final_dist[i], b = g[i], e = fabs(a - b);
            mae += e; mse += e * e; if (e > maxe) maxe = e;
            st += a; sg += b; stg += a * b; st2 += a * a; sg2 += b * b;
        }
        double den  = sqrt((n * st2 - st * st) * (n * sg2 - sg * sg));
        double corr = (den != 0.0) ? (n * stg - st * sg) / den : 0.0;
#if PIPELINE_MODE == MODE_BASELINE
        printf("\n[VALIDATION vs PyTorch]  golden mean %.4f\n", sg / n);
#else
        printf("\n[SIMILARITY to the UNFILTERED PyTorch distance map - not an accuracy figure]  golden mean %.4f\n", sg / n);
#endif
        printf("MAE=%.6f  RMSE=%.6f  MaxErr=%.6f  Corr=%.6f\n", mae / n, sqrt(mse / n), maxe, corr);
    }

    printf("\nTiming (s)        T1        T2\n");
    printf("  PL filter  %9.3f %9.3f\n", b1.filter,   b2.filter);
    printf("  backbone   %9.2f %9.2f\n", b1.backbone, b2.backbone);
    printf("  decoder    %9.2f %9.2f\n", b1.decoder,  b2.decoder);
    printf("  CBAM (PL)  %9.3f %9.3f\n", b1.cbam,     b2.cbam);
    printf("  post-processing %.3f s | total %.1f s\n", t_post, ps_timer_now_s() - t_start);
    printf("============================================================\n");
    return 0;
}
