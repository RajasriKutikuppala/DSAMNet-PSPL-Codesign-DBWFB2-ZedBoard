#include "cbam_accel.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ============================================================================
// C-simulation testbench for the CBAM IP, four real cases:
//   1. original image,  branch 1 (cbam0) : the case that always passed
//   2. original image,  branch 2 (cbam1) : decoder channel mean up to 2.23 -> overflowed the old 28-bit channel sum
//   3. filtered (LL) image, branch 1
//   4. filtered (LL) image, branch 2     : channel mean up to 2.05
// Inputs and reference outputs are the REAL PyTorch decoder / CBAM tensors (made by
// reference/make_cbam_goldens.py, except case 1 which uses your original golden files).
// ============================================================================
static const char *SHARED = "C:/Users/HP/Desktop/BTP_RemoteSensing/DSAMNet";
static const char *DATA   = "C:/Users/HP/Desktop/BTP_RemoteSensing/DSAMNet_Filter_Codesign/data";

static bool load_binary(const char *filename, void *buffer, size_t bytes)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) { printf("  ERROR: cannot open %s\n", filename); return false; }
    size_t n = fread(buffer, 1, bytes, fp);
    fclose(fp);
    if (n != bytes) { printf("  ERROR: %s expected %zu bytes, got %zu\n", filename, bytes, n); return false; }
    return true;
}

static const int N = DEC_OC * DEC_H * DEC_W;

static bool run_case(const char *label, const char *dec_file, const char *weights_file, const char *gold_file)
{
    static data_t in[N], after_ca[N], out[N];
    static data_t weights[CBAM_WEIGHT_COUNT];
    static float f_in[N], f_gold[N], f_w[CBAM_WEIGHT_COUNT];

    printf("\n=== %s ===\n", label);
    if (!load_binary(dec_file, f_in, N * sizeof(float)))           return false;
    if (!load_binary(weights_file, f_w, CBAM_WEIGHT_COUNT * sizeof(float))) return false;
    if (!load_binary(gold_file, f_gold, N * sizeof(float)))        return false;
    for (int i = 0; i < N; i++) in[i] = data_t(f_in[i]);
    for (int i = 0; i < CBAM_WEIGHT_COUNT; i++) weights[i] = data_t(f_w[i]);

    // largest channel mean of this input (the old channel sum wrapped above 2.0)
    double max_cm = 0;
    for (int c = 0; c < DEC_OC; c++) {
        double s = 0;
        for (int p = 0; p < DEC_H * DEC_W; p++) s += f_in[c * DEC_H * DEC_W + p];
        s /= (DEC_H * DEC_W);
        if (s > max_cm) max_cm = s;
    }

    cbam_accel(in, weights, after_ca, out);

    double mae = 0, mse = 0, maxerr = 0, st = 0, sg = 0, stg = 0, st2 = 0, sg2 = 0;
    for (int i = 0; i < N; i++) {
        double a = (double)out[i], b = f_gold[i], e = fabs(a - b);
        mae += e; mse += e * e; if (e > maxerr) maxerr = e;
        st += a; sg += b; stg += a * b; st2 += a * a; sg2 += b * b;
    }
    mae /= N; mse = sqrt(mse / N);
    double den = sqrt((N * st2 - st * st) * (N * sg2 - sg * sg));
    double corr = (den != 0) ? (N * stg - st * sg) / den : 0;
    printf("  largest channel mean of the input: %.3f %s\n", max_cm, max_cm > 2.0 ? "(above 2.0: the old 28-bit sum would wrap)" : "");
    printf("  out mean %.6f | golden mean %.6f\n", st / N, sg / N);
    printf("  Corr=%.6f  MAE=%.6f  RMSE=%.6f  MaxErr=%.6f\n", corr, mae, mse, maxerr);
    bool ok = (corr >= 0.9999) && (mae <= 0.003);
    printf("  -> %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

int main()
{
    char dec[512], w0[512], w1[512], gold[512];
    snprintf(w0, sizeof(w0), "%s/hls_weights_flat/weights_cbam0_flat.bin", SHARED);
    snprintf(w1, sizeof(w1), "%s/hls_weights_flat/weights_cbam1_flat.bin", SHARED);

    bool all = true;

    snprintf(dec,  sizeof(dec),  "%s/golden_reference/golden_decoder_branch1.bin", SHARED);
    snprintf(gold, sizeof(gold), "%s/golden_reference/golden_cbam0_branch1.bin", SHARED);
    all &= run_case("1. original image, branch 1 (cbam0)", dec, w0, gold);

    snprintf(dec,  sizeof(dec),  "%s/g_base_b2_dec.bin", DATA);
    snprintf(gold, sizeof(gold), "%s/g_base_b2_cbam.bin", DATA);
    all &= run_case("2. original image, branch 2 (cbam1)", dec, w1, gold);

    snprintf(dec,  sizeof(dec),  "%s/g_llup_b1_dec.bin", DATA);
    snprintf(gold, sizeof(gold), "%s/g_llup_b1_cbam.bin", DATA);
    all &= run_case("3. filtered (LL) image, branch 1 (cbam0)", dec, w0, gold);

    snprintf(dec,  sizeof(dec),  "%s/g_llup_b2_dec.bin", DATA);
    snprintf(gold, sizeof(gold), "%s/g_llup_b2_cbam.bin", DATA);
    all &= run_case("4. filtered (LL) image, branch 2 (cbam1)", dec, w1, gold);

    printf("\n%s\n", all ? "ALL CASES PASSED" : "SOME CASES FAILED");
    return all ? 0 : 1;
}
