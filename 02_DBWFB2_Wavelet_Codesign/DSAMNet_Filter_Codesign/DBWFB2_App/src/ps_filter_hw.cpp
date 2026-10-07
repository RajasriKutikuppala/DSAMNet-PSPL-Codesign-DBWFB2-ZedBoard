#include "ps_filter_hw.h"
#include "ps_timer.h"
#include "xaxidma.h"
#include "xil_cache.h"
#include "xil_io.h"
#include "xstatus.h"
#include <cstdio>
#include <cstring>

#ifdef XPAR_XAXIDMA_0_BASEADDR
#define PS_DMA_BASE XPAR_XAXIDMA_0_BASEADDR
#else
#define PS_DMA_BASE 0x41E00000u
#endif

#define DMA_TIMEOUT_S 2.0

// Staging buffers in DDR, cache-line aligned, so cache maintenance is exact
alignas(64) static int8_t g_in [FILTER_IN_BYTES];
alignas(64) static int8_t g_out[FILTER_OUT_BYTES];

static XAxiDma g_dma;
static bool    g_ready = false;

static void print_dma_status(const char *why)
{
    printf("[FILTER] %s\n", why);
    printf("[FILTER]   MM2S status = 0x%08X   S2MM status = 0x%08X\n",
           (unsigned)Xil_In32(PS_DMA_BASE + 0x04), (unsigned)Xil_In32(PS_DMA_BASE + 0x34));
    printf("[FILTER]   (status bits: 0 halted, 1 idle, 4 internal err, 5 slave err, 6 decode err, 12 IOC)\n");
}

bool ps_filter_init()
{
    if (g_ready) return true;

    XAxiDma_Config *cfg = XAxiDma_LookupConfig(PS_DMA_BASE);
    if (!cfg) {
        printf("[FILTER] ERROR: no AXI DMA found at 0x%08X in the platform\n", (unsigned)PS_DMA_BASE);
        return false;
    }
    if (XAxiDma_CfgInitialize(&g_dma, cfg) != XST_SUCCESS) {
        printf("[FILTER] ERROR: XAxiDma_CfgInitialize failed\n");
        return false;
    }
    if (XAxiDma_HasSg(&g_dma)) {
        printf("[FILTER] ERROR: DMA was built with scatter-gather (expected simple mode)\n");
        return false;
    }
    XAxiDma_IntrDisable(&g_dma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DEVICE_TO_DMA);
    XAxiDma_IntrDisable(&g_dma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DMA_TO_DEVICE);

    g_ready = true;
    printf("[FILTER] DMA initialised at 0x%08X (simple mode)\n", (unsigned)PS_DMA_BASE);
    return true;
}

bool ps_filter_run(const int8_t *in, int8_t *out, FilterTiming *t)
{
    if (!g_ready) { printf("[FILTER] ERROR: ps_filter_init() not called\n"); return false; }

    // ---- prepare buffers -------------------------------------------------------------
    double t0 = ps_timer_now_s();
    memcpy(g_in, in, FILTER_IN_BYTES);
    for (int i = 0; i < FILTER_OUT_BYTES; i++) g_out[i] = (int8_t)(i * 37 + 11);   // known junk
    Xil_DCacheFlushRange((UINTPTR)g_in,  FILTER_IN_BYTES);     // PL must see the pixels
    Xil_DCacheFlushRange((UINTPTR)g_out, FILTER_OUT_BYTES);    // push junk to DDR, clean lines
    Xil_DCacheInvalidateRange((UINTPTR)g_out, FILTER_OUT_BYTES);
    double t1 = ps_timer_now_s();

    // ---- start: receiver first, then sender ----------------------------------------
    if (XAxiDma_SimpleTransfer(&g_dma, (UINTPTR)g_out, FILTER_OUT_BYTES, XAXIDMA_DEVICE_TO_DMA) != XST_SUCCESS) {
        print_dma_status("ERROR: S2MM transfer could not be started");
        return false;
    }
    if (XAxiDma_SimpleTransfer(&g_dma, (UINTPTR)g_in, FILTER_IN_BYTES, XAXIDMA_DMA_TO_DEVICE) != XST_SUCCESS) {
        print_dma_status("ERROR: MM2S transfer could not be started");
        return false;
    }

    // ---- wait for both channels ----------------------------------------------------
    while (XAxiDma_Busy(&g_dma, XAXIDMA_DEVICE_TO_DMA) || XAxiDma_Busy(&g_dma, XAXIDMA_DMA_TO_DEVICE)) {
        if (ps_timer_now_s() - t1 > DMA_TIMEOUT_S) {
            print_dma_status("ERROR: DMA timeout (the filter did not finish)");
            return false;
        }
    }
    double t2 = ps_timer_now_s();

    // ---- fetch result --------------------------------------------------------------
    Xil_DCacheInvalidateRange((UINTPTR)g_out, FILTER_OUT_BYTES);
    memcpy(out, g_out, FILTER_OUT_BYTES);
    double t3 = ps_timer_now_s();

    if (t) { t->prep_s = t1 - t0; t->dma_s = t2 - t1; t->post_s = t3 - t2; }
    return true;
}
