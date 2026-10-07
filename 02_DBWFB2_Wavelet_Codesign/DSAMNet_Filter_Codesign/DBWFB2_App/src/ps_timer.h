#ifndef PS_TIMER_H
#define PS_TIMER_H
// 64-bit timer from the Cortex-A9 global timer (0xF8F00200). It runs at CPU clock / 2
// (333.33 MHz here) and does not wrap during a run. No Xilinx driver needed.
#include <stdint.h>

#define PS_GT_BASE        0xF8F00200u
#define PS_GT_HZ_NOPRESC  333333343.0     // XPAR_CPU_CORE_CLOCK_FREQ_HZ / 2

static inline void ps_timer_init(void)
{
    volatile uint32_t *ctrl = (volatile uint32_t *)(PS_GT_BASE + 0x08);
    *ctrl |= 1u;                          // enable counting, keep the other bits
}

static inline uint64_t ps_timer_ticks(void)
{
    volatile uint32_t *lo = (volatile uint32_t *)(PS_GT_BASE + 0x00);
    volatile uint32_t *hi = (volatile uint32_t *)(PS_GT_BASE + 0x04);
    uint32_t h1, l, h2;
    do { h1 = *hi; l = *lo; h2 = *hi; } while (h1 != h2);
    return ((uint64_t)h1 << 32) | l;
}

static inline double ps_timer_now_s(void)
{
    volatile uint32_t *ctrl = (volatile uint32_t *)(PS_GT_BASE + 0x08);
    double presc = 1.0 + (double)((*ctrl >> 8) & 0xFFu);
    return (double)ps_timer_ticks() * presc / PS_GT_HZ_NOPRESC;
}
#endif
