#ifndef PS_CBAM_HW_H
#define PS_CBAM_HW_H

#include "ps_dsamnet.h"

// Runs CBAM on the PL accelerator.
//   in      : DEC_OC x DEC_H x DEC_W decoder output   (float, PS format)
//   weights : CBAM_WEIGHT_COUNT floats
//   out     : DEC_OC x DEC_H x DEC_W CBAM output      (float, PS format)
//
// The IP works in ap_fixed<16,4> (Q4.12, 16 bits/element), so this function
// converts float -> Q4.12 into aligned DDR staging buffers, runs the IP, and
// converts the result back to float. Returns false on timeout or if the
// hardware did not write the whole output buffer.
bool ps_cbam_hw(const data_t *in, const data_t *weights, data_t *out);

#endif
