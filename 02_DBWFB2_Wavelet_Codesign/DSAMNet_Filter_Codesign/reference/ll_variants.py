"""
Torch-free helpers for the accuracy study: the exact input variants that the board pipeline feeds to
DSAMNet, plus the metric code.  Everything here is plain numpy so it can be tested without PyTorch.

  to_int8        float image in [-1,1]  -> int8 exactly like the board's image*_int8.bin  (round(127 x), clip +-127)
  ll_image_fast  int8 (3,H,W)           -> int8 LL (3,H/2,W/2); bit-identical to dbwfb2_ll_ref.ll_image / the PL filter
  avgpool2       int8 (3,H,W)           -> float 2x2 average (the control arm: same data reduction, no wavelet)
  upsample_ps    LL / pooled image      -> (3,2H,2W) bilinear, exactly the PS rule in ps_ll_prep.cpp
"""
import numpy as np

TAPS = np.array([27, -17, -78, 273, 614, 273, -78, -17, 27], dtype=np.int64)


def to_int8(x):
    """x: float array in [-1,1] -> int8, same rule as the board data (scale 1/127)."""
    return np.clip(np.rint(x * 127.0), -127, 127).astype(np.int8)


def ll_plane_fast(P):
    """one int8 plane (H,W) -> int8 (H/2,W/2). Vertical pass, then horizontal pass; edge-replicated; (sum+512)>>10; clip."""
    H, W = P.shape
    Pp = np.pad(P.astype(np.int64), ((4, 4), (0, 0)), mode="edge")
    V = np.zeros((H // 2, W), dtype=np.int64)
    for k in range(9):
        V += TAPS[k] * Pp[k:k + H:2]
    V = (V + 512) >> 10
    Vp = np.pad(V, ((0, 0), (4, 4)), mode="edge")
    L = np.zeros((H // 2, W // 2), dtype=np.int64)
    for k in range(9):
        L += TAPS[k] * Vp[:, k:k + W:2]
    L = (L + 512) >> 10
    return np.clip(L, -128, 127).astype(np.int8)


def ll_image_fast(img):
    return np.stack([ll_plane_fast(img[c]) for c in range(img.shape[0])])


def avgpool2(img):
    """int8 (3,H,W) -> float32 (3,H/2,W/2), plain 2x2 mean."""
    c, h, w = img.shape
    return img.astype(np.float32).reshape(c, h // 2, 2, w // 2, 2).mean(axis=(2, 4)).astype(np.float32)


def _lerp_coords(n_out, n_in, shift):
    u = (np.arange(n_out, dtype=np.float32) - np.float32(shift)) * np.float32(0.5)
    u = np.clip(u, 0.0, n_in - 1.0).astype(np.float32)
    i0 = np.floor(u).astype(np.int64)
    i1 = np.minimum(i0 + 1, n_in - 1)
    w = (u - i0).astype(np.float32)
    return i0, i1, w


def upsample_ps(a, shift=0.0):
    """
    a: (3,h,w) -> float32 (3,2h,2w), values divided by 127 (network input scale).
    Output pixel y reads source coordinate (y - shift)/2, linear interpolation, edge repeated.
    shift = 0.0 for the wavelet LL (LL pixel i is centred on input pixel 2i, this is the PS code);
    shift = 0.5 for a 2x2 average (pixel i is centred between input pixels 2i and 2i+1).
    """
    a = a.astype(np.float32)
    c, h, w = a.shape
    y0, y1, wy = _lerp_coords(2 * h, h, shift)
    x0, x1, wx = _lerp_coords(2 * w, w, shift)
    p00 = a[:, y0][:, :, x0]; p01 = a[:, y0][:, :, x1]
    p10 = a[:, y1][:, :, x0]; p11 = a[:, y1][:, :, x1]
    top = p00 + (p01 - p00) * wx[None, None, :]
    bot = p10 + (p11 - p10) * wx[None, None, :]
    out = top + (bot - top) * wy[None, :, None]
    return (out / np.float32(127.0)).astype(np.float32)


def corr(a, b):
    a = np.asarray(a, dtype=np.float64).ravel(); b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


class Confusion:
    """accumulates TP/FP/FN/TN for several thresholds"""
    def __init__(self, thresholds):
        self.t = list(thresholds)
        self.tp = np.zeros(len(self.t), dtype=np.int64); self.fp = self.tp.copy()
        self.fn = self.tp.copy(); self.tn = self.tp.copy()

    def add(self, dist, label):
        lab = label.astype(bool)
        for i, th in enumerate(self.t):
            pred = dist > th
            self.tp[i] += int(np.count_nonzero(pred & lab)); self.fp[i] += int(np.count_nonzero(pred & ~lab))
            self.fn[i] += int(np.count_nonzero(~pred & lab)); self.tn[i] += int(np.count_nonzero(~pred & ~lab))

    def table(self):
        rows = []
        for i, th in enumerate(self.t):
            tp, fp, fn, tn = (int(self.tp[i]), int(self.fp[i]), int(self.fn[i]), int(self.tn[i]))
            iou = tp / max(tp + fp + fn, 1); f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1); oa = (tp + tn) / max(tp + fp + fn + tn, 1)
            rows.append((th, iou, f1, pr, rc, oa))
        return rows


def pl_channel_avg_max(d, wrap=True):
    """
    The first step of the PL CBAM (channel attention), as the HLS code does it.
      d : float32 decoder output (C,H,W)
      data_t = ap_fixed<16,4>  -> values are integers in units of 2^-12
      the per-channel sum is accumulated in acc_t = ap_fixed<28,16> -> 28 bits, WRAPS when it reaches 2^15 (a channel mean of 2.0)
      avg = data_t(sum / (H*W))  (truncated toward minus infinity)
    returns avg (C,), max (C,), number of channels whose sum wrapped
    """
    c, h, w = d.shape
    q = np.clip(np.rint(d.astype(np.float64) * 4096.0), -32768, 32767).astype(np.int64).reshape(c, -1)
    s = q.sum(axis=1)
    wrapped = ((s + (1 << 27)) % (1 << 28)) - (1 << 27) if wrap else s
    avg = np.floor_divide(wrapped, h * w).astype(np.float64) / 4096.0
    mx = q.max(axis=1).astype(np.float64) / 4096.0
    return avg.astype(np.float32), mx.astype(np.float32), int(np.count_nonzero(wrapped != s))