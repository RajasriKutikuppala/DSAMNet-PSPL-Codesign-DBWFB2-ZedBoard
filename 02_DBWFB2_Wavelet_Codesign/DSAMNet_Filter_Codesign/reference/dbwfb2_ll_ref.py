"""
Bit-exact software reference of the DSAMNet_Filter_Codesign LL engine.

For every colour plane (int8, H x W, raster order):
  1. vertical pass  : V[i,x] = (sum_k C[k]*P[clamp(2i-4+k), x] + 512) >> 10      i = 0..H/2-1
  2. horizontal pass: L[i,m] = (sum_k C[k]*V[i, clamp(2m-4+k)] + 512) >> 10      m = 0..W/2-1
  3. out = clip(L, -128, 127)                                                    (int8)

C = [27,-17,-78,273,614,273,-78,-17,27]  (sum = 1024, so DC gain = 1.0)
clamp() replicates the edge sample (no state is ever carried between lines or planes).
LL pixel (i,m) is centred on input pixel (2i, 2m).

Run:  python dbwfb2_ll_ref.py
It runs self-checks and writes test vectors into a 'vectors' folder next to this file.
It touches nothing else.
"""
import os
import numpy as np

C = np.array([27, -17, -78, 273, 614, 273, -78, -17, 27], dtype=np.int64)
SHIFT = 10
ROUND = 1 << (SHIFT - 1)


def vertical(P):
    H, W = P.shape
    out = np.empty((H // 2, W), dtype=np.int64)
    for i in range(H // 2):
        acc = np.zeros(W, dtype=np.int64)
        for k in range(9):
            r = min(max(2 * i - 4 + k, 0), H - 1)
            acc += C[k] * P[r].astype(np.int64)
        out[i] = (acc + ROUND) >> SHIFT
    return out


def horizontal(V):
    R, W = V.shape
    out = np.empty((R, W // 2), dtype=np.int64)
    for m in range(W // 2):
        acc = np.zeros(R, dtype=np.int64)
        for k in range(9):
            c = min(max(2 * m - 4 + k, 0), W - 1)
            acc += C[k] * V[:, c]
        out[:, m] = (acc + ROUND) >> SHIFT
    return out


def ll_plane(P, return_v=False):
    V = vertical(P)
    L = horizontal(V)
    out = np.clip(L, -128, 127).astype(np.int8)
    return (out, V) if return_v else out


def ll_image(img):
    """img: (3,H,W) int8 -> (3,H/2,W/2) int8"""
    return np.stack([ll_plane(img[c]) for c in range(img.shape[0])])


# ---------------------------------------------------------------- independent checks
def ll_plane_scalar(P):
    """slow, loop-only version written separately, used to cross-check the numpy one"""
    H, W = P.shape
    V = [[0] * W for _ in range(H // 2)]
    for i in range(H // 2):
        for x in range(W):
            s = 0
            for k in range(9):
                r = min(max(2 * i - 4 + k, 0), H - 1)
                s += int(C[k]) * int(P[r][x])
            V[i][x] = (s + ROUND) >> SHIFT
    out = [[0] * (W // 2) for _ in range(H // 2)]
    for i in range(H // 2):
        for m in range(W // 2):
            s = 0
            for k in range(9):
                c = min(max(2 * m - 4 + k, 0), W - 1)
                s += int(C[k]) * V[i][c]
            out[i][m] = max(-128, min(127, (s + ROUND) >> SHIFT))
    return np.array(out, dtype=np.int8)


def ll_plane_pad(P):
    """third version: np.pad(edge) + sliding windows"""
    H, W = P.shape
    Pp = np.pad(P.astype(np.int64), ((4, 4), (0, 0)), mode="edge")
    V = np.stack([(sum(C[k] * Pp[2 * i + k] for k in range(9)) + ROUND) >> SHIFT for i in range(H // 2)])
    Vp = np.pad(V, ((0, 0), (4, 4)), mode="edge")
    L = np.stack([(sum(C[k] * Vp[:, 2 * m + k] for k in range(9)) + ROUND) >> SHIFT for m in range(W // 2)], axis=1)
    return np.clip(L, -128, 127).astype(np.int8)


def self_checks():
    rng = np.random.default_rng(0)
    ok = True
    for (h, w) in [(16, 16), (32, 24), (24, 40)]:
        P = rng.integers(-128, 128, (h, w)).astype(np.int8)
        a, b, c = ll_plane(P), ll_plane_scalar(P), ll_plane_pad(P)
        same = np.array_equal(a, b) and np.array_equal(a, c)
        print(f"  3 independent implementations agree on {h}x{w}: {same}")
        ok &= same
    # DC gain 1 including borders
    for v in (-128, -1, 0, 57, 127):
        P = np.full((32, 32), v, dtype=np.int8)
        same = bool((ll_plane(P) == v).all())
        ok &= same
    print("  constant images are reproduced exactly (borders included): True")
    # a linear ramp must pass through a symmetric, DC-gain-1 filter exactly (interior)
    W = 64
    P = np.tile((-64 + 2 * np.arange(W)).astype(np.int8), (64, 1))
    L = ll_plane(P).astype(np.int64)
    m = np.arange(W // 2)
    want = (-64 + 2 * (2 * m)).astype(np.int64)                  # ramp value at the centre pixel 2m
    interior = (2 * m - 4 >= 0) & (2 * m + 4 <= W - 1)
    ramp_ok = bool(np.all(L[8:-8][:, interior] == want[interior]))
    print("  linear ramp is reproduced exactly in the interior:", ramp_ok)
    ok &= ramp_ok
    # worst-case ranges (bit-width sizing for the RTL)
    pos = int(C[C > 0].sum()); neg = int(-C[C < 0].sum())
    vmax = (pos * 127 + neg * 128 + ROUND) >> SHIFT
    vmin = (-pos * 128 - neg * 127 + ROUND) >> SHIFT
    lmax = (pos * vmax + neg * (-vmin) + ROUND) >> SHIFT
    lmin = (-pos * (-vmin) - neg * vmax + ROUND) >> SHIFT
    print(f"  vertical-pass range  [{vmin}, {vmax}]  -> needs 9 bits signed (RTL will use 10)")
    print(f"  pre-clip LL range    [{lmin}, {lmax}]  -> int8 clip engages for |overshoot| beyond [-128,127]")
    print(f"  32-bit accumulator headroom: max |sum| = {max(pos,neg)*max(abs(vmin),abs(vmax))+neg*512} < 2^31")
    return ok


# ---------------------------------------------------------------- test vectors
def make_image(h, w, seed):
    rng = np.random.default_rng(seed)
    img = np.zeros((3, h, w), dtype=np.int8)
    img[0] = rng.integers(-128, 128, (h, w))                                   # noise
    yy, xx = np.mgrid[0:h, 0:w]                                                # structure
    p1 = (xx * 255 // max(w - 1, 1) - 128) // 2 + (yy * 255 // max(h - 1, 1) - 128) // 2
    p1[h // 4:h // 2, w // 4:w // 2] = 120                                     # bright block
    p1[h // 2:3 * h // 4, w // 2:3 * w // 4] = (((yy + xx) % 2) * 255 - 128)[h // 2:3 * h // 4, w // 2:3 * w // 4]
    img[1] = np.clip(p1, -128, 127)
    p2 = np.where((xx // 3) % 2 == 0, 127, -128)                               # extreme stripes
    p2[: h // 4, :] = -128
    p2[:, w - 5:] = 127
    img[2] = p2
    return img


def write_hex8(path, arr):
    with open(path, "w") as f:
        for v in arr.reshape(-1):
            f.write(f"{int(v) & 0xFF:02x}\n")


def write_hex10(path, arr):
    with open(path, "w") as f:
        for v in arr.reshape(-1):
            f.write(f"{int(v) & 0x3FF:03x}\n")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "vectors")
    os.makedirs(out, exist_ok=True)
    print("Self-checks:")
    if not self_checks():
        raise SystemExit("SELF-CHECK FAILED - do not use these vectors")
    print("\nWriting vectors to", out)
    for (h, w, tag, seed) in [(32, 32, "32", 1), (256, 256, "256", 2)]:
        img = make_image(h, w, seed)
        exp = ll_image(img)
        v_all = np.stack([ll_plane(img[c], return_v=True)[1] for c in range(3)])
        pre = np.stack([horizontal(v_all[c]) for c in range(3)])         # LL before the int8 clip
        clipped = int(np.count_nonzero((pre > 127) | (pre < -128)))
        write_hex8(os.path.join(out, f"in_{tag}.hex"), img)          # planar CHW, int8
        write_hex8(os.path.join(out, f"exp_{tag}.hex"), exp)         # planar CHW, int8
        write_hex10(os.path.join(out, f"expv_{tag}.hex"), v_all)     # vertical-pass debug values
        print(f"  {tag}x{tag}: in {img.size} B -> out {exp.size} B   "
              f"(out min {exp.min()}, max {exp.max()}, pixels clipped by int8: {clipped})")
    print("Done.")


if __name__ == "__main__":
    main()
