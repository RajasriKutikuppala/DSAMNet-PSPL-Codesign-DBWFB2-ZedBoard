"""
eval_ft.py : fair before/after test of the fine-tuned DSAMNet on pairs it NEVER trained or validated on.

Compares, on the first N test pairs (default 200 = the same pairs as the earlier accuracy_check result):
  1. original weights, normal sharp image            (the reference, ~78% F1)
  2. original weights, board-style filtered image    (what you had before fine-tuning)
  3. fine-tuned weights, board-style filtered image  (after fine-tuning)

Run from the reference folder, in the dsamnet_eval environment:
    python eval_ft.py                    (200 pairs, about 10 minutes on CPU)
    python eval_ft.py --pairs 500
Pairs 0-999 were never used for training or for picking the best epoch.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--root", default=r"C:\Users\HP\Desktop\BTP_RemoteSensing\DSAMNet")
ap.add_argument("--orig", default=None, help="original checkpoint (default <root>/epochs/netCD_epoch_12.pt)")
ap.add_argument("--ft", default=None, help="fine-tuned checkpoint (default <root>/epochs/netCD_ll_ft_best.pt)")
ap.add_argument("--pairs", type=int, default=200)
opt = ap.parse_args()

ROOT = Path(opt.root)
ORIG = Path(opt.orig) if opt.orig else ROOT / "epochs" / "netCD_epoch_12.pt"
FT = Path(opt.ft) if opt.ft else ROOT / "epochs" / "netCD_ll_ft_best.pt"
if opt.pairs > 1000:
    sys.exit("Use at most 1000 pairs: pairs 1000+ were used for fine-tuning / validation.")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from model.dsamnet import DSAMNet      # noqa: E402
import ll_variants as V                # noqa: E402

THRESHOLDS = [0.8, 1.0, 1.2, 1.4]
TEST = ROOT / "SYSU-CD" / "test"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(path):
    m = DSAMNet(2).to(dev, dtype=torch.float)
    sd = torch.load(path, map_location="cpu")
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    m.load_state_dict(sd)
    return m.eval()


def load_rgb(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return ((a - 0.5) / 0.5).transpose(2, 0, 1).copy()


def board_input(x):
    return V.upsample_ps(V.ll_image_fast(V.to_int8(x)), 0.0)


@torch.no_grad()
def dist(model, a, b):
    t1 = torch.from_numpy(np.ascontiguousarray(a))[None].to(dev, dtype=torch.float)
    t2 = torch.from_numpy(np.ascontiguousarray(b))[None].to(dev, dtype=torch.float)
    return model(t1, t2)[0][0, 0].cpu().numpy()


def main():
    m_orig, m_ft = load_model(ORIG), load_model(FT)
    names = sorted(p.name for p in (TEST / "time1").iterdir() if p.suffix.lower() in (".png", ".jpg", ".tif"))
    conf = {k: V.Confusion(THRESHOLDS) for k in ("orig_sharp", "orig_filtered", "ft_filtered")}
    t0 = time.time()
    for i in range(opt.pairs):
        n = names[i]
        x1, x2 = load_rgb(TEST / "time1" / n), load_rgb(TEST / "time2" / n)
        lab = np.asarray(Image.open(TEST / "label" / n))
        if lab.ndim == 3:
            lab = lab[..., 0]
        gt = lab > 0
        f1, f2 = board_input(x1), board_input(x2)
        conf["orig_sharp"].add(dist(m_orig, x1, x2), gt)
        conf["orig_filtered"].add(dist(m_orig, f1, f2), gt)
        conf["ft_filtered"].add(dist(m_ft, f1, f2), gt)
        if (i + 1) % 20 == 0 or i + 1 == opt.pairs:
            el = time.time() - t0
            print("  %d/%d pairs (%.0f s, about %.0f s left)" % (i + 1, opt.pairs, el, el / (i + 1) * (opt.pairs - i - 1)), flush=True)

    labels = {"orig_sharp": "original weights, sharp image (reference)",
              "orig_filtered": "original weights, filtered image (before fine-tuning)",
              "ft_filtered": "fine-tuned weights, filtered image (after fine-tuning)"}
    print("\nResults on the first %d test pairs (never used for training)" % opt.pairs)
    for k in conf:
        print("\n  " + labels[k])
        print("    thr     IoU      F1    prec  recall")
        for th, iou, f1, pr, rc, oa in conf[k].table():
            print("    %.2f  %.4f  %.4f  %.4f  %.4f" % (th, iou, f1, pr, rc))
    print("\nBest F1:")
    for k in conf:
        b = max(conf[k].table(), key=lambda r: r[2])
        p = [r for r in conf[k].table() if abs(r[0] - 1.0) < 1e-9][0]
        print("  %-55s  F1 %.4f at thr %.2f   |   F1 %.4f at thr 1.00 (paper value)" % (labels[k], b[2], b[0], p[2]))


if __name__ == "__main__":
    main()
