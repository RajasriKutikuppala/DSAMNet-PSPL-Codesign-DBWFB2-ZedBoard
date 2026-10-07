"""
accuracy_check.py : what does the FROZEN DSAMNet do with the wavelet-LL front end?  (PyTorch, float, on your PC)

It rebuilds, in PyTorch, exactly the inputs the board pipeline feeds to the network and measures:

  A) "explain the board" check on pair 00000:  decoder range, per-channel means (CBAM overflow risk),
     CBAM output means, distance mean, changed pixels, correlation with the normal float output -
     printed next to the numbers your board printed, so you can see whether the board is faithful.
  B) accuracy against the ground-truth labels (IoU, F1, precision, recall) on the first N test pairs.

Variants
  float      original image, exactly as the PyTorch golden run                (reference)
  int8       the same image quantised to int8 like the board data              (how much does int8 alone change?)
  ll_up      DBWFB2 LL (bit-exact with the PL filter) -> upsample to 256 -> DSAMNet     = board MODE 1
  avg_up     2x2 average pooling -> upsample to 256 -> DSAMNet                           = control arm
  ll_nopool  DBWFB2 LL 128x128 -> DSAMNet with the max-pool removed                      = board MODE 2

Run in the dsamnet_eval conda environment, from this folder:
  python accuracy_check.py                 (pair 00000 only, about a minute)
  python accuracy_check.py --pairs 200     (adds IoU/F1 on 200 test pairs)
  python accuracy_check.py --pairs all     (all test pairs - slow on a CPU)
It only READS your model/dataset and writes results into ..\\data .
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(r"C:\Users\HP\Desktop\BTP_RemoteSensing\DSAMNet")
TEST_DIR = ROOT / "SYSU-CD" / "test"
CKPT = ROOT / "epochs" / "netCD_epoch_12.pt"
INT8_DIR = ROOT / "hls_golden_inputs"
HERE = Path(__file__).resolve().parent
DATA_OUT = HERE.parent / "data"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from model.dsamnet import DSAMNet          # noqa: E402  (your repository)
import ll_variants as V                    # noqa: E402
from dbwfb2_ll_ref import ll_image as ll_image_ref   # noqa: E402  (bit-exact reference of the PL filter)

THRESHOLDS = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0]

# numbers printed by the board (pair 00000), for the side-by-side table
BOARD = {
    "baseline (mode 0)": dict(cbam1=0.074819, cbam2=0.099124, dist_mean=1.0960, changed=38207, corr=0.988358),
    "ll_up (mode 1)":    dict(cbam1=0.156242, cbam2=0.167712, dist_mean=1.1502, changed=42550, corr=0.656396,
                              dec_max1=5.9768, dec_max2=5.3310),
}


# ----------------------------------------------------------------------------------------- data
def load_rgb(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return ((a - 0.5) / 0.5).transpose(2, 0, 1).copy()          # same as ToTensor + Normalize(0.5, 0.5)


def load_pair(name):
    x1 = load_rgb(TEST_DIR / "time1" / name)
    x2 = load_rgb(TEST_DIR / "time2" / name)
    lab = np.asarray(Image.open(TEST_DIR / "label" / name))
    if lab.ndim == 3:
        lab = lab[..., 0]
    return x1, x2, lab > 0


def make_variants(x1, x2):
    q1, q2 = V.to_int8(x1), V.to_int8(x2)
    out = {
        "float": (x1, x2),
        "int8": (q1.astype(np.float32) / 127.0, q2.astype(np.float32) / 127.0),
    }
    l1, l2 = V.ll_image_fast(q1), V.ll_image_fast(q2)
    out["ll_up"] = (V.upsample_ps(l1, 0.0), V.upsample_ps(l2, 0.0))
    out["avg_up"] = (V.upsample_ps(V.avgpool2(q1), 0.5), V.upsample_ps(V.avgpool2(q2), 0.5))
    out["ll_nopool"] = (l1.astype(np.float32) / 127.0, l2.astype(np.float32) / 127.0)
    return out, (q1, q2, l1, l2)


# ----------------------------------------------------------------------------------------- model
def find_maxpool(module):
    for name, child in module.named_children():
        if isinstance(child, nn.MaxPool2d):
            return module, name
        r = find_maxpool(child)
        if r is not None:
            return r
    return None


class NoPool:
    """temporarily replace the stem max-pool by identity (board mode 2)"""
    def __init__(self, model):
        self.loc = find_maxpool(model.backbone)
        self.old = None

    def available(self):
        return self.loc is not None

    def __enter__(self):
        parent, name = self.loc
        self.old = getattr(parent, name)
        setattr(parent, name, nn.Identity())
        return self

    def __exit__(self, *exc):
        parent, name = self.loc
        setattr(parent, name, self.old)


@torch.no_grad()
def forward_all(model, x1, x2, dev):
    t1 = torch.from_numpy(np.ascontiguousarray(x1))[None].to(dev)
    t2 = torch.from_numpy(np.ascontiguousarray(x2))[None].to(dev)
    a, f2, f3, f4 = model.backbone(t1)
    d1 = model.decoder(a, f2, f3, f4)
    c1 = model.cbam0(d1)
    b, g2, g3, g4 = model.backbone(t2)
    d2 = model.decoder(b, g2, g3, g4)
    c2 = model.cbam1(d2)
    dist = torch.sqrt(((c1 - c2 + 1e-6) ** 2).sum(dim=1, keepdim=True))       # = F.pairwise_distance over channels
    dist = F.interpolate(dist, size=(256, 256), mode="bilinear", align_corners=True)
    return dist[0, 0].cpu().numpy(), d1, d2, c1, c2


def run_variant(model, name, pair, dev):
    x1, x2 = pair
    if name == "ll_nopool":
        np_ctx = NoPool(model)
        if not np_ctx.available():
            return None
        with np_ctx:
            return forward_all(model, x1, x2, dev)
    return forward_all(model, x1, x2, dev)


@torch.no_grad()
def cbam_pl_emulation(cbam, d, wrap):
    """CBAM with the PL's channel-sum behaviour (wrap=True) or with a wide accumulator (wrap=False); rest in float."""
    avg, mx, n_wrapped = V.pl_channel_avg_max(d[0].cpu().numpy(), wrap)
    avg_t = torch.from_numpy(avg)[None, :, None, None].to(d.device)
    mx_t = torch.from_numpy(mx)[None, :, None, None].to(d.device)
    avg_out = cbam.ca.fc2(cbam.ca.relu1(cbam.ca.fc1(avg_t)))
    max_out = cbam.ca.fc2(cbam.ca.relu1(cbam.ca.fc1(mx_t)))
    ca = cbam.ca.sigmoid(avg_out + max_out)
    x = ca * d
    return cbam.sa(x) * x, n_wrapped


@torch.no_grad()
def distance_from(c1, c2):
    dist = torch.sqrt(((c1 - c2 + 1e-6) ** 2).sum(dim=1, keepdim=True))
    dist = F.interpolate(dist, size=(256, 256), mode="bilinear", align_corners=True)
    return dist[0, 0].cpu().numpy()


# ----------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="0", help="number of test pairs for IoU/F1, or 'all' (default 0 = only the pair-00000 diagnostics)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu"))
    print(f"device: {dev}")
    DATA_OUT.mkdir(parents=True, exist_ok=True)

    model = DSAMNet(n_class=2, backbone="resnet18", output_stride=16, f_c=64)
    state = torch.load(CKPT, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(dev).eval()

    names = sorted(p.name for p in (TEST_DIR / "time1").iterdir() if p.suffix.lower() == ".png")
    print(f"{len(names)} test pairs found; first pair: {names[0]}")

    # ========================================================== A) pair 00000
    x1, x2, lab0 = load_pair(names[0])
    variants, (q1, q2, l1, l2) = make_variants(x1, x2)

    print("\n--- consistency checks on pair", names[0], "---")
    f1 = INT8_DIR / "image1_int8.bin"
    if f1.exists():
        board_q1 = np.fromfile(f1, dtype=np.int8).reshape(3, 256, 256)
        print(f"my int8 conversion vs your image1_int8.bin : {int(np.count_nonzero(board_q1 != q1))} different values of {q1.size}")
    r1 = DATA_OUT / "image1_ll_ref.bin"
    if r1.exists():
        ref_ll = np.fromfile(r1, dtype=np.int8).reshape(3, 128, 128)
        print(f"my LL vs the file the PL was checked against: {int(np.count_nonzero(ref_ll != l1))} different values of {l1.size}")
    print(f"fast LL vs bit-exact reference (dbwfb2_ll_ref)   : {int(np.count_nonzero(ll_image_ref(q1) != l1))} different values")

    results = {}
    t0 = time.time()
    for name in ["float", "int8", "ll_up", "avg_up", "ll_nopool"]:
        r = run_variant(model, name, variants[name], dev)
        if r is None:
            print(f"[{name}] skipped: no MaxPool2d found inside model.backbone")
            continue
        results[name] = r
    print(f"(five variants x two branches in {time.time() - t0:.0f} s)")

    # does my reconstruction of the final distance equal model.forward()?
    with torch.no_grad():
        ref_dist = model(torch.from_numpy(x1)[None].to(dev), torch.from_numpy(x2)[None].to(dev))[0][0, 0].cpu().numpy()
    dd = float(np.abs(ref_dist - results["float"][0]).max())
    print(f"my distance reconstruction vs model.forward(): max difference {dd:.2e}  {'OK' if dd < 1e-3 else '<-- MISMATCH, tell Claude'}")

    base = results["float"][0]
    print("\n--- pair", names[0], ": PyTorch (float) values ---")
    print(f"{'variant':10s} {'dec max T1/T2':>15s} {'max chan-mean T1/T2':>21s} {'CBAM mean T1/T2':>18s} {'dist mean':>10s} {'changed>1':>10s} {'corr vs float':>14s}")
    for name, (dist, d1, d2, c1, c2) in results.items():
        chm1, chm2 = float(d1.mean(dim=(2, 3)).max()), float(d2.mean(dim=(2, 3)).max())
        print(f"{name:10s} {float(d1.max()):7.3f}/{float(d2.max()):<7.3f} {chm1:9.3f}/{chm2:<9.3f} "
              f"{float(c1.mean()):8.4f}/{float(c2.mean()):<8.4f} {dist.mean():10.4f} {int((dist > 1.0).sum()):10d} {V.corr(dist, base):14.4f}")
    print("\n--- what your BOARD printed for the same pair ---")
    for k, b in BOARD.items():
        extra = f"  decoder max {b['dec_max1']}/{b['dec_max2']}" if "dec_max1" in b else ""
        print(f"{k:18s} CBAM mean {b['cbam1']:.4f}/{b['cbam2']:.4f}  dist mean {b['dist_mean']:.4f}  changed {b['changed']}  corr vs golden {b['corr']:.4f}{extra}")

    print("\n--- reading the table ---")
    if "ll_up" in results:
        chm = max(float(results["ll_up"][1].mean(dim=(2, 3)).max()), float(results["ll_up"][2].mean(dim=(2, 3)).max()))
        cu = V.corr(results["ll_up"][0], base)
        print(f"* PyTorch's own correlation of the filtered (ll_up) distance map with the normal one is {cu:.3f}; the board printed 0.656.")
        print("  Close to 0.656  -> the board is faithful; the lower correlation is a genuine effect of feeding a blurred image to the frozen network.")
        print("  Far from 0.656  -> something differs between the board and PyTorch for filtered input; send this output to Claude.")
        print(f"* Largest per-channel decoder mean in ll_up is {chm:.3f}. The CBAM channel sum is mean x 16384 and the PL accumulator overflows above 32768, i.e. mean > 2.0.")
        print("  " + ("RISK: close to or above 2.0 - the PL CBAM could be wrong for this input." if chm > 1.8 else "OK: well below 2.0, no accumulator overflow expected."))
        print(f"* PyTorch CBAM output means for ll_up: T1 {float(results['ll_up'][3].mean()):.4f}, T2 {float(results['ll_up'][4].mean()):.4f}  (board: 0.1562 / 0.1677)")
        # the board's own baseline already differs a little from PyTorch (distance mean 1.096 vs 0.900), so compare RATIOS
        bb, bf = BOARD["baseline (mode 0)"], BOARD["ll_up (mode 1)"]
        pt_dist = float(results["ll_up"][0].mean()) / float(base.mean())
        pt_c1 = float(results["ll_up"][3].mean()) / float(results["float"][3].mean())
        pt_c2 = float(results["ll_up"][4].mean()) / float(results["float"][4].mean())
        b_dist, b_c1, b_c2 = bf["dist_mean"] / bb["dist_mean"], bf["cbam1"] / bb["cbam1"], bf["cbam2"] / bb["cbam2"]
        print("* filtered / unfiltered ratios (robust to the board's small baseline offset):")
        print(f"    distance mean   board x{b_dist:.2f}   PyTorch x{pt_dist:.2f}")
        print(f"    CBAM mean T1    board x{b_c1:.2f}   PyTorch x{pt_c1:.2f}")
        print(f"    CBAM mean T2    board x{b_c2:.2f}   PyTorch x{pt_c2:.2f}")
        faithful = (abs(pt_c1 / b_c1 - 1) < 0.25 and abs(pt_c2 / b_c2 - 1) < 0.25 and abs(cu - 0.656) < 0.12)
        print("  VERDICT: " + ("the board's filtered-input behaviour is CONSISTENT with PyTorch - the lower correlation is a real effect of the blur."
                               if faithful else
                               "board and PyTorch DISAGREE on filtered input - something in the PL/PS path may be wrong; send this output to Claude."))

    # ---- does the PL's channel-sum wrap-around explain the board's numbers? ----
    print("\n--- emulating the PL CBAM channel sum (ap_fixed<28,16> wraps when a channel mean exceeds 2.0) ---")
    print(f"{'variant':8s} {'accumulator':14s} {'channels wrapped T1/T2':>23s} {'CBAM mean T1/T2':>18s} {'dist mean':>10s} {'changed>1':>10s} {'corr vs float':>14s}")
    emu = {}
    for name in ["float", "ll_up"]:
        if name not in results:
            continue
        _, d1, d2, _, _ = results[name]
        for wrap in (True, False):
            c1e, n1 = cbam_pl_emulation(model.cbam0, d1, wrap)
            c2e, n2 = cbam_pl_emulation(model.cbam1, d2, wrap)
            de = distance_from(c1e, c2e)
            emu[(name, wrap)] = (float(c1e.mean()), float(c2e.mean()), float(de.mean()), int((de > 1.0).sum()))
            print(f"{name:8s} {'wraps (as PL)' if wrap else 'wide (no wrap)':14s} {n1:11d}/{n2:<11d} "
                  f"{float(c1e.mean()):8.4f}/{float(c2e.mean()):<8.4f} {de.mean():10.4f} {int((de > 1.0).sum()):10d} {V.corr(de, base):14.4f}")
    for name, key in (("float", "baseline (mode 0)"), ("ll_up", "ll_up (mode 1)")):
        if (name, True) in emu:
            b = BOARD[key]; e = emu[(name, True)]
            print(f"  board {key:18s}: CBAM mean {b['cbam1']:.4f}/{b['cbam2']:.4f}  dist mean {b['dist_mean']:.4f}  changed {b['changed']}")
            ok = abs(e[2] / b["dist_mean"] - 1) < 0.08 and abs(e[1] / b["cbam2"] - 1) < 0.08
            print(f"  -> emulation with the wrap {'REPRODUCES' if ok else 'does NOT reproduce'} the board's {key} numbers "
                  f"(dist mean {e[2]:.4f} vs {b['dist_mean']:.4f}, T2 CBAM mean {e[1]:.4f} vs {b['cbam2']:.4f})")
    np.save(DATA_OUT / "pair0_dist_float.npy", base)
    if "ll_up" in results:
        np.save(DATA_OUT / "pair0_dist_ll_up.npy", results["ll_up"][0])

    # ========================================================== B) accuracy over pairs
    if args.pairs == "0":
        print("\n(run with --pairs 200 or --pairs all for IoU / F1 against the labels)")
        return
    n_pairs = len(names) if args.pairs == "all" else min(int(args.pairs), len(names))
    use = [v for v in ["float", "ll_up", "avg_up", "ll_nopool"] if v in results]
    conf = {v: V.Confusion(THRESHOLDS) for v in use}
    corr_sum = {v: 0.0 for v in use}
    print(f"\n--- accuracy on {n_pairs} test pairs (variants: {', '.join(use)}) ---")
    t0 = time.time()
    for i in range(n_pairs):
        xa, xb, lab = load_pair(names[i])
        var, _ = make_variants(xa, xb)
        ref = None
        for v in use:
            dist = run_variant(model, v, var[v], dev)[0]
            conf[v].add(dist, lab)
            if v == "float":
                ref = dist
            corr_sum[v] += V.corr(dist, ref) if ref is not None else 1.0
        if (i + 1) % 10 == 0 or i + 1 == n_pairs:
            el = time.time() - t0
            print(f"  {i + 1}/{n_pairs} pairs  ({el:.0f} s, about {el / (i + 1) * (n_pairs - i - 1):.0f} s left)")

    rows_out = []
    print(f"\n{'variant':10s} {'thr':>5s} {'IoU':>7s} {'F1':>7s} {'prec':>7s} {'recall':>7s} {'acc':>7s}   mean corr vs float")
    for v in use:
        tab = conf[v].table()
        best = max(tab, key=lambda r: r[2])
        paper = [r for r in tab if abs(r[0] - 1.0) < 1e-9][0]
        for tag, r in (("paper", paper), ("best-F1", best)):
            print(f"{v:10s} {r[0]:5.2f} {r[1]:7.4f} {r[2]:7.4f} {r[3]:7.4f} {r[4]:7.4f} {r[5]:7.4f}   {corr_sum[v] / n_pairs:.4f}   ({tag})")
            rows_out.append([v, tag, r[0], r[1], r[2], r[3], r[4], r[5], corr_sum[v] / n_pairs, n_pairs])
    with open(DATA_OUT / "accuracy_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "which", "threshold", "IoU", "F1", "precision", "recall", "accuracy", "mean_corr_vs_float", "pairs"])
        w.writerows(rows_out)
    print(f"\nsaved {DATA_OUT / 'accuracy_results.csv'}")


if __name__ == "__main__":
    main()