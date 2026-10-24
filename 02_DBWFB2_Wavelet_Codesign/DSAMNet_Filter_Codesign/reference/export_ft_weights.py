"""
export_ft_weights.py : turn a PyTorch checkpoint into the flat weight files the board app embeds.

It folds every BatchNorm into the convolution before it (same as the board needs), and writes the SAME six files
and the SAME layer order as your existing  DSAMNet\\hls_weights_flat  folder - but into a NEW folder, so your
originals are never touched.

STEP 1  prove the exporter is right (no files are written, nothing changes):
    python export_ft_weights.py --validate
        -> exports the ORIGINAL checkpoint in memory and compares it with your existing hls_weights_flat files.
           Every line must say OK. If any line says MISMATCH, stop and send the output to Claude.

STEP 2  export the fine-tuned weights + the matching reference output:
    python export_ft_weights.py
        -> writes  DSAMNet\\hls_weights_flat_ft\\*.bin   (6 files)
           and     DSAMNet\\golden_reference_ft\\final_dist_256x256.bin
              (PyTorch output of the FINE-TUNED model for the board's own filtered input of pair 00000;
               the board compares its result with this file)

Run from the reference folder, in the dsamnet_eval environment.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--root", default=r"C:\Users\HP\Desktop\BTP_RemoteSensing\DSAMNet")
ap.add_argument("--orig", default=None, help="original checkpoint (default <root>/epochs/netCD_epoch_12.pt)")
ap.add_argument("--ft", default=None, help="fine-tuned checkpoint (default <root>/epochs/netCD_ll_ft_best.pt)")
ap.add_argument("--validate", action="store_true", help="only check the exporter against your existing flat files")
opt = ap.parse_args()

ROOT = Path(opt.root)
ORIG = Path(opt.orig) if opt.orig else ROOT / "epochs" / "netCD_epoch_12.pt"
FT = Path(opt.ft) if opt.ft else ROOT / "epochs" / "netCD_ll_ft_best.pt"
FLAT_ORIG = ROOT / "hls_weights_flat"
FLAT_FT = ROOT / "hls_weights_flat_ft"
GOLD_FT = ROOT / "golden_reference_ft"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from model.dsamnet import DSAMNet      # noqa: E402
import ll_variants as V                # noqa: E402

EXPECT = {"weights_backbone_flat.bin": 15885504, "biases_backbone_flat.bin": 5824,
          "weights_decoder_flat.bin": 993280, "biases_decoder_flat.bin": 704,
          "weights_cbam0_flat.bin": 1122, "weights_cbam1_flat.bin": 1122}


def load_model(path):
    m = DSAMNet(2)
    sd = torch.load(path, map_location="cpu")
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    m.load_state_dict(sd)
    return m.eval()


def folded_convs(module):
    """Every Conv2d of `module`, in definition order, with the BatchNorm that directly follows it folded in.
    Returns [(name, weight float32 [O,I,kh,kw], bias float32 [O])]."""
    mods = list(module.named_modules())
    out = []
    for i, (name, m) in enumerate(mods):
        if not isinstance(m, nn.Conv2d):
            continue
        w = m.weight.detach().double()
        b = m.bias.detach().double() if m.bias is not None else torch.zeros(w.shape[0], dtype=torch.double)
        nxt = mods[i + 1][1] if i + 1 < len(mods) else None
        if isinstance(nxt, nn.BatchNorm2d):
            s = nxt.weight.detach().double() / torch.sqrt(nxt.running_var.detach().double() + nxt.eps)
            w = w * s[:, None, None, None]
            b = (b - nxt.running_mean.detach().double()) * s + nxt.bias.detach().double()
        out.append((name, w.float().numpy(), b.float().numpy()))
    return out


def export(model):
    bb = folded_convs(model.backbone)
    dc = folded_convs(model.decoder)
    files = {
        "weights_backbone_flat.bin": np.concatenate([w.ravel() for _, w, _ in bb]),
        "biases_backbone_flat.bin": np.concatenate([b.ravel() for _, _, b in bb]),
        "weights_decoder_flat.bin": np.concatenate([w.ravel() for _, w, _ in dc]),
        "biases_decoder_flat.bin": np.concatenate([b.ravel() for _, _, b in dc]),
    }
    for k, cb in enumerate((model.cbam0, model.cbam1)):
        files["weights_cbam%d_flat.bin" % k] = np.concatenate([
            cb.ca.fc1.weight.detach().numpy().ravel(),
            cb.ca.fc2.weight.detach().numpy().ravel(),
            cb.sa.conv1.weight.detach().numpy().ravel()]).astype(np.float32)
    info = "backbone convs: %d, decoder convs: %d" % (len(bb), len(dc))
    return {k: v.astype(np.float32) for k, v in files.items()}, info


def check_sizes(files):
    ok = True
    for k, n in EXPECT.items():
        good = files[k].size == n
        ok &= good
        print("  %-28s %9d values (expected %d)  %s" % (k, files[k].size, n, "OK" if good else "WRONG SIZE"))
    return ok


def main():
    if opt.validate:
        print("exporting the ORIGINAL checkpoint in memory and comparing it with", FLAT_ORIG)
        files, info = export(load_model(ORIG))
        print(" ", info)
        allok = check_sizes(files)
        for k, a in files.items():
            ref = np.fromfile(FLAT_ORIG / k, dtype=np.float32)
            if ref.size != a.size:
                print("  %-28s MISMATCH: your file has %d values, the exporter made %d" % (k, ref.size, a.size)); allok = False; continue
            d = float(np.abs(ref - a).max())
            good = d < 1e-4
            allok &= good
            print("  %-28s max difference %.3e   %s" % (k, d, "OK" if good else "MISMATCH"))
        print("\nRESULT:", "exporter matches your existing weights - safe to run STEP 2." if allok
              else "NOT matching - do not use the exported files; send this output to Claude.")
        return

    print("exporting the FINE-TUNED checkpoint", FT)
    files, info = export(load_model(FT))
    print(" ", info)
    if not check_sizes(files):
        sys.exit("wrong sizes - nothing written")
    FLAT_FT.mkdir(exist_ok=True)
    for k, a in files.items():
        a.tofile(FLAT_FT / k)
    print("wrote 6 files to", FLAT_FT)

    # reference output for the board: fine-tuned model, board-style filtered input of the board's own pair
    q1 = np.fromfile(ROOT / "hls_golden_inputs" / "image1_int8.bin", dtype=np.int8).reshape(3, 256, 256)
    q2 = np.fromfile(ROOT / "hls_golden_inputs" / "image2_int8.bin", dtype=np.int8).reshape(3, 256, 256)
    x1, x2 = V.upsample_ps(V.ll_image_fast(q1), 0.0), V.upsample_ps(V.ll_image_fast(q2), 0.0)
    m = load_model(FT)
    with torch.no_grad():
        d = m(torch.from_numpy(np.ascontiguousarray(x1))[None].float(),
              torch.from_numpy(np.ascontiguousarray(x2))[None].float())[0][0, 0].numpy().astype(np.float32)
    GOLD_FT.mkdir(exist_ok=True)
    d.tofile(GOLD_FT / "final_dist_256x256.bin")
    print("wrote", GOLD_FT / "final_dist_256x256.bin", "  distance mean %.4f, pixels above 1.0: %d" % (d.mean(), int((d > 1.0).sum())))
    g0 = ROOT / "golden_reference" / "final_dist_256x256.bin"
    if g0.exists():
        o = np.fromfile(g0, dtype=np.float32)
        print("for reference, the old UNFILTERED golden has distance mean %.4f" % o.mean())


if __name__ == "__main__":
    main()
