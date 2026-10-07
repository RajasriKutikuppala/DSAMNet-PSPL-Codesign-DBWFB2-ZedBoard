"""
make_cbam_goldens.py : real PyTorch decoder inputs and CBAM outputs for the HLS C-simulation of the overflow fix.

For test pair 00000 it writes, into ..\\data, the float32 tensors (64 x 128 x 128) of
   g_base_b2_*  : original image, branch 2  (decoder output 'dec' and CBAM output 'cbam')
   g_llup_b1_*  : filtered (LL, upsampled) image, branch 1
   g_llup_b2_*  : filtered (LL, upsampled) image, branch 2
(the original-image branch 1 already exists in your DSAMNet\\golden_reference folder).
Run in the dsamnet_eval environment, from this folder:   python make_cbam_goldens.py
It only READS your model and dataset.
"""
import numpy as np
import torch

import accuracy_check as A      # same model loading, same input variants as the accuracy study


def save(path, t):
    arr = t[0].detach().cpu().numpy().astype(np.float32)
    assert arr.shape == (64, 128, 128), arr.shape
    arr.tofile(path)
    cm = arr.reshape(64, -1).mean(axis=1)
    print(f"  {path.name:22s} {arr.nbytes // 1024} KB   max {arr.max():.3f}   largest channel mean {cm.max():.3f}")


def main():
    dev = torch.device("cpu")
    model = A.DSAMNet(n_class=2, backbone="resnet18", output_stride=16, f_c=64)
    state = torch.load(A.CKPT, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(dev).eval()
    A.DATA_OUT.mkdir(parents=True, exist_ok=True)

    names = sorted(p.name for p in (A.TEST_DIR / "time1").iterdir() if p.suffix.lower() == ".png")
    x1, x2, _ = A.load_pair(names[0])
    variants, _ = A.make_variants(x1, x2)
    print("pair", names[0])
    for tag, var, branches in (("base", "float", (2,)), ("llup", "ll_up", (1, 2))):
        _, d1, d2, c1, c2 = A.run_variant(model, var, variants[var], dev)
        for b in branches:
            d, c = (d1, c1) if b == 1 else (d2, c2)
            save(A.DATA_OUT / f"g_{tag}_b{b}_dec.bin", d)
            save(A.DATA_OUT / f"g_{tag}_b{b}_cbam.bin", c)
    print("done")


if __name__ == "__main__":
    main()
