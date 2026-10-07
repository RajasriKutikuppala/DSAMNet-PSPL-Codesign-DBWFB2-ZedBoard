"""
Makes the two reference LL files that the hardware test compares against.

Reads  : <SRC>/image1_int8.bin and image2_int8.bin   (3x256x256 int8, planar) - READ ONLY
Writes : <this project>/data/image1_ll_ref.bin and image2_ll_ref.bin (3x128x128 int8)

Run:  python make_ll_golden.py
Optional: python make_ll_golden.py <other source folder>
"""
import os
import sys
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from dbwfb2_ll_ref import ll_image          # the bit-exact reference

SRC = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\HP\Desktop\BTP_RemoteSensing\DSAMNet\hls_golden_inputs"
OUT = os.path.join(os.path.dirname(here), "data")
os.makedirs(OUT, exist_ok=True)

for k in (1, 2):
    src = os.path.join(SRC, f"image{k}_int8.bin")
    raw = np.fromfile(src, dtype=np.int8)
    if raw.size != 3 * 256 * 256:
        raise SystemExit(f"{src}: expected {3*256*256} bytes, found {raw.size}")
    img = raw.reshape(3, 256, 256)
    ll = ll_image(img)
    dst = os.path.join(OUT, f"image{k}_ll_ref.bin")
    ll.tofile(dst)
    print(f"image{k}: input range [{img.min()}, {img.max()}] -> LL {ll.shape}, "
          f"range [{ll.min()}, {ll.max()}], mean {ll.mean():.3f}  ->  {dst} ({ll.size} bytes)")
print("Done.")
