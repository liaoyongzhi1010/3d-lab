"""E-679 verification: compute baseline-vs-GT and fixed-vs-GT PSNR per eval frame.

GT: medium_scenes/<scene>/images/gt_{idx}.png
Pred: <out>/{baseline,fixed}_{idx}.png
Both resized to GT size, 5% border crop, then PSNR. Also flags black/NaN.
"""

import argparse
import json
import os

import numpy as np
from PIL import Image

EVAL_IDXS = [15, 30, 50, 69, 84, 99]


def crop_border(arr, frac=0.05):
    h, w = arr.shape[:2]
    dh, dw = int(round(h * frac)), int(round(w * frac))
    return arr[dh : h - dh, dw : w - dw]


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse <= 1e-12:
        return 99.0
    return 10.0 * np.log10((255.0**2) / mse)


def load_resize(path, size):
    img = Image.open(path).convert("RGB").resize(size, Image.Resampling.LANCZOS)
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scene-images", required=True, help="medium_scenes/<scene>/images"
    )
    ap.add_argument("--out", required=True, help="dir with baseline_/fixed_ pngs")
    ap.add_argument("--name", default="")
    args = ap.parse_args()

    rows = []
    for idx in EVAL_IDXS:
        gt_path = os.path.join(args.scene_images, f"gt_{idx}.png")
        gt = Image.open(gt_path).convert("RGB")
        gt_size = gt.size
        gt_np = crop_border(np.asarray(gt))

        row = {"idx": idx}
        for kind in ("baseline", "fixed"):
            p = os.path.join(args.out, f"{kind}_{idx:03d}.png")
            arr_full = np.asarray(Image.open(p).convert("RGB"))
            pred = crop_border(load_resize(p, gt_size))
            row[f"{kind}_psnr"] = round(psnr(pred, gt_np), 3)
            row[f"{kind}_mean"] = round(float(arr_full.mean()), 2)
            row[f"{kind}_nan"] = bool(np.isnan(arr_full).any())
            row[f"{kind}_black"] = bool(arr_full.max() < 3)
        rows.append(row)

    result = {"name": args.name, "out": args.out, "frames": rows}
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
