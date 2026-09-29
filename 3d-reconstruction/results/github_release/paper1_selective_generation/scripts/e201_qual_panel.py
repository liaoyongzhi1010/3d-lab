"""E-201: conference-grade qualitative panels for Paper 1.

For a scene with full per-frame teacher renders (from --save_teacher_npy), build a
comparison figure:
    columns = selected frames
    rows    = [GT, Gen3R baseline, Ours (selective), Disocclusion mask,
               |baseline-GT| error, |ours-GT| error]
This mirrors the GT | baseline | ours style of recent world-model papers and adds
error maps + the disocclusion mask so the improvement region is explicit.

Inputs (server):
  teacher_dir/gt_<sid>.npy         [F,3,560,560]
  teacher_dir/baseline_<sid>.npy   [F,3,560,560]
  teacher_dir/adaptive2_<sid>.npy  [F,3,560,560]
  data/<sid>/visibility.npy        [F,70,70]

Usage:
  python e201_qual_panel.py --teacher_dir /home/data/E-147_.../teacher_npy \
    --sid train_xxx --out /home/data/E-201_panels
"""

import os
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw


def load_npy_frame(arr, k, size=256):
    x = arr[k]
    if x.max() > 1.5:
        x = x / 255.0
    x = np.clip(x, 0, 1).transpose(1, 2, 0)
    im = Image.fromarray((x * 255).astype(np.uint8)).resize(
        (size, size), Image.BILINEAR
    )
    return np.asarray(im).astype(np.uint8)


def mask_frame(vis, k, size=256):
    v = vis[k].astype(np.float32)
    inv = 1.0 - v
    im = Image.fromarray((inv * 255).astype(np.uint8)).resize(
        (size, size), Image.NEAREST
    )
    m = np.asarray(im).astype(np.uint8)
    return np.stack(
        [m, np.zeros_like(m), np.zeros_like(m)], axis=-1
    )  # red = disocclusion


def err_map(pred, gt):
    e = np.abs(pred.astype(np.float32) - gt.astype(np.float32)).mean(-1)
    e = np.clip(e / 128.0, 0, 1)
    # viridis-like: map to heat
    r = np.clip(1.5 - np.abs(4 * e - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * e - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * e - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def label(img, text):
    im = Image.fromarray(img.copy())
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, max(70, 8 * len(text)), 16], fill=(0, 0, 0))
    d.text((2, 2), text, fill=(0, 255, 0))
    return np.asarray(im)


def hcat(imgs, gap=4):
    g = np.ones((imgs[0].shape[0], gap, 3), dtype=np.uint8) * 255
    parts = []
    for i, im in enumerate(imgs):
        parts.append(im)
        if i < len(imgs) - 1:
            parts.append(g)
    return np.concatenate(parts, axis=1)


def vcat(rows, gap=4):
    g = np.ones((gap, rows[0].shape[1], 3), dtype=np.uint8) * 255
    parts = []
    for i, r in enumerate(rows):
        parts.append(r)
        if i < len(rows) - 1:
            parts.append(g)
    return np.concatenate(parts, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_dir", required=True)
    ap.add_argument("--sid", required=True)
    ap.add_argument("--data", default="/home/data/gen3r_re10k/re10k")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", default="6,18,30,42,48")
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sid = args.sid
    td = args.teacher_dir
    gt = np.load(os.path.join(td, f"gt_{sid}.npy"))
    base = np.load(os.path.join(td, f"baseline_{sid}.npy"))
    ours = np.load(os.path.join(td, f"adaptive2_{sid}.npy"))
    vis = np.load(os.path.join(args.data, sid, "visibility.npy")).astype(np.float32)
    frames = [int(x) for x in args.frames.split(",")]
    frames = [k for k in frames if k < len(gt)]
    sz = args.size

    def row(getter, name):
        cells = [getter(k) for k in frames]
        cells[0] = label(cells[0], name)
        return hcat(cells)

    r_gt = row(lambda k: load_npy_frame(gt, k, sz), "GT")
    r_base = row(lambda k: load_npy_frame(base, k, sz), "Gen3R")
    r_ours = row(lambda k: load_npy_frame(ours, k, sz), "Ours")
    r_mask = row(lambda k: mask_frame(vis, k, sz), "Disocc")
    r_ebase = row(
        lambda k: err_map(load_npy_frame(base, k, sz), load_npy_frame(gt, k, sz)),
        "Err base",
    )
    r_eours = row(
        lambda k: err_map(load_npy_frame(ours, k, sz), load_npy_frame(gt, k, sz)),
        "Err ours",
    )

    full = vcat([r_gt, r_base, r_ours, r_mask, r_ebase, r_eours])
    outp = os.path.join(args.out, f"panel_{sid}.png")
    Image.fromarray(full).save(outp)
    print(f"saved {outp} shape={full.shape}")


if __name__ == "__main__":
    main()
