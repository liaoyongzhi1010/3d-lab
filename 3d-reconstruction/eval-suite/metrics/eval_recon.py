"""E-052 external-images metric driver.

Feeds pre-exported (pred, gt) image pairs from ANY method into the OFFICIAL
Flash3D/CATSplat evaluator (evaluation/evaluator.py). We DO NOT reimplement any
metric here: we import the official `Evaluator` (PSNR/SSIM/LPIPS-VGG, 5% border
crop, margin=0.05) so numbers are directly comparable to the papers.

Expected export layout (Flash3D/CATSplat convention, verified 2026-07-21):
  <root>/<scene>/pred/000.png 001.png 002.png 003.png
  <root>/<scene>/gt/  000.png 001.png 002.png 003.png
where 000=src, 001=tgt5, 002=tgt10, 003=tgt_rand
(target_frame_ids=[1,2,3], add_source_frame_id -> [0,1,2,3]).

Images are full-res (e.g. 384x256), NOT pre-cropped; the official Evaluator
applies the 5% crop itself. This matches how CATSplat/Flash3D score their own
renders, so cross-method numbers align to each paper's Table.

Usage:
  python _e052_eval_external.py \
    --root "/root/projects/CATSplat/exp/evaluate/wide700/Put your ply file path" \
    --method catsplat \
    --out /home/data/E-052_external/catsplat_wide700.json
  # optional: --glob-depth to auto-find the scene dir, --n-scenes to cap
"""

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/projects/flash3d")

import torch
import numpy as np
from PIL import Image

from evaluation.evaluator import Evaluator  # OFFICIAL metric, do not reimplement

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# frame-index -> eval-bucket name (Flash3D/CATSplat convention)
FRAME_NAMES = {0: "src", 1: "tgt5", 2: "tgt10", 3: "tgt_rand"}


def load_img(path):
    """PNG -> float tensor (1,3,H,W) in [0,1]."""
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)[None]
    return t.to(DEVICE)


def find_scene_dirs(root):
    """A scene dir is any dir that contains both pred/ and gt/ subdirs."""
    root = Path(root)
    scenes = []
    for p in sorted(root.rglob("pred")):
        if p.is_dir() and (p.parent / "gt").is_dir():
            scenes.append(p.parent)
    return scenes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root", required=True, help="dir containing <scene>/{pred,gt}/00X.png"
    )
    ap.add_argument("--method", required=True, help="method tag for the report")
    ap.add_argument("--out", required=True, help="output json path")
    ap.add_argument("--n-scenes", type=int, default=0, help="cap #scenes (0=all)")
    ap.add_argument(
        "--no-crop",
        action="store_true",
        help="disable 5% border crop (NOT recommended)",
    )
    args = ap.parse_args()

    evaluator = Evaluator(crop_border=not args.no_crop).to(DEVICE).eval()

    scene_dirs = find_scene_dirs(args.root)
    if args.n_scenes > 0:
        scene_dirs = scene_dirs[: args.n_scenes]
    print(
        f"[external] method={args.method} scenes={len(scene_dirs)} crop={not args.no_crop}",
        flush=True,
    )
    if not scene_dirs:
        raise SystemExit(f"no <scene>/{{pred,gt}} found under {args.root}")

    # per-bucket accumulators
    acc = {name: {"psnr": [], "ssim": [], "lpips": []} for name in FRAME_NAMES.values()}
    n_scene_ok = 0
    n_pair = 0
    manifest = []

    for sd in scene_dirs:
        pred_dir, gt_dir = sd / "pred", sd / "gt"
        pred_imgs = sorted(
            glob.glob(str(pred_dir / "*.png")) + glob.glob(str(pred_dir / "*.jpg"))
        )
        used = 0
        for pf in pred_imgs:
            idx = int(Path(pf).stem)
            name = FRAME_NAMES.get(idx)
            if name is None:
                continue
            gf = gt_dir / Path(pf).name
            if not gf.exists():
                continue
            with torch.no_grad():
                pred = load_img(pf)
                gt = load_img(gf)
                if pred.shape != gt.shape:
                    pred = torch.nn.functional.interpolate(
                        pred, gt.shape[-2:], mode="bilinear", align_corners=False
                    )
                m = evaluator(pred, gt)
            for k in ("psnr", "ssim", "lpips"):
                acc[name][k].append(m[k])
            used += 1
            n_pair += 1
        if used > 0:
            n_scene_ok += 1
            manifest.append({"scene": sd.name, "frames": used})
        if n_scene_ok % 50 == 0 and used > 0:
            print(f"  [{n_scene_ok}] {sd.name} frames={used}", flush=True)

    def mean(x):
        return float(np.mean(x)) if x else float("nan")

    # per-bucket means
    buckets = {}
    for name, d in acc.items():
        buckets[name] = {k: mean(v) for k, v in d.items()}
    # novel-mean = mean over tgt5/tgt10/tgt_rand (source excluded, per protocol)
    novel = {}
    for k in ("psnr", "ssim", "lpips"):
        vals = []
        for name in ("tgt5", "tgt10", "tgt_rand"):
            vals += acc[name][k]
        novel[k] = mean(vals)

    summary = {
        "method": args.method,
        "root": args.root,
        "n_scenes": n_scene_ok,
        "n_pairs": n_pair,
        "crop_border": not args.no_crop,
        "metric_source": "official evaluation/evaluator.py (PSNR/SSIM/LPIPS-VGG, 5% crop)",
        "buckets": buckets,
        "novel_mean": novel,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "manifest": manifest}, f, indent=2)

    print(
        f"\n=== EXTERNAL METRICS [{args.method}] {n_scene_ok} scenes / {n_pair} pairs ===",
        flush=True,
    )
    for name in ("src", "tgt5", "tgt10", "tgt_rand"):
        b = buckets[name]
        print(
            f"  {name:9s}  PSNR {b['psnr']:.2f}  SSIM {b['ssim']:.3f}  LPIPS {b['lpips']:.3f}",
            flush=True,
        )
    print(
        f"  {'novel':9s}  PSNR {novel['psnr']:.2f}  SSIM {novel['ssim']:.3f}  LPIPS {novel['lpips']:.3f}",
        flush=True,
    )
    print(f"  -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
