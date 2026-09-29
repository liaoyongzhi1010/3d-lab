"""E-052 generation-quality metrics for external-exported image pairs.

Conference-standard generative metrics on top of the per-image metrics:
  - FID / KID : cleanfid (same package/weights as our prior E-051 FID runs,
                weights at /tmp/inception-2015-12-05.pt) -> historically consistent
  - DISTS     : DISTS_pytorch (the exact package latentSplat uses)
  - LPIPS     : net='vgg' (all papers agree)
Reported PER GAP BUCKET (tgt5/tgt10/tgt_rand) + pooled novel. Full-image (no crop).

We DO NOT reimplement any metric. FID via cleanfid (torchmetrics FID needs a
weight download that hangs in this env; cleanfid uses the local /tmp weight and
matches our earlier FID numbers).

Input layout (Flash3D/CATSplat export convention):
  <root>/<scene>/pred/00X.png , <root>/<scene>/gt/00X.png
  000=src 001=tgt5 002=tgt10 003=tgt_rand

Usage:
  python _e052_eval_genquality.py \
    --root "<export dir>" --method catsplat \
    --out /home/data/E-052_external/catsplat_genq.json \
    --workdir /home/data/E-052_external/_genq_tmp/catsplat
"""

import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/root/projects/flash3d")

import torch
import numpy as np
from PIL import Image

import lpips as lpips_lib
from DISTS_pytorch import DISTS

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
FRAME_NAMES = {0: "src", 1: "tgt5", 2: "tgt10", 3: "tgt_rand"}
NOVEL = ("tgt5", "tgt10", "tgt_rand")


def load_float(path):
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)[None].to(DEVICE)


def find_scene_dirs(root):
    root = Path(root)
    out = []
    for p in sorted(root.rglob("pred")):
        if p.is_dir() and (p.parent / "gt").is_dir():
            out.append(p.parent)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--workdir",
        required=True,
        help="temp dir to gather per-bucket images for cleanfid",
    )
    ap.add_argument("--n-scenes", type=int, default=0)
    args = ap.parse_args()

    lpips_fn = lpips_lib.LPIPS(net="vgg").to(DEVICE).eval()
    dists_fn = DISTS().to(DEVICE).eval()

    scenes = find_scene_dirs(args.root)
    if args.n_scenes > 0:
        scenes = scenes[: args.n_scenes]
    print(f"[genq] method={args.method} scenes={len(scenes)}", flush=True)
    if not scenes:
        raise SystemExit(f"no <scene>/{{pred,gt}} under {args.root}")

    # prepare per-bucket folders for cleanfid (pred/ and gt/ per bucket)
    workdir = Path(args.workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    for name in FRAME_NAMES.values():
        (workdir / name / "pred").mkdir(parents=True, exist_ok=True)
        (workdir / name / "gt").mkdir(parents=True, exist_ok=True)

    lp = {n: [] for n in FRAME_NAMES.values()}
    ds = {n: [] for n in FRAME_NAMES.values()}

    n_pair = 0
    n_sc = 0
    for sd in scenes:
        for pf in sorted(glob.glob(str(sd / "pred" / "*.png"))):
            idx = int(Path(pf).stem)
            name = FRAME_NAMES.get(idx)
            if name is None:
                continue
            gf = sd / "gt" / Path(pf).name
            if not gf.exists():
                continue
            with torch.no_grad():
                pred = load_float(pf)
                gt = load_float(gf)
                if pred.shape != gt.shape:
                    pred = torch.nn.functional.interpolate(
                        pred, gt.shape[-2:], mode="bilinear", align_corners=False
                    )
                lp[name].append(lpips_fn(pred * 2 - 1, gt * 2 - 1).item())
                ds[name].append(dists_fn(gt, pred, require_grad=False).item())
            tag = f"{sd.name}_{idx}.png"
            shutil.copy(pf, workdir / name / "pred" / tag)
            shutil.copy(gf, workdir / name / "gt" / tag)
            n_pair += 1
        n_sc += 1
        if n_sc % 50 == 0:
            print(
                f"  ...{n_sc}/{len(scenes)} scenes, {n_pair} pairs (lpips/dists done)",
                flush=True,
            )

    print("  [FID/KID via cleanfid] ...", flush=True)
    from cleanfid import fid as cfid

    def mean(x):
        return float(np.mean(x)) if x else float("nan")

    def fid_kid_dirs(gt_dir, pred_dir, n):
        if n < 10:
            return float("nan"), float("nan")
        f = cfid.compute_fid(str(gt_dir), str(pred_dir), verbose=False)
        k = cfid.compute_kid(str(gt_dir), str(pred_dir), verbose=False)
        return float(f), float(k)

    buckets = {}
    for name in FRAME_NAMES.values():
        n = len(lp[name])
        f, k = fid_kid_dirs(workdir / name / "gt", workdir / name / "pred", n)
        buckets[name] = {
            "lpips": mean(lp[name]),
            "dists": mean(ds[name]),
            "fid": f,
            "kid": k,
            "n": n,
        }
        print(f"    {name}: FID {f:.2f} KID {k:.4f}", flush=True)

    # pooled novel: gather all novel buckets into one folder pair
    nov_gt = workdir / "_novel" / "gt"
    nov_pred = workdir / "_novel" / "pred"
    nov_gt.mkdir(parents=True, exist_ok=True)
    nov_pred.mkdir(parents=True, exist_ok=True)
    for name in NOVEL:
        for f in glob.glob(str(workdir / name / "pred" / "*.png")):
            shutil.copy(f, nov_pred / f"{name}_{Path(f).name}")
        for f in glob.glob(str(workdir / name / "gt" / "*.png")):
            shutil.copy(f, nov_gt / f"{name}_{Path(f).name}")
    novel_lp = sum([lp[n] for n in NOVEL], [])
    novel_ds = sum([ds[n] for n in NOVEL], [])
    nf, nk = fid_kid_dirs(nov_gt, nov_pred, len(novel_lp))
    novel = {
        "lpips": mean(novel_lp),
        "dists": mean(novel_ds),
        "fid": nf,
        "kid": nk,
        "n": len(novel_lp),
    }

    summary = {
        "method": args.method,
        "root": args.root,
        "n_pairs": n_pair,
        "metric_source": "FID/KID=cleanfid(local /tmp weight), DISTS=DISTS_pytorch(latentSplat), LPIPS=vgg; full-image, no crop",
        "buckets": buckets,
        "novel_overall": novel,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"\n=== GEN-QUALITY [{args.method}] {n_pair} pairs (full-image) ===", flush=True
    )
    print(
        f"  {'bucket':9s} {'FID':>8s} {'KID':>9s} {'LPIPS':>7s} {'DISTS':>7s}  n",
        flush=True,
    )
    for name in ("tgt5", "tgt10", "tgt_rand"):
        b = buckets[name]
        print(
            f"  {name:9s} {b['fid']:8.2f} {b['kid']:9.4f} {b['lpips']:7.3f} {b['dists']:7.3f}  {b['n']}",
            flush=True,
        )
    print(
        f"  {'novel':9s} {novel['fid']:8.2f} {novel['kid']:9.4f} {novel['lpips']:7.3f} {novel['dists']:7.3f}  {novel['n']}",
        flush=True,
    )
    print(f"  -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
