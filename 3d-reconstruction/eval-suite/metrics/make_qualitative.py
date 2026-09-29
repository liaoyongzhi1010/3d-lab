"""SV3D-Eval-Suite : qualitative gap-degradation montage.

Turns the FID/LPIPS numbers into an eyes-on picture of the core finding:
  small view change (tgt5) -> ok, medium (tgt10) -> worse, large (tgt_rand /
  wide700) -> disocclusion holes are smeared/stretched while GT stays sharp.

Anti-cherry-pick: scenes are chosen at fixed LPIPS QUANTILES of a sort bucket
(default tgt_rand), not hand-picked. The quantile + per-cell LPIPS are printed
on the figure so the selection is auditable.

Layout: one scene = 2 rows (pred / gt) x 4 cols (src, tgt5, tgt10, tgt_rand).
Reading top-to-bottom per column shows where pred starts to deviate from GT.

Input layout (row-level unique dirs, see docs/USAGE.md):
  <root>/<row_dir>/pred/00X.png , <root>/<row_dir>/gt/00X.png
  000=src 001=tgt5 002=tgt10 003=tgt_rand

Usage (run on server, flash3d venv has lpips):
  python metrics/make_qualitative.py \
    --root /home/data/E-052_official/flash3d_present_imgs2 \
    --method Flash3D --title "Flash3D | official MINE present split" \
    --out /home/data/E-052_official/qual_flash3d_present.png
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import lpips as lpips_lib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
FRAMES = [(0, "src"), (1, "tgt5"), (2, "tgt10"), (3, "tgt_rand")]
BUCKET_IDX = {"src": 0, "tgt5": 1, "tgt10": 2, "tgt_rand": 3}


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def to_t(arr):
    return torch.from_numpy(arr).permute(2, 0, 1)[None].to(DEVICE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--sort-bucket", default="tgt_rand", choices=list(BUCKET_IDX))
    ap.add_argument(
        "--quantiles",
        default="0.5,0.75,0.9,0.97",
        help="LPIPS quantiles of sort-bucket to sample scenes at",
    )
    ap.add_argument("--max-scan", type=int, default=0, help="0=all dirs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    quants = [float(x) for x in args.quantiles.split(",")]
    lp_fn = lpips_lib.LPIPS(net="vgg").to(DEVICE).eval()

    root = Path(args.root)
    dirs = [p for p in sorted(root.iterdir()) if (p / "pred").is_dir()]
    if args.max_scan > 0:
        import random

        random.seed(args.seed)
        dirs = sorted(random.sample(dirs, min(args.max_scan, len(dirs))))
    print(
        f"[qual] scanning {len(dirs)} dirs, sort by {args.sort_bucket} LPIPS ...",
        flush=True,
    )

    sidx = BUCKET_IDX[args.sort_bucket]
    scored = []
    with torch.no_grad():
        for i, d in enumerate(dirs):
            pf = d / "pred" / f"{sidx:03d}.png"
            gf = d / "gt" / f"{sidx:03d}.png"
            if not (pf.exists() and gf.exists()):
                continue
            v = lp_fn(to_t(load(pf)) * 2 - 1, to_t(load(gf)) * 2 - 1).item()
            scored.append((v, d))
            if (i + 1) % 500 == 0:
                print(f"  ...{i + 1}/{len(dirs)}", flush=True)
    scored.sort(key=lambda x: x[0])
    n = len(scored)
    print(
        f"[qual] scored {n} scenes. {args.sort_bucket} LPIPS "
        f"min={scored[0][0]:.3f} med={scored[n // 2][0]:.3f} max={scored[-1][0]:.3f}",
        flush=True,
    )

    picks = []
    seen = set()
    for q in quants:
        j = min(n - 1, max(0, int(round(q * (n - 1)))))
        while j in seen and j < n - 1:
            j += 1
        seen.add(j)
        picks.append((q, scored[j][0], scored[j][1]))

    nrows = len(picks) * 2
    ncols = len(FRAMES)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.6, nrows * 1.9))
    if nrows == 1:
        axes = axes[None, :]

    with torch.no_grad():
        for r, (q, qv, d) in enumerate(picks):
            for c, (fidx, fname) in enumerate(FRAMES):
                pf = d / "pred" / f"{fidx:03d}.png"
                gf = d / "gt" / f"{fidx:03d}.png"
                pred = load(pf) if pf.exists() else np.zeros((256, 384, 3), np.float32)
                gt = load(gf) if gf.exists() else np.zeros((256, 384, 3), np.float32)
                lp = (
                    lp_fn(to_t(pred) * 2 - 1, to_t(gt) * 2 - 1).item()
                    if pf.exists() and gf.exists()
                    else float("nan")
                )
                ax_p = axes[r * 2, c]
                ax_g = axes[r * 2 + 1, c]
                ax_p.imshow(np.clip(pred, 0, 1))
                ax_g.imshow(np.clip(gt, 0, 1))
                for a in (ax_p, ax_g):
                    a.set_xticks([])
                    a.set_yticks([])
                # column headers on very top row
                if r == 0:
                    ax_p.set_title(fname, fontsize=11, fontweight="bold")
                # per-cell LPIPS on pred (the one that degrades)
                if fname != "src":
                    ax_p.set_xlabel(f"LPIPS {lp:.3f}", fontsize=8)
                # row labels on left column
                if c == 0:
                    ax_p.set_ylabel(f"pred\nP{int(q * 100)}", fontsize=9)
                    ax_g.set_ylabel("gt", fontsize=9)

    sup = (
        args.title
        or f"{args.method}: gap-degradation (scenes at {args.sort_bucket} LPIPS quantiles)"
    )
    fig.suptitle(sup, fontsize=13, fontweight="bold", y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"[qual] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
