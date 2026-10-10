"""E-206: publication teaser figure (fig0) for Paper 1.

Layout (single row story):
  [ source input ]  ||  novel views: Gen3R (top) vs Ours (bottom), 3 target frames
with the invisible-region gain annotated. Uses teacher_npy renders (gt/baseline/
adaptive2) so it needs no GPU. Produces PDF+PNG.

Usage:
  python e206_teaser.py --teacher_dir <dir> --sid <scene> --data <re10k> \
    --frames 12,30,48 --out results/paper_figs
"""

import os, json, argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_stack(path):
    a = np.load(path).astype(np.float32)
    if a.max() > 1.5:
        a = a / 255.0
    return np.clip(a, 0, 1)


def chw2hwc(x):
    return np.transpose(x, (1, 2, 0))


def psnr_inv(pred, gt, inv):
    m = inv > 0.5
    if m.sum() < 10:
        return None
    mse = (((pred - gt) ** 2).mean(0) * m).sum() / m.sum()
    return float(10 * np.log10(1.0 / max(mse, 1e-10)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_dir", required=True)
    ap.add_argument("--sid", required=True)
    ap.add_argument("--data", default="/home/data/gen3r_re10k/re10k")
    ap.add_argument("--frames", default="12,30,48")
    ap.add_argument("--out", default="results/paper_figs")
    ap.add_argument("--src_png", default=None, help="optional source input image path")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sid = args.sid
    gt = load_stack(os.path.join(args.teacher_dir, f"gt_{sid}.npy"))
    base = load_stack(os.path.join(args.teacher_dir, f"baseline_{sid}.npy"))
    ours = load_stack(os.path.join(args.teacher_dir, f"adaptive2_{sid}.npy"))
    vis = np.load(os.path.join(args.data, sid, "visibility.npy")).astype(np.float32)
    frames = [int(x) for x in args.frames.split(",") if int(x) < len(gt)]

    ncol = 1 + len(frames)
    fig, axes = plt.subplots(3, ncol, figsize=(3.0 * ncol, 8.4))
    # column 0: source (frame 0 GT) spanning as "input"
    for r in range(3):
        axes[r, 0].axis("off")
    axes[0, 0].imshow(chw2hwc(gt[0]))
    axes[0, 0].set_title("Input view", fontsize=13, fontweight="bold")
    axes[0, 0].axis("off")
    axes[1, 0].text(
        0.5,
        0.5,
        "Novel views →",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )
    axes[2, 0].axis("off")

    row_titles = ["GT", "Gen3R (baseline)", "Ours (selective)"]
    for ci, k in enumerate(frames):
        col = ci + 1
        from PIL import Image

        H, W = gt.shape[2], gt.shape[3]
        inv = (
            1.0
            - np.asarray(
                Image.fromarray((vis[k] * 255).astype(np.uint8)).resize(
                    (W, H), Image.NEAREST
                )
            ).astype(np.float32)
            / 255.0
        )
        pb = psnr_inv(base[k], gt[k], inv)
        po = psnr_inv(ours[k], gt[k], inv)
        for r, img in enumerate([gt[k], base[k], ours[k]]):
            ax = axes[r, col]
            ax.imshow(chw2hwc(img))
            ax.set_xticks([])
            ax.set_yticks([])
            if ci == 0:
                ax.set_ylabel(row_titles[r], fontsize=12, fontweight="bold")
            if r == 0:
                ax.set_title(f"target t={k}", fontsize=11)
        # annotate invisible PSNR gain on ours row
        if pb is not None and po is not None:
            axes[2, col].set_xlabel(
                f"disocc PSNR {pb:.1f}→{po:.1f} dB (+{po - pb:.1f})",
                fontsize=10,
                color="tab:green",
                fontweight="bold",
            )

    fig.suptitle(
        "Selective Geometry-Guided Generation: sharper disoccluded content, visible regions untouched",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out, f"fig0_teaser.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.out}/fig0_teaser.[pdf,png] for {sid}")


if __name__ == "__main__":
    main()
