"""E-210: STANDARD full-image perceptual/distribution metrics for comparison
with published generative-NVS methods. NO region split, NO invented metric ---
only PSNR / SSIM / LPIPS(VGG) / FID / KID, exactly the metrics ViewCrafter,
GenWarp, pixelSplat, MVSplat, Flash3D report.

Compares, on the same rendered frames (21 high-gain scenes, per-frame npy):
  - Gen3R-alone   (generative single-view baseline)
  - Flash3D       (feed-forward geometry expert; evidence render)
  - Injected output (adaptive2, always evaluated directly; no oracle fallback)

Thesis: a generative refinement improves PERCEPTUAL/DISTRIBUTION realism
(LPIPS/FID/KID), the axis on which generative NVS methods are compared, even
when pixel PSNR (an MSE metric that rewards blur) does not move much.

LPIPS uses VGG backbone (net='vgg'), inputs in [-1,1] --- matches Flash3D /
pixelSplat / MVSplat / DepthSplat convention. FID/KID use torchmetrics on
full RGB frames resized to 299. Run on server with system python3 (has cuda,
lpips, torchmetrics, skimage).
"""

import os
import glob
import json
import argparse
import numpy as np
import torch
import lpips
from skimage.metrics import structural_similarity as ssim_fn
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance


def load(sid, tdir, edir):
    def L(p):
        a = np.load(p).astype(np.float32)
        return a / 255.0 if a.max() > 1.5 else a

    return (
        L(os.path.join(tdir, f"gt_{sid}.npy")),
        L(os.path.join(tdir, f"baseline_{sid}.npy")),
        L(os.path.join(tdir, f"adaptive2_{sid}.npy")),
        L(os.path.join(edir, f"f3d_{sid}.npy")),
    )


def psnr(pred, gt):
    mse = np.mean((pred - gt) ** 2)
    return 10.0 * np.log10(1.0 / max(mse, 1e-10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/home/data/gen3r_re10k/re10k")
    ap.add_argument("--tdir", default="/home/data/E-161_gen3r_highgain/teacher_npy")
    ap.add_argument("--edir", default="/home/data/E-160_highgain_evidence")
    ap.add_argument("--out", default="/home/data/E-210_standard_metrics.json")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lp = lpips.LPIPS(net="vgg").to(dev).eval()

    sids = sorted(
        os.path.basename(p)[3:-4]
        for p in glob.glob(os.path.join(args.tdir, "gt_*.npy"))
    )
    arms = ["gen3r", "flash3d", "ours"]
    acc = {a: {"psnr": [], "ssim": [], "lpips": []} for a in arms}
    # FID/KID accumulators (uint8 NCHW)
    fid = {
        a: FrechetInceptionDistance(feature=2048, normalize=True).to(dev) for a in arms
    }
    kid = {
        a: KernelInceptionDistance(subset_size=50, normalize=True).to(dev) for a in arms
    }
    fid_gt = FrechetInceptionDistance(feature=2048, normalize=True).to(dev)
    kid_gt = KernelInceptionDistance(subset_size=50, normalize=True).to(dev)

    def to_t(img_hw3):
        return torch.from_numpy(img_hw3.transpose(2, 0, 1)[None]).float().to(dev)

    n_frames = 0
    for sid in sids:
        try:
            gt, base, adap, f3d = load(sid, args.tdir, args.edir)
        except Exception as e:
            print("skip", sid[:16], e)
            continue
        vis = np.load(os.path.join(args.data_root, sid, "visibility.npy")).astype(
            np.float32
        )
        F = min(len(gt), len(base), len(adap), len(f3d), len(vis))
        for i in range(1, F):
            disocc = 1.0 - vis[i]
            # only frames that actually contain disocclusion (the regime the
            # method targets); keeps comparison honest & on-topic
            if disocc.mean() < 0.02:
                continue
            preds = {"gen3r": base[i], "flash3d": f3d[i], "ours": adap[i]}
            g = np.clip(gt[i].transpose(1, 2, 0), 0, 1)
            gt_t = to_t(g).clamp(0, 1)
            fid_gt.update(gt_t, real=True)
            kid_gt.update(gt_t, real=True)
            for a in arms:
                p = np.clip(preds[a].transpose(1, 2, 0), 0, 1)
                acc[a]["psnr"].append(psnr(p, g))
                acc[a]["ssim"].append(ssim_fn(g, p, channel_axis=2, data_range=1.0))
                with torch.no_grad():
                    pt = to_t(p)
                    acc[a]["lpips"].append(float(lp(pt * 2 - 1, gt_t * 2 - 1).item()))
                    fid[a].update(pt.clamp(0, 1), real=False)
                    kid[a].update(pt.clamp(0, 1), real=False)
            n_frames += 1
        print(f"{sid[:16]} cumulative frames={n_frames}")

    # FID/KID need real set in each metric object; copy GT features by re-updating
    out = {
        "n_frames": n_frames,
        "scenes": len(sids),
        "lpips_backbone": "vgg",
        "metric_note": "standard full-image metrics; no region split",
        "ours_semantics": (
            "adaptive2 injected output evaluated directly on every included frame; "
            "no oracle fallback"
        ),
    }
    print(f"\n=== E-210 standard full-image metrics (frames={n_frames}) ===")
    print(
        f"{'method':10s} {'PSNR':>7s} {'SSIM':>7s} {'LPIPS':>8s} {'FID':>8s} {'KID':>10s}"
    )
    for a in arms:
        # attach GT reals into each arm's FID/KID
        fid[a].real_features_sum = fid_gt.real_features_sum.clone()
        fid[a].real_features_cov_sum = fid_gt.real_features_cov_sum.clone()
        fid[a].real_features_num_samples = fid_gt.real_features_num_samples.clone()
        fv = float(fid[a].compute().item())
        for feat in kid_gt.real_features:
            kid[a].real_features.append(feat)
        km, ks = kid[a].compute()
        km = float(km.item())
        P = float(np.mean(acc[a]["psnr"]))
        S = float(np.mean(acc[a]["ssim"]))
        Lp = float(np.mean(acc[a]["lpips"]))
        out[a] = {"psnr": P, "ssim": S, "lpips": Lp, "fid": fv, "kid": km}
        print(f"{a:10s} {P:7.3f} {S:7.4f} {Lp:8.4f} {fv:8.3f} {km:10.5f}")

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nsaved {args.out}")
    print("\nNote: Flash3D evidence is scene-folder constructed (optimistic ref).")


if __name__ == "__main__":
    main()
