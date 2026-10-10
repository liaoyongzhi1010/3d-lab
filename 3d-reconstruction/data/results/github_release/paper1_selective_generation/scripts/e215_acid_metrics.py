"""E-215: ACID generalization — mechanism check + standard full-image metrics.

Second dataset (ACID, outdoor aerial) to test whether the RE10K findings
generalize. Uses the same pipeline outputs (Gen3R baseline, adaptive2 injected
output, Flash3D evidence); the released aggregate contains 8 valid ACID scenes.

Part A (oracle diagnostic): scene-level correlation between the target-GT
quality gap (f3d_inv - base_inv) and actual teacher gain
(teacher_inv - base_inv). This is not a deployable selection signal.

Part B (standard metrics): full-image PSNR/SSIM/LPIPS(VGG)/FID for
Gen3R / Flash3D / injected output, same as E-210 (direct adaptive2 evaluation,
no oracle fallback, no region split, no invented metric).

npy naming: teacher_npy/{gt,baseline,adaptive2}_<sid>.npy (no test_ prefix);
evidence E-213_acid_f3d/f3d_test_<sid>.npy (with test_ prefix).
Run on server system python3.
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


def L(p):
    a = np.load(p).astype(np.float32)
    return a / 255.0 if a.max() > 1.5 else a


def psnr(pred, gt):
    return 10.0 * np.log10(1.0 / max(np.mean((pred - gt) ** 2), 1e-10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results", default="/home/data/E-214_acid_gen3r_results/e012_vesg.json"
    )
    ap.add_argument("--tdir", default="/home/data/E-214_acid_gen3r_results/teacher_npy")
    ap.add_argument("--edir", default="/home/data/E-213_acid_f3d")
    ap.add_argument("--scene_root", default="/home/data/E-212_acid_gen3r")
    ap.add_argument("--out", default="/home/data/E-215_acid_metrics.json")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Part A: mechanism (scene-level) ----
    ps = json.load(open(args.results))["per_scene"]
    bl, ad = ps["baseline"], ps["adaptive2"]
    base_inv, teach_inv = [], []
    for sc in bl:
        if (
            sc in ad
            and bl[sc].get("invis_psnr") is not None
            and ad[sc].get("invis_psnr") is not None
        ):
            base_inv.append(bl[sc]["invis_psnr"])
            teach_inv.append(ad[sc]["invis_psnr"])
    base_inv = np.array(base_inv)
    teach_inv = np.array(teach_inv)
    gain = teach_inv - base_inv
    print(f"=== ACID mechanism (N={len(gain)} scenes) ===")
    print(
        f"  mean base_inv={base_inv.mean():.2f}  teacher_inv={teach_inv.mean():.2f}  gain={gain.mean():+.2f} dB"
    )
    print(
        f"  scenes with teacher>baseline (invisible): {int((gain > 0).sum())}/{len(gain)}"
    )

    # ---- Part B: standard full-image metrics ----
    lp = lpips.LPIPS(net="vgg").to(dev).eval()
    sids = sorted(
        os.path.basename(p)[3:-4]
        for p in glob.glob(os.path.join(args.tdir, "gt_*.npy"))
    )
    arms = ["gen3r", "flash3d", "ours"]
    acc = {a: {"psnr": [], "ssim": [], "lpips": []} for a in arms}
    fid = {
        a: FrechetInceptionDistance(feature=2048, normalize=True).to(dev) for a in arms
    }
    fid_gt = FrechetInceptionDistance(feature=2048, normalize=True).to(dev)
    f3d_inv_all, base_inv_frame, teach_inv_frame = [], [], []

    def t3(hw3):
        return torch.from_numpy(hw3.transpose(2, 0, 1)[None]).float().to(dev)

    n = 0
    for sid in sids:
        try:
            gt = L(os.path.join(args.tdir, f"gt_{sid}.npy"))
            base = L(os.path.join(args.tdir, f"baseline_{sid}.npy"))
            adap = L(os.path.join(args.tdir, f"adaptive2_{sid}.npy"))
            f3d = L(os.path.join(args.edir, f"f3d_test_{sid}.npy"))
        except Exception as e:
            print("skip", sid[:12], e)
            continue
        vis = np.load(
            os.path.join(args.scene_root, f"test_{sid}", "visibility.npy")
        ).astype(np.float32)
        Fn = min(len(gt), len(base), len(adap), len(f3d), len(vis))
        for i in range(1, Fn):
            if (1.0 - vis[i]).mean() < 0.02:
                continue
            g = np.clip(gt[i].transpose(1, 2, 0), 0, 1)
            gt_t = t3(g)
            fid_gt.update(gt_t.clamp(0, 1), real=True)
            preds = {"gen3r": base[i], "flash3d": f3d[i], "ours": adap[i]}
            for a in arms:
                p = np.clip(preds[a].transpose(1, 2, 0), 0, 1)
                acc[a]["psnr"].append(psnr(p, g))
                acc[a]["ssim"].append(ssim_fn(g, p, channel_axis=2, data_range=1.0))
                with torch.no_grad():
                    pt = t3(p)
                    acc[a]["lpips"].append(float(lp(pt * 2 - 1, gt_t * 2 - 1).item()))
                    fid[a].update(pt.clamp(0, 1), real=False)
            n += 1
        print(f"{sid[:12]} frames={n}")

    out = {
        "mechanism": {
            "n_scenes": len(gain),
            "mean_gain": float(gain.mean()),
            "win": int((gain > 0).sum()),
            "base_inv": float(base_inv.mean()),
            "teacher_inv": float(teach_inv.mean()),
        },
        "n_frames": n,
        "lpips_backbone": "vgg",
    }
    print(f"\n=== ACID standard full-image metrics (frames={n}) ===")
    print(f"{'method':10s} {'PSNR':>7s} {'SSIM':>7s} {'LPIPS':>8s} {'FID':>8s}")
    for a in arms:
        fid[a].real_features_sum = fid_gt.real_features_sum.clone()
        fid[a].real_features_cov_sum = fid_gt.real_features_cov_sum.clone()
        fid[a].real_features_num_samples = fid_gt.real_features_num_samples.clone()
        fv = float(fid[a].compute().item())
        P = float(np.mean(acc[a]["psnr"]))
        S = float(np.mean(acc[a]["ssim"]))
        Lp = float(np.mean(acc[a]["lpips"]))
        out[a] = {"psnr": P, "ssim": S, "lpips": Lp, "fid": fv}
        print(f"{a:10s} {P:7.3f} {S:7.4f} {Lp:8.4f} {fv:8.3f}")
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
