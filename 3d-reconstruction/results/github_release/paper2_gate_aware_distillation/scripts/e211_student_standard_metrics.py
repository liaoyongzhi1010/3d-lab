"""E-211: Full-image metrics for the Paper-2 fast student.

This is a fixed custom quality-vs-speed diagnostic: 21 disocclusion-heavy scenes,
759 frames, and shared preprocessing for all components. It is not the official
Flash3D protocol and does not support a cross-protocol benchmark ranking. Metrics
are PSNR/SSIM/LPIPS(VGG)/FID, with no region split or invented metric.

Components measured on the same diagnostic:
  - Gen3R baseline (generative, slow)
  - Flash3D evidence (feed-forward geometry component)
  - Teacher = Gen3R + selective injection (slow: 247 s/scene)
  - Student (fast feed-forward CNN distilled from teacher: 0.087 s/scene)

Paper 2 measures the quality-speed tradeoff: the student improves LPIPS/FID over
the evidence component at feed-forward cost but trails the teacher on both.
Loads student.pt (TinyStudent, 8->48->3). Run with system python3
(cuda+lpips+torchmetrics+torch-fidelity+skimage).
"""

import os
import glob
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from skimage.metrics import structural_similarity as ssim_fn
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

SIZE = 256


class TinyStudent(nn.Module):
    def __init__(self, in_ch=8, hidden=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


def L(p):
    a = np.load(p).astype(np.float32)
    return a / 255.0 if a.max() > 1.5 else a


def rs(t, size, nn_=False):
    return F.interpolate(
        t[None],
        size=(size, size),
        mode="nearest" if nn_ else "bilinear",
        align_corners=None if nn_ else False,
    )[0]


def psnr(pred, gt):
    return 10.0 * np.log10(1.0 / max(np.mean((pred - gt) ** 2), 1e-10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/home/data/gen3r_re10k/re10k")
    ap.add_argument("--tdir", default="/home/data/E-161_gen3r_highgain/teacher_npy")
    ap.add_argument("--edir", default="/home/data/E-160_highgain_evidence")
    ap.add_argument("--student", default="/home/data/E-031b_final/student.pt")
    ap.add_argument("--out", default="/home/data/E-211_student_standard_metrics.json")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lp = lpips.LPIPS(net="vgg").to(dev).eval()
    net = TinyStudent().to(dev).eval()
    sd = torch.load(args.student, map_location=dev)
    net.load_state_dict(sd)

    sids = sorted(
        os.path.basename(p)[3:-4]
        for p in glob.glob(os.path.join(args.tdir, "gt_*.npy"))
    )
    arms = ["gen3r", "flash3d", "teacher", "student"]
    acc = {a: {"psnr": [], "ssim": [], "lpips": []} for a in arms}
    fid = {
        a: FrechetInceptionDistance(feature=2048, normalize=True).to(dev) for a in arms
    }
    kid = {
        a: KernelInceptionDistance(subset_size=50, normalize=True).to(dev) for a in arms
    }
    fid_gt = FrechetInceptionDistance(feature=2048, normalize=True).to(dev)
    kid_gt = KernelInceptionDistance(subset_size=50, normalize=True).to(dev)

    def t3(hw3):
        return torch.from_numpy(hw3.transpose(2, 0, 1)[None]).float().to(dev)

    n = 0
    for sid in sids:
        try:
            gt = L(os.path.join(args.tdir, f"gt_{sid}.npy"))
            base = L(os.path.join(args.tdir, f"baseline_{sid}.npy"))
            teach = L(os.path.join(args.tdir, f"adaptive2_{sid}.npy"))
            f3d = L(os.path.join(args.edir, f"f3d_{sid}.npy"))
        except Exception as e:
            print("skip", sid[:14], e)
            continue
        vis = np.load(os.path.join(args.data_root, sid, "visibility.npy")).astype(
            np.float32
        )
        Fn = min(len(gt), len(base), len(teach), len(f3d), len(vis))
        for i in range(1, Fn):
            if (1.0 - vis[i]).mean() < 0.02:
                continue
            # tensors at SIZE
            g = rs(torch.from_numpy(gt[i]).to(dev).clamp(0, 1), SIZE)
            bb = rs(torch.from_numpy(base[i]).to(dev).clamp(0, 1), SIZE)
            ff = rs(torch.from_numpy(f3d[i]).to(dev).clamp(0, 1), SIZE)
            tt = rs(torch.from_numpy(teach[i]).to(dev).clamp(0, 1), SIZE)
            v = rs(torch.from_numpy(vis[i])[None].to(dev), SIZE, nn_=True)
            inv = 1.0 - v
            with torch.no_grad():
                inp = torch.cat([ff, bb, v, inv], dim=0)[None]
                r = net(inp)[0]
                stu = v * bb + inv * (ff + r)
                stu = stu.clamp(0, 1)
            preds = {"gen3r": bb, "flash3d": ff, "teacher": tt, "student": stu}
            gt_t = g[None]
            fid_gt.update(gt_t, real=True)
            kid_gt.update(gt_t, real=True)
            gnp = g.permute(1, 2, 0).cpu().numpy()
            for a in arms:
                pt = preds[a][None]
                pnp = preds[a].permute(1, 2, 0).cpu().numpy()
                acc[a]["psnr"].append(psnr(pnp, gnp))
                acc[a]["ssim"].append(ssim_fn(gnp, pnp, channel_axis=2, data_range=1.0))
                with torch.no_grad():
                    acc[a]["lpips"].append(float(lp(pt * 2 - 1, gt_t * 2 - 1).item()))
                    fid[a].update(pt.clamp(0, 1), real=False)
                    kid[a].update(pt.clamp(0, 1), real=False)
            n += 1
        print(f"{sid[:14]} frames={n}")

    runtime = {"gen3r": 247.0, "flash3d": 0.09, "teacher": 247.0, "student": 0.087}
    out = {"n_frames": n, "scenes": len(sids), "lpips_backbone": "vgg", "size": SIZE}
    print(f"\n=== E-211 student standard metrics (frames={n}) ===")
    print(
        f"{'method':10s} {'PSNR':>7s} {'SSIM':>7s} {'LPIPS':>8s} {'FID':>8s} {'time/scene':>11s}"
    )
    for a in arms:
        fid[a].real_features_sum = fid_gt.real_features_sum.clone()
        fid[a].real_features_cov_sum = fid_gt.real_features_cov_sum.clone()
        fid[a].real_features_num_samples = fid_gt.real_features_num_samples.clone()
        fv = float(fid[a].compute().item())
        P = float(np.mean(acc[a]["psnr"]))
        S = float(np.mean(acc[a]["ssim"]))
        Lp = float(np.mean(acc[a]["lpips"]))
        out[a] = {"psnr": P, "ssim": S, "lpips": Lp, "fid": fv, "time_s": runtime[a]}
        print(f"{a:10s} {P:7.3f} {S:7.4f} {Lp:8.4f} {fv:8.3f} {runtime[a]:10.3f}s")
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
