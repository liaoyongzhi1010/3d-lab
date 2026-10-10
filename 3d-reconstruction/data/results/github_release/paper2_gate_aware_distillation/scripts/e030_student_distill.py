"""E-030: Image-level disocclusion-prior distillation student.

Second-paper MVP: distill the slow teacher (Paper-1 adaptive2/learned output) into a
fast feed-forward image-level student.

This is intentionally NOT the final 3DGS adapter. It is a proof-of-concept that a
small feed-forward network can learn the teacher's disocclusion corrections while
leaving visible pixels untouched.

Data layout expected from E-021 panels:
  frames_dir/test_<sid>/
    baseline_f000.png ... baseline_f048.png
    adaptive2_f000.png ... adaptive2_f048.png   (teacher)
    gt_f000.png ... gt_f048.png
Flash3D evidence:
  f3d_dir/f3d_<sid>.npy  [49,3,560,560]
Visibility:
  data/test_<sid>/visibility.npy [49,70,70]

Student input per key frame:
  concat[f3d_rgb(3), baseline_rgb(3), vis(1), inv(1)] = 8 channels
Output:
  residual(3), out = baseline in visible + (f3d + residual) in invisible
Loss:
  L = L1(out, teacher) on invisible + 0.2*L1(out, GT) on invisible + 2.0*L1(out, baseline) on visible

Run on server or local with torch/PIL:
  python e030_student_distill.py --frames_dir /home/data/E-021_panels/frames \
    --f3d_dir /home/data/E-013_f3d_evidence --data /home/data/gen3r_re10k/re10k \
    --out /home/data/E-030_student
"""

import os
import glob
import json
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def load_png(path, size=256):
    im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0).permute(2, 0, 1)


def load_vis(scene_dir, frame_id, size=256):
    v = np.load(os.path.join(scene_dir, "visibility.npy"))[frame_id].astype(np.float32)
    im = Image.fromarray((v * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)
    return torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0)[None]


def load_f3d(f3d_path, frame_id, size=256):
    arr = np.load(f3d_path)
    if arr.ndim != 4:
        raise ValueError(f"bad f3d shape {arr.shape}: {f3d_path}")
    x = torch.from_numpy(arr[frame_id].astype(np.float32))
    if x.max() > 1.5:
        x = x / 255.0
    x = F.interpolate(x[None], size=(size, size), mode="bilinear", align_corners=False)[
        0
    ]
    return x.clamp(0, 1)


def psnr_mask(pred, gt, mask):
    m = mask.expand_as(pred)
    denom = m.sum().clamp(min=1.0)
    mse = (((pred - gt) ** 2) * m).sum() / denom
    return float(10 * torch.log10(1.0 / mse.clamp(min=1e-10)))


def build_samples(args):
    samples = []
    for scene_name in sorted(os.listdir(args.frames_dir)):
        fdir = os.path.join(args.frames_dir, scene_name)
        if not os.path.isdir(fdir):
            continue
        sid = scene_name.replace("test_", "")
        f3dp = os.path.join(args.f3d_dir, f"f3d_{sid}.npy")
        scene_dir = os.path.join(args.data, f"test_{sid}")
        if not os.path.exists(f3dp) or not os.path.exists(scene_dir):
            continue
        for gt_path in sorted(glob.glob(os.path.join(fdir, "gt_f*.png"))):
            k = int(os.path.basename(gt_path)[4:7])
            b_path = os.path.join(fdir, f"baseline_f{k:03d}.png")
            t_path = os.path.join(fdir, f"adaptive2_f{k:03d}.png")
            if not (os.path.exists(b_path) and os.path.exists(t_path)):
                continue
            samples.append((sid, k, f3dp, scene_dir, b_path, t_path, gt_path))
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", default="/home/data/E-021_panels/frames")
    ap.add_argument("--f3d_dir", default="/home/data/E-013_f3d_evidence")
    ap.add_argument("--data", default="/home/data/gen3r_re10k/re10k")
    ap.add_argument("--out", default="/home/data/E-030_student")
    ap.add_argument("--holdout", default="249fd0890d439aa9,edaf13c2d419ff89")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    hold = set([x for x in args.holdout.split(",") if x])
    samples = build_samples(args)
    train_s = [s for s in samples if s[0] not in hold]
    hold_s = [s for s in samples if s[0] in hold]
    print(f"samples train={len(train_s)} holdout={len(hold_s)} hold={sorted(hold)}")

    def load_sample(s):
        sid, k, f3dp, scene_dir, bp, tp, gp = s
        f3d = load_f3d(f3dp, k, args.size)
        base = load_png(bp, args.size)
        teacher = load_png(tp, args.size)
        gt = load_png(gp, args.size)
        vis = load_vis(scene_dir, k, args.size)
        inv = 1.0 - vis
        inp = torch.cat([f3d, base, vis, inv], dim=0)
        return inp, f3d, base, teacher, gt, vis, inv

    net = TinyStudent().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    # preload small dataset
    train_data = [
        tuple(x.to(dev) for x in load_sample(s)) + (s[0], s[1]) for s in train_s
    ]
    hold_data = [
        tuple(x.to(dev) for x in load_sample(s)) + (s[0], s[1]) for s in hold_s
    ]

    for ep in range(args.epochs):
        total = 0.0
        for inp, f3d, base, teacher, gt, vis, inv, sid, k in train_data:
            resid = net(inp[None])[0]
            inv_out = (f3d + resid).clamp(0, 1)
            out = vis * base + inv * inv_out
            loss = (
                (torch.abs(out - teacher) * inv).mean()
                + 0.2 * (torch.abs(out - gt) * inv).mean()
                + 2.0 * (torch.abs(out - base) * vis).mean()
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss)
        if ep % 100 == 0 or ep == args.epochs - 1:
            print(f"ep{ep:03d} loss={total / max(1, len(train_data)):.5f}")

    torch.save(net.state_dict(), os.path.join(args.out, "student.pt"))

    def eval_set(name, data):
        print(f"\n[{name}] n={len(data)}")
        rows = []
        with torch.no_grad():
            for inp, f3d, base, teacher, gt, vis, inv, sid, k in data:
                resid = net(inp[None])[0]
                out = vis * base + inv * (f3d + resid).clamp(0, 1)
                r = {
                    "sid": sid,
                    "k": k,
                    "base_inv": psnr_mask(base, gt, inv),
                    "f3d_inv": psnr_mask(f3d, gt, inv),
                    "teach_inv": psnr_mask(teacher, gt, inv),
                    "stud_inv": psnr_mask(out, gt, inv),
                    "stud_vis": psnr_mask(out, gt, vis),
                    "base_vis": psnr_mask(base, gt, vis),
                }
                rows.append(r)
                print(
                    f"{sid[:8]} f{k:03d} base={r['base_inv']:.2f} f3d={r['f3d_inv']:.2f} teacher={r['teach_inv']:.2f} student={r['stud_inv']:.2f} visΔ={r['stud_vis'] - r['base_vis']:+.2f}"
                )
        if rows:
            for key in ["base_inv", "f3d_inv", "teach_inv", "stud_inv"]:
                print(f"  mean {key}: {np.mean([r[key] for r in rows]):.2f}")
            print(
                f"  mean student-base inv Δ: {np.mean([r['stud_inv'] - r['base_inv'] for r in rows]):+.2f}"
            )
            print(
                f"  mean student vis Δ: {np.mean([r['stud_vis'] - r['base_vis'] for r in rows]):+.2f}"
            )
        return rows

    all_rows = {
        "train": eval_set("train", train_data),
        "holdout": eval_set("holdout", hold_data),
    }
    json.dump(all_rows, open(os.path.join(args.out, "results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
