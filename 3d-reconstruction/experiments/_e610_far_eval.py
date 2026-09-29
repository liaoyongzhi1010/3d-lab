"""E-610: re-evaluate E-609 cached renders on FAR-from-source frames only.

E-609 full-trajectory average had Flash3D winning because early frames are near the
source (Flash3D warps them near-perfectly). SS paper evaluates TARGET views far from
source (large disocclusion) -- the generative regime. We re-score the SAME cached
renders (flash3d_video / render_video ss / render_video grounded) but only on the
LAST portion of the trajectory (frames with index >= frac*N), zero re-run.

Reports Flash3D vs SS vs gated on far frames, per scene + mean.

Run (viewcrafter env):
  cd /root/projects/Scene-Splatter
  /root/miniconda3/envs/viewcrafter/bin/python _e610_far_eval.py --frac 0.5
"""

import os
import sys
import json
import glob
import math
import argparse
import numpy as np
from PIL import Image
import imageio.v2 as iio

SS_ROOT = "/root/projects/Scene-Splatter"
SCENES = os.path.join(SS_ROOT, "gen3r_ss_scenes")
sys.path.insert(0, SS_ROOT)
import torch
import lpips as lpips_lib
from skimage.metrics import structural_similarity as ssim_fn

_LP = lpips_lib.LPIPS(net="vgg").cuda().eval()
INTERVAL = 2


def crop5(x):
    H, W = x.shape[:2]
    ch, cw = int(math.ceil(0.05 * H)), int(math.ceil(0.05 * W))
    return x[ch : H - ch, cw : W - cw]


def psnr(a, b):
    v = ((a - b) ** 2).mean()
    return 100.0 if v < 1e-10 else float(-10 * np.log10(v))


def lp(a, b):
    pa = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).cuda() * 2 - 1
    pb = torch.from_numpy(b).permute(2, 0, 1).unsqueeze(0).cuda() * 2 - 1
    with torch.no_grad():
        return float(_LP(pa, pb).item())


def eval_video(video_path, scene, frac):
    if not os.path.exists(video_path):
        return None
    frames = [
        np.asarray(f).astype(np.float32) / 255 for f in iio.get_reader(video_path)
    ]
    H, W = frames[0].shape[:2]
    n = len(frames)
    start = int(frac * n)
    sd = os.path.join(SCENES, scene)
    ps, ss_, ls = [], [], []
    for k in range(start, n):
        gp = os.path.join(sd, "images", f"gt_frame_{k * INTERVAL:03d}.png")
        if not os.path.exists(gp):
            continue
        gt = (
            np.asarray(Image.open(gp).convert("RGB").resize((W, H))).astype(np.float32)
            / 255
        )
        pc, gc = crop5(frames[k]), crop5(gt)
        ps.append(psnr(pc, gc))
        ss_.append(float(ssim_fn(pc, gc, channel_axis=2, data_range=1.0)))
        ls.append(lp(pc, gc))
    if not ps:
        return None
    return {
        "psnr": float(np.mean(ps)),
        "ssim": float(np.mean(ss_)),
        "lpips": float(np.mean(ls)),
        "n": len(ps),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--frac",
        type=float,
        default=0.5,
        help="eval frames with index >= frac*N (far from source)",
    )
    a = ap.parse_args()
    e609 = json.load(open("/home/data/E-609/results.json"))
    out = {}
    for scene, modes in e609.items():
        out[scene] = {}
        # flash3d + ss from ss dir; grounded from grounded dir
        ssdir = modes.get("ss", {}).get("dir")
        grdir = modes.get("grounded", {}).get("dir")
        if ssdir:
            f3 = eval_video(os.path.join(ssdir, "flash3d_video.mp4"), scene, a.frac)
            rv = sorted(glob.glob(os.path.join(ssdir, "render_video_*.mp4")))
            ss = eval_video(rv[-1], scene, a.frac) if rv else None
            out[scene]["flash3d"] = f3
            out[scene]["ss"] = ss
        if grdir:
            rv = sorted(glob.glob(os.path.join(grdir, "render_video_*.mp4")))
            gd = eval_video(rv[-1], scene, a.frac) if rv else None
            out[scene]["grounded"] = gd
        line = f"[{scene}]"
        for m in ["flash3d", "ss", "grounded"]:
            if out[scene].get(m):
                line += f" {m}={out[scene][m]['psnr']:.2f}/{out[scene][m]['lpips']:.3f}"
        print(line, flush=True)

    print(f"\n===== E-610 FAR-frame eval (frac>={a.frac}) =====")
    for m in ["flash3d", "ss", "grounded"]:
        ps = [out[s][m]["psnr"] for s in out if out[s].get(m)]
        ls = [out[s][m]["lpips"] for s in out if out[s].get(m)]
        if ps:
            print(
                f"  {m:9s}: PSNR {np.mean(ps):.2f} LPIPS {np.mean(ls):.3f} (n={len(ps)})"
            )
    json.dump(out, open(f"/home/data/E-609/far_frac{a.frac}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
