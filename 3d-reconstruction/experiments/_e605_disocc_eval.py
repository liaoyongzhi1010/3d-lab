"""E-605: Flash3D on the DISOCC protocol (validate protocol vs SS Table 1).

SS Table 1: Flash3D Easy 17.94 / Hard 14.41. My earlier 23.75 was wrong (full-49-frame
average incl. near-source frames). The correct protocol evaluates the TARGET
(large-disocclusion) frame. This script renders Flash3D's feed-forward gaussians at the
target pose of each disocc_scenes/* and compares to gt_target.png, split by easy/hard.
If Flash3D lands near 17.94/14.41, the protocol is aligned and we can then run SS+gate.

Run (viewcrafter env):
  cd /root/projects/Scene-Splatter
  /root/miniconda3/envs/viewcrafter/bin/python _e605_disocc_eval.py
"""

import os
import sys
import gzip
import glob
import json
import pickle
import math
import numpy as np
import torch
from PIL import Image

sys.argv_backup = sys.argv[:]
sys.argv = ["x"]
SS_ROOT = "/root/projects/Scene-Splatter"
os.chdir(SS_ROOT)
sys.path.insert(0, SS_ROOT)

from gaussianSplatting.gaussian_renderer import render
from gaussianSplatting.arguments import PipelineParams
from argparse import ArgumentParser
import lpips as lpips_lib
from skimage.metrics import structural_similarity as ssim_fn
from scenesplatter import image2gaussian

_LP = lpips_lib.LPIPS(net="vgg").cuda().eval()


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


def main():
    sys.argv = sys.argv_backup
    device = torch.device("cuda:0")
    from hydra import initialize, compose
    from omegaconf import open_dict

    with initialize(config_path="configs", version_base=None):
        cfg = compose(config_name="config")
    with open_dict(cfg):
        cfg.model.depth.root = os.path.join(SS_ROOT, "flash3d", "UniDepth")
    cfg.dataset.height = 576
    cfg.dataset.width = 1024
    ckpt = os.path.join(SS_ROOT, "flash3d", "model_re10k_v2.pth")

    scenes = sorted(glob.glob(os.path.join(SS_ROOT, "disocc_scenes", "*/")))
    parser = ArgumentParser()
    pp = PipelineParams(parser)
    pipe = pp.extract(parser.parse_args([]))
    res = {"easy": [], "hard": []}
    for sd in scenes:
        name = os.path.basename(sd.rstrip("/"))
        band = "easy" if name.startswith("easy") else "hard"
        cam0 = os.path.join(sd, "cameras", "camera0.pickle.gz")
        img0 = os.path.join(sd, "images", "0.png")
        gtp = os.path.join(sd, "images", "gt_target.png")
        if not (os.path.exists(cam0) and os.path.exists(gtp)):
            continue
        nfr = len(pickle.load(gzip.open(cam0, "rb"))["poses"])
        g, il, cl = image2gaussian(
            cfg,
            device,
            num_render_frames=nfr,
            image_path=img0,
            camera_path=cam0,
            ckpt_dir=ckpt,
        )
        # target = last camera in trajectory
        with torch.no_grad():
            pkg = render(cl, len(cl) - 1, g, pipe)
        pred = pkg["render"].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        H, W = pred.shape[:2]
        gt = (
            np.asarray(Image.open(gtp).convert("RGB").resize((W, H))).astype(np.float32)
            / 255.0
        )
        pc, gc = crop5(pred), crop5(gt)
        r = {
            "scene": name,
            "psnr": psnr(pc, gc),
            "ssim": float(ssim_fn(pc, gc, channel_axis=2, data_range=1.0)),
            "lpips": lp(pc, gc),
        }
        res[band].append(r)
        print(
            f"  [{band}] {name}: psnr={r['psnr']:.2f} ssim={r['ssim']:.3f} lpips={r['lpips']:.3f}",
            flush=True,
        )

    print("\n===== E-605 Flash3D on DISOCC protocol =====")
    for band in ["easy", "hard"]:
        if res[band]:
            p = np.mean([x["psnr"] for x in res[band]])
            s = np.mean([x["ssim"] for x in res[band]])
            l = np.mean([x["lpips"] for x in res[band]])
            print(
                f"  {band}: PSNR {p:.2f} SSIM {s:.3f} LPIPS {l:.3f} (n={len(res[band])})"
            )
    print("  (SS paper Table1: Flash3D Easy 17.94/0.682/0.160, Hard 14.41/0.599/0.370)")
    os.makedirs("/home/data/E-605_disocc", exist_ok=True)
    json.dump(res, open("/home/data/E-605_disocc/flash3d.json", "w"), indent=2)


if __name__ == "__main__":
    main()
