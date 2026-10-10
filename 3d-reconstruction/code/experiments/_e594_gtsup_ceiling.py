"""E-594: Oracle ceiling of GENERATION-SUPERVISED per-scene reconstruction.

CORRECT paradigm (user-confirmed, NOT inpainting): a generator provides multi-view
supervision -> per-scene optimize ONE coherent 3DGS -> render novel views ->
standard whole-image PSNR/SSIM/LPIPS. This is Scene-Splatter's optimize_gaussian /
Lyra's self-distillation.

Before spending ViewCrafter compute, measure the ORACLE ceiling with zero diffusion
cost: if the multi-view supervision were PERFECT (= GT frames), how much does
per-scene 3DGS optimization improve NOVEL-view quality over the Flash3D feed-forward
baseline? This bounds what any generator can achieve and validates the paradigm.

Protocol (per scene, held-out to avoid cheating):
  - Flash3D forward from frame-0 -> initial 3DGS (feed-forward baseline).
  - HELD-IN frames = {0, 2, 4, ...} (even). HELD-OUT = {1, 3, 5, ...} (odd).
  - Variant FF  : Flash3D initial gaussians, NO optimization -> eval on held-out.
  - Variant OPT : optimize the SAME gaussians on held-in GT frames -> eval held-out.
  Standard whole-image metrics (5% crop, VGG-LPIPS) on held-out (novel) views.

If OPT >> FF on held-out, generation-supervised per-scene reconstruction has real
ceiling -> worth building (make the generator approach GT). If not, paradigm is weak.

Run (viewcrafter env):
  cd /root/projects/Scene-Splatter
  /root/miniconda3/envs/viewcrafter/bin/python _e594_gtsup_ceiling.py \
    --scenes test_0a9f2831a3e73de8 test_5f75672448394958 --iters 1500
"""

import os
import sys
import gzip
import glob
import json
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.argv_backup = sys.argv[:]
sys.argv = ["x"]  # gaussianSplatting arg parsers read sys.argv

SS_ROOT = "/root/projects/Scene-Splatter"
os.chdir(SS_ROOT)
sys.path.insert(0, SS_ROOT)

from omegaconf import OmegaConf
from gaussianSplatting.scene import GaussianModel
from gaussianSplatting.gaussian_renderer import render
from gaussianSplatting.arguments import ModelParams, PipelineParams, OptimizationParams
from gaussianSplatting.utils.loss_utils import l1_loss, ssim
from argparse import ArgumentParser
import lpips as lpips_lib
from skimage.metrics import structural_similarity as ssim_fn

sys.path.insert(0, os.path.join(SS_ROOT))
from scenesplatter import image2gaussian

_LP = None


def lpips_vgg(a, b, device):
    global _LP
    if _LP is None:
        _LP = lpips_lib.LPIPS(net="vgg").to(device).eval()
    pa = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device).float() * 2 - 1
    pb = torch.from_numpy(b).permute(2, 0, 1).unsqueeze(0).to(device).float() * 2 - 1
    with torch.no_grad():
        return float(_LP(pa, pb).item())


def crop5(x):
    import math

    H, W = x.shape[:2]
    ch, cw = int(math.ceil(0.05 * H)), int(math.ceil(0.05 * W))
    return x[ch : H - ch, cw : W - cw]


def psnr(a, b):
    v = ((a - b) ** 2).mean()
    return 100.0 if v < 1e-10 else float(-10 * np.log10(v))


def gt_of(scene_dir, k, w, h):
    p = os.path.join(scene_dir, "images", f"gt_frame_{k:03d}.png")
    return (
        np.asarray(Image.open(p).convert("RGB").resize((w, h))).astype(np.float32)
        / 255.0
    )


def optimize_on(gaussians, images, cameras, held_in_idx, iterations, device):
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args([])
    opt = op.extract(args)
    pipe = pp.extract(args)
    opt.iterations = iterations
    opt.position_lr_max_steps = iterations
    cameras_extent = 10
    gaussians.training_setup(opt)
    for iteration in range(1, iterations + 1):
        gaussians.update_learning_rate(iteration)
        vi = held_in_idx[iteration % len(held_in_idx)]
        pkg = render(cameras, vi, gaussians, pipe)
        ri, vpt, visf, radii = (
            pkg["render"],
            pkg["viewspace_points"],
            pkg["visibility_filter"],
            pkg["radii"],
        )
        tgt = images[vi].type(ri.dtype)
        Ll1 = l1_loss(ri, tgt)
        Ls = 1.0 - ssim(ri, tgt)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Ls
        loss.backward()
        with torch.no_grad():
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visf] = torch.max(
                    gaussians.max_radii2D[visf], radii[visf]
                )
                gaussians.add_densification_stats(vpt, visf)
                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    st = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, 0.005, cameras_extent, st
                    )
                if iteration % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity()
            if iteration < iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
    return gaussians


def eval_on(gaussians, cameras, scene_dir, held_out_idx, w, h, device):
    parser = ArgumentParser()
    pp = PipelineParams(parser)
    args = parser.parse_args([])
    pipe = pp.extract(args)
    res = {"psnr": [], "ssim": [], "lpips": []}
    with torch.no_grad():
        for k in held_out_idx:
            pkg = render(cameras, k, gaussians, pipe)
            pred = pkg["render"].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            gt = gt_of(scene_dir, k, w, h)
            pc, gc = crop5(pred), crop5(gt)
            res["psnr"].append(psnr(pc, gc))
            res["ssim"].append(float(ssim_fn(pc, gc, channel_axis=2, data_range=1.0)))
            res["lpips"].append(lpips_vgg(pc, gc, device))
    return {m: float(np.mean(v)) for m, v in res.items()}


def main():
    sys.argv = sys.argv_backup
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--nframes", type=int, default=25)
    a = ap.parse_args()
    device = torch.device("cuda:0")
    from hydra import initialize, compose

    with initialize(config_path="configs", version_base=None):
        cfg = compose(config_name="config")
    # UniDepth hub path: SS flash3d does abspath('..')+cfg.model.depth.root; pin absolute
    from omegaconf import open_dict

    with open_dict(cfg):
        cfg.model.depth.root = os.path.join(SS_ROOT, "flash3d", "UniDepth")
    # neutralize the abspath('..') prepend by making it already absolute (code joins abspath('..'), root;
    # os.path.join(abs, abspath) returns the second if absolute) -> safe.
    ckpt = os.path.join(SS_ROOT, "flash3d", "model_re10k_v2.pth")
    allres = {}
    for scene in a.scenes:
        sd = os.path.join(SS_ROOT, "gen3r_ss_scenes", scene)
        img0 = os.path.join(sd, "images", "0.png")
        cam0 = os.path.join(sd, "cameras", "camera0.pickle.gz")
        w = cfg.dataset.width
        h = cfg.dataset.height
        gaussians, image_list, camera_list = image2gaussian(
            cfg,
            device,
            num_render_frames=a.nframes,
            image_path=img0,
            camera_path=cam0,
            ckpt_dir=ckpt,
        )
        n = len(camera_list)
        held_in = [i for i in range(n) if i % 2 == 0]
        held_out = [i for i in range(1, n, 2)]
        # supervision images = GT frames (oracle)
        gt_imgs = []
        for k in range(n):
            g = gt_of(sd, k, w, h)
            gt_imgs.append(torch.from_numpy(g).permute(2, 0, 1).unsqueeze(0).to(device))
        gt_imgs = [x.squeeze(0) if x.dim() == 4 else x for x in gt_imgs]
        # FF baseline eval (no optimization)
        ff = eval_on(gaussians, camera_list, sd, held_out, w, h, device)
        # OPT eval (optimize on held-in GT)
        gaussians = optimize_on(
            gaussians, [g for g in gt_imgs], camera_list, held_in, a.iters, device
        )
        opt = eval_on(gaussians, camera_list, sd, held_out, w, h, device)
        allres[scene] = {"FF": ff, "OPT": opt, "n": n}
        print(f"\n[{scene}] held_out={len(held_out)}")
        print(
            f"  FF : psnr={ff['psnr']:.2f} ssim={ff['ssim']:.3f} lpips={ff['lpips']:.3f}"
        )
        print(
            f"  OPT: psnr={opt['psnr']:.2f} ssim={opt['ssim']:.3f} lpips={opt['lpips']:.3f}",
            flush=True,
        )

    def avg(key, sub):
        return float(np.mean([allres[s][key][sub] for s in allres]))

    print("\n===== E-594 GT-supervised per-scene ceiling =====")
    for key in ["FF", "OPT"]:
        print(
            f"  {key}: psnr={avg(key, 'psnr'):.2f} ssim={avg(key, 'ssim'):.3f} lpips={avg(key, 'lpips'):.3f}"
        )
    os.makedirs("/home/data/E-594_gtsup", exist_ok=True)
    json.dump(allres, open("/home/data/E-594_gtsup/result.json", "w"), indent=2)
    print("saved /home/data/E-594_gtsup/result.json")


if __name__ == "__main__":
    main()
