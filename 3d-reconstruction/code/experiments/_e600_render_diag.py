"""E-600: Localize WHERE Scene-Splatter loses 8dB vs Flash3D (render-conversion vs generation).

SS (15.65) << Flash3D (23.75). SS = Flash3D init gaussians -> GaussianModel ->
gaussianSplatting render + ViewCrafter generation + per-scene optimization. We must
find which step costs the 8dB before designing a fix.

Per scene, all frames, standard whole-image metrics (5% crop, VGG-LPIPS):
  A_f3d_render : image_list[k] = Flash3D's OWN color_gauss render      (= flash3d_video)
  B_gs_render  : gaussianSplatting render of the SAME converted gaussians, NO opt, NO gen
  C_gs_selfopt : B + per-scene optimize on Flash3D's own renders (no generation)

If B << A -> the Flash3D->GaussianModel->gsplat CONVERSION is broken (coordinate/
render mismatch), and that alone explains SS's collapse (generation is a red herring).
If B ~ A but C << B -> optimization itself hurts. If B~A, C~A -> generation is the culprit.

Run (viewcrafter env):
  cd /root/projects/Scene-Splatter
  /root/miniconda3/envs/viewcrafter/bin/python _e600_render_diag.py --scene test_0a9f2831a3e73de8
"""

import os
import sys
import json
import math
import argparse
import numpy as np
import torch
from PIL import Image

sys.argv_backup = sys.argv[:]
sys.argv = ["x"]
SS_ROOT = "/root/projects/Scene-Splatter"
os.chdir(SS_ROOT)
sys.path.insert(0, SS_ROOT)

from gaussianSplatting.gaussian_renderer import render
from gaussianSplatting.arguments import PipelineParams, OptimizationParams
from gaussianSplatting.utils.loss_utils import l1_loss, ssim
from argparse import ArgumentParser
import lpips as lpips_lib
from skimage.metrics import structural_similarity as ssim_fn
from scenesplatter import image2gaussian

_LP = None


def lp(a, b, device):
    global _LP
    if _LP is None:
        _LP = lpips_lib.LPIPS(net="vgg").to(device).eval()
    pa = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device).float() * 2 - 1
    pb = torch.from_numpy(b).permute(2, 0, 1).unsqueeze(0).to(device).float() * 2 - 1
    with torch.no_grad():
        return float(_LP(pa, pb).item())


def crop5(x):
    H, W = x.shape[:2]
    ch, cw = int(math.ceil(0.05 * H)), int(math.ceil(0.05 * W))
    return x[ch : H - ch, cw : W - cw]


def psnr(a, b):
    v = ((a - b) ** 2).mean()
    return 100.0 if v < 1e-10 else float(-10 * np.log10(v))


def gt_of(sd, k, w, h):
    p = os.path.join(sd, "images", f"gt_frame_{k:03d}.png")
    return (
        np.asarray(Image.open(p).convert("RGB").resize((w, h))).astype(np.float32)
        / 255.0
    )


def main():
    sys.argv = sys.argv_backup
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="test_0a9f2831a3e73de8")
    ap.add_argument("--iters", type=int, default=1000)
    a = ap.parse_args()
    device = torch.device("cuda:0")
    from hydra import initialize, compose

    with initialize(config_path="configs", version_base=None):
        cfg = compose(config_name="config")
    from omegaconf import open_dict

    with open_dict(cfg):
        cfg.model.depth.root = os.path.join(SS_ROOT, "flash3d", "UniDepth")
    cfg.dataset.height = 576
    cfg.dataset.width = 1024
    ckpt = os.path.join(SS_ROOT, "flash3d", "model_re10k_v2.pth")
    sd = os.path.join(SS_ROOT, "gen3r_ss_scenes", a.scene)
    img0 = os.path.join(sd, "images", "0.png")
    cam0 = os.path.join(sd, "cameras", "camera0.pickle.gz")
    import gzip, pickle

    nfr = len(pickle.load(gzip.open(cam0, "rb"))["poses"])
    w, h = cfg.dataset.width, cfg.dataset.height

    gaussians, image_list, camera_list = image2gaussian(
        cfg,
        device,
        num_render_frames=nfr,
        image_path=img0,
        camera_path=cam0,
        ckpt_dir=ckpt,
    )
    n = min(len(camera_list), nfr)

    parser = ArgumentParser()
    pp = PipelineParams(parser)
    args = parser.parse_args([])
    pipe = pp.extract(args)

    A, B = {"p": [], "l": []}, {"p": [], "l": []}
    with torch.no_grad():
        for k in range(n):
            gt = crop5(gt_of(sd, k, w, h))
            # A: Flash3D own render
            im = image_list[k]
            im = im.squeeze(0) if im.dim() == 4 else im
            a_img = crop5(im.permute(1, 2, 0).clamp(0, 1).cpu().numpy())
            A["p"].append(psnr(a_img, gt))
            A["l"].append(lp(a_img, gt, device))
            # B: gaussianSplatting render of converted gaussians
            pkg = render(camera_list, k, gaussians, pipe)
            b_img = crop5(pkg["render"].permute(1, 2, 0).clamp(0, 1).cpu().numpy())
            B["p"].append(psnr(b_img, gt))
            B["l"].append(lp(b_img, gt, device))

    # C: per-scene optimize on Flash3D's own renders (no generation)
    parser = ArgumentParser()
    op = OptimizationParams(parser)
    pp2 = PipelineParams(parser)
    args2 = parser.parse_args([])
    opt = op.extract(args2)
    pipe2 = pp2.extract(args2)
    opt.iterations = a.iters
    opt.position_lr_max_steps = a.iters
    sup = [
        (image_list[k].squeeze(0) if image_list[k].dim() == 4 else image_list[k]).to(
            device
        )
        for k in range(n)
    ]
    gaussians.training_setup(opt)
    for it in range(1, a.iters + 1):
        gaussians.update_learning_rate(it)
        vi = it % n
        pkg = render(camera_list, vi, gaussians, pipe2)
        ri, vpt, visf, radii = (
            pkg["render"],
            pkg["viewspace_points"],
            pkg["visibility_filter"],
            pkg["radii"],
        )
        tgt = sup[vi].type(ri.dtype)
        loss = (1.0 - opt.lambda_dssim) * l1_loss(ri, tgt) + opt.lambda_dssim * (
            1.0 - ssim(ri, tgt)
        )
        loss.backward()
        with torch.no_grad():
            if it < opt.densify_until_iter:
                gaussians.max_radii2D[visf] = torch.max(
                    gaussians.max_radii2D[visf], radii[visf]
                )
                gaussians.add_densification_stats(vpt, visf)
                if it > opt.densify_from_iter and it % opt.densification_interval == 0:
                    st = 20 if it > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, 0.005, 10, st
                    )
                if it % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity()
            if it < a.iters:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
    C = {"p": [], "l": []}
    with torch.no_grad():
        for k in range(n):
            gt = crop5(gt_of(sd, k, w, h))
            pkg = render(camera_list, k, gaussians, pipe2)
            c_img = crop5(pkg["render"].permute(1, 2, 0).clamp(0, 1).cpu().numpy())
            C["p"].append(psnr(c_img, gt))
            C["l"].append(lp(c_img, gt, device))

    def m(d, key):
        return float(np.mean(d[key]))

    res = {
        "A_f3d_render": {"psnr": m(A, "p"), "lpips": m(A, "l")},
        "B_gs_render_noopt": {"psnr": m(B, "p"), "lpips": m(B, "l")},
        "C_gs_selfopt_nogen": {"psnr": m(C, "p"), "lpips": m(C, "l")},
        "n_frames": n,
    }
    print(f"\n===== E-600 render diagnosis [{a.scene}] =====")
    for k, v in res.items():
        if isinstance(v, dict):
            print(f"  {k:22s}: psnr={v['psnr']:.2f} lpips={v['lpips']:.3f}")
    os.makedirs("/home/data/E-600_render_diag", exist_ok=True)
    json.dump(res, open(f"/home/data/E-600_render_diag/{a.scene}.json", "w"), indent=2)
    print("saved")


if __name__ == "__main__":
    main()
