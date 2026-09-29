"""E-598: per-scene reconstruction under DIFFERENT supervision sources (the study).

Same held-out protocol as E-594. We compare what supervision the per-scene 3DGS
optimization sees, to quantify how far each source gets toward the GT oracle:
  --sup gt        : real GT frames        (ORACLE upper bound, E-594 = 17.31)
  --sup ff        : Flash3D's own initial renders (NO new info, lower bound)
  --sup video     : frames from a supervision video (e.g. ViewCrafter output_video),
                    aligned to held-in cameras -> the REAL generation-supervised number.

held_in = {0,2,4,...} used for supervision; held_out = {1,3,...} always evaluated
against REAL GT (standard whole-image PSNR/SSIM/LPIPS, 5% crop, VGG). This makes all
supervision sources directly comparable on identical novel views.

Run (viewcrafter env):
  cd /root/projects/Scene-Splatter
  /root/miniconda3/envs/viewcrafter/bin/python _e598_sup_study.py \
    --scenes test_0a9f2831a3e73de8 --iters 1000 --sup ff
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import torch
from PIL import Image
import imageio.v2 as iio

sys.argv_backup = sys.argv[:]
sys.argv = ["x"]
SS_ROOT = "/root/projects/Scene-Splatter"
os.chdir(SS_ROOT)
sys.path.insert(0, SS_ROOT)

from gaussianSplatting.gaussian_renderer import render
from gaussianSplatting.arguments import ModelParams, PipelineParams, OptimizationParams
from gaussianSplatting.utils.loss_utils import l1_loss, ssim
from argparse import ArgumentParser
import lpips as lpips_lib
from skimage.metrics import structural_similarity as ssim_fn
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


def gt_of(sd, k, w, h):
    p = os.path.join(sd, "images", f"gt_frame_{k:03d}.png")
    return (
        np.asarray(Image.open(p).convert("RGB").resize((w, h))).astype(np.float32)
        / 255.0
    )


def read_video_frames(path, w, h):
    frames = []
    for f in iio.get_reader(path):
        im = Image.fromarray(f).convert("RGB").resize((w, h))
        frames.append(np.asarray(im).astype(np.float32) / 255.0)
    return frames


def optimize_on(gaussians, images, cameras, held_in_idx, iterations, device):
    parser = ArgumentParser()
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args([])
    opt = op.extract(args)
    pipe = pp.extract(args)
    opt.iterations = iterations
    opt.position_lr_max_steps = iterations
    gaussians.training_setup(opt)
    for it in range(1, iterations + 1):
        gaussians.update_learning_rate(it)
        vi = held_in_idx[it % len(held_in_idx)]
        pkg = render(cameras, vi, gaussians, pipe)
        ri, vpt, visf, radii = (
            pkg["render"],
            pkg["viewspace_points"],
            pkg["visibility_filter"],
            pkg["radii"],
        )
        tgt = images[vi].type(ri.dtype)
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
            if it < iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
    return gaussians


def eval_on(gaussians, cameras, sd, held_out_idx, w, h, device):
    parser = ArgumentParser()
    pp = PipelineParams(parser)
    args = parser.parse_args([])
    pipe = pp.extract(args)
    res = {"psnr": [], "ssim": [], "lpips": []}
    with torch.no_grad():
        for k in held_out_idx:
            pkg = render(cameras, k, gaussians, pipe)
            pred = pkg["render"].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            gt = gt_of(sd, k, w, h)
            pc, gc = crop5(pred), crop5(gt)
            res["psnr"].append(psnr(pc, gc))
            res["ssim"].append(float(ssim_fn(pc, gc, channel_axis=2, data_range=1.0)))
            res["lpips"].append(lpips_vgg(pc, gc, device))
    return {m: float(np.mean(v)) for m, v in res.items()}


def main():
    sys.argv = sys.argv_backup
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--nframes", type=int, default=25)
    ap.add_argument("--sup", choices=["gt", "ff", "video"], default="ff")
    ap.add_argument(
        "--split",
        choices=["interp", "extrap"],
        default="interp",
        help="interp: held_out=odd adjacent (easy). extrap: held_in=first half, held_out=second half (exposes generator value).",
    )
    ap.add_argument(
        "--video_dir", default=None, help="dir with output_video_*.mp4 for --sup video"
    )
    a = ap.parse_args()
    device = torch.device("cuda:0")
    from hydra import initialize, compose

    with initialize(config_path="configs", version_base=None):
        cfg = compose(config_name="config")
    from omegaconf import open_dict

    with open_dict(cfg):
        cfg.model.depth.root = os.path.join(SS_ROOT, "flash3d", "UniDepth")
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
        if a.split == "extrap":
            # held_in = first 60% (near, continuous), held_out = last 40% (extrapolation)
            cut = int(round(n * 0.6))
            held_in = list(range(0, cut))
            held_out = list(range(cut, n))
        else:
            held_in = [i for i in range(n) if i % 2 == 0]
            held_out = [i for i in range(1, n, 2)]
        # build supervision images
        if a.sup == "gt":
            sup = [
                torch.from_numpy(gt_of(sd, k, w, h)).permute(2, 0, 1).to(device)
                for k in range(n)
            ]
        elif a.sup == "ff":
            sup = [
                image_list[k].squeeze(0).to(device)
                if image_list[k].dim() == 4
                else image_list[k].to(device)
                for k in range(n)
            ]
        else:
            vids = sorted(glob.glob(os.path.join(a.video_dir, "output_video_*.mp4")))
            frames = []
            for v in vids:
                frames += read_video_frames(v, w, h)
            frames = frames[:n]
            while len(frames) < n:
                frames.append(frames[-1])
            sup = [
                torch.from_numpy(frames[k]).permute(2, 0, 1).to(device)
                for k in range(n)
            ]
        ff = eval_on(gaussians, camera_list, sd, held_out, w, h, device)
        gaussians = optimize_on(gaussians, sup, camera_list, held_in, a.iters, device)
        opt = eval_on(gaussians, camera_list, sd, held_out, w, h, device)
        allres[scene] = {"FF": ff, "OPT": opt, "sup": a.sup}
        print(f"\n[{scene}] sup={a.sup} held_out={len(held_out)}")
        print(
            f"  FF : psnr={ff['psnr']:.2f} ssim={ff['ssim']:.3f} lpips={ff['lpips']:.3f}"
        )
        print(
            f"  OPT: psnr={opt['psnr']:.2f} ssim={opt['ssim']:.3f} lpips={opt['lpips']:.3f}",
            flush=True,
        )

    def avg(k, s):
        return float(np.mean([allres[x][k][s] for x in allres]))

    print(f"\n===== E-598 sup={a.sup} ({len(allres)} scenes) =====")
    for k in ["FF", "OPT"]:
        print(
            f"  {k}: psnr={avg(k, 'psnr'):.2f} ssim={avg(k, 'ssim'):.3f} lpips={avg(k, 'lpips'):.3f}"
        )
    os.makedirs("/home/data/E-598_sup_study", exist_ok=True)
    json.dump(allres, open(f"/home/data/E-598_sup_study/{a.sup}.json", "w"), indent=2)
    print(f"saved /home/data/E-598_sup_study/{a.sup}.json")


if __name__ == "__main__":
    main()
