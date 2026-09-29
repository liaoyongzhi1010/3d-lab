"""E-679: render held-out EVAL frames from flash3d_init.ply.

Mirror of _e663_rerender_from_ply.py's camera-load + render path, but WITHOUT
the eval-frame exclusion: here we intentionally render the eval poses
[15,30,50,69,84,99] so a per-pixel router can be trained leave-one-scene-out.

Run (viewcrafter env):
  python _e679_render_evalframes.py --render-dir <dir with flash3d_init.ply + camera_list.pt> --out <dir>
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

SS_ROOT = "/root/projects/Scene-Splatter"
sys.path.insert(0, SS_ROOT)

EVAL_IDXS = [15, 30, 50, 69, 84, 99]


def move_cameras_to_cuda(cameras):
    for camera in cameras:
        for key, value in list(camera.items()):
            if torch.is_tensor(value):
                camera[key] = value.cuda()
    return cameras


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--init-ply", default=None, help="defaults to <render-dir>/flash3d_init.ply"
    )
    ap.add_argument(
        "--idxs", default=None, help="comma-separated; defaults to eval frames"
    )
    args = ap.parse_args()
    sys.argv = [sys.argv[0]]

    from argparse import ArgumentParser
    from gaussianSplatting.arguments import PipelineParams
    from gaussianSplatting.gaussian_renderer import render
    from gaussianSplatting.scene.gaussian_model import GaussianModel

    os.makedirs(args.out, exist_ok=True)
    init_ply = args.init_ply or os.path.join(args.render_dir, "flash3d_init.ply")
    cameras = move_cameras_to_cuda(
        torch.load(os.path.join(args.render_dir, "camera_list.pt"))
    )
    if args.idxs:
        idxs = [int(x) for x in args.idxs.split(",") if x != ""]
    else:
        idxs = list(EVAL_IDXS)

    gaussian = GaussianModel(3)
    gaussian.load_ply(init_ply)
    parser = ArgumentParser()
    pipe = PipelineParams(parser).extract(parser.parse_args([]))

    for j in idxs:
        with torch.no_grad():
            pkg = render(cameras, j, gaussian, pipe)
            rgb = pkg["render"].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        out_name = f"baseline_{j:03d}.png"
        Image.fromarray((rgb * 255).astype(np.uint8)).save(
            os.path.join(args.out, out_name)
        )
        print(
            f"[e679] view {j}: mean={rgb.mean():.4f} shape={rgb.shape} -> {out_name}",
            flush=True,
        )

    print(f"[e679] rendered {len(idxs)} eval frames -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
