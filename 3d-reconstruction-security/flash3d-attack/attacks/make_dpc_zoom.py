"""DPC attack qualitative figures with RED-BOX zoom insets on worst-degraded regions.

For each scene, renders clean vs attacked target views, auto-detects the region where
attacked deviates most from clean (via a blurred abs-diff heatmap), draws a red box there
on BOTH clean and attacked, and appends a magnified crop (zoom inset) side by side so the
degradation is visually obvious for presentations.

Layout per scene (one PNG):
  Row: [ Source(clean) | Source(attacked) | tgt_rand(clean)+redbox | tgt_rand(attacked)+redbox | zoom(clean) | zoom(attacked) ]

Run (Flash3D venv):
  python attacks/make_dpc_zoom.py --scenes 7 12 20 33 41 --epsilon 0.031373 \
    --lambda_src 8.0 --steps 100 --out_dir results/qual_zoom_indoor
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/root/projects/flash3d")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hydra import compose, initialize_config_dir
from models.model import GaussianPredictor
from datasets.util import create_datasets

from attacks.dpc_attack import sample_auxiliary_poses, DPCConfig
from attacks.make_dpc_perceptual import (
    _load,
    _make_cfg,
    _render_single,
    LPIPSLoss,
    dpc_perceptual_attack,
    _to_img,
    _psnr,
)

RED = (255, 40, 40)


def _find_worst_region(clean_img, att_img, box_frac=0.28):
    """Return (x0,y0,x1,y1) of the box (box_frac of min dim) with max mean abs-diff."""
    H, W = clean_img.shape[:2]
    bs = int(min(H, W) * box_frac)
    diff = np.abs(clean_img.astype(np.float32) - att_img.astype(np.float32)).mean(
        axis=2
    )
    # integral image for fast box-sum
    ii = np.zeros((H + 1, W + 1), dtype=np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(diff, axis=0), axis=1)

    def box_sum(y0, x0, y1, x1):
        return ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]

    best, best_xy = -1.0, (0, 0)
    step = max(8, bs // 6)
    for y0 in range(0, H - bs + 1, step):
        for x0 in range(0, W - bs + 1, step):
            s = box_sum(y0, x0, y0 + bs, x0 + bs)
            if s > best:
                best, best_xy = s, (x0, y0)
    x0, y0 = best_xy
    return x0, y0, x0 + bs, y0 + bs


def _draw_box(img, box, color=RED, width=4):
    im = Image.fromarray(img.copy())
    d = ImageDraw.Draw(im)
    d.rectangle(box, outline=color, width=width)
    return np.array(im)


def _crop_zoom(img, box, out_hw):
    x0, y0, x1, y1 = box
    crop = img[y0:y1, x0:x1]
    im = Image.fromarray(crop).resize((out_hw[1], out_hw[0]), Image.NEAREST)
    arr = np.array(im)
    # red border on the zoom to tie it to the box
    arr[:4, :] = RED
    arr[-4:, :] = RED
    arr[:, :4] = RED
    arr[:, -4:] = RED
    return arr


def _label(img, text):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
        )
    except Exception:
        font = ImageFont.load_default()
    d.rectangle([(0, 0), (im.width, 26)], fill=(0, 0, 0))
    d.text((4, 3), text, fill=(255, 255, 255), font=font)
    return np.array(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--split", default="splits/re10k_mine_filtered/test_files_present.txt"
    )
    ap.add_argument("--scenes", type=int, nargs="+", default=[7, 12, 20, 33, 41])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epsilon", type=float, default=0.031373)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lambda_src", type=float, default=8.0)
    ap.add_argument("--n_aux_poses", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda:0")
    cfg = _make_cfg(args.split)
    model = _load(cfg, device)
    _, loader = create_datasets(cfg, split="test")
    lpips_fn = LPIPSLoss(device).eval()

    scene_set = set(args.scenes)
    max_scene = max(args.scenes)
    idx = -1
    for inputs in loader:
        idx += 1
        if idx > max_scene:
            break
        if idx not in scene_set:
            continue
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
        if ("color", 1, 0) not in inputs:
            continue
        inputs["target_frame_ids"] = [1, 2, 3]
        clean_src = inputs[("color_aug", 0, 0)].detach()

        def render_fn(image, pose):
            return _render_single(model, inputs, image, pose)

        print(f"scene {idx}: attacking eps={args.epsilon:.3f} ...", flush=True)
        attacked_src = dpc_perceptual_attack(
            clean_src,
            render_fn,
            lpips_fn,
            epsilon=args.epsilon,
            steps=args.steps,
            lambda_src=args.lambda_src,
            n_aux_poses=args.n_aux_poses,
            device=device,
        )

        with torch.no_grad():
            ci = dict(inputs)
            ci[("color_aug", 0, 0)] = clean_src
            ci["target_frame_ids"] = [1, 2, 3]
            out_c = model(ci)
            ai = dict(inputs)
            ai[("color_aug", 0, 0)] = attacked_src
            ai["target_frame_ids"] = [1, 2, 3]
            out_a = model(ai)

        # widest target = frame 3 (tgt_rand) if present, else last available
        fid = (
            3
            if ("color_gauss", 3, 0) in out_c
            else max(f for f in [1, 2, 3] if ("color_gauss", f, 0) in out_c)
        )
        src_c = _to_img(out_c[("color_gauss", 0, 0)])
        src_a = _to_img(out_a[("color_gauss", 0, 0)])
        tgt_c = _to_img(out_c[("color_gauss", fid, 0)])
        tgt_a = _to_img(out_a[("color_gauss", fid, 0)])

        H, W = tgt_c.shape[:2]
        box = _find_worst_region(tgt_c, tgt_a)
        src_psnr = _psnr(src_c, src_a)
        tgt_psnr = _psnr(tgt_c, tgt_a)

        zoom_h = H
        zoom_c = _crop_zoom(tgt_c, box, (zoom_h, zoom_h))
        zoom_a = _crop_zoom(tgt_a, box, (zoom_h, zoom_h))
        tgt_c_box = _draw_box(tgt_c, box)
        tgt_a_box = _draw_box(tgt_a, box)

        panels = [
            _label(src_c, "Source clean"),
            _label(src_a, f"Source attacked {src_psnr:.1f}dB (imperceptible)"),
            _label(tgt_c_box, "Novel clean"),
            _label(tgt_a_box, f"Novel attacked {tgt_psnr:.1f}dB"),
            _label(zoom_c, "ZOOM clean"),
            _label(zoom_a, "ZOOM attacked"),
        ]
        Hh = min(p.shape[0] for p in panels)
        Ww = min(p.shape[1] for p in panels)
        grid = np.concatenate([p[:Hh, :Ww] for p in panels], axis=1)
        out_path = os.path.join(args.out_dir, f"scene{idx}_zoom.png")
        Image.fromarray(grid).save(out_path, quality=95)
        print(
            f"  -> {out_path} | src={src_psnr:.1f}dB tgt={tgt_psnr:.1f}dB box={box}",
            flush=True,
        )


if __name__ == "__main__":
    main()
