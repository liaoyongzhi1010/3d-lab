"""Generate visually striking DPC figures: depth map comparison + structural artifacts.

Two panels per scene:
  Panel A: Source image (clean vs attacked side-by-side) — should look identical
  Panel B: Depth map (clean vs attacked, viridis colormap) — should look dramatically different
  Panel C: Novel-view render (clean vs attacked) at the widest target — show structural damage

Run on server (Flash3D venv):
  python /root/flash3d-attack/attacks/make_dpc_visual_v3.py --scenes 0 3 7 12 20 \
    --out_dir /root/flash3d-attack/results/qual_v3
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/root/projects/flash3d")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hydra import compose, initialize_config_dir
from models.model import GaussianPredictor
from datasets.util import create_datasets

from attacks.dpc_attack import DPCConfig, dpc_attack

FLASH3D_CKPT = "/root/projects/flash3d/checkpoints/model_re10k_v2.pth"


def _load(cfg, device):
    model = GaussianPredictor(cfg).to(device)
    state = torch.load(FLASH3D_CKPT, map_location="cpu")
    sd = state["model"] if "model" in state else state
    current = model.state_dict()
    filtered = {}
    for k, v in sd.items():
        if "backproject_depth" in k:
            if k in current:
                filtered[k] = current[k].clone()
        else:
            filtered[k] = v
    model.load_state_dict(filtered, strict=False)
    model.set_eval()
    return model


def _make_cfg(split):
    with initialize_config_dir(
        config_dir="/root/projects/flash3d/configs", version_base=None
    ):
        return compose(
            config_name="config",
            overrides=[
                "+experiment=layered_re10k",
                "+dataset.crop_border=true",
                f"dataset.test_split_path={split}",
                "model.depth.version=v1",
                "data_loader.batch_size=1",
                "data_loader.num_workers=1",
            ],
        )


def _render_single(model, inputs, src_image, relative_pose):
    local = dict(inputs)
    local[("color_aug", 0, 0)] = src_image
    local["target_frame_ids"] = [1]
    outputs = model.models["unidepth_extended"](local)
    model.compute_gauss_means(local, outputs)
    B = src_image.shape[0]
    device = src_image.device
    dtype = outputs["gauss_means"].dtype
    outputs[("cam_T_cam", 0, 1)] = (
        relative_pose.to(device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
    )
    model.render_images(local, outputs)
    return outputs[("color_gauss", 1, 0)]


def _full_forward(model, inputs, src_img):
    """Full forward: returns outputs dict with renders + depth."""
    eval_inputs = {
        kk: (vv.clone() if isinstance(vv, torch.Tensor) else vv)
        for kk, vv in inputs.items()
    }
    eval_inputs[("color_aug", 0, 0)] = src_img.detach()
    eval_inputs["target_frame_ids"] = [1, 2, 3]
    with torch.no_grad():
        out = model(eval_inputs)
    return out


def _to_img(t):
    if t.dim() == 4:
        t = t[0]
    t = t.detach().clamp(0, 1).cpu().numpy()
    return (t.transpose(1, 2, 0) * 255).astype(np.uint8)


def _depth_to_colormap(depth_tensor, vmin=None, vmax=None):
    """Convert depth (B,1,H,W) or (1,H,W) to a turbo/viridis-like colormap image."""
    if depth_tensor.dim() == 4:
        d = depth_tensor[0, 0]
    elif depth_tensor.dim() == 3:
        d = depth_tensor[0]
    else:
        d = depth_tensor
    d = d.detach().cpu().numpy()
    if vmin is None:
        vmin = np.percentile(d, 2)
    if vmax is None:
        vmax = np.percentile(d, 98)
    d_norm = np.clip((d - vmin) / (vmax - vmin + 1e-8), 0, 1)
    # Turbo-like colormap (R-Y-G-C-B progression)
    r = np.clip(np.where(d_norm < 0.5, 2 * d_norm * 2, 2 - 2 * d_norm), 0, 1)
    # Simplified turbo: use a proper LUT
    cm = _turbo_colormap(d_norm)
    return (cm * 255).astype(np.uint8)


def _turbo_colormap(x):
    """Attempt at turbo colormap using piecewise interpolation."""
    # Anchor colors for turbo (simplified 8-point)
    anchors = np.array(
        [
            [0.18995, 0.07176, 0.23217],  # 0.0 dark blue
            [0.09140, 0.20640, 0.53371],  # ~0.14
            [0.12897, 0.56560, 0.55060],  # ~0.29 teal
            [0.21468, 0.78971, 0.33490],  # ~0.43 green
            [0.60099, 0.89640, 0.12599],  # ~0.57 yellow-green
            [0.88989, 0.79380, 0.10437],  # ~0.71 yellow
            [0.97610, 0.46711, 0.05725],  # ~0.86 orange
            [0.70567, 0.01555, 0.15023],  # 1.0 dark red
        ]
    )
    positions = np.linspace(0, 1, len(anchors))
    h, w = x.shape
    flat = x.flatten()
    rgb = np.zeros((flat.shape[0], 3))
    for c in range(3):
        rgb[:, c] = np.interp(flat, positions, anchors[:, c])
    return rgb.reshape(h, w, 3)


def _psnr(a, b):
    """PSNR between two uint8 images."""
    diff = (a.astype(np.float32) / 255.0 - b.astype(np.float32) / 255.0) ** 2
    mse = diff.mean()
    if mse < 1e-10:
        return 99.0
    return -10.0 * np.log10(mse)


def _add_label(img_array, text, position="top"):
    """Add white text on dark strip at top."""
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
        )
    except Exception:
        font = ImageFont.load_default()
    if position == "top":
        draw.rectangle([(0, 0), (img.width, 28)], fill=(0, 0, 0))
        draw.text((6, 4), text, fill=(255, 255, 255), font=font)
    return np.array(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--split", default="splits/re10k_mine_filtered/test_files_present.txt"
    )
    ap.add_argument("--scenes", type=int, nargs="+", default=[0, 3, 7, 12, 20])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epsilon", type=float, default=16.0 / 255.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--lambda_src", type=float, default=3.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda:0")
    cfg = _make_cfg(args.split)
    pad = cfg.dataset.pad_border_aug
    model = _load(cfg, device)
    _, loader = create_datasets(cfg, split="test")
    dpc_cfg = DPCConfig(
        epsilon=args.epsilon, steps=args.steps, lambda_src=args.lambda_src
    )

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

        attacked_src, delta, info = dpc_attack(clean_src, render_fn, dpc_cfg)

        # Full forward for both
        out_clean = _full_forward(model, inputs, clean_src)
        out_att = _full_forward(model, inputs, attacked_src)

        # Source images
        src_clean_img = _to_img(clean_src)
        src_att_img = _to_img(attacked_src)

        # Depth maps (use shared vmin/vmax for fair comparison)
        scale = model.cfg.model.scales[0]
        d_clean = out_clean[("depth", scale)]
        d_att = out_att[("depth", scale)]
        # Unpad
        if pad:
            d_clean_u = d_clean[:, :, pad:-pad, pad:-pad]
            d_att_u = d_att[:, :, pad:-pad, pad:-pad]
        else:
            d_clean_u = d_clean
            d_att_u = d_att
        # Shared range for depth visualization
        d_all = torch.cat([d_clean_u.flatten(), d_att_u.flatten()])
        vmin = float(d_all.quantile(0.02))
        vmax = float(d_all.quantile(0.98))
        depth_clean_img = _depth_to_colormap(d_clean_u, vmin, vmax)
        depth_att_img = _depth_to_colormap(d_att_u, vmin, vmax)

        # Novel view renders (pick the worst target)
        psnrs = {}
        for fid in [1, 2, 3]:
            pk = ("color_gauss", fid, 0)
            if pk in out_clean and pk in out_att:
                rc = _to_img(out_clean[pk])
                ra = _to_img(out_att[pk])
                psnrs[fid] = (_psnr(rc, ra), rc, ra)
        # Pick worst (lowest PSNR = most damaged)
        worst_fid = min(psnrs, key=lambda f: psnrs[f][0])
        worst_psnr, novel_clean, novel_att = psnrs[worst_fid]
        fid_names = {1: "tgt5", 2: "tgt10", 3: "tgt_rand"}

        # Resize all to same height
        H = min(src_clean_img.shape[0], depth_clean_img.shape[0], novel_clean.shape[0])
        W = min(src_clean_img.shape[1], depth_clean_img.shape[1], novel_clean.shape[1])

        def _resize(img, h, w):
            return np.array(Image.fromarray(img).resize((w, h), Image.LANCZOS))

        src_clean_img = _resize(src_clean_img, H, W)
        src_att_img = _resize(src_att_img, H, W)
        depth_clean_img = _resize(depth_clean_img, H, W)
        depth_att_img = _resize(depth_att_img, H, W)
        novel_clean = _resize(novel_clean, H, W)
        novel_att = _resize(novel_att, H, W)

        # Add labels
        src_clean_img = _add_label(src_clean_img, "Source (clean)")
        src_att_img = _add_label(src_att_img, "Source (attacked) — looks the same!")
        depth_clean_img = _add_label(depth_clean_img, "Depth (clean)")
        depth_att_img = _add_label(
            depth_att_img, "Depth (attacked) — completely wrong!"
        )
        novel_clean = _add_label(novel_clean, f"Novel {fid_names[worst_fid]} (clean)")
        novel_att = _add_label(
            novel_att,
            f"Novel {fid_names[worst_fid]} (attacked) PSNR={worst_psnr:.1f}dB",
        )

        # Layout: 3 rows x 2 cols
        # Row 1: source clean | source attacked
        # Row 2: depth clean | depth attacked
        # Row 3: novel clean | novel attacked
        row1 = np.concatenate([src_clean_img, src_att_img], axis=1)
        row2 = np.concatenate([depth_clean_img, depth_att_img], axis=1)
        row3 = np.concatenate([novel_clean, novel_att], axis=1)

        grid = np.concatenate([row1, row2, row3], axis=0)

        out_path = os.path.join(args.out_dir, f"scene{idx}_v3.png")
        Image.fromarray(grid).save(out_path, quality=95)

        src_psnr = _psnr(_to_img(clean_src)[:H, :W], _to_img(attacked_src)[:H, :W])
        print(
            f"scene {idx}: saved {out_path} | "
            f"src_psnr={src_psnr:.1f} novel({fid_names[worst_fid]})_psnr={worst_psnr:.1f} | "
            f"linf={info['linf']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
